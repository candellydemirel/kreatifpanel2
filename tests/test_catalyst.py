"""Öngörü motoru testleri (ağ yok; sabit örnek yanıtlar ve sentetik fiyatlar)."""

import json

import numpy as np
import pandas as pd
import pytest

from kreatifbot.intel.catalyst import (
    Fundamentals, FundamentalsProvider, InsightEngine, InsightStore, build_insight, candidate_bases,
    detect_catalysts, is_rumor, score_catalysts, technical_check,
)
from kreatifbot.intel.config import CatalystConfig, IntelConfig
from kreatifbot.intel.execution import PaperVenue
from kreatifbot.intel.news import NewsItem, NewsMonitor, NewsStore
from kreatifbot.intel.signal_store import SignalStore
from kreatifbot.telegram import TelegramClient, TelegramNotifier

from .conftest import make_ohlcv
from .test_intel import FakeMarket, cfg_1h, resample, with_taker

NOW = pd.Timestamp.now(tz="UTC")


def item(title, sym, hours_ago=1.0, source="CoinDesk", iid=None):
    return NewsItem(iid or f"{title}{source}", source, title, published_at=(NOW - pd.Timedelta(hours=hours_ago))
                    .isoformat(), symbols=[sym])


def uptrend_1h(n=300, seed=2, start_price=10.0):
    """Son 24 saatte hacmi artan yükseliş (sentetik, yalnızca test)."""
    df = make_ohlcv(n, seed=seed, start=start_price)
    k = np.linspace(1.0, 1.25, n)
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].iloc[0] * k * (df[c] / df["close"].iloc[0]) ** 0.05
    df["high"] = df[["open", "close"]].max(axis=1) * 1.003
    df["low"] = df[["open", "close"]].min(axis=1) * 0.997
    df["volume"] = 1_000_000.0
    df.loc[df.index[-24:], "volume"] = 2_500_000.0
    df["quote_volume"] = df["volume"] * df["close"]
    shift = NOW.floor("h") - pd.to_datetime(df["open_time"].iloc[-1], utc=True) - pd.Timedelta(hours=1)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True) + shift
    df["close_time"] = df["open_time"] + pd.Timedelta(minutes=59, seconds=59.999)
    return df


def test_detect_catalysts_and_rumor():
    assert ("PARTNERSHIP", 1.0) in detect_catalysts("Chainlink partners with SWIFT on settlement")
    assert any(t == "PARTNERSHIP" for t, _ in detect_catalysts("XYZ signs agreement with Visa"))
    assert any(t == "INSTITUTIONAL" for t, _ in detect_catalysts("XYZ signs agreement with Visa"))
    assert detect_catalysts("Mainnet goes live for ABC")[0][0] == "MAINNET_UPGRADE"
    assert detect_catalysts("ABC token unlock of $200M next week")[0] == ("TOKEN_UNLOCK", -0.6)
    assert detect_catalysts("Price analysis for today") == []
    assert is_rumor("ABC reportedly in talks with Google") and not is_rumor("ABC partners with Google")


def test_score_catalysts_sources_decay_and_negatives():
    cfg = CatalystConfig()
    one = score_catalysts("ABC", [item("ABC partners with Microsoft", "ABC")], cfg)
    two = score_catalysts("ABC", [item("ABC partners with Microsoft", "ABC"),
                                  item("ABC strikes partnership with Microsoft", "ABC", source="Decrypt")], cfg)
    assert 0 < one.score < two.score and two.sources == ["CoinDesk", "Decrypt"]
    old = score_catalysts("ABC", [item("ABC partners with Microsoft", "ABC", hours_ago=60)], cfg)
    assert old.score < one.score  # eski haber etkisi azalır
    expired = score_catalysts("ABC", [item("ABC partners with Microsoft", "ABC", hours_ago=100)], cfg)
    assert expired.score == 0
    rumor = score_catalysts("ABC", [item("ABC reportedly partners with Microsoft", "ABC")], cfg)
    assert rumor.score < one.score and rumor.rumor_only
    neg = score_catalysts("ABC", [item("ABC protocol exploited, funds drained", "ABC")], cfg)
    assert neg.score == 0 and neg.negative > 50
    other = score_catalysts("XYZ", [item("ABC partners with Microsoft", "ABC")], cfg)
    assert other.score == 0


class _R:
    def __init__(self, js, status=200):
        self._js, self.status_code = js, status

    def json(self):
        return self._js


class FakeAPISession:
    def __init__(self, llama=True, cg=True):
        self.llama, self.cg, self.calls = llama, cg, []

    def get(self, url, params=None, timeout=None, headers=None):
        self.calls.append(url)
        if "llama" in url:
            if not self.llama:
                return _R({}, 503)
            return _R([{"name": "Abc Protocol", "symbol": "ABC", "tvl": 250_000_000, "change_7d": 18.0,
                        "mcap": 200_000_000, "category": "Dexes"},
                       {"name": "small", "symbol": "ABC", "tvl": 1_000, "change_7d": 0}])
        if "search" in url:
            if not self.cg:
                return _R({}, 429)
            return _R({"coins": [{"id": "abc-fake", "symbol": "ABC", "market_cap_rank": 900},
                                 {"id": "abc-real", "symbol": "ABC", "market_cap_rank": 85}]})
        if "/coins/abc-real" in url:
            return _R({"market_cap_rank": 85, "genesis_date": "2020-01-01", "developer_data":
                       {"commit_count_4_weeks": 120}, "links": {"whitepaper": "https://abc.io/wp.pdf"},
                       "categories": ["DeFi"]})
        return _R({}, 404)


def test_fundamentals_provider():
    fp = FundamentalsProvider(FakeAPISession())
    fd = fp.get("ABC")
    assert fd.tvl == 250_000_000 and fd.mcap_tvl == pytest.approx(0.8)
    assert fd.dev_commits_4w == 120 and fd.rank == 85 and fd.whitepaper.endswith("wp.pdf")
    assert 60 < fd.score <= 100 and fd.coverage > 0.8
    assert any("Whitepaper içeriği otomatik değerlendirilmez" in n for n in fd.notes)
    fp.get("ABC")
    assert sum("llama" in u for u in fp.session.calls) == 1  # önbellek
    none = FundamentalsProvider(FakeAPISession(llama=False, cg=False)).get("ABC")
    assert none.score is None and none.coverage == 0  # sahte değer yok


def test_technical_check_and_priced_in():
    cfg = CatalystConfig(min_quote_volume_24h=1000)
    df = uptrend_1h()
    h4 = resample(df, "4h")
    ok = technical_check(df, h4, (NOW - pd.Timedelta(hours=2)).isoformat(), cfg)
    assert ok.ok, ok.notes
    assert ok.rvol_24h > 2 and ok.move_since_news_pct is not None
    chased = technical_check(df, h4, (NOW - pd.Timedelta(hours=250)).isoformat(), CatalystConfig(
        min_quote_volume_24h=1000, max_chase_pct=5))
    assert not chased.ok and any("fiyatlanmış" in n for n in chased.notes)
    down = df.iloc[::-1].reset_index(drop=True)
    down["open_time"], down["close_time"] = df["open_time"], df["close_time"]
    bad = technical_check(down, None, "", cfg)
    assert not bad.ok


def test_build_insight_signals():
    cfg = CatalystConfig(min_quote_volume_24h=1000)
    df = uptrend_1h()
    good_news = [item("ABC partners with Microsoft", "ABC", 2), item("ABC strikes partnership with Microsoft", "ABC",
                                                                    3, "Decrypt"),
                 item("ABC mainnet goes live", "ABC", 4, "Cointelegraph")]
    fd = FundamentalsProvider(FakeAPISession()).get("ABC")
    ins = build_insight("ABCUSDT", "ABC", good_news, fd, df, resample(df, "4h"), cfg)
    assert ins.signal == "AL", ins.explain()
    assert ins.stop < ins.entry < ins.targets[0] and ins.potential >= cfg.min_potential
    assert "ortaklık" in ins.explain() and "Whitepaper" in ins.explain()
    weak = build_insight("ABCUSDT", "ABC", [item("ABC reportedly partners with Microsoft", "ABC")], fd, df, None, cfg)
    assert weak.signal != "AL"
    hacked = build_insight("ABCUSDT", "ABC", good_news + [item("ABC exploited, $50M drained", "ABC", 1)], fd, df,
                           None, cfg)
    assert hacked.signal != "AL" and any("Olumsuz" in w for w in hacked.warnings)
    no_data = build_insight("ABCUSDT", "ABC", good_news, Fundamentals("ABC"), None, None, cfg)
    assert no_data.signal != "AL" and no_data.technical.score is None


def test_candidate_bases_only_tradable():
    items = [item("ABC partners with Microsoft", "ABC"), item("XYZ partners with Google", "XYZ"),
             item("Bitcoin price analysis", "BTC")]
    assert candidate_bases(items, CatalystConfig(), {"ABC", "BTC"}) == ["ABC"]


def test_insight_store_dedupe_and_outcomes(tmp_path):
    cfg = CatalystConfig(min_quote_volume_24h=1000)
    df = uptrend_1h()
    store = InsightStore(tmp_path / "i.db")
    ins = build_insight("ABCUSDT", "ABC", [item("ABC partners with Microsoft", "ABC")], Fundamentals("ABC"), df,
                        None, cfg)
    ins.created_at = (NOW - pd.Timedelta(hours=80)).isoformat()
    assert store.save(ins)
    assert not store.save(ins)  # aynı sinyal tekrar kaydedilmez

    class Client:
        def klines(self, symbol, interval, limit=80, start_time=None):
            k = df.tail(80).reset_index(drop=True).copy()
            k["close"] = ins.technical.price * np.linspace(1.0, 1.10, len(k))
            k["high"], k["low"] = k["close"] * 1.01, k["close"] * 0.99
            return k

    assert store.update_outcomes(Client()) == 1
    row = store.frame().iloc[0]
    assert row["ret_24h"] == pytest.approx((1 + 0.10 * 23 / 79 - 1) * 100, rel=1e-3) and row["ret_72h"] > row["ret_24h"]
    perf = store.performance()
    assert not perf.empty and (perf["n"] == 1).all() and "güvenilir değil" in perf["not"].iat[0]


def test_engine_opens_and_expires_news_catalyst_position(tmp_path):
    from kreatifbot.intel.live_engine import IntelligentBotEngine
    cfg = cfg_1h()
    cfg.btc_context = cfg.use_futures_context = False
    cfg.listing.enabled = False
    cfg.news.poll_seconds = 0
    cfg.catalyst.scan_minutes = 0
    cfg.catalyst.min_quote_volume_24h = 1000
    cfg.allowed_symbols = ["BTCUSDT", "ABCUSDT"]
    market = FakeMarket(with_taker(make_ohlcv(400, seed=9)))
    abc = uptrend_1h()
    orig_k, orig_p = market.klines, market.price

    def klines(symbol, interval, limit=500, **kw):
        if symbol == "ABCUSDT":
            return abc.tail(limit).reset_index(drop=True) if interval == "1h" else resample(abc, "4h").tail(limit)
        return orig_k(symbol, interval, limit, **kw)

    market.klines = klines
    abc_px = float(abc["close"].iloc[-1])
    market.price = lambda s: abc_px if s == "ABCUSDT" else orig_p(s)
    base_depth = market.depth
    market.depth = lambda s, n=100: {"bids": [[str(abc_px * 0.9999), "1e6"]], "asks": [[str(abc_px * 1.0001), "1e6"]]} \
        if s == "ABCUSDT" else base_depth(s, n)
    market.exchange_info = lambda: {"symbols": [
        {"symbol": "BTCUSDT", "status": "TRADING", "quoteAsset": "USDT", "baseAsset": "BTC"},
        {"symbol": "ABCUSDT", "status": "TRADING", "quoteAsset": "USDT", "baseAsset": "ABC"}]}
    nstore = NewsStore(tmp_path / "n.db")
    nstore.add([item("ABC partners with Microsoft", "ABC", 2, iid="1"),
                item("ABC strikes partnership with Microsoft", "ABC", 3, "Decrypt", iid="2"),
                item("ABC mainnet goes live", "ABC", 4, "Cointelegraph", iid="3")])

    class NoNet:
        def get(self, *a, **k):
            return _R({}, 503)

    mon = NewsMonitor(nstore, NoNet(), rss_feeds={}, use_binance=False)
    ie = InsightEngine(cfg.catalyst, market, FundamentalsProvider(FakeAPISession()))
    events = []
    eng = IntelligentBotEngine(cfg, ["BTCUSDT"], market, PaperVenue("SPOT", 10000, cfg),
                               SignalStore(tmp_path / "s.db"), news_monitor=mon, insight_engine=ie,
                               insight_store=InsightStore(tmp_path / "i.db"), kline_limit=380,
                               on_event=lambda k, p: events.append((k, p)))
    eng.tick()
    assert any(k == "insight" and p.signal == "AL" for k, p in events), [p for k, p in events if k == "log"]
    assert "ABCUSDT" in eng.positions and eng.positions["ABCUSDT"].strategy == "news_catalyst"
    # süre dolunca kapanır
    pos = eng.positions["ABCUSDT"]
    pos.opened_at = (NOW - pd.Timedelta(hours=cfg.catalyst.max_hold_hours + 1)).isoformat()
    eng._last_insight_scan = 10**12
    eng.tick()
    assert "ABCUSDT" not in eng.positions and eng.trades[-1].exit_reason == "TIME_EXPIRY"


def test_disallowed_symbol_not_traded(tmp_path):
    cfg = IntelConfig()
    assert cfg.catalyst.any_binance_pair is False  # varsayılan: yalnızca izinli semboller


def test_telegram_insight_format():
    cfg = CatalystConfig(min_quote_volume_24h=1000)
    df = uptrend_1h()
    ins = build_insight("ABCUSDT", "ABC", [item("ABC partners with <Microsoft>", "ABC")], Fundamentals("ABC"), df,
                        None, cfg)
    n = TelegramNotifier(TelegramClient("T"), "1", commands=False)
    txt = n.format_event("insight", ins)
    assert "Öngörü: ABCUSDT" in txt and "&lt;Microsoft&gt;" in txt
    _ = json
