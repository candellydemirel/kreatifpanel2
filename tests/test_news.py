"""Haber ve yeni listeleme modülü testleri (ağ yok; sabit örnek yanıtlarla)."""

import time

import numpy as np
import pandas as pd
import pytest

from kreatifbot.binance_client import parse_symbol_rules
from kreatifbot.intel.config import IntelConfig, ListingConfig
from kreatifbot.intel.execution import PaperVenue
from kreatifbot.intel.listing import evaluate_listing, listing_backtest, simulate_listing
from kreatifbot.intel.news import (
    ListingEvent, NewsItem, NewsMonitor, NewsStore, classify, diff_exchange_info, extract_symbols,
    fetch_binance_announcements, parse_rss,
)
from kreatifbot.intel.signal_store import SignalStore
from kreatifbot.telegram import TelegramClient, TelegramNotifier

from .test_intel import SPOT_SYMBOL, FakeMarket, cfg_1h, with_taker
from .conftest import make_ohlcv

NOW = pd.Timestamp.now(tz="UTC")
RSS = f"""<?xml version="1.0"?><rss version="2.0"><channel><title>X</title>
<item><title>Bitcoin surges to record high as ETF inflows grow</title><link>https://ex.com/a</link>
<guid>a1</guid><pubDate>{NOW.strftime('%a, %d %b %Y %H:%M:%S +0000')}</pubDate>
<description>&lt;p&gt;BTC rally&lt;/p&gt;</description></item>
<item><title>DeFi protocol on Solana exploited, $40M drained</title><link>https://ex.com/b</link>
<guid>b2</guid><pubDate>{NOW.strftime('%a, %d %b %Y %H:%M:%S +0000')}</pubDate></item>
</channel></rss>"""
ATOM = f"""<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
<entry><title>SEC sues exchange over XRP sales</title><link href="https://ex.com/c"/>
<id>c3</id><updated>{NOW.isoformat()}</updated></entry></feed>"""
BINANCE_LIST = {"code": "000000", "data": {"catalogs": [{"catalogId": 48, "articles": [
    {"id": 1, "code": "abc", "title": "Binance Will List Fooken (FOO) with Seed Tag Applied",
     "releaseDate": int(NOW.timestamp() * 1000)}]}]}}
BINANCE_DELIST = {"code": "000000", "data": {"catalogs": [{"catalogId": 161, "articles": [
    {"id": 2, "code": "def", "title": "Binance Will Delist BAR (BAR) on 2026-10-10",
     "releaseDate": int(NOW.timestamp() * 1000)}]}]}}


class _R:
    def __init__(self, status=200, text="", js=None):
        self.status_code, self.text, self._js = status, text, js

    def json(self):
        if self._js is None:
            raise ValueError("json yok")
        return self._js


class FakeSession:
    def __init__(self, fail=()):
        self.fail = set(fail)

    def get(self, url, params=None, timeout=None, headers=None):
        for key in self.fail:
            if key in url:
                return _R(503)
        if "cms/article" in url:
            return _R(js=BINANCE_DELIST if params and params.get("catalogId") == 161 else BINANCE_LIST)
        if "coindesk" in url:
            return _R(text=RSS)
        if "cointelegraph" in url:
            return _R(text=ATOM)
        return _R(text="<bozuk")


def test_classify_and_symbols():
    assert classify("Binance Will Delist XYZ")[0] == "DELISTING"
    cat, sev, sent = classify("Protocol hacked, funds stolen")
    assert cat == "HACK" and sev == 3 and sent < 0
    assert classify("Binance Will List Foo (FOO)")[0] == "LISTING"
    assert classify("Fed holds interest rate")[0] == "MACRO"
    assert classify("Quiet day")[2] == 0.0
    assert extract_symbols("Binance Will List Fooken (FOO) and (BAR)") == ["FOO", "BAR"]
    assert "BTC" in extract_symbols("Bitcoin rallies") and "SOL" in extract_symbols("Solana network")
    assert extract_symbols("THE SEC AND ETF", known_bases={"ETH"}) == []
    assert set(extract_symbols("ETH and PEPE pump", known_bases={"ETH", "PEPE"})) == {"ETH", "PEPE"}


def test_parse_rss_atom_and_binance():
    items = parse_rss(RSS, "CoinDesk")
    assert len(items) == 2 and items[0].url == "https://ex.com/a" and "BTC" in items[0].symbols
    assert items[1].category == "HACK" and "SOL" in items[1].symbols and items[1].age_hours < 1
    atom = parse_rss(ATOM, "Cointelegraph")
    assert atom[0].category == "REGULATION_NEG" and atom[0].url == "https://ex.com/c"
    b = fetch_binance_announcements(FakeSession(), 48)
    assert b[0].category == "LISTING" and b[0].symbols == ["FOO"] and b[0].url.endswith("/abc")
    d = fetch_binance_announcements(FakeSession(), 161)
    assert d[0].category == "DELISTING" and d[0].symbols == ["BAR"]


def test_exchange_info_diff():
    syms = [{"symbol": "BTCUSDT", "status": "TRADING", "quoteAsset": "USDT", "baseAsset": "BTC"}]
    cur, ev = diff_exchange_info({}, syms)
    assert cur == {"BTCUSDT": "TRADING"} and ev == []  # ilk anlık görüntüde olay yok
    syms2 = syms + [{"symbol": "NEWUSDT", "status": "TRADING", "quoteAsset": "USDT", "baseAsset": "NEW"},
                    {"symbol": "NEWBTC", "status": "TRADING", "quoteAsset": "BTC", "baseAsset": "NEW"}]
    cur2, ev2 = diff_exchange_info(cur, syms2)
    assert [(e.symbol, e.kind) for e in ev2] == [("NEWUSDT", "NEW_SYMBOL")]
    syms3 = [dict(s, status="BREAK") if s["symbol"] == "BTCUSDT" else s for s in syms2]
    _, ev3 = diff_exchange_info(cur2, syms3)
    assert ev3[0].kind == "HALTED"


def test_monitor_poll_dedupe_and_context(tmp_path):
    store = NewsStore(tmp_path / "n.db")
    mon = NewsMonitor(store, FakeSession(fail=("decrypt",)))
    new, events = mon.poll([{"symbol": "BARUSDT", "status": "TRADING", "quoteAsset": "USDT", "baseAsset": "BAR"}])
    assert len(new) >= 4
    assert mon.source_status["Decrypt"].startswith("ERİŞİLEMİYOR")
    assert mon.source_status["CoinDesk"] == "OK"
    assert any(e.kind == "ANNOUNCED" and e.symbol == "FOOUSDT" for e in events)
    again, ev2 = mon.poll()
    assert again == [] and ev2 == []  # tekrar eden haberler kaydedilmez
    ctx = mon.context("BARUSDT")
    assert ctx.block_long and ctx.force_exit and "Delist" in ctx.reasons[0]
    sol = mon.context("SOLUSDT")
    assert sol.block_long and not sol.force_exit
    btc = mon.context("BTCUSDT")
    assert not btc.block_long and btc.sentiment > 0  # olumlu haber işlem açtırmaz, sadece bilgi
    assert store.listings()[0].symbol == "FOOUSDT"


def test_decision_blocks_long_on_negative_news(data_prepared):
    eng, prep = data_prepared
    from kreatifbot.intel.news import NewsContext
    ctx = NewsContext("BTCUSDT", items=[NewsItem("x", "t", "hack")], block_long=True, reasons=["Olumsuz haber"])
    blocked = [eng.decide(prep, i, news_ctx=ctx) for i in range(300, len(prep.f))]
    assert not any(d.is_trade for d in blocked)
    assert any("NEWS_RISK" in d.no_trade_reasons for d in blocked)


@pytest.fixture(scope="module")
def data_prepared():
    from kreatifbot.intel.decision import DecisionEngine
    cfg = cfg_1h()
    eng = DecisionEngine(cfg)
    return eng, eng.prepare(with_taker(make_ohlcv(1500, seed=5)), "BTCUSDT", "1h")


# ------------------------------------------------------------------ listeleme
def listing_frame(breakout_at=30, n=200, chase=False, start=None, base=1.0):
    """1 dk sentetik listeleme verisi: 25 dk dalgalı aralık, sonra hacimli kırılım ve yükseliş."""
    rng = np.random.default_rng(3)
    start = start or (pd.Timestamp.now(tz="UTC").floor("min") - pd.Timedelta(minutes=n))
    close = np.full(n, base) + rng.normal(0, 0.004, n)
    close[breakout_at:] = base * 1.03 + np.linspace(0, 0.08, n - breakout_at)
    if chase:
        close[breakout_at:] = base * 1.2
    open_ = np.concatenate([[base], close[:-1]])
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    vol = np.full(n, 100_000.0)
    vol[breakout_at] = 600_000.0
    ot = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({"open_time": ot, "open": open_, "high": high, "low": low, "close": close, "volume": vol,
                         "close_time": ot + pd.Timedelta(seconds=59.999), "quote_volume": vol * close})


def test_evaluate_listing_rules():
    cfg = ListingConfig()
    df = listing_frame()
    early = evaluate_listing(df, "NEWUSDT", cfg, i=10)
    assert not early.ok and "Yetersiz" in early.reasons[0]
    sig = evaluate_listing(df, "NEWUSDT", cfg, i=30)
    assert sig.ok, sig.reasons
    assert sig.stop < sig.range_low * 1.001 and sig.targets[0] > sig.entry
    late = evaluate_listing(df, "NEWUSDT", cfg, i=60)
    assert not late.ok  # taze kırılım değil
    chase = evaluate_listing(listing_frame(chase=True), "NEWUSDT", cfg, i=30)
    assert not chase.ok and any("kovalanmaz" in r for r in chase.reasons)
    thin = evaluate_listing(df, "NEWUSDT", ListingConfig(min_quote_volume=1e12), i=30)
    assert not thin.ok
    bad_book = evaluate_listing(df, "NEWUSDT", cfg, i=30, book_stats={"available": True, "spread_pct": 2.0,
                                                                       "bid_depth_quote": 1e6, "ask_depth_quote": 1e6})
    assert not bad_book.ok and "Spread" in bad_book.reasons[0]
    assert "LONG ADAYI" in sig.explain()


def test_listing_backtest_costs_and_no_fake_results():
    cfg = ListingConfig()
    costs = IntelConfig().costs
    trade, msg = simulate_listing(listing_frame(), "NEWUSDT", cfg, costs)
    assert msg == "OK" and trade.entry > 1.0 and trade.gross_ret_pct > trade.net_ret_pct
    res = listing_backtest({"A": listing_frame(), "B": listing_frame(chase=True), "C": pd.DataFrame()}, cfg, costs)
    assert res["summary"]["listeleme"] == 3 and res["summary"]["işlem"] == 1
    assert res["notes"]["C"].startswith("DATA_UNAVAILABLE")
    assert "uyarı" in res["summary"]


def test_live_engine_listing_trade_and_delist_exit(tmp_path):
    from kreatifbot.intel.live_engine import IntelligentBotEngine
    cfg = cfg_1h()
    cfg.btc_context = cfg.use_futures_context = False
    cfg.data_quality.min_history_bars = 200
    cfg.news.poll_seconds = 0
    market = FakeMarket(with_taker(make_ohlcv(400, seed=9)))
    lst = listing_frame(breakout_at=180, n=181, start=pd.Timestamp.now(tz="UTC").floor("min") - pd.Timedelta(
        minutes=182))
    orig = market.klines

    def klines(symbol, interval, limit=500, **kw):
        if symbol == "NEWUSDT":
            return lst
        return orig(symbol, interval, limit, **kw)

    market.klines = klines
    market.exchange_info = lambda: {"symbols": [
        {"symbol": "BTCUSDT", "status": "TRADING", "quoteAsset": "USDT", "baseAsset": "BTC"},
        {"symbol": "NEWUSDT", "status": "TRADING", "quoteAsset": "USDT", "baseAsset": "NEW"}]}
    rules = {"NEWUSDT": parse_symbol_rules({**SPOT_SYMBOL, "symbol": "NEWUSDT", "baseAsset": "NEW",
                                            "filters": [{"filterType": "LOT_SIZE", "minQty": "1", "maxQty": "1e9",
                                                         "stepSize": "1"},
                                                        {"filterType": "PRICE_FILTER", "tickSize": "0.0001"},
                                                        {"filterType": "NOTIONAL", "minNotional": "5"}]})}
    market.symbol_rules = lambda s: rules.get(s) or parse_symbol_rules({**SPOT_SYMBOL, "symbol": s})
    price_now = float(lst["close"].iloc[-1])
    market.depth = lambda s, n=100: {"bids": [[str(price_now * 0.9995), "1e7"]], "asks": [[str(price_now * 1.0005),
                                                                                             "1e7"]]} \
        if s == "NEWUSDT" else {"bids": [[str(market.px * 0.9999), "1000"]], "asks": [[str(market.px * 1.0001), "1000"]]}
    base_price = market.price
    market.price = lambda s: price_now if s == "NEWUSDT" else base_price(s)
    store = NewsStore(tmp_path / "n.db")
    mon = NewsMonitor(store, FakeSession(fail=("cms", "coindesk", "cointelegraph", "decrypt")), rss_feeds={})
    mon.poll(market.exchange_info()["symbols"][:1])  # ilk anlık görüntü (NEWUSDT yok)
    events = []
    eng = IntelligentBotEngine(cfg, ["BTCUSDT"], market, PaperVenue("SPOT", 10000, cfg), SignalStore(tmp_path / "s.db"),
                               news_monitor=mon, kline_limit=380, state_path=tmp_path / "st.json",
                               on_event=lambda k, p: events.append((k, p)))
    mon.last_poll = 0
    eng.tick()
    kinds = [k for k, _ in events]
    assert "listing" in kinds, [p for k, p in events if k == "log"]
    assert "NEWUSDT" in eng.positions, [str(p) for k, p in events if k in ("log", "listing_signal")]
    pos = eng.positions["NEWUSDT"]
    assert pos.strategy == "new_listing" and pos.stop_loss < pos.entry_price
    # Delist duyurusu → pozisyon kapanır
    store.add([NewsItem("dl1", "Binance duyuru", "Binance Will Delist New (NEW)", published_at=NOW.isoformat(),
                        symbols=["NEW"], category="DELISTING", severity=3, sentiment=-0.8)])
    mon.last_poll = 0
    eng.tick()
    assert "NEWUSDT" not in eng.positions
    assert eng.trades[-1].exit_reason == "EMERGENCY_EXIT"


def test_telegram_formats_news_and_listing():
    n = TelegramNotifier(TelegramClient("T"), "1", commands=False)
    txt = n.format_event("news", NewsItem("1", "CoinDesk", "Protocol <hacked>", url="https://x", symbols=["SOL"],
                                          category="HACK", severity=3, sentiment=-0.7))
    assert "🚨" in txt and "&lt;hacked&gt;" in txt and "SOL" in txt
    ev = ListingEvent("NEWUSDT", "NEW", NOW.isoformat(), "NOW_TRADING", "TRADING", "exchangeInfo")
    assert "İşleme açıldı" in n.format_event("listing", ev)
    off = TelegramNotifier(TelegramClient("T"), "1", notify={"news": False}, commands=False)
    assert off.format_event("listing", ev) is None


def test_listing_config_validation():
    c = IntelConfig()
    c.listing.tp_fractions = [0.5, 0.2, 0.2]
    assert any("listing" in e for e in c.validate())
    _ = time
