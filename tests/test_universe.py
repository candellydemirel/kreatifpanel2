from kreatifbot.intel.universe import UniverseSelector, select_universe


def _t(sym, qv, chg):
    return {"symbol": sym, "quoteVolume": str(qv), "priceChangePercent": str(chg)}


TICKERS = [
    _t("BTCUSDT", 2e9, 1.0), _t("ETHUSDT", 1e9, 0.5), _t("SOLUSDT", 5e8, 2.0), _t("USDCUSDT", 3e9, 0.0),
    _t("FDUSDUSDT", 1e9, 0.0), _t("BTCUPUSDT", 1e8, 9.0), _t("PEPEUSDT", 3e8, 18.0), _t("WIFUSDT", 6e7, -15.0),
    _t("TINYUSDT", 1e6, 40.0), _t("ETHBTC", 1e9, 3.0), _t("ARBUSDT", 8e7, 1.0), _t("HALTUSDT", 9e8, 30.0),
]


def test_select_mixes_volume_and_movers_and_filters():
    tradable = {t["symbol"] for t in TICKERS} - {"HALTUSDT"}
    syms, why = select_universe(TICKERS, "USDT", 6, 20_000_000, tradable, always=["BTCUSDT"])
    assert syms[0] == "BTCUSDT" and len(syms) == 6
    assert "PEPEUSDT" in syms and "WIFUSDT" in syms          # hareketli coinler de seçilir
    for bad in ("USDCUSDT", "FDUSDUSDT", "BTCUPUSDT", "TINYUSDT", "ETHBTC", "HALTUSDT"):
        assert bad not in syms
    assert why["WIFUSDT"].startswith("Hareket") and why["ETHUSDT"].startswith("Hacim")


class _Client:
    def __init__(self, fail=False):
        self.fail = fail

    def ticker_24h(self):
        if self.fail:
            raise ConnectionError("ağ yok")
        return TICKERS

    def exchange_info(self):
        return {"symbols": [{"symbol": t["symbol"], "status": "TRADING", "quoteAsset": "USDT"} for t in TICKERS]}


def test_selector_keeps_positions_and_handles_failure():
    sel = UniverseSelector(_Client(), n=4, base_symbols=["ETHUSDT"])
    syms = sel.refresh(keep=["ARBUSDT"])
    assert syms[:2] == ["ETHUSDT", "ARBUSDT"] and len(syms) == 4 and not sel.due()
    bad = UniverseSelector(_Client(fail=True), n=4)
    assert bad.refresh() is None and "alınamadı" in bad.status


def test_live_readiness_warns_and_universe_switches_symbols(tmp_path):
    from kreatifbot.intel.execution import PaperVenue
    from kreatifbot.intel.live_engine import IntelligentBotEngine

    from .conftest import make_ohlcv
    from .test_intel import FakeMarket, cfg_1h

    cfg = cfg_1h()
    cfg.btc_context = cfg.use_futures_context = False
    cfg.news.enabled = cfg.listing.enabled = cfg.catalyst.enabled = False

    class LiveZero(PaperVenue):
        is_live = True

        def available_balance(self):
            return 0.0

    logs = []
    sel = UniverseSelector(_Client(), n=4, base_symbols=["BTCUSDT"])
    eng = IntelligentBotEngine(cfg, ["BTCUSDT"], FakeMarket(make_ohlcv(400)), LiveZero("SPOT", 0, cfg), None,
                               state_path=tmp_path / "s.json", universe=sel,
                               on_event=lambda k, p: logs.append(p) if k == "log" else None)
    eng._live_readiness()
    text = "\n".join(str(x) for x in logs)
    assert "gerçek emir AÇMAZ" in text and "Fonlama cüzdanından" in text
    eng._universe_tick()
    assert eng.symbols[0] == "BTCUSDT" and len(eng.symbols) == 4 and "PEPEUSDT" in eng.symbols


def test_engine_logs_plain_summary(tmp_path):
    from types import SimpleNamespace

    from kreatifbot.intel.config import OLD_DEFAULT_TIMEFRAMES, config_from_dict
    from kreatifbot.intel.execution import PaperVenue
    from kreatifbot.intel.live_engine import IntelligentBotEngine

    from .conftest import make_ohlcv
    from .test_intel import FakeMarket, cfg_1h

    cfg = cfg_1h()
    logs = []
    eng = IntelligentBotEngine(cfg, ["BTCUSDT"], FakeMarket(make_ohlcv(300)), PaperVenue("SPOT", 1000, cfg), None,
                               state_path=tmp_path / "s.json",
                               on_event=lambda k, p: logs.append(str(p)) if k == "log" else None)
    nt = SimpleNamespace(symbol="SOLUSDT", is_trade=False, no_trade_reasons=["EDGE_BELOW_COSTS"],
                         long_score=float("nan"), short_score=40.0)
    eng._summary_add(nt)
    eng._summary_add(SimpleNamespace(**{**nt.__dict__, "long_score": 63.0, "no_trade_reasons": ["LOW_CONFIDENCE"]}))
    eng._summary_maybe_log()
    text = "\n".join(logs)
    assert "ilk tarama" in text and "Avantaj maliyetin altında" in text and "SOLUSDT LONG (alış) 63/100" in text
    # eski varsayılan zaman dilimleri yeni varsayılana taşınır
    assert config_from_dict({"timeframes": OLD_DEFAULT_TIMEFRAMES}).timeframes.entry == "15m"
