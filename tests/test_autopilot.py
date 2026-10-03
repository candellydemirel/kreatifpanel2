"""Otomatik pilot bakımı testleri (sentetik veri yalnızca kod doğruluğu içindir)."""

import pandas as pd

from kreatifbot.intel.autopilot import MaintenanceResult, run_maintenance
from kreatifbot.intel.execution import PaperVenue
from kreatifbot.intel.signal_store import SignalStore
from kreatifbot.intel.types import Regime
from kreatifbot.intel.router import route
from kreatifbot.telegram import TelegramClient, TelegramNotifier

from .conftest import make_ohlcv
from .test_intel import FakeMarket, cfg_1h, resample, with_taker


class HistClient:
    """end_time ve limit'e uyan sahte Binance istemcisi (geçmiş sayfalama için)."""

    def __init__(self, df):
        self.df = df

    def klines(self, symbol, interval, limit=500, end_time=None, **kw):
        d = self.df if interval == "1h" else resample(self.df, {"4h": "4h", "1d": "1D"}.get(interval, "4h"))
        if end_time is not None:
            ms = (pd.to_datetime(d["open_time"], utc=True) - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(milliseconds=1)
            d = d[ms <= end_time]
        return d.tail(limit).reset_index(drop=True)


def _cfg():
    cfg = cfg_1h()
    cfg.btc_context = False
    cfg.use_futures_context = False
    cfg.autopilot.history_bars = 1800
    cfg.ml.min_train_samples = 40
    cfg.autopilot.min_holdout_samples = 10
    return cfg


def test_maintenance_gate_and_stats():
    df = with_taker(make_ohlcv(2000, seed=5))
    c = HistClient(df)
    strict = _cfg()
    strict.autopilot.min_auc = 0.99  # neredeyse imkânsız eşik → model devreye alınmamalı
    res = run_maintenance(strict, ["BTCUSDT"], c, c)
    assert res.stats and res.health is not None and not res.errors, res.errors
    assert "BTCUSDT" not in res.metas
    assert any("doğrulamayı geçemedi" in line or "deterministik" in line or "yetersiz" in line for line in res.lines)
    loose = _cfg()
    loose.autopilot.min_auc = 0.0
    res2 = run_maintenance(loose, ["BTCUSDT"], c, c)
    assert "BTCUSDT" in res2.metas and any("AKTİF" in line for line in res2.lines)
    assert "Otomatik bakım tamamlandı" in res2.summary()


def test_maintenance_skips_unavailable_data():
    class Empty:
        def klines(self, *a, **k):
            return pd.DataFrame()
    res = run_maintenance(_cfg(), ["BTCUSDT"], Empty(), Empty())
    assert not res.metas and not res.stats and res.errors and "DATA_UNAVAILABLE" in res.errors[0]


def test_engine_applies_maintenance(tmp_path, monkeypatch):
    from kreatifbot.intel import autopilot as ap_mod
    from kreatifbot.intel.live_engine import IntelligentBotEngine
    cfg = _cfg()
    cfg.data_quality.min_history_bars = 200
    market = FakeMarket(with_taker(make_ohlcv(400, seed=9)))
    fake = MaintenanceResult(finished_at="2026-01-01T00:00:00", metas={}, stats={"trend_following": {"n": 50}},
                             health={"trend_following": "PAUSED"}, lines=["test"])
    monkeypatch.setattr(ap_mod, "run_maintenance", lambda *a, **k: fake)
    events = []
    eng = IntelligentBotEngine(cfg, ["BTCUSDT"], market, PaperVenue("SPOT", 10000, cfg), SignalStore(tmp_path / "s.db"),
                               kline_limit=380, on_event=lambda k, p: events.append((k, p)))
    eng.engine.meta = object()  # önceki model
    eng.run_maintenance_now()
    assert eng.engine.stats == fake.stats and eng.engine.health == fake.health
    assert any(k == "maintenance" for k, _ in events)
    r = route(Regime.STRONG_BULL, eng.engine.strategies, cfg, "SPOT", eng.engine.health)
    assert "trend_following" not in r.active["LONG"] and "PAUSED" in r.inactive["trend_following"]
    eng.tick()
    assert eng.engine.meta is None  # doğrulanmış model yoksa deterministik mod
    n = TelegramNotifier(TelegramClient("T"), "1", commands=False)
    assert "Otomatik bakım" in n.format_event("maintenance", fake)


def test_engine_starts_maintenance_thread(tmp_path, monkeypatch):
    from kreatifbot.intel import autopilot as ap_mod
    from kreatifbot.intel.live_engine import IntelligentBotEngine
    cfg = _cfg()
    cfg.autopilot.first_run_delay_s = 0
    calls = []
    monkeypatch.setattr(ap_mod, "run_maintenance", lambda *a, **k: calls.append(1) or MaintenanceResult())
    market = FakeMarket(make_ohlcv(400))
    eng = IntelligentBotEngine(cfg, ["BTCUSDT"], market, PaperVenue("SPOT", 10000, cfg), None, poll_seconds=5)
    eng.start()
    import time
    end = time.time() + 10
    while not calls and time.time() < end:
        time.sleep(0.05)
    eng.stop(wait=True)
    assert calls


def test_learned_risk_rules():
    from types import SimpleNamespace

    from kreatifbot.intel.research import learned_risk

    def t(key, pnl):
        return SimpleNamespace(to_dict=lambda: {"strategy": key, "net_pnl": pnl})
    trades = ([t("good", 3.0)] * 15 + [t("good", -1.0)] * 10 + [t("bad", 1.0)] * 5 + [t("bad", -2.0)] * 10
              + [t("few", 5.0)] * 3)
    lr = learned_risk(trades, ["good", "bad", "few", "none"])
    assert lr["good"][0] == 1.0 and lr["bad"][0] == 0.25 and lr["few"][0] == 0.5 and lr["none"][0] == 0.5
    assert all(0 < m <= 1.0 for m, _ in lr.values())  # çarpan riski asla artırmaz


def test_maintenance_learns_and_persists_risk():
    from kreatifbot.intel.autopilot import load_learned
    df = with_taker(make_ohlcv(2000, seed=5))
    c = HistClient(df)
    res = run_maintenance(_cfg(), ["BTCUSDT"], c, c)
    assert res.learned and all(0 < m <= 1.0 for m, _ in res.learned.values())
    assert load_learned() == {k: (float(m), w) for k, (m, w) in res.learned.items()}
    assert "Öğrenilen risk" in res.summary()


def test_learned_risk_not_written_without_data():
    from kreatifbot.intel.autopilot import load_learned

    class Empty:
        def klines(self, *a, **k):
            return pd.DataFrame()
    res = run_maintenance(_cfg(), ["BTCUSDT"], Empty(), Empty())
    assert not res.learned and load_learned() == {}
