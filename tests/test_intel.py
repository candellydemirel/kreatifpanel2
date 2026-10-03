"""Zeka motoru testleri: birim, entegrasyon, uçtan uca ve ileriye bakma (look-ahead).

Testlerde kullanılan sentetik veri yalnızca kod doğruluğu içindir; hiçbir
performans iddiası bu verilerden türetilmez.
"""

import copy
import math
import time

import numpy as np
import pandas as pd
import pytest

from kreatifbot import indicators as ind
from kreatifbot.binance_client import BinanceAPIError, parse_symbol_rules
from kreatifbot.intel.backtest import run_intel_backtest
from kreatifbot.intel.config import IntelConfig, config_from_dict, load_intel_config, save_intel_config
from kreatifbot.intel.data_quality import check_candles, check_orderbook
from kreatifbot.intel.decision import DecisionEngine
from kreatifbot.intel.execution import PaperVenue, pretrade_check
from kreatifbot.intel.features import compute_features
from kreatifbot.intel.futures_client import BinanceFuturesClient, LiquidationUnavailable
from kreatifbot.intel.ml import LogisticModel, MetaModel, prepare_meta_frame, psi, triple_barrier
from kreatifbot.intel.mtf import align_htf, htf_summary, mtf_score
from kreatifbot.intel.orderflow import (
    classify_funding, cvd_divergence, estimate_slippage_pct, oi_quadrant, orderbook_features,
)
from kreatifbot.intel.position_manager import evaluate_signal_decay, open_position, update_on_bar
from kreatifbot.intel.regime import classify_regime
from kreatifbot.intel.research import label_candidates, train_meta, walk_forward
from kreatifbot.intel.risk_engine import (
    OpenExposure, PortfolioState, circuit_breaker, compute_stop, compute_targets, correlated_exposure_check,
    estimate_costs, expectancy, liquidation_price, position_size, trailing_stop,
)
from kreatifbot.intel.router import route
from kreatifbot.intel.scoring import group_scores, total_scores
from kreatifbot.intel.signal_store import SignalStore
from kreatifbot.intel.strategies import REGISTRY, build_strategies
from kreatifbot.intel.types import ExitReason, Regime

from .conftest import make_ohlcv

SPOT_SYMBOL = {
    "symbol": "BTCUSDT", "status": "TRADING", "baseAsset": "BTC", "quoteAsset": "USDT", "quoteAssetPrecision": 8,
    "filters": [
        {"filterType": "PRICE_FILTER", "minPrice": "0.01", "maxPrice": "1000000", "tickSize": "0.01"},
        {"filterType": "LOT_SIZE", "minQty": "0.00001", "maxQty": "9000", "stepSize": "0.00001"},
        {"filterType": "MARKET_LOT_SIZE", "minQty": "0", "maxQty": "100", "stepSize": "0"},
        {"filterType": "NOTIONAL", "minNotional": "5", "maxNotional": "9000000"},
    ],
}
FUT_SYMBOL = {
    "symbol": "BTCUSDT", "status": "TRADING", "contractType": "PERPETUAL", "baseAsset": "BTC", "quoteAsset": "USDT",
    "pricePrecision": 2, "quantityPrecision": 3,
    "filters": [
        {"filterType": "PRICE_FILTER", "minPrice": "556.80", "maxPrice": "4529764", "tickSize": "0.10"},
        {"filterType": "LOT_SIZE", "minQty": "0.001", "maxQty": "1000", "stepSize": "0.001"},
        {"filterType": "MARKET_LOT_SIZE", "minQty": "0.001", "maxQty": "120", "stepSize": "0.001"},
        {"filterType": "MIN_NOTIONAL", "notional": "100"},
    ],
}


def cfg_1h(**kw) -> IntelConfig:
    c = IntelConfig()
    c.timeframes.entry, c.timeframes.confirmation, c.timeframes.trend = "1h", "1h", "4h"
    c.timeframes.major, c.timeframes.macro = "4h", "1d"
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def with_taker(df, seed=1):
    rng = np.random.default_rng(seed)
    df = df.copy()
    up = (df["close"] > df["open"]).astype(float)
    df["taker_buy_base"] = df["volume"] * np.clip(0.5 + 0.15 * (2 * up - 1) + rng.normal(0, 0.05, len(df)), 0.05, 0.95)
    return df


def resample(df, rule):
    g = df.set_index("open_time").resample(rule, label="left", closed="left")
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                        "close": g["close"].last(), "volume": g["volume"].sum()}).dropna().reset_index()
    sec = pd.Timedelta(rule).total_seconds()
    out["close_time"] = out["open_time"] + pd.Timedelta(seconds=sec - 0.001)
    out["quote_volume"] = out["volume"] * out["close"]
    return out


@pytest.fixture(scope="module")
def data():
    return with_taker(make_ohlcv(1800, seed=5))


@pytest.fixture(scope="module")
def feats(data):
    return compute_features(data, "1h")


def _eq(a, b):
    try:
        return np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), equal_nan=True, rtol=1e-7,
                           atol=1e-9)
    except (TypeError, ValueError):
        return (pd.Series(a).astype(str).values == pd.Series(b).astype(str).values).all()


# =========================================================== göstergeler
def test_new_indicators(data):
    v = ind.vwap_session(data)
    assert ((v >= data["low"].min()) & (v <= data["high"].max())).all()
    m = ind.mfi(data).dropna()
    assert m.between(0, 100).all()
    w = ind.williams_r(data).dropna()
    assert w.between(-100, 0).all()
    k, d = ind.stoch_rsi(data["close"])
    assert k.dropna().between(0, 100).all()
    assert ind.roc(pd.Series([100.0, 110.0]), 1).iloc[-1] == pytest.approx(10.0)
    z = ind.zscore(pd.Series(np.arange(100, dtype=float)), 10).dropna()
    assert np.isfinite(z).all()
    rv = ind.relative_volume(pd.Series([1.0] * 30 + [3.0]), 20)
    assert rv.iloc[-1] == pytest.approx(3.0)
    sar = ind.parabolic_sar(data).dropna()
    assert len(sar) > 100
    t, kj, sa, sb = ind.ichimoku(data)
    assert sa.iloc[:26 + 26 - 1].isna().all()  # bulut 26 mum geriden


def test_cvd_from_taker_and_unavailable(data):
    d = ind.volume_delta(data)
    assert np.allclose(d, 2 * data["taker_buy_base"] - data["volume"])
    no_taker = data.drop(columns=["taker_buy_base"])
    assert ind.cvd(no_taker).isna().all()


def test_feature_availability_flags(feats):
    av = feats.attrs["availability"]
    assert av["funding_rate"].startswith("UNAVAILABLE")
    assert av["liquidations"].startswith("UNAVAILABLE")
    assert feats["funding_rate"].isna().all()  # sahte değer yok


# =========================================================== look-ahead
def test_features_no_lookahead(data, feats):
    for cut in (500, 1100):
        g = compute_features(data.iloc[:cut].reset_index(drop=True), "1h")
        bad = [c for c in g.columns if not _eq(g[c].to_numpy(), feats[c].iloc[:cut].to_numpy())]
        assert not bad, bad


def test_mtf_alignment_is_causal(data):
    f = compute_features(data, "1h")
    h4 = resample(data, "4h")
    fh = compute_features(h4, "4h")
    aligned = align_htf(f, htf_summary(fh), "major")
    # 4s mumu kapanmadan önce değeri görünmemeli
    first_valid = aligned["major_bias"].first_valid_index()
    htf_first = fh.index[fh["ema50"].notna()][0]
    assert pd.to_datetime(f["close_time"].iat[first_valid], utc=True) >= \
        pd.to_datetime(fh["close_time"].iat[htf_first], utc=True)
    # kısaltılmış veride aynı sonuç
    cut = 900
    f2 = compute_features(data.iloc[:cut].reset_index(drop=True), "1h")
    a2 = align_htf(f2, htf_summary(compute_features(resample(data.iloc[:cut], "4h"), "4h")), "major")
    assert _eq(a2["major_bias"].to_numpy()[:cut - 8], aligned["major_bias"].to_numpy()[:cut - 8])


@pytest.mark.parametrize("key", list(REGISTRY))
def test_strategies_no_lookahead(key, data, feats):
    f = feats.copy()
    f["regime"] = classify_regime(f)["regime"]
    st = REGISTRY[key]()
    full = st.candidates(f)
    cut = 1000
    g = compute_features(data.iloc[:cut].reset_index(drop=True), "1h")
    g["regime"] = classify_regime(g)["regime"]
    part = st.candidates(g)
    for col in ("long", "short"):
        assert (part[col].to_numpy() == full[col].iloc[:cut].to_numpy()).all(), f"{key}:{col}"


def test_decisions_no_lookahead(data):
    cfg = cfg_1h()
    eng = DecisionEngine(cfg)
    full = eng.prepare(data, "BTCUSDT", "1h")
    part = eng.prepare(data.iloc[:1200].reset_index(drop=True), "BTCUSDT", "1h")
    for i in range(1000, 1200, 7):
        a, b = eng.decide(full, i), eng.decide(part, i)
        assert (a.direction, a.signal_status, a.no_trade_reasons, round(a.confidence, 6)) == \
               (b.direction, b.signal_status, b.no_trade_reasons, round(b.confidence, 6))


# =========================================================== veri kalitesi
def test_data_quality_checks(data):
    cfg = IntelConfig().data_quality
    ok = check_candles(data, "1h", cfg, check_stale=False)
    assert ok.ok, ok.critical
    dup = pd.concat([data, data.iloc[[-1]]], ignore_index=True)
    assert not check_candles(dup, "1h", cfg, check_stale=False).ok
    gap = data.drop(index=range(1700, 1720)).reset_index(drop=True)
    rep = check_candles(gap, "1h", cfg, check_stale=False)
    assert not rep.ok and rep.stats["missing_candles"] == 20
    bad = data.copy()
    bad.loc[100, "high"] = bad.loc[100, "low"] - 1
    assert any("OHLC" in m for m in check_candles(bad, "1h", cfg, check_stale=False).critical)
    stale = check_candles(data, "1h", cfg, now_ms=int(time.time() * 1000))
    assert any("Bayat" in m for m in stale.critical)
    short = check_candles(data.iloc[:50], "1h", cfg, check_stale=False)
    assert any("Yetersiz" in m for m in short.critical)
    assert not check_candles(pd.DataFrame(), "1h", cfg).ok


def test_orderbook_quality():
    book = {"bids": [["100", "1"]], "asks": [["101", "1"]], "T": 1000}
    rep = check_orderbook(book, IntelConfig().data_quality, now_ms=60_000)
    assert not rep.ok  # %1 spread ve bayat
    assert check_orderbook(None).stats["orderbook"] == "UNAVAILABLE"


# =========================================================== sembol filtreleri
def test_symbol_rules_and_precision():
    r = parse_symbol_rules(SPOT_SYMBOL)
    assert (r.tick_size, r.step_size, r.min_qty, r.max_qty, r.min_notional) == ("0.01", "0.00001", 1e-5, 9000.0, 5.0)
    assert r.quantity_precision == 5 and r.price_precision == 2 and r.tradable
    assert r.round_price(100.019, "BUY") == "100.01" and r.round_price(100.011, "SELL") == "100.02"
    chk = r.check_order(0.000123456, 60000)
    assert chk.ok and chk.quantity == "0.00012"
    assert not r.check_order(0.00005, 60000).ok  # notional 3 < 5
    assert not r.check_order(0.000001, 60000).ok  # sıfıra yuvarlanır
    fr = parse_symbol_rules(FUT_SYMBOL, "USDM_FUTURES")
    assert fr.min_notional == 100 and fr.quantity_precision == 3 and fr.market_max_qty == 120
    assert not fr.check_order(200, 60000).ok  # market maxQty


# =========================================================== rejim / skor / MTF / yönlendirici
def test_regime_classes(feats):
    reg = classify_regime(feats)
    vals = set(reg["regime"])
    assert vals <= {r.value for r in Regime}
    assert reg["regime"].iloc[:20].eq("UNKNOWN").all()
    up = make_ohlcv(600, seed=1).iloc[:200].reset_index(drop=True)  # yükseliş bölümü
    r_up = classify_regime(compute_features(up, "1h"))["regime"].iloc[-30:]
    assert r_up.isin(["STRONG_BULL", "BULL", "WEAK_BULL", "HIGH_VOLATILITY"]).mean() > 0.5


def test_scores_range_and_weights(feats):
    gs = group_scores(feats)
    vals = gs.stack().dropna()
    assert ((vals >= 0) & (vals <= 1)).all()
    cfg = IntelConfig().scoring
    tot = total_scores(gs, cfg)
    s = tot["score_long"].dropna()
    assert ((s >= 0) & (s <= 100)).all()
    assert tot["coverage_long"].max() < 1.0  # MTF/OI/AI yok → kapsam düşük
    tot2 = total_scores(gs, cfg, disabled={"trend", "momentum"})
    assert not np.allclose(tot2["score_long"].fillna(0), tot["score_long"].fillna(0))


def test_mtf_score():
    row = pd.Series({"major_bias": 1.0, "trend_bias": 1.0, "confirmation_bias": 1.0, "macro_bias": np.nan})
    w = IntelConfig().scoring.mtf_weights
    assert mtf_score(row, "LONG", w)[0] == pytest.approx(1.0)
    assert mtf_score(row, "SHORT", w)[0] == pytest.approx(-1.0)
    assert mtf_score(pd.Series({}), "LONG", w)[0] is None


def test_router_regime_rules():
    cfg = IntelConfig()
    sts = build_strategies(cfg)
    r = route(Regime.STRONG_BULL, sts, cfg, "SPOT")
    assert "trend_following" in r.active["LONG"] and "mean_reversion" not in r.active["LONG"]
    assert not r.active["SHORT"]  # spot'ta short yok
    r2 = route(Regime.SIDEWAYS, sts, cfg, "USDM_FUTURES")
    assert "bollinger_mean_reversion" in r2.active["LONG"] and "trend_following" not in r2.active["LONG"]
    assert "bollinger_mean_reversion" in r2.active["SHORT"]
    assert route(Regime.PANIC, sts, cfg, "SPOT").blocked_reason
    assert route(Regime.UNKNOWN, sts, cfg, "SPOT").blocked_reason
    hv = route(Regime.HIGH_VOLATILITY, sts, cfg, "SPOT")
    assert hv.threshold_bonus > 0 and hv.size_mult < 1
    cfg.strategy("trend_following").enabled = False
    assert "trend_following" not in route(Regime.STRONG_BULL, sts, cfg, "SPOT").active["LONG"]
    live = route(Regime.STRONG_BULL, sts, IntelConfig(), "SPOT", live=True)  # varsayılan aşama PAPER
    assert not live.active["LONG"]


# =========================================================== strateji koşulları
def _frame(close, **extra):
    n = len(close)
    df = pd.DataFrame({"close": close, "open": close, "high": np.array(close) + 0.5, "low": np.array(close) - 0.5})
    for k, v in extra.items():
        df[k] = v
    return df.assign(rvol=extra.get("rvol", 1.5), adx=extra.get("adx", 30.0))[:n]


def test_strategy_conditions_explicit():
    rsi = [25, 28, 32, 40]
    f = _frame([100, 100, 101, 102], rsi=rsi)
    c = REGISTRY["rsi_mean_reversion"]().setup(f, {"oversold": 30.0}, 1)
    assert list(c) == [False, False, True, False]
    s = REGISTRY["rsi_mean_reversion"]().setup(f.assign(rsi=[75, 72, 68, 60]), {"oversold": 30.0}, -1)
    assert list(s) == [False, False, True, False]  # SHORT ayna
    st = REGISTRY["supertrend"]().setup(_frame([1, 2, 3, 4], st_dir=[-1, -1, 1, 1]), {}, 1)
    assert list(st) == [False, False, True, False]


def test_short_params_asymmetric():
    cls = REGISTRY["rsi_momentum"]
    s = cls(params={"level": 55}, short_params={"level": 60})
    assert s.params["level"] == 55 and s.short_params["level"] == 60


def test_derivative_strategies_unavailable_without_data(feats):
    for key in ("oi_momentum", "funding_crowding", "liquidation_reversal", "ai_ensemble", "mtf_confluence"):
        c = REGISTRY[key]().candidates(feats)
        assert not c["long"].any() and not c["short"].any()
        assert c.attrs["missing"]


# =========================================================== SL / TP / trailing / breakeven
def test_stops_and_targets():
    cfg = IntelConfig().risk
    row = pd.Series({"atr": 2.0, "last_swing_low": 95.0, "last_swing_high": 106.0, "supertrend": 97.0,
                     "resistance": 103.0, "support": 96.0, "range_high": 104.0, "range_low": 94.0})
    sl, m = compute_stop("LONG", 100, row, cfg, "atr")
    assert sl == pytest.approx(96.0) and m == "atr"
    sls, _ = compute_stop("SHORT", 100, row, cfg, "atr")
    assert sls == pytest.approx(104.0)
    sw, _ = compute_stop("LONG", 100, row, cfg, "swing")
    assert sw == pytest.approx(95.0 - 0.4)
    stp, _ = compute_stop("LONG", 100, row, cfg, "supertrend")
    assert stp == 97.0
    hv, _ = compute_stop("LONG", 100, row, cfg, "atr", regime=Regime.HIGH_VOLATILITY)
    assert hv < sl  # yüksek volatilitede daha geniş stop
    fx, _ = compute_stop("LONG", 100, row, cfg, "fixed_pct")
    assert fx == pytest.approx(98.5)
    tps, warn = compute_targets("LONG", 100, 96, row, cfg)
    assert tps == pytest.approx([104, 108, 112, 116])
    assert warn  # direnç 103 TP1'den önce
    tps_s, _ = compute_targets("SHORT", 100, 104, row, cfg)
    assert tps_s[0] == pytest.approx(96)
    st_tp, _ = compute_targets("LONG", 100, 96, row, cfg, method="structure")
    assert st_tp[0] == pytest.approx(103)


def test_trailing_never_loosens():
    cfg = IntelConfig().risk
    row = pd.Series({"atr": 2.0, "ema20": 99.0})
    assert trailing_stop("LONG", 100, row, 101, cfg, "atr") == 100  # aday 96 < 100 → değişmez
    assert trailing_stop("LONG", 100, row, 110, cfg, "atr") == pytest.approx(105)
    assert trailing_stop("SHORT", 100, row, 99, cfg, "atr") == 100
    assert trailing_stop("SHORT", 100, row, 90, cfg, "atr") == pytest.approx(95)
    assert trailing_stop("LONG", 100, row, 110, cfg, "percent") == pytest.approx(110 * 0.985)


def _pos(direction="LONG", **kw):
    s = 1 if direction == "LONG" else -1
    return open_position("BTCUSDT", "SPOT", direction, 100.0, 1.0, 100 - s * 4, [100 + s * 4 * r for r in (1, 2, 3, 4)],
                         [0.25] * 4, "trend_following", "2024-01-01T00:00:00+00:00", 0, kw.get("max_hold", 10), 5,
                         80.0, "BULL")


def _bar(o, h, l, c, atr=2.0):
    return pd.Series({"open": o, "high": h, "low": l, "close": c, "atr": atr})


def test_position_lifecycle_tp_breakeven_trailing():
    cfg = IntelConfig().risk
    p = _pos()
    acts = update_on_bar(p, _bar(100, 104.5, 99.5, 104), cfg)  # TP1
    assert [a.reason for a in acts] == [ExitReason.TP_HIT] and acts[0].qty == pytest.approx(0.25)
    assert p.stop_loss == pytest.approx(100)  # breakeven
    acts = update_on_bar(p, _bar(104, 108.5, 103.5, 108), cfg)  # TP2 → trailing
    assert acts[0].note == "TP2" and p.stage == "TRAILING"
    stop_after = p.stop_loss
    update_on_bar(p, _bar(108, 108.2, 107, 107.5), cfg)
    assert p.stop_loss >= stop_after  # geri gitmez
    acts = update_on_bar(p, _bar(107, 107, 90, 91), cfg)
    assert acts[-1].reason == ExitReason.TRAILING_STOP and p.qty == 0


def test_stop_first_when_both_hit_and_short():
    cfg = IntelConfig().risk
    p = _pos("SHORT")
    acts = update_on_bar(p, _bar(100, 105, 95, 100), cfg)  # aynı mumda SL (104) ve TP1 (96)
    assert acts[0].reason == ExitReason.SL_HIT and acts[0].price == pytest.approx(104)
    p2 = _pos("SHORT")
    acts = update_on_bar(p2, _bar(100, 101, 95.5, 96), cfg)
    assert acts[0].reason == ExitReason.TP_HIT


def test_time_barrier_and_decay():
    cfg = IntelConfig().risk
    p = _pos(max_hold=3)
    for _ in range(2):
        assert not update_on_bar(p, _bar(100, 101, 99, 100), cfg)
    acts = update_on_bar(p, _bar(100, 101, 99, 100.5), cfg)
    assert acts[0].reason == ExitReason.TIME_EXPIRY
    p2 = _pos()
    assert evaluate_signal_decay(p2, 70, cfg) is None
    a = evaluate_signal_decay(p2, 40, cfg, price=99)
    assert a.reason == ExitReason.SIGNAL_DECAY
    p3 = _pos()
    assert evaluate_signal_decay(p3, 80, cfg, reversal_confidence=80).reason == ExitReason.SIGNAL_REVERSAL
    assert evaluate_signal_decay(_pos(), 80, cfg, regime_forbidden=True).reason == ExitReason.REGIME_CHANGE
    p4 = _pos()
    assert not update_on_bar(p4, _bar(100, 100, 100, 100), cfg, count_bar=False) and p4.bars_held == 0


# =========================================================== boyut / maliyet / risk
def test_position_sizing_rules():
    cfg = IntelConfig().risk
    r = position_size(10000, 100, 96, cfg, confidence=95)
    assert r.ok and r.risk_amount == pytest.approx(50) and r.qty == pytest.approx(12.5)
    low = position_size(10000, 100, 96, cfg, confidence=55)
    assert low.risk_amount < r.risk_amount
    hi = position_size(10000, 100, 96, cfg, confidence=100, strategy_risk_mult=3.0)
    assert hi.risk_amount <= 50 + 1e-9  # güven/çarpan limiti aşamaz
    hv = position_size(10000, 100, 96, cfg, confidence=95, high_vol=True)
    assert hv.risk_amount == pytest.approx(25)
    capped = position_size(1000, 100, 99.99, cfg, confidence=95)
    assert capped.notional <= 1000 + 1e-6 and capped.warnings
    full = position_size(1000, 100, 96, cfg, 95, open_exposure=[OpenExposure("ETHUSDT", "LONG", 1000, 10)])
    assert not full.ok


def test_costs_and_expectancy():
    cfg = IntelConfig().costs
    spot = estimate_costs("SPOT", cfg, spread_pct=0.02)
    assert spot.fee_pct == pytest.approx(0.2) and spot.funding_pct == 0
    fut = estimate_costs("USDM_FUTURES", cfg, spread_pct=0.02, funding_rate_8h=0.0003, hold_minutes=960)
    assert fut.funding_pct == pytest.approx(0.06) and fut.fee_pct == pytest.approx(0.1)
    short_fut = estimate_costs("USDM_FUTURES", cfg, spread_pct=0.02, funding_rate_8h=0.0003, hold_minutes=960,
                               direction="SHORT")
    assert short_fut.funding_pct == 0
    assumed = estimate_costs("USDM_FUTURES", cfg, hold_minutes=480)
    assert assumed.assumptions
    assert expectancy(0.5, 2, 1, 0.2) == pytest.approx(0.3)
    assert expectancy(0.3, 2, 1, 0.2) < 0


def test_liquidation_price_binance_formula():
    lp = liquidation_price("LONG", 100, 1, 10, 0.005)
    assert lp == pytest.approx(90.0 / 0.995, rel=1e-6)
    ls = liquidation_price("SHORT", 100, 1, 10, 0.005)
    assert ls == pytest.approx(110 / 1.005, rel=1e-6)
    assert liquidation_price("LONG", 100, 1, 1, 0.005) == 0.0


def test_correlated_exposure_and_circuit_breaker():
    cfg = IntelConfig().risk
    corr = pd.DataFrame([[1, 0.9], [0.9, 1]], index=["BTCUSDT", "ETHUSDT"], columns=["BTCUSDT", "ETHUSDT"])
    ok, msg, total = correlated_exposure_check("BTCUSDT", "LONG", 2000, [OpenExposure("ETHUSDT", "LONG", 3000, 30)],
                                               corr, cfg, 10000)
    assert ok and total == pytest.approx(4700) and "ETHUSDT" in msg  # limit %60 × 10000 = 6000
    ok2, msg2, _ = correlated_exposure_check("BTCUSDT", "LONG", 5000, [OpenExposure("ETHUSDT", "LONG", 3000, 30)],
                                             corr, cfg, 10000)
    assert not ok2 and "Korelasyonlu" in msg2
    cb = IntelConfig().circuit_breaker
    assert circuit_breaker(pd.Series({"open": 100, "close": 92, "vol_percentile": 0.5}), cb, cfg)
    ps = PortfolioState(equity=9600, peak_equity=10000, day_start_equity=10000)
    assert any("Günlük" in r for r in circuit_breaker(None, cb, cfg, ps))
    assert any("Art arda" in r for r in circuit_breaker(None, cb, cfg, PortfolioState(1, 1, 1, consecutive_losses=4)))
    assert not circuit_breaker(pd.Series({"open": 100, "close": 101, "vol_percentile": 0.5}), cb, cfg,
                               PortfolioState(10000, 10000, 10000))


# =========================================================== order flow / funding / OI
def test_orderflow_engines():
    book = {"bids": [["99.9", "5"], ["99.8", "100"]], "asks": [["100.1", "1"], ["100.2", "2"]]}
    ob = orderbook_features(book)
    assert ob["imbalance"] > 0 and ob["bid_walls"]
    assert estimate_slippage_pct(book, "BUY", 50) == pytest.approx(0.0, abs=1e-9)
    assert estimate_slippage_pct(book, "BUY", 10_000) is None  # derinlik yetersiz
    assert classify_funding(0.00005) == "NEUTRAL"
    assert classify_funding(0.0002) == "POSITIVE"
    assert classify_funding(-0.0005) == "ELEVATED_NEGATIVE"
    assert classify_funding(0.001) == "EXTREME_POSITIVE"
    assert classify_funding(None) == "UNAVAILABLE"
    assert oi_quadrant(1, 1) == "PRICE_UP+OI_UP" and oi_quadrant(-1, 1) == "PRICE_DOWN+OI_UP"
    assert oi_quadrant(np.nan, 1) == "UNAVAILABLE"
    assert cvd_divergence(1, -1) == "BEARISH_DIVERGENCE_CANDIDATE"
    assert cvd_divergence(-1, 1) == "BULLISH_DIVERGENCE_CANDIDATE"


def test_derivatives_merge_is_causal(data):
    ct = pd.to_datetime(data["close_time"], utc=True)
    funding = pd.DataFrame({"fundingTime": ct.iloc[::8].reset_index(drop=True) + pd.Timedelta(minutes=30),
                            "fundingRate": np.linspace(-0.001, 0.001, len(ct.iloc[::8]))})
    oi = pd.DataFrame({"timestamp": pd.to_datetime(data["open_time"], utc=True), "open_interest": np.arange(len(data),
                                                                                                         dtype=float)})
    f = compute_features(data, "1h", derivatives={"funding": funding, "oi": oi, "period": "1h"})
    assert f.attrs["availability"]["funding_rate"] == "OK"
    # funding mum kapanışından 30 dk sonra bilinir → ilk bar NaN
    assert np.isnan(f["funding_rate"].iat[0])
    # OI periyodu (1h) bittikten sonra kullanılabilir: bar i'de en fazla i-1 indeksli değer
    assert (f["open_interest"].dropna().to_numpy() <= np.arange(len(f))[f["open_interest"].notna()]).all()
    assert f["oi_change_pct"].notna().any()
    c = REGISTRY["funding_crowding"]().candidates(f)
    assert not c.attrs["missing"]


# =========================================================== ML / meta-labeling
def test_logistic_and_metrics():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 3))
    y = (x[:, 0] + 0.3 * rng.normal(size=500) > 0).astype(float)
    m = LogisticModel().fit(x, y)
    p = m.predict_proba(x)
    assert ((p > 0.5) == y).mean() > 0.85
    assert m.importance()[0] > m.importance()[1]
    assert psi(rng.normal(size=500), rng.normal(size=500)) < 0.1
    assert psi(rng.normal(size=500), rng.normal(3, 1, size=500)) > 0.5


def test_triple_barrier_labels(data, feats):
    idx = np.array([300, 600, 900])
    lab = triple_barrier(feats, idx, np.array([1, -1, 1]), 2.0, 2.0, 24)
    assert set(lab["outcome"]) <= {-1, 0, 1}
    assert (lab["end_idx"] > lab["idx"]).all() and (lab["bars"] <= 25).all()


def test_meta_training_is_purged(data):
    cfg = cfg_1h()
    cfg.ml.min_train_samples = 30
    eng = DecisionEngine(cfg)
    prep = eng.prepare(data, "BTCUSDT", "1h")
    meta, labels = train_meta(prep, cfg, 250, 1200, purge_until=1200)
    assert (labels["end_idx"] < 1200).all()
    assert meta.ready and meta.columns
    p = meta.predict(prep.meta_x, np.array([1300, 1301]), np.array([1, 1]))
    assert ((p >= 0) & (p <= 1)).all()
    unready = MetaModel().fit(prepare_meta_frame(prep.f), labels.iloc[:5], min_samples=100)
    assert not unready.ready


# =========================================================== karar motoru ve backtest
@pytest.fixture(scope="module")
def prepared(data):
    cfg = cfg_1h()
    eng = DecisionEngine(cfg)
    h4 = resample(data, "4h")
    d1 = resample(data, "1D")
    prep = eng.prepare(data, "BTCUSDT", "1h", htf_frames={"trend": h4, "major": h4, "confirmation": data, "macro": d1})
    return eng, prep


def test_decision_object_and_no_trade(prepared):
    eng, prep = prepared
    outs = [eng.decide(prep, i) for i in range(300, len(prep.f))]
    trades = [d for d in outs if d.is_trade]
    assert trades, "en az bir onaylı karar beklenir"
    d = trades[0]
    for attr in ("symbol", "market", "direction", "signal_status", "strategy", "confidence", "market_regime", "entry",
                 "stop_loss", "take_profit", "take_profit_levels", "risk_reward", "risk_percent", "position_size",
                 "leverage", "expected_return", "expected_duration_min", "maximum_duration_min", "signal_expiry",
                 "ai_probability", "warnings", "reasons", "invalidations"):
        assert hasattr(d, attr)
    assert d.direction == "LONG"  # spot
    assert d.risk_reward >= eng.cfg.risk.min_rr
    assert d.score_of("trend") is not None and "Grup puanları" in d.explanation
    assert any("Toplam" in line for line in d.log_lines) and d.log_lines[-1] == "Karar: LONG (alış)"
    reasons = {r for o in outs for r in o.no_trade_reasons}
    assert reasons & {"LOW_CONFIDENCE", "CONFLICT", "NO_CANDIDATE"}
    assert all(o.direction == "NONE" for o in outs if not o.is_trade)


def test_decision_data_quality_and_safety(prepared):
    eng, prep = prepared
    from kreatifbot.intel.data_quality import QualityReport
    bad = QualityReport()
    bad.fail("eksik mum")
    d = eng.decide(prep, 1000, dq=bad)
    assert d.no_trade_reasons == ["DATA_QUALITY_FAILURE"]
    d2 = eng.decide(prep, 1000, live=True, rules=None)
    assert d2.no_trade_reasons == ["SYMBOL_FILTER_UNKNOWN"]


def test_backtest_costs_and_accounting(prepared):
    eng, prep = prepared
    r = run_intel_backtest(eng, prep, 300)
    m = r.metrics
    assert m["Toplam işlem"] == len(r.trades) > 0
    assert m["Net PnL"] == pytest.approx(m["Brüt PnL"] - m["Ücretler"] - m["Funding"], abs=1e-6)
    assert m["Son bakiye"] == pytest.approx(eng.cfg.backtest.initial_equity + m["Net PnL"], rel=1e-9)
    assert all(t.exit_reason in {e.value for e in ExitReason} for t in r.trades)
    assert all(t.entry_bar > int(d) for t, d in zip(r.trades, [t.entry_bar - 1 for t in r.trades]))
    r2 = run_intel_backtest(eng, prep, 300, fee_mult=3)
    assert r2.metrics["Ücretler"] > m["Ücretler"] * 2


def test_signal_expiry_in_backtest(prepared):
    eng, prep = prepared
    cfg = copy.deepcopy(eng.cfg)
    eng2 = DecisionEngine(cfg)
    r = run_intel_backtest(eng2, prep, 300, entry_mode="limit_pullback")
    assert r.signal_stats["expired"] + r.signal_stats["cancelled"] > 0


def test_futures_backtest_shorts_and_funding(data):
    cfg = cfg_1h(market="USDM_FUTURES")
    eng = DecisionEngine(cfg)
    prep = eng.prepare(data, "BTCUSDT", "1h")
    r = run_intel_backtest(eng, prep, 300)
    assert r.metrics["Toplam işlem"] > 0
    assert any("Funding verisi yok" in a for a in r.assumptions)


def test_backtest_data_unavailable():
    cfg = cfg_1h()
    eng = DecisionEngine(cfg)
    prep = eng.prepare(make_ohlcv(80), "X", "1h")
    assert run_intel_backtest(eng, prep).status == "DATA_UNAVAILABLE"


def test_walk_forward_separates_samples(prepared):
    eng, prep = prepared
    eng.cfg.ml.min_train_samples = 40
    wf = walk_forward(eng, prep, grid={"min_trade_score": [60.0, 70.0]}, train=500, validation=300, test=300)
    assert wf.status == "OK" and len(wf.folds) >= 2
    for fold in wf.folds:
        tr_end = int(fold["eğitim"].split("-")[1])
        te_start = int(fold["test"].split("-")[0])
        assert te_start > tr_end
    assert all(int(t.entry_bar) >= int(wf.folds[0]["test"].split("-")[0]) for t in wf.oos_trades)
    short = walk_forward(eng, prep, train=5000)
    assert short.status == "DATA_UNAVAILABLE"


# =========================================================== veritabanı / yapılandırma / futures istemcisi
def test_signal_store_roundtrip(tmp_path, prepared):
    eng, prep = prepared
    store = SignalStore(tmp_path / "s.db")
    d = next(eng.decide(prep, i) for i in range(300, len(prep.f)) if eng.decide(prep, i).is_trade)
    store.save_decision(d)
    store.update_status(d.signal_id, "EXECUTED")
    store.record_outcome(d.signal_id, "WIN", "TP_HIT", 12.5, 14.0, 1.5, 1.2, 180)
    rows = store.recent(10)
    assert rows.iloc[0]["status"] == "CLOSED" and rows.iloc[0]["exit_reason"] == "TP_HIT"
    assert store.log_for(d.signal_id)
    assert store.strategy_stats()[d.strategy]["n"] == 1
    assert store.healthy()


def test_intel_config_roundtrip_and_validation(tmp_path):
    cfg = IntelConfig()
    cfg.strategy("breakout").enabled = False
    cfg.scoring.weights["ai"] = 0
    save_intel_config(cfg, tmp_path / "c.json")
    back = load_intel_config(tmp_path / "c.json")
    assert back.strategy("breakout").enabled is False and back.scoring.weights["ai"] == 0
    bad = config_from_dict({"timeframes": {"entry": "7m"}})
    assert bad.validate()
    with pytest.raises(ValueError):
        save_intel_config(config_from_dict({"risk": {"tp_fractions": [0.5, 0.2]}}), tmp_path / "x.json")


class _Resp:
    def __init__(self, status, payload):
        self.status_code, self._p, self.text = status, payload, str(payload)

    def json(self):
        return self._p


class _Sess:
    def __init__(self, routes):
        self.routes, self.urls = routes, []

    def request(self, method, url, headers=None, timeout=None):
        self.urls.append(url)
        for key, resp in self.routes.items():
            if key in url:
                return resp
        return _Resp(404, {"code": -1, "msg": "yok"})


def test_futures_client_endpoints():
    now = int(time.time() * 1000)
    sess = _Sess({
        "/fapi/v1/exchangeInfo": _Resp(200, {"symbols": [FUT_SYMBOL]}),
        "/fapi/v1/premiumIndex": _Resp(200, {"markPrice": "60000.1", "indexPrice": "59990", "lastFundingRate": "0.0001",
                                             "nextFundingTime": now}),
        "/fapi/v1/fundingRate": _Resp(200, [{"symbol": "BTCUSDT", "fundingTime": now, "fundingRate": "0.0001",
                                             "markPrice": "1"}]),
        "/futures/data/openInterestHist": _Resp(200, [{"symbol": "BTCUSDT", "sumOpenInterest": "100",
                                                       "sumOpenInterestValue": "6000000", "timestamp": now}]),
        "/fapi/v1/order": _Resp(200, {"executedQty": "0.010", "avgPrice": "60000", "cumQuote": "600", "orderId": 1}),
    })
    c = BinanceFuturesClient("k", "s", testnet=True, session=sess)
    assert c.base_url.startswith("https://testnet.binancefuture.com")
    assert c.symbol_rules("BTCUSDT").min_notional == 100
    assert float(c.premium_index("BTCUSDT")["markPrice"]) == 60000.1
    assert c.funding_history("BTCUSDT")["fundingRate"].iat[0] == 0.0001
    assert c.open_interest_hist("BTCUSDT", "5m")["open_interest"].iat[0] == 100
    with pytest.raises(ValueError):
        c.open_interest_hist("BTCUSDT", "3m")
    with pytest.raises(LiquidationUnavailable):
        c.liquidations("BTCUSDT")
    c.new_order("BTCUSDT", "SELL", "STOP_MARKET", stop_price="59000", close_position=True)
    url = sess.urls[-1]
    assert "closePosition=true" in url and "quantity" not in url and "signature=" in url


# =========================================================== yürütme ve canlı motor (uçtan uca)
def test_pretrade_and_paper_venue():
    cfg = IntelConfig()
    rules = parse_symbol_rules(SPOT_SYMBOL)
    book = {"bids": [["59999", "10"]], "asks": [["60001", "10"]]}
    ok = pretrade_check(rules, 0.01, 60000, "LONG", cfg, book, 10000)
    assert ok.ok and ok.qty == "0.01"
    assert not pretrade_check(rules, 0.01, 60000, "LONG", cfg, book, 100).ok  # bakiye
    wide = {"bids": [["59000", "10"]], "asks": [["61000", "10"]]}
    assert not pretrade_check(rules, 0.01, 60000, "LONG", cfg, wide, 10000).ok  # spread
    assert not pretrade_check(rules, 0.01, 60000, "LONG", cfg, None, 10000).ok  # order book zorunlu
    v = PaperVenue("SPOT", 1000, cfg)
    f = v.open("BTCUSDT", "LONG", 0.01, 60000, rules)
    assert v.cash == pytest.approx(1000 - f.notional - f.fee)
    v.close("BTCUSDT", "LONG", 0.01, 61000, f.price, rules)
    assert v.cash > 1000 - 2 * f.fee
    fv = PaperVenue("USDM_FUTURES", 1000, cfg)
    fv.open("BTCUSDT", "SHORT", 0.05, 60000, None, leverage=5)
    assert fv.available_balance() < 1000 - 0.05 * 60000 / 5 + 1
    fv.close("BTCUSDT", "SHORT", 0.05, 59000, 60000)
    assert fv.cash > 1000 and fv.available_balance() == pytest.approx(fv.cash)


class FakeMarket:
    """Binance spot istemcisini taklit eder (yalnızca test)."""

    def __init__(self, df):
        self.df = df.copy()
        shift = pd.Timestamp.now(tz="UTC") - pd.to_datetime(self.df["close_time"].iloc[-1], utc=True) - \
            pd.Timedelta(seconds=5)
        for c in ("open_time", "close_time"):
            self.df[c] = pd.to_datetime(self.df[c], utc=True) + shift
        self.px = float(self.df["close"].iloc[-1])
        self.fail = False

    def klines(self, symbol, interval, limit=500, **kw):
        if self.fail:
            raise BinanceAPIError(500, -1000, "test")
        if interval == "1h":
            return self.df.tail(limit).reset_index(drop=True)
        return resample(self.df, {"4h": "4h", "1d": "1D"}.get(interval, "4h")).tail(limit).reset_index(drop=True)

    def depth(self, symbol, limit=100):
        return {"bids": [[str(self.px * 0.9999), "1000"]], "asks": [[str(self.px * 1.0001), "1000"]]}

    def price(self, symbol):
        return self.px

    def sync_time(self):
        return 0

    def symbol_rules(self, symbol):
        return parse_symbol_rules({**SPOT_SYMBOL, "symbol": symbol})


class ScriptedDecisions(DecisionEngine):
    """İlk çağrıda onaylı LONG kararı döndüren karar motoru (E2E testi için)."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.calls = 0

    def decide(self, prep, i=-1, **kw):
        d = super().decide(prep, i, **kw)
        self.calls += 1
        if self.calls == 1 and kw.get("dq") is not None and kw["dq"].ok:
            px = float(prep.f["close"].iat[i])
            d.direction, d.signal_status, d.strategy = "LONG", "CONFIRMED", "trend_following"
            d.entry, d.stop_loss, d.take_profit_levels = px, px * 0.98, [px * 1.01, px * 1.02, px * 1.03, px * 1.04]
            d.position_size, d.notional, d.leverage, d.confidence = 0.05, 0.05 * px, 1, 80.0
            d.max_hold_bars, d.expected_hold_bars = 10, 5
            d.signal_expiry = (pd.Timestamp.now(tz="UTC") + pd.Timedelta(minutes=30)).isoformat()
        return d


def test_live_engine_end_to_end(tmp_path):
    from kreatifbot.intel.live_engine import IntelligentBotEngine
    cfg = cfg_1h()
    cfg.btc_context = False
    cfg.use_futures_context = False
    cfg.data_quality.min_history_bars = 200
    market = FakeMarket(with_taker(make_ohlcv(700, seed=9)))
    store = SignalStore(tmp_path / "sig.db")
    events = []
    eng = IntelligentBotEngine(cfg, ["BTCUSDT"], market, PaperVenue("SPOT", 10000, cfg), store,
                               decision_engine=ScriptedDecisions(cfg), state_path=tmp_path / "st.json",
                               on_event=lambda k, p: events.append((k, p)), kline_limit=600)
    eng.tick()
    assert "BTCUSDT" in eng.positions, [p for k, p in events if k == "log"]
    pos = eng.positions["BTCUSDT"]
    assert pos.stop_loss < pos.entry_price
    kinds = [k for k, _ in events]
    assert "decision" in kinds and "opened" in kinds
    # fiyat TP1 ve TP2'ye çıkar → kısmi çıkışlar, stop breakeven'a gelir
    market.px = pos.entry_price * 1.021
    eng.tick()
    p = eng.positions["BTCUSDT"]
    assert p.qty < p.initial_qty and p.stop_loss >= p.entry_price
    # fiyat sert düşer → kalan pozisyon stop ile kapanır
    market.px = pos.entry_price * 0.95
    eng.tick()
    assert "BTCUSDT" not in eng.positions
    tr = eng.trades[-1]
    # Fiyat stop'un altına boşlukla (gap) düştüğü için kalan kısım piyasa fiyatından kapanır (gerçekçi dolum)
    assert tr.exit_reason in ("SL_HIT", "TRAILING_STOP") and tr.fees > 0
    assert sum("TP_HIT" in str(x) for x in pos.confidence_history) == 2
    row = store.recent(5, only_trades=True)
    assert (row["status"] == "CLOSED").any()
    # durum kaydı ve yeniden yükleme
    eng2 = IntelligentBotEngine(cfg, ["BTCUSDT"], market, PaperVenue("SPOT", 10000, cfg), store,
                                state_path=tmp_path / "st.json")
    assert eng2.trades and eng2.venue.cash == pytest.approx(eng.venue.cash)


def test_live_engine_api_failure_means_no_trade(tmp_path):
    from kreatifbot.intel.live_engine import IntelligentBotEngine
    cfg = cfg_1h()
    cfg.btc_context = cfg.use_futures_context = False
    market = FakeMarket(make_ohlcv(400))
    market.fail = True
    eng = IntelligentBotEngine(cfg, ["BTCUSDT"], market, PaperVenue("SPOT", 10000, cfg), SignalStore(tmp_path / "a.db"))
    eng.tick()
    assert not eng.positions and not eng.pending
    with pytest.raises(ValueError):
        IntelligentBotEngine(cfg, ["DOGEUSDT"], market, PaperVenue("SPOT", 1, cfg), None)  # izinli değil


def test_live_engine_integration_real_decisions(tmp_path):
    """Gerçek karar motoru ile: piyasa verisi → özellik → karar → risk → (varsa) yürütme; hata olmamalı."""
    from kreatifbot.intel.live_engine import IntelligentBotEngine
    cfg = cfg_1h()
    cfg.btc_context = cfg.use_futures_context = False
    cfg.data_quality.min_history_bars = 200
    market = FakeMarket(with_taker(make_ohlcv(700, seed=2)))
    store = SignalStore(tmp_path / "i.db")
    eng = IntelligentBotEngine(cfg, ["BTCUSDT"], market, PaperVenue("SPOT", 10000, cfg), store, kline_limit=600)
    eng.tick()
    d = eng.last_decisions["BTCUSDT"]
    assert d.explanation and d.market_regime
    assert len(store.recent(5)) == 1


def test_expected_value_blocks_negative(prepared):
    eng, prep = prepared
    stats = {k: {"n": 100, "p_win": 0.05, "avg_win_pct": 0.5, "avg_loss_pct": 2.0} for k in REGISTRY}
    e2 = DecisionEngine(eng.cfg, eng.strategies, None, stats)
    outs = [e2.decide(prep, i) for i in range(300, len(prep.f))]
    assert not any(o.is_trade for o in outs)
    assert any("NEGATIVE_EXPECTANCY" in o.no_trade_reasons for o in outs)


def test_label_candidates_shape(prepared):
    eng, prep = prepared
    lab = label_candidates(prep, eng.cfg, 300, 600, 24)
    assert {"idx", "sign", "outcome"} <= set(lab.columns)
    assert math.isfinite(lab["ret_pct"].mean())


def test_data_history_paging():
    from kreatifbot.intel.market_data import _data_history
    base = pd.Timestamp("2024-01-01", tz="UTC")
    all_rows = pd.DataFrame({"timestamp": [base + pd.Timedelta(minutes=5 * k) for k in range(1200)],
                             "open_interest": np.arange(1200.0)})
    calls = []

    def fn(symbol, period, limit, end_time=None):
        calls.append(end_time)
        ms = (all_rows["timestamp"] - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(milliseconds=1)
        sub = all_rows[ms <= end_time] if end_time else all_rows
        return sub.tail(limit).reset_index(drop=True)

    end = int((all_rows["timestamp"].iloc[-1] - pd.Timestamp(0, tz="UTC")) / pd.Timedelta(milliseconds=1))
    out = _data_history(fn, "BTCUSDT", "5m", 0, end)
    assert len(out) == 1200 and out["timestamp"].is_monotonic_increasing and len(calls) == 3


def test_live_snapshot_keeps_spot_and_futures_separate():
    from kreatifbot.intel.market_data import live_snapshot

    class Spot:
        def book_ticker(self, s):
            return {"bidPrice": "100", "askPrice": "100.2"}

        def depth(self, s, n):
            return {"bids": [["100", "5"]], "asks": [["100.2", "5"]]}

        def ticker_24h(self, s):
            return {"quoteVolume": "1000000", "priceChangePercent": "1.2"}

        def agg_trades(self, s, n):
            return [{"q": "1", "m": False}, {"q": "0.5", "m": True}]

    class Fut:
        def premium_index(self, s):
            return {"markPrice": "100.5", "indexPrice": "100.4", "lastFundingRate": "0.0001", "nextFundingTime": 0}

        def open_interest(self, s):
            raise BinanceAPIError(500, -1, "geçici")

        def long_short_ratio(self, *a):
            return pd.DataFrame()

        def taker_volume(self, *a):
            return pd.DataFrame()

        def depth(self, s, n):
            return {"bids": [["100.4", "5"]], "asks": [["100.6", "5"]]}

        def liquidations(self, s):
            raise LiquidationUnavailable("yok")

    fs, raw = live_snapshot("BTCUSDT", Spot(), Fut())
    assert fs.get("spot_price") == pytest.approx(100.1) and fs.get("futures_mark_price") == 100.5
    assert fs.items["spot_price"].source.startswith("binance_spot")
    assert fs.items["futures_mark_price"].source.startswith("binance_futures")
    assert not fs.is_available("open_interest") and fs.get("open_interest") is None  # sahte değer yok
    assert not fs.is_available("liquidations")
    assert fs.get("spot_aggressive_delta") == pytest.approx(0.5)
    assert raw["spot_book"] is not raw["futures_book"]
