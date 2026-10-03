"""Araştırma modu: walk-forward, sağlamlık, ablasyon, kalibrasyon, tutma süresi,
çıkış/giriş optimizasyonu, özellik korelasyonu, strateji sağlığı ve S&C.

İlke: Aynı veride hem optimize edip hem performans ilan edilmez. Parametreler
eğitim penceresinde seçilir, doğrulama penceresinde onaylanır, yalnızca
örneklem dışı (test) sonuçlar raporlanır. Tek bir parametre kombinasyonunda
çalışan strateji sağlam kabul edilmez.
"""

from __future__ import annotations

import copy
import itertools
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..utils import INTERVALS
from .backtest import IntelBacktestResult, performance_metrics, run_intel_backtest, strategy_stats_from_trades
from .config import IntelConfig
from .decision import DecisionEngine, Prepared
from .ml import MetaModel, brier, calibration_table, directional_matrix, ece, permutation_importance, triple_barrier
from .scoring import group_scores, total_scores
from .types import BEAR_REGIMES, BULL_REGIMES, RANGE_REGIMES, Regime, StrategyHealth

DEFAULT_GRID = {"min_trade_score": [60.0, 65.0, 70.0], "stop_atr_mult": [1.5, 2.0, 2.5]}


def set_param(cfg: IntelConfig, key: str, value) -> IntelConfig:
    c = copy.deepcopy(cfg)
    if key == "min_trade_score":
        c.scoring.min_trade_score = float(value)
    elif key == "stop_atr_mult":
        c.risk.stop_atr_mult = float(value)
        c.risk.stop_atr_mult_high_vol = max(float(value), c.risk.stop_atr_mult_high_vol * float(value) /
                                            max(cfg.risk.stop_atr_mult, 1e-9))
    elif key == "tp_scale":
        c.risk.tp_levels_r = [r * float(value) for r in cfg.risk.tp_levels_r]
    elif key == "trailing_atr_mult":
        c.risk.trailing_atr_mult = float(value)
    elif key == "conflict_margin":
        c.scoring.conflict_margin = float(value)
    elif key == "meta_threshold":
        c.ml.meta_threshold = float(value)
    else:
        raise KeyError(key)
    return c


def _objective(m: dict) -> float:
    """Seçim ölçütü: yeterli işlem yoksa ceza; net getiri / (1 + |maks düşüş|)."""
    n = m.get("Toplam işlem", 0)
    if n < 5:
        return -1e9 + n
    return m["Net getiri (%)"] / (1 + abs(m["Maks. düşüş (%)"]))


def _engine_like(engine: DecisionEngine, cfg: IntelConfig, meta=None, stats=None) -> DecisionEngine:
    e = DecisionEngine(cfg, engine.strategies, meta, stats, engine.health)
    return e


# ---------------------------------------------------------------------- meta model eğitimi (purged)
def label_candidates(prep: Prepared, cfg: IntelConfig, start: int, end: int, max_bars: int) -> pd.DataFrame:
    idx, signs = [], []
    for key, c in prep.candidates.items():
        for col, s in (("long", 1), ("short", -1)):
            hits = np.flatnonzero(c[col][start:end]) + start
            idx += hits.tolist()
            signs += [s] * len(hits)
    if not idx:
        return pd.DataFrame(columns=["idx", "sign", "outcome", "ret_pct", "bars", "end_idx", "r_multiple"])
    order = np.argsort(idx)
    idx, signs = np.array(idx)[order], np.array(signs)[order]
    uniq = pd.DataFrame({"i": idx, "s": signs}).drop_duplicates()
    fee = 2 * (cfg.costs.spot_taker_fee_pct if prep.market == "SPOT" else cfg.costs.futures_taker_fee_pct)
    tp_r = cfg.risk.tp_levels_r[min(1, len(cfg.risk.tp_levels_r) - 1)]
    return triple_barrier(prep.f, uniq["i"].to_numpy(), uniq["s"].to_numpy(), cfg.risk.stop_atr_mult, tp_r,
                          max_bars, fee)


def train_meta(prep: Prepared, cfg: IntelConfig, start: int, end: int, purge_until: int,
               version: str = "") -> tuple[MetaModel, pd.DataFrame]:
    """[start, end) aralığındaki adaylardan, sonucu purge_until'den ÖNCE belli olanlarla eğitir."""
    max_bars = max(1, int(720 * 60 / INTERVALS[prep.interval]))
    labels = label_candidates(prep, cfg, start, end, max_bars)
    labels = labels[labels["end_idx"] < purge_until]
    if prep.market == "SPOT" and not cfg.backtest.allow_short_on_spot:
        labels = labels[labels["sign"] == 1]
    meta = MetaModel().fit(prep.meta_x, labels, cfg.ml.model, cfg.ml.l2, cfg.ml.min_train_samples, version)
    return meta, labels


# ---------------------------------------------------------------------- walk-forward
@dataclass
class WalkForwardResult:
    folds: list = field(default_factory=list)
    oos_trades: list = field(default_factory=list)
    oos_metrics: dict = field(default_factory=dict)
    param_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    calibration: pd.DataFrame = field(default_factory=pd.DataFrame)
    feature_importance: pd.DataFrame = field(default_factory=pd.DataFrame)
    importance_stability: pd.DataFrame = field(default_factory=pd.DataFrame)
    notes: list = field(default_factory=list)
    status: str = "OK"


def walk_forward(engine: DecisionEngine, prep: Prepared, grid: dict | None = None, train: int | None = None,
                 validation: int | None = None, test: int | None = None, mode: str | None = None,
                 warmup: int = 250, use_meta: bool = True, progress=None) -> WalkForwardResult:
    cfg = engine.cfg
    bt = cfg.backtest
    train, validation, test = train or bt.walk_forward_train, validation or bt.walk_forward_validation, \
        test or bt.walk_forward_test
    mode = mode or bt.walk_forward_mode
    grid = grid or DEFAULT_GRID
    n = len(prep.f)
    res = WalkForwardResult()
    if n < warmup + train + validation + test:
        res.status = "DATA_UNAVAILABLE"
        res.notes.append(f"Walk-forward için en az {warmup + train + validation + test} mum gerekli (mevcut {n}).")
        return res
    combos = [dict(zip(grid, vals)) for vals in itertools.product(*grid.values())]
    s0 = warmup
    fold_id = 0
    equity = cfg.backtest.initial_equity
    oos_eq_parts = []
    importances = []
    cal_p, cal_y = [], []
    while True:
        tr_start = warmup if mode == "expanding" else s0
        tr_end = s0 + train
        va_end = tr_end + validation
        te_end = min(va_end + test, n)
        if va_end + 50 > n:
            break
        fold_id += 1
        if progress:
            progress(f"Kat {fold_id}: eğitim {tr_start}-{tr_end}, doğrulama {tr_end}-{va_end}, test {va_end}-{te_end}")
        # 1) Eğitim: parametre taraması (meta modelsiz)
        scored = []
        for p in combos:
            c = cfg
            for k, v in p.items():
                c = set_param(c, k, v)
            r = run_intel_backtest(_engine_like(engine, c), prep, tr_start, tr_end, record_decisions=False)
            scored.append((_objective(r.metrics), p, r))
        scored.sort(key=lambda t: t[0], reverse=True)
        # 2) Doğrulama: en iyi 3 aday
        best = None
        for obj_tr, p, r_tr in scored[:3]:
            c = cfg
            for k, v in p.items():
                c = set_param(c, k, v)
            r_va = run_intel_backtest(_engine_like(engine, c), prep, tr_end, va_end, record_decisions=False)
            cand = (_objective(r_va.metrics), p, c, r_tr, r_va)
            if best is None or cand[0] > best[0]:
                best = cand
        _, params, chosen_cfg, r_tr, r_va = best
        # 3) Meta model: eğitim+doğrulama adayları, sonucu test başlangıcından önce belli olanlar
        meta = None
        if use_meta and cfg.ml.enabled:
            meta, labels = train_meta(prep, chosen_cfg, tr_start, va_end, va_end, version=f"wf{fold_id}")
            if not meta.ready:
                res.notes.append(f"Kat {fold_id}: meta model eğitilemedi ({meta.metrics.get('status')})")
                meta = None
        stats = strategy_stats_from_trades(r_tr.trades + r_va.trades, prep.interval)
        # 4) Test (örneklem dışı)
        r_te = run_intel_backtest(_engine_like(engine, chosen_cfg, meta, stats), prep, va_end, te_end,
                                  initial_equity=equity, record_decisions=True)
        equity = r_te.metrics.get("Son bakiye", equity)
        oos_eq_parts.append(r_te.equity)
        res.oos_trades += r_te.trades
        if meta is not None:
            te_labels = label_candidates(prep, chosen_cfg, va_end, te_end, max(1, int(720 * 60 /
                                                                                     INTERVALS[prep.interval])))
            if prep.market == "SPOT":
                te_labels = te_labels[te_labels["sign"] == 1]
            te_labels = te_labels[te_labels["outcome"] != 0]
            if len(te_labels) >= 20:
                p_te = meta.predict(prep.meta_x, te_labels["idx"].to_numpy(), te_labels["sign"].to_numpy())
                y_te = (te_labels["outcome"].to_numpy() == 1).astype(float)
                cal_p.append(p_te)
                cal_y.append(y_te)
                x_te = directional_matrix(prep.meta_x, te_labels["idx"].to_numpy(), te_labels["sign"].to_numpy(),
                                          meta.columns)
                imp = permutation_importance(meta.model, x_te, y_te, meta.columns + ["direction"], repeats=3)
                imp["fold"] = fold_id
                importances.append(imp)
        res.folds.append({
            "kat": fold_id, "eğitim": f"{tr_start}-{tr_end}", "doğrulama": f"{tr_end}-{va_end}",
            "test": f"{va_end}-{te_end}", **{f"param:{k}": v for k, v in params.items()},
            "eğitim_getiri_%": r_tr.metrics.get("Net getiri (%)", 0), "doğrulama_getiri_%": r_va.metrics.get(
                "Net getiri (%)", 0), "test_getiri_%": r_te.metrics.get("Net getiri (%)", 0),
            "test_işlem": r_te.metrics.get("Toplam işlem", 0), "meta_model": meta.version if meta else "yok",
        })
        if te_end >= n:
            break
        s0 += test
    if not res.folds:
        res.status = "DATA_UNAVAILABLE"
        return res
    eq = pd.concat(oos_eq_parts).sort_index() if oos_eq_parts else pd.Series(dtype=float)
    eq = eq[~eq.index.duplicated(keep="last")]
    res.oos_metrics = performance_metrics(res.oos_trades, eq, cfg.backtest.initial_equity, prep.interval) \
        if len(eq) else {}
    res.param_table = pd.DataFrame(res.folds)
    if cal_p:
        p, y = np.concatenate(cal_p), np.concatenate(cal_y)
        res.calibration = calibration_table(p, y)
        res.notes.append(f"Örneklem dışı kalibrasyon: Brier {brier(p, y):.3f}, ECE {ece(p, y):.3f}, n={len(p)}")
    if importances:
        allimp = pd.concat(importances)
        res.feature_importance = allimp.groupby("feature")["auc_drop"].mean().sort_values(ascending=False) \
            .reset_index()
        res.importance_stability = allimp.groupby("feature")["auc_drop"].agg(["mean", "std", "count"]) \
            .sort_values("mean", ascending=False).reset_index()
    pcols = [c for c in res.param_table.columns if c.startswith("param:")]
    if pcols and len(res.param_table) > 1:
        unstable = [c for c in pcols if res.param_table[c].nunique() == len(res.param_table)]
        if unstable:
            res.notes.append("Parametre seçimi her katta değişti (kararsız): " + ", ".join(unstable))
    return res


# ---------------------------------------------------------------------- sağlamlık
@dataclass
class RobustnessReport:
    perturbation: pd.DataFrame
    threshold: pd.DataFrame
    fee_stress: pd.DataFrame
    slippage_stress: pd.DataFrame
    monte_carlo: dict
    regimes: pd.DataFrame
    periods: pd.DataFrame
    per_strategy_regime: pd.DataFrame
    verdict: str
    notes: list = field(default_factory=list)


def _row(name, r: IntelBacktestResult) -> dict:
    m = r.metrics
    return {"senaryo": name, "işlem": m.get("Toplam işlem", 0), "net_getiri_%": m.get("Net getiri (%)", 0.0),
            "kâr_faktörü": m.get("Kâr faktörü", 0.0), "maks_düşüş_%": m.get("Maks. düşüş (%)", 0.0),
            "sharpe": m.get("Sharpe", 0.0), "ücret": m.get("Ücretler", 0.0)}


def regime_bucket(r: str) -> str:
    try:
        reg = Regime(r)
    except ValueError:
        return "Bilinmiyor"
    if reg in BULL_REGIMES:
        return "Bull"
    if reg in BEAR_REGIMES:
        return "Bear"
    if reg == Regime.HIGH_VOLATILITY or reg == Regime.PANIC:
        return "High Volatility"
    if reg == Regime.LOW_VOLATILITY:
        return "Low Volatility"
    if reg in RANGE_REGIMES:
        return "Sideways"
    return "Bilinmiyor"


def monte_carlo(trades: list, initial: float, runs: int = 500, seed: int = 11) -> dict:
    if len(trades) < 10:
        return {"status": "Yetersiz işlem (en az 10)"}
    pnl = np.array([t.net_pnl for t in trades])
    rng = np.random.default_rng(seed)
    dds, finals, exps = [], [], []
    for _ in range(runs):
        seq = rng.permutation(pnl)
        eq = initial + np.cumsum(seq)
        peak = np.maximum.accumulate(np.concatenate([[initial], eq]))[1:]
        dds.append(float(((eq / peak) - 1).min() * 100))
        boot = rng.choice(pnl, size=len(pnl), replace=True)
        finals.append(float(initial + boot.sum()))
        exps.append(float(boot.mean()))
    return {"status": "OK", "runs": runs,
            "maks_düşüş_medyan_%": float(np.median(dds)), "maks_düşüş_p95_%": float(np.percentile(dds, 5)),
            "son_bakiye_p5": float(np.percentile(finals, 5)), "son_bakiye_medyan": float(np.median(finals)),
            "beklenti_GA95": (float(np.percentile(exps, 2.5)), float(np.percentile(exps, 97.5))),
            "zarar_olasılığı": float((np.array(finals) < initial).mean())}


def robustness(engine: DecisionEngine, prep: Prepared, start: int = 250, progress=None) -> RobustnessReport:
    cfg = engine.cfg
    notes = []

    def run(c, **kw):
        return run_intel_backtest(_engine_like(engine, c), prep, start, None, record_decisions=False, **kw)

    base = run(cfg)
    if progress:
        progress("Parametre pertürbasyonu...")
    pert_rows = [_row("temel", base)]
    for key, vals in (("stop_atr_mult", [cfg.risk.stop_atr_mult * k for k in (0.8, 0.9, 1.1, 1.2)]),
                      ("tp_scale", [0.8, 0.9, 1.1, 1.2]),
                      ("trailing_atr_mult", [cfg.risk.trailing_atr_mult * k for k in (0.8, 1.2)]),
                      ("conflict_margin", [cfg.scoring.conflict_margin * k for k in (0.5, 1.5)])):
        for v in vals:
            pert_rows.append(_row(f"{key}={v:.2f}", run(set_param(cfg, key, v))))
    pert = pd.DataFrame(pert_rows)
    if progress:
        progress("Eşik duyarlılığı...")
    thr = pd.DataFrame([_row(f"min_skor={t}", run(set_param(cfg, "min_trade_score", t)))
                        for t in (55, 60, 65, 70, 75, 80, 85)])
    if progress:
        progress("Ücret ve kayma stresi...")
    fee = pd.DataFrame([_row(f"ücret x{k}", run(cfg, fee_mult=k)) for k in (1, 2, 3)])
    slip = pd.DataFrame([_row(f"kayma x{k}", run(cfg, slippage_mult=k)) for k in (1, 2, 4)])
    mc = monte_carlo(base.trades, cfg.backtest.initial_equity, cfg.backtest.monte_carlo_runs)
    # rejim ve dönem ayrımı
    tdf = base.trades_frame()
    if not tdf.empty:
        tdf["rejim_grubu"] = tdf["regime"].map(regime_bucket)
        regimes = tdf.groupby("rejim_grubu").agg(işlem=("net_pnl", "size"), net_pnl=("net_pnl", "sum"),
                                                 kazanma_oranı=("net_pnl", lambda x: (x > 0).mean() * 100),
                                                 ort_R=("r_multiple", "mean")).reset_index()
        psr = tdf.groupby(["strategy", "rejim_grubu"]).agg(işlem=("net_pnl", "size"), net_pnl=("net_pnl", "sum"),
                                                           ort_R=("r_multiple", "mean")).reset_index()
    else:
        regimes = psr = pd.DataFrame()
    n = len(prep.f)
    cuts = np.linspace(start, n, 4).astype(int)
    per_rows = []
    for k in range(3):
        r = run_intel_backtest(_engine_like(engine, cfg), prep, int(cuts[k]), int(cuts[k + 1]), record_decisions=False)
        per_rows.append(_row(f"dönem {k + 1} ({prep.f['open_time'].iat[cuts[k]]:%Y-%m-%d}…)", r))
    periods = pd.DataFrame(per_rows)

    # karar
    def positive(df):
        return df[(df["işlem"] >= 5)]["net_getiri_%"] > 0 if len(df) else pd.Series(dtype=bool)

    pp = positive(pert)
    share = float(pp.mean()) if len(pp) else 0.0
    checks = {
        "pertürbasyonların ≥%60'ı kârlı": share >= 0.6,
        "ücret x2 sonrası kârlı": bool(fee.iloc[1]["net_getiri_%"] > 0) if len(fee) > 1 else False,
        "kayma x2 sonrası kârlı": bool(slip.iloc[1]["net_getiri_%"] > 0) if len(slip) > 1 else False,
        "3 dönemin ≥2'si kârlı": int((periods["net_getiri_%"] > 0).sum()) >= 2,
        "Monte Carlo zarar olasılığı < %30": mc.get("zarar_olasılığı", 1.0) < 0.3,
    }
    passed = sum(checks.values())
    verdict = ("SAĞLAM ADAY" if passed == len(checks) else "KISMEN SAĞLAM" if passed >= 3 else "SAĞLAM DEĞİL") + \
        f" ({passed}/{len(checks)} kontrol): " + "; ".join(f"{'✔' if v else '✘'} {k}" for k, v in checks.items())
    if base.metrics.get("Toplam işlem", 0) < 30:
        notes.append("Uyarı: 30'dan az işlem; sonuçlar istatistiksel olarak güvenilir değil.")
    notes.append("Bu analiz tüm veri üzerinde yapıldı; performans iddiası için walk-forward (örneklem dışı) sonuçlarını "
                 "kullanın.")
    return RobustnessReport(pert, thr, fee, slip, mc, regimes, periods, psr, verdict, notes)


# ---------------------------------------------------------------------- ablasyon
ABLATIONS = {
    "RSI olmadan": {"neutral": {"rsi": 50.0, "stochrsi_k": 50.0, "stochrsi_d": 50.0}},
    "MACD olmadan": {"neutral": {"macd_hist": 0.0}, "copy": {"macd": "macd_signal"}},
    "Hacim olmadan": {"neutral": {"rvol": 1.0, "obv_slope": 0.0, "vol_accel": 0.0, "volume_spike": False},
                      "groups": {"volume"}},
    "OI olmadan": {"nan": ["open_interest", "oi_change_pct", "oi_accel"]},
    "Funding olmadan": {"nan": ["funding_rate"]},
    "CVD/Delta olmadan": {"nan": ["delta", "cvd", "cvd_slope", "taker_buy_ratio"], "groups": {"orderflow"}},
    "VWAP olmadan": {"copy": {"vwap": "close"}},
    "ADX olmadan": {"neutral": {"adx": 20.0}, "copy": {"di_minus": "di_plus"}},
    "MTF olmadan": {"drop_prefix": ["major_", "trend_", "confirmation_", "macro_"], "groups": {"mtf"}},
    "AI olmadan": {"drop": ["ml_prob_long", "ml_prob_short"], "groups": {"ai"}, "no_meta": True},
}


def rebuild(engine: DecisionEngine, prep: Prepared, f: pd.DataFrame) -> Prepared:
    p = copy.copy(prep)
    p.f = f
    p.gs = group_scores(f)
    p.totals = total_scores(p.gs, engine.cfg.scoring)
    cands = {}
    for st in engine.strategies:
        c = st.candidates(f)
        cands[st.key] = {k: c[k].to_numpy() for k in ("long", "short", "long_strength", "short_strength",
                                                       "long_invalid", "short_invalid")}
    p.candidates = cands
    return p


def ablation(engine: DecisionEngine, prep: Prepared, start: int = 250, end: int | None = None,
             progress=None) -> pd.DataFrame:
    full = run_intel_backtest(engine, prep, start, end, record_decisions=False)
    rows = [{"model": "Tam model", **_row("", full)}]
    for name, spec in ABLATIONS.items():
        if progress:
            progress(f"Ablasyon: {name}")
        f = prep.f.copy()
        for c, v in spec.get("neutral", {}).items():
            if c in f:
                f[c] = v
        for c, src in spec.get("copy", {}).items():
            if c in f and src in f:
                f[c] = f[src]
        for c in spec.get("nan", []):
            if c in f:
                f[c] = np.nan
        drop = [c for c in spec.get("drop", []) if c in f]
        for pref in spec.get("drop_prefix", []):
            drop += [c for c in f.columns if c.startswith(pref) and c != "trend_following"]
        f = f.drop(columns=list(set(drop)))
        p2 = rebuild(engine, prep, f)
        e2 = _engine_like(engine, engine.cfg, None if spec.get("no_meta") else engine.meta, engine.stats)
        r = run_intel_backtest(e2, p2, start, end, record_decisions=False,
                               disabled_groups=spec.get("groups"))
        rows.append({"model": name, **_row("", r)})
    df = pd.DataFrame(rows).drop(columns=["senaryo"])
    base = df.iloc[0]
    df["Δ net_getiri_%"] = df["net_getiri_%"] - base["net_getiri_%"]
    df["Δ kâr_faktörü"] = df["kâr_faktörü"].replace(math.inf, np.nan) - (base["kâr_faktörü"]
                                                                         if math.isfinite(base["kâr_faktörü"]) else np.nan)
    df["yorum"] = np.where(df["Δ net_getiri_%"] < -0.5, "Özellik katkı sağlıyor (çıkarınca kötüleşti)",
                           np.where(df["Δ net_getiri_%"] > 0.5, "Özellik zarar veriyor olabilir (çıkarınca iyileşti)",
                                    "Belirgin katkı yok"))
    df.loc[0, "yorum"] = "Referans"
    return df


# ---------------------------------------------------------------------- kalibrasyon ve korelasyon
def confidence_calibration(trades: list) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame([t.to_dict() for t in trades])
    bins = [0, 50, 65, 75, 85, 101]
    labels = ["0-49", "50-64", "65-74", "75-84", "85-100"]
    df["bant"] = pd.cut(df["confidence"], bins=bins, labels=labels, right=False)
    out = df.groupby("bant", observed=True).agg(işlem=("net_pnl", "size"),
                                                gerçekleşen_kazanma_=("net_pnl", lambda x: (x > 0).mean() * 100),
                                                ort_R=("r_multiple", "mean"), ort_güven=("confidence", "mean"))
    if df["ai_probability"].notna().sum() >= 10:
        p = df["ai_probability"].dropna().to_numpy()
        y = (df.loc[df["ai_probability"].notna(), "net_pnl"] > 0).astype(float).to_numpy()
        out.attrs["ai_calibration"] = calibration_table(p, y, 5)
        out.attrs["ai_brier"] = brier(p, y)
    return out.reset_index()


def feature_correlation(prep: Prepared, threshold: float = 0.8) -> tuple[pd.DataFrame, list]:
    f = prep.f
    cols = {
        "EMA20-50": (f["ema20"] - f["ema50"]) / f["atr"], "SMA20-50": (f["sma20"] - f["sma50"]) / f["atr"],
        "MACD": f["macd"] / f["atr"], "EMA50 eğim": f["ema50_slope"], "RSI": f["rsi"], "Stoch": f["stoch_k"],
        "CCI": f["cci"], "Williams%R": f["willr"], "MFI": f["mfi"], "ROC": f["roc"], "ADX": f["adx"],
        "ATR%": f["atr_pct"], "BB genişlik": f["bb_width"], "RVOL": f["rvol"], "Z-score": f["zscore"],
    }
    gs = prep.gs
    for g in ("trend", "momentum", "volume", "structure", "mtf", "orderflow", "oi_funding"):
        if gs[f"{g}_long"].notna().sum() > 50:
            cols[f"skor:{g}"] = gs[f"{g}_long"]
    corr = pd.DataFrame(cols).corr()
    pairs = []
    names = list(corr.columns)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            v = corr.iat[a, b]
            if np.isfinite(v) and abs(v) >= threshold:
                pairs.append((names[a], names[b], round(float(v), 2)))
    return corr, pairs


# ---------------------------------------------------------------------- optimizasyon (IS / OOS)
EXIT_MODES = ["default", "fixed_tp_sl", "atr_tp_sl", "trailing_only", "breakeven", "time_only", "signal_reversal",
              "regime_exit"]
ENTRY_MODES = ["next_open", "limit_pullback", "retest", "vwap_reclaim"]


def _is_oos(engine, prep, start, split=0.6, **kw):
    n = len(prep.f)
    mid = int(start + (n - start) * split)
    r_is = run_intel_backtest(engine, prep, start, mid, record_decisions=False, **kw)
    r_oos = run_intel_backtest(engine, prep, mid, n, record_decisions=False, **kw)
    return r_is, r_oos


def exit_optimization(engine: DecisionEngine, prep: Prepared, start: int = 250, progress=None) -> pd.DataFrame:
    rows = []
    for mode in EXIT_MODES:
        if progress:
            progress(f"Çıkış modu: {mode}")
        r_is, r_oos = _is_oos(engine, prep, start, exit_mode=mode)
        rows.append({"çıkış": mode, "IS_işlem": r_is.metrics.get("Toplam işlem", 0),
                     "IS_getiri_%": r_is.metrics.get("Net getiri (%)", 0), "OOS_işlem": r_oos.metrics.get("Toplam işlem", 0),
                     "OOS_getiri_%": r_oos.metrics.get("Net getiri (%)", 0),
                     "OOS_PF": r_oos.metrics.get("Kâr faktörü", 0), "OOS_maksDD_%": r_oos.metrics.get("Maks. düşüş (%)", 0),
                     "sağlam": r_is.metrics.get("Net getiri (%)", 0) > 0 and r_oos.metrics.get("Net getiri (%)", 0) > 0})
    return pd.DataFrame(rows)


def entry_optimization(engine: DecisionEngine, prep: Prepared, start: int = 250, progress=None) -> pd.DataFrame:
    rows = []
    for mode in ENTRY_MODES:
        if progress:
            progress(f"Giriş modu: {mode}")
        r_is, r_oos = _is_oos(engine, prep, start, entry_mode=mode)
        rows.append({"giriş": mode, "IS_işlem": r_is.metrics.get("Toplam işlem", 0),
                     "IS_getiri_%": r_is.metrics.get("Net getiri (%)", 0), "OOS_işlem": r_oos.metrics.get("Toplam işlem", 0),
                     "OOS_getiri_%": r_oos.metrics.get("Net getiri (%)", 0), "OOS_ücret": r_oos.metrics.get("Ücretler", 0),
                     "gerçekleşmeyen_sinyal": r_oos.signal_stats.get("expired", 0) + r_oos.signal_stats.get("cancelled", 0),
                     "sağlam": r_is.metrics.get("Net getiri (%)", 0) > 0 and r_oos.metrics.get("Net getiri (%)", 0) > 0})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- strateji sağlığı ve tutma süresi
def strategy_health(trades: list, recent: int = 20, baseline_trades: list | None = None) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame([t.to_dict() for t in trades])
    base = pd.DataFrame([t.to_dict() for t in baseline_trades]) if baseline_trades else df
    rows = []
    for key, g in df.groupby("strategy"):
        last = g.tail(recent)
        gp, gl = last.loc[last["net_pnl"] > 0, "net_pnl"].sum(), -last.loc[last["net_pnl"] <= 0, "net_pnl"].sum()
        pf = gp / gl if gl > 0 else (math.inf if gp > 0 else 0.0)
        wr = (last["net_pnl"] > 0).mean() * 100
        b = base[base["strategy"] == key]
        base_wr = (b["net_pnl"] > 0).mean() * 100 if len(b) else wr
        state, why = StrategyHealth.ACTIVE, "Normal"
        if len(last) >= 10 and pf < 0.7 and last["net_pnl"].sum() < 0:
            state, why = StrategyHealth.PAUSED, f"Son {len(last)} işlemde PF {pf:.2f}"
        elif len(last) >= 10 and (pf < 1.0 or wr < base_wr - 15):
            state, why = StrategyHealth.DEGRADED, f"PF {pf:.2f}, kazanma %{wr:.0f} (temel %{base_wr:.0f})"
        rows.append({"strateji": key, "durum": state.value, "neden": why, "son_işlem": len(last), "PF": pf,
                     "kazanma_%": wr})
    return pd.DataFrame(rows)


def learned_risk(trades: list, keys: list | None = None, min_trades: int = 10) -> dict[str, tuple[float, str]]:
    """Gerçek geçmiş veriyle yapılan backtest sonuçlarından (maliyet dahil) her strateji için risk çarpanı.

    Çarpan yalnızca riski AZALTIR (0.25-1.0): az örnekte temkinli, zarar edende çok düşük,
    istikrarlı kâr edende tam risk. Sonuç bir olasılık değerlendirmesidir, kâr garantisi değildir.
    """
    out: dict[str, tuple[float, str]] = {}
    df = pd.DataFrame([t.to_dict() for t in trades]) if trades else pd.DataFrame()
    groups = dict(tuple(df.groupby("strategy"))) if not df.empty else {}
    for key in sorted(set(keys or []) | set(groups)):
        g = groups.get(key)
        if g is None or len(g) == 0:
            out[key] = (0.5, "Geçmiş testte hiç işlem yok → temkinli yarım risk")
            continue
        n = len(g)
        gp, gl = g.loc[g["net_pnl"] > 0, "net_pnl"].sum(), -g.loc[g["net_pnl"] <= 0, "net_pnl"].sum()
        pf = gp / gl if gl > 0 else (math.inf if gp > 0 else 0.0)
        net = g["net_pnl"].sum()
        pf_txt = "∞" if math.isinf(pf) else f"{pf:.2f}"
        if n < min_trades:
            out[key] = (0.5, f"Yalnızca {n} işlem (en az {min_trades} gerekli) → temkinli yarım risk")
        elif net <= 0 or pf < 1.0:
            out[key] = (0.25, f"{n} işlemde maliyet sonrası zarar (kâr faktörü {pf_txt}) → çeyrek risk")
        elif pf < 1.3:
            out[key] = (0.5, f"{n} işlemde zayıf kâr (kâr faktörü {pf_txt}) → yarım risk")
        elif pf < 1.7 or n < 20:
            out[key] = (0.75, f"{n} işlemde kâr (kâr faktörü {pf_txt}) → %75 risk")
        else:
            out[key] = (1.0, f"{n} işlemde istikrarlı kâr (kâr faktörü {pf_txt}) → tam risk")
    return out


def holding_recommendation(trades: list, interval: str, min_winners: int = 10) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame([t.to_dict() for t in trades])
    rows = []
    for key, g in df.groupby("strategy"):
        w = g[g["net_pnl"] > 0]["holding_min"]
        h = g["holding_min"]
        rows.append({"strateji": key, "işlem": len(g), "ort_dk": h.mean(), "medyan_dk": h.median(),
                     "p25_dk": h.quantile(.25), "p75_dk": h.quantile(.75), "p90_dk": h.quantile(.9), "maks_dk": h.max(),
                     "önerilen_maks_dk": float(w.quantile(.9)) if len(w) >= min_winners else np.nan,
                     "not": "Kazananların %90'ı bu sürede sonuçlandı" if len(w) >= min_winners
                     else f"Yetersiz kazanan işlem (<{min_winners}); yapılandırma korunur"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- araştırma soruları
QUESTIONS = {
    "why": "Bu sinyal neden oluştu?",
    "bull_factors": "En güçlü bullish faktörler hangileri?",
    "against": "Hangi faktörler karşı çıktı?",
    "regime_perf": "Bu strateji mevcut rejimde nasıl performans gösterdi?",
    "holding": "Benzer işlemler ortalama kaç saat açık kaldı?",
    "top_exit": "Bu stratejinin en sık çıkış sebebi ne?",
    "funding": "Funding gerçekten katkı sağlıyor mu?",
    "oi": "OI gerçekten katkı sağlıyor mu?",
    "cvd": "CVD gerçekten katkı sağlıyor mu?",
    "edge_after_fees": "İşlem ücretleri sonrası edge kalıyor mu?",
    "threshold": "Güven eşiği değişirse ne oluyor?",
    "atr": "ATR çarpanı değişirse ne oluyor?",
}


@dataclass
class ResearchContext:
    decision: object = None
    backtest: IntelBacktestResult | None = None
    walk_forward: WalkForwardResult | None = None
    robustness: RobustnessReport | None = None
    ablation: pd.DataFrame | None = None


def answer(q: str, ctx: ResearchContext) -> str:
    d, bt = ctx.decision, ctx.backtest
    need_bt = "Önce bu sembol için gerçek Binance verisiyle backtest çalıştırın."
    if q == "why":
        if d is None:
            return "Henüz bir karar üretilmedi (Zeka Motoru → Karar Üret)."
        return d.explanation
    if q in ("bull_factors", "against"):
        if d is None or not d.scores:
            return "Karar nesnesi yok veya skor hesaplanmadı."
        items = [(g, v[0] / v[1]) for g, v in d.scores.items() if v[0] is not None and v[1] > 0]
        items.sort(key=lambda t: t[1], reverse=(q == "bull_factors"))
        label = "destekleyen" if q == "bull_factors" else "zayıf/karşı"
        sel = [f"{g}: %{s * 100:.0f}" for g, s in items[:4]]
        extra = [w for w in d.warnings] if q == "against" else []
        return f"{d.direction} yönü için en {label} gruplar: " + ", ".join(sel) + \
            ("\nUyarılar: " + "; ".join(extra) if extra else "")
    if bt is None or bt.status != "OK":
        return need_bt
    tdf = bt.trades_frame()
    if q == "regime_perf":
        if d is None or tdf.empty:
            return "Karar veya işlem yok."
        g = tdf[(tdf["strategy"] == d.strategy)]
        same = g[g["regime"] == d.market_regime]
        if same.empty:
            return f"{d.strategy} için {d.market_regime} rejiminde geçmiş işlem yok."
        return (f"{d.strategy} / {d.market_regime}: {len(same)} işlem, kazanma %{(same['net_pnl'] > 0).mean() * 100:.0f}, "
                f"net {same['net_pnl'].sum():+.2f}, ort. R {same['r_multiple'].mean():+.2f}")
    if q == "holding":
        if tdf.empty:
            return "İşlem yok."
        g = tdf[tdf["strategy"] == d.strategy] if d is not None and d.strategy in set(tdf["strategy"]) else tdf
        return (f"{len(g)} benzer işlem: ortalama {g['holding_min'].mean() / 60:.1f} saat, medyan "
                f"{g['holding_min'].median() / 60:.1f} saat, %90'ı {g['holding_min'].quantile(.9) / 60:.1f} saat içinde.")
    if q == "top_exit":
        g = tdf[tdf["strategy"] == d.strategy] if d is not None and not tdf.empty and d.strategy in set(tdf["strategy"]) \
            else tdf
        if g.empty:
            return "İşlem yok."
        vc = g["exit_reason"].value_counts()
        return "Çıkış nedenleri: " + ", ".join(f"{k} {v}" for k, v in vc.items())
    if q in ("funding", "oi", "cvd"):
        if ctx.ablation is None:
            return "Ablasyon testi çalıştırılmadı (Araştırma → Ablasyon)."
        name = {"funding": "Funding olmadan", "oi": "OI olmadan", "cvd": "CVD/Delta olmadan"}[q]
        row = ctx.ablation[ctx.ablation["model"] == name]
        if row.empty:
            return "Sonuç yok."
        r = row.iloc[0]
        return (f"{name}: net getiri farkı {r['Δ net_getiri_%']:+.2f} puan → {r['yorum']}. "
                "(Veri yoksa bu özellik zaten kullanılmamıştır; fark 0 çıkar.)")
    if q == "edge_after_fees":
        m = bt.metrics
        txt = (f"Brüt PnL {m['Brüt PnL']:+.2f}, ücretler {m['Ücretler']:.2f}, funding {m['Funding']:.2f} → "
               f"net {m['Net PnL']:+.2f} ({m['Toplam işlem']} işlem).")
        if ctx.robustness is not None:
            fs = ctx.robustness.fee_stress
            txt += " Ücret stresi: " + ", ".join(f"{r.senaryo} {r._3:+.2f}%" for r in fs.itertuples())
        return txt + (" Edge ücretlerden sonra pozitif." if m["Net PnL"] > 0 else " Ücretlerden sonra edge YOK.")
    if q in ("threshold", "atr"):
        if ctx.robustness is None:
            return "Sağlamlık testi çalıştırılmadı (Araştırma → Sağlamlık)."
        if q == "threshold":
            t = ctx.robustness.threshold
            return "Güven eşiği duyarlılığı:\n" + "\n".join(
                f"  {r.senaryo}: {r.işlem} işlem, net %{r._3:+.2f}, PF {r.kâr_faktörü:.2f}" for r in t.itertuples())
        p = ctx.robustness.perturbation
        p = p[p["senaryo"].str.startswith("stop_atr_mult") | (p["senaryo"] == "temel")]
        return "ATR stop çarpanı duyarlılığı:\n" + "\n".join(
            f"  {r.senaryo}: {r.işlem} işlem, net %{r._3:+.2f}, maks DD %{r._5:.2f}" for r in p.itertuples())
    return "Bilinmeyen soru."
