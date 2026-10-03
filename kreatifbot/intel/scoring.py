"""Uyum (confluence) ve sinyal skoru motoru.

LONG ve SHORT için ayrı 0-100 skor üretir. Her grup aynı bilgi kaynağını yalnızca
bir kez kullanır (çifte sayım yok): EMA/SMA/MACD aynı trend bilgisini taşıdığı
için trend grubu EMA dizilimi+eğim+ADX+VWAP'tan; momentum grubu RSI+MACD
histogramı+ROC'tan tek bir ortalama olarak hesaplanır.

Veri yoksa grup UNAVAILABLE olur (0 sayılmaz); kapsam (coverage) düşerse karar
reddedilir. Eşikler yalnızca başlangıç değeridir, kârlı oldukları varsayılmaz.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ScoringConfig
from .orderflow import classify_funding

GROUPS = ["trend", "momentum", "volume", "volatility", "structure", "mtf", "orderflow", "oi_funding", "ai"]


def _clip01(x):
    return np.clip(x, 0.0, 1.0)


def _dir_scores(f: pd.DataFrame, s: int) -> dict[str, pd.Series]:
    close = f["close"]
    out = {}
    # ---- trend (EMA dizilimi + eğim + ADX/DI + VWAP)
    align = (((f["ema20"] - f["ema50"]) * s > 0).astype(float) +
             (((f["ema50"] - f["ema200"]) * s > 0) & f["ema200"].notna()).astype(float)) / \
        (1 + f["ema200"].notna().astype(float))
    slope = _clip01(f["ema50_slope"] * s / 1.0)
    adx_dir = _clip01((f["adx"] - 15) / 25) * ((f["di_plus"] - f["di_minus"]) * s > 0)
    vwap = ((close - f["vwap"]) * s > 0).astype(float)
    trend = 0.35 * align + 0.2 * slope + 0.25 * adx_dir + 0.2 * vwap
    out["trend"] = trend.where(f["ema50"].notna() & f["adx"].notna())

    # ---- momentum (RSI + MACD hist + ROC, tek grup)
    r = f["rsi"] if s > 0 else 100 - f["rsi"]
    rsi_part = _clip01((r - 45) / 25)
    rsi_part = rsi_part.where(r < 78, 0.5)  # aşırı alımda momentum puanı sınırlı
    hist = f["macd_hist"] * s
    macd_part = 0.5 * (hist > 0) + 0.5 * (hist > hist.shift())
    roc_part = _clip01(f["roc"] * s / 2)
    out["momentum"] = ((rsi_part + macd_part + roc_part) / 3).where(f["rsi"].notna() & f["macd_hist"].notna())

    # ---- hacim
    cdir = ((close - f["open"]) * s > 0).astype(float)
    rv = _clip01((f["rvol"] - 0.8) / 1.2) * (0.3 + 0.7 * cdir)
    obv = (f["obv_slope"] * s > 0).astype(float)
    out["volume"] = (0.7 * rv + 0.3 * obv).where(f["rvol"].notna())

    # ---- volatilite (yönden bağımsız uygunluk)
    vp = f["vol_percentile"]
    vol = pd.Series(np.select([vp.between(0.15, 0.85), vp < 0.97], [1.0, 0.5], 0.0), index=f.index)
    out["volatility"] = vol.where(vp.notna())

    # ---- yapı
    st = pd.Series(f["structure_trend"], index=f.index).astype(float) * s
    st_part = (st + 1) / 2
    bos = (f["bos_bull"] if s > 0 else f["bos_bear"]).astype(float).rolling(10, min_periods=1).max()
    opp = f["resistance"] if s > 0 else f["support"]
    room = ((opp - close) * s / f["atr"].replace(0, np.nan))
    room_part = _clip01(room / 2).fillna(1.0)  # karşı seviye yoksa alan açık
    out["structure"] = (0.5 * st_part + 0.2 * bos + 0.3 * room_part).where(f["atr"].notna())

    # ---- MTF
    if any(c in f for c in ("major_bias", "trend_bias", "confirmation_bias", "macro_bias")):
        parts, weights = [], []
        for role, w in (("major", 0.35), ("trend", 0.30), ("confirmation", 0.20), ("macro", 0.15)):
            c = f"{role}_bias"
            if c in f:
                parts.append(f[c] * s * w)
                weights.append(f[c].notna() * w)
        num = pd.concat(parts, axis=1).sum(axis=1, min_count=1)
        den = pd.concat(weights, axis=1).sum(axis=1).replace(0, np.nan)
        out["mtf"] = ((num / den) + 1) / 2
    else:
        out["mtf"] = pd.Series(np.nan, index=f.index)

    # ---- order flow (kline taker delta/CVD)
    if f["delta"].notna().any():
        d3 = np.sign(f["delta"].rolling(3).sum()) * s
        cs = np.sign(f["cvd_slope"]) * s
        tk = (f["taker_buy_ratio"] - 0.5) * s * 4
        of = (0.35 * (d3 > 0) + 0.35 * (cs > 0) + 0.3 * _clip01(tk + 0.5))
        out["orderflow"] = of.where(f["delta"].notna())
    else:
        out["orderflow"] = pd.Series(np.nan, index=f.index)

    # ---- OI / funding
    oi_ok = f["oi_change_pct"].notna()
    price_up = (f["roc"] * s > 0)
    oi_up = f["oi_change_pct"] > 0
    oi_part = pd.Series(np.select([price_up & oi_up, price_up & ~oi_up, ~price_up & ~oi_up], [1.0, 0.55, 0.4], 0.15),
                        index=f.index).where(oi_ok)
    fr = f["funding_rate"]
    fund_part = pd.Series(np.nan, index=f.index)
    if fr.notna().any():
        signed = fr * s  # LONG için pozitif funding = long kalabalığı (riskli)
        fund_part = pd.Series(np.select([signed >= 0.00075, signed >= 0.0003, signed <= -0.0003], [0.2, 0.45, 0.8],
                                        0.6), index=f.index).where(fr.notna())
    combined = pd.concat([oi_part, fund_part], axis=1).mean(axis=1, skipna=True)
    out["oi_funding"] = combined

    # ---- AI
    col = "ml_prob_long" if s > 0 else "ml_prob_short"
    out["ai"] = f[col] if col in f else pd.Series(np.nan, index=f.index)
    return out


def group_scores(f: pd.DataFrame) -> pd.DataFrame:
    """Her mum için LONG/SHORT grup skorları (0..1, veri yoksa NaN)."""
    cols = {}
    for direction, s in (("long", 1), ("short", -1)):
        for g, series in _dir_scores(f, s).items():
            cols[f"{g}_{direction}"] = series.astype(float)
    return pd.DataFrame(cols, index=f.index)


def total_scores(gs: pd.DataFrame, cfg: ScoringConfig, disabled: set | None = None) -> pd.DataFrame:
    """Ağırlıklı 0-100 skor ve kapsam. `disabled` ablasyon testinde grupları kapatır."""
    disabled = disabled or set()
    out = {}
    w_all = sum(w for g, w in cfg.weights.items() if w > 0 and g not in disabled)
    for d in ("long", "short"):
        num = pd.Series(0.0, index=gs.index)
        den = pd.Series(0.0, index=gs.index)
        for g in GROUPS:
            w = cfg.weights.get(g, 0.0)
            if w <= 0 or g in disabled:
                continue
            v = gs[f"{g}_{d}"]
            num += (v.fillna(0) * w)
            den += v.notna() * w
        out[f"score_{d}"] = (num / den.replace(0, np.nan) * 100)
        out[f"coverage_{d}"] = den / w_all if w_all else 0.0
    return pd.DataFrame(out, index=gs.index)


def group_points(row: pd.Series, direction: str, cfg: ScoringConfig) -> dict[str, tuple]:
    """Açıklama/log için grup puanları: grup -> (puan, ağırlık) ya da (None, ağırlık)."""
    d = direction.lower()
    out = {}
    for g in GROUPS:
        w = cfg.weights.get(g, 0.0)
        v = row.get(f"{g}_{d}")
        out[g] = (None if v is None or pd.isna(v) else float(v) * w, w)
    return out


_ = classify_funding
