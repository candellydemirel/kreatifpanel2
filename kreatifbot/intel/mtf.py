"""Çoklu zaman dilimi (MTF) analizi.

Üst zaman dilimi bilgisi giriş zaman dilimine yalnızca o üst mum KAPANDIKTAN
sonra aktarılır (merge_asof, backward, close_time) → ileriye bakma yok.
Alt zaman dilimi sinyali üst bağlamı kör şekilde geçersiz kılamaz: üst zaman
dilimleri ters yöndeyse MTF skoru düşer veya çatışma (NO TRADE) oluşur.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import htf_bias
from .regime import classify_regime


def htf_summary(f: pd.DataFrame) -> pd.DataFrame:
    reg = classify_regime(f)
    return pd.DataFrame({
        "close_time": pd.to_datetime(f["close_time"], utc=True).dt.as_unit("ns"),
        "bias": htf_bias(f),
        "regime": reg["regime"],
        "adx": f["adx"],
        "rsi": f["rsi"],
    })


def align_htf(entry_f: pd.DataFrame, htf: pd.DataFrame, role: str) -> pd.DataFrame:
    """htf_summary çıktısını giriş zaman dilimine nedensel olarak hizalar."""
    left = pd.DataFrame({"close_time": pd.to_datetime(entry_f["close_time"], utc=True).dt.as_unit("ns"),
                         "_row": np.arange(len(entry_f))})
    right = htf.rename(columns={c: f"{role}_{c}" for c in htf.columns if c != "close_time"})
    right = right.rename(columns={"close_time": "_htf_close"}).sort_values("_htf_close")
    merged = pd.merge_asof(left.sort_values("close_time"), right, left_on="close_time",
                           right_on="_htf_close", direction="backward").sort_values("_row")
    merged.index = entry_f.index
    return merged.drop(columns=["close_time", "_row", "_htf_close"])


def add_mtf(entry_f: pd.DataFrame, htf_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """htf_frames: rol -> o zaman diliminin özellik tablosu."""
    out = entry_f.copy()
    out["entry_bias"] = htf_bias(entry_f)
    for role, f in htf_frames.items():
        if f is None or f.empty:
            continue
        aligned = align_htf(entry_f, htf_summary(f), role)
        for c in aligned.columns:
            out[c] = aligned[c]
    return out


def mtf_score(row: pd.Series, direction: str, weights: dict) -> tuple[float | None, dict]:
    """Yöne göre -1..1 MTF uyum skoru ve rol bazında detay. Hiç üst TF yoksa None."""
    sign = 1 if direction == "LONG" else -1
    total_w, acc, detail = 0.0, 0.0, {}
    for role, w in weights.items():
        b = row.get(f"{role}_bias")
        if b is None or pd.isna(b):
            detail[role] = None
            continue
        detail[role] = float(b)
        acc += w * sign * float(b)
        total_w += w
    if total_w == 0:
        return None, detail
    return acc / total_w, detail


def bias_label(b) -> str:
    if b is None or pd.isna(b):
        return "UNAVAILABLE"
    if b >= 0.5:
        return "Bullish"
    if b >= 0.15:
        return "Hafif bullish"
    if b <= -0.5:
        return "Bearish"
    if b <= -0.15:
        return "Hafif bearish"
    return "Nötr"
