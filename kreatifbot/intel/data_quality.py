"""Veri kalitesi motoru: sinyal üretmeden önce kritik veri sorunlarını tespit eder.

Kritik sorun varsa karar NO TRADE / DATA_QUALITY_FAILURE olur.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..utils import INTERVALS
from .config import DataQualityConfig


def to_ms(series) -> pd.Series:
    """Zaman damgalarını çözünürlükten bağımsız (ns/us/ms) milisaniyeye çevirir."""
    ts = pd.to_datetime(series, utc=True)
    return ((ts - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(milliseconds=1)).astype("int64")


@dataclass
class QualityReport:
    ok: bool = True
    critical: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def fail(self, msg: str):
        self.ok = False
        self.critical.append(msg)

    def warn(self, msg: str):
        self.warnings.append(msg)

    def merge(self, other: "QualityReport", prefix: str = ""):
        if not other.ok:
            self.ok = False
        self.critical += [f"{prefix}{m}" for m in other.critical]
        self.warnings += [f"{prefix}{m}" for m in other.warnings]
        self.stats.update({f"{prefix}{k}": v for k, v in other.stats.items()})


def check_candles(df: pd.DataFrame, interval: str, cfg: DataQualityConfig | None = None,
                  now_ms: int | None = None, check_stale: bool = True, min_history: int | None = None) -> QualityReport:
    cfg = cfg or DataQualityConfig()
    rep = QualityReport()
    if df is None or df.empty:
        rep.fail("Mum verisi yok (DATA_UNAVAILABLE)")
        return rep
    need = cfg.min_history_bars if min_history is None else min_history
    if len(df) < need:
        rep.fail(f"Yetersiz geçmiş: {len(df)} < {need} mum")

    step_ms = INTERVALS[interval] * 1000
    ot = to_ms(df["open_time"])
    dup = int(ot.duplicated().sum())
    if dup:
        rep.fail(f"{dup} adet tekrarlanan mum")
    diffs = np.diff(ot.to_numpy())
    if (diffs <= 0).any():
        rep.fail("Zaman damgaları sıralı değil")
    if interval not in ("1M",):
        gaps = diffs[diffs > step_ms]
        missing = int(((gaps // step_ms) - 1).sum()) if len(gaps) else 0
        expected = len(df) + missing
        ratio = missing / expected if expected else 0
        rep.stats["missing_candles"] = missing
        rep.stats["missing_ratio"] = ratio
        recent = diffs[-50:]
        recent_missing = int(((recent[recent > step_ms] // step_ms) - 1).sum()) if len(recent) else 0
        if ratio > cfg.max_missing_ratio:
            rep.fail(f"Eksik mum oranı yüksek: %{ratio * 100:.2f} ({missing} mum)")
        elif missing:
            rep.warn(f"{missing} eksik mum (zaman boşluğu)")
        if recent_missing > cfg.max_recent_gap_bars:
            rep.fail(f"Son 50 mumda {recent_missing} eksik mum")

    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    invalid = (h < np.maximum(o, c) - 1e-12) | (l > np.minimum(o, c) + 1e-12) | (l <= 0) | (h < l) | \
        ~np.isfinite(o + h + l + c)
    n_invalid = int(invalid.sum())
    if n_invalid:
        rep.fail(f"{n_invalid} geçersiz OHLC mumu")
    vol = df["volume"].to_numpy(dtype=float)
    if (vol < 0).any() or not np.isfinite(vol).all():
        rep.fail("Geçersiz hacim değeri")

    if len(df) > 60:
        z = (vol[-1] - vol[-51:-1].mean()) / (vol[-51:-1].std() or 1)
        rep.stats["last_volume_z"] = float(z)
        if z > cfg.abnormal_volume_z:
            rep.warn(f"Anormal hacim (z={z:.1f})")
        tr = h[-51:] - l[-51:]
        atr_proxy = tr[:-1].mean()
        if atr_proxy > 0 and tr[-1] / atr_proxy > cfg.abnormal_range_atr:
            rep.warn(f"Anormal mum aralığı ({tr[-1] / atr_proxy:.1f}x ortalama)")
        if (vol[-20:] == 0).sum() >= 10:
            rep.fail("Son 20 mumun yarısında hacim yok (likidite yok)")

    if check_stale:
        now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
        last_close = int(pd.to_datetime(df["close_time"].iloc[-1], utc=True).value // 10**6)
        age = now_ms - last_close
        rep.stats["last_candle_age_s"] = age / 1000
        if age > cfg.stale_multiplier * step_ms:
            rep.fail(f"Bayat veri: son mum {age / 1000:.0f} sn önce kapandı")
    return rep


def check_orderbook(book: dict | None, cfg: DataQualityConfig | None = None, now_ms: int | None = None,
                    fetched_ms: int | None = None) -> QualityReport:
    cfg = cfg or DataQualityConfig()
    rep = QualityReport()
    if not book or not book.get("bids") or not book.get("asks"):
        rep.warn("Order book verisi yok (UNAVAILABLE)")
        rep.stats["orderbook"] = "UNAVAILABLE"
        return rep
    bid = float(book["bids"][0][0])
    ask = float(book["asks"][0][0])
    if bid <= 0 or ask <= 0 or ask < bid:
        rep.fail("Geçersiz order book (ask < bid)")
        return rep
    mid = (bid + ask) / 2
    spread_pct = (ask - bid) / mid * 100
    rep.stats["spread_pct"] = spread_pct
    if spread_pct > cfg.max_spread_pct:
        rep.fail(f"Anormal spread: %{spread_pct:.3f}")
    ts = book.get("T") or book.get("E") or fetched_ms
    if ts and now_ms:
        age = (now_ms - int(ts)) / 1000
        rep.stats["orderbook_age_s"] = age
        if age > cfg.orderbook_max_age_s:
            rep.fail(f"Bayat order book ({age:.0f} sn)")
    return rep


def check_symbol_status(rules) -> QualityReport:
    rep = QualityReport()
    if rules is None:
        rep.fail("Sembol filtreleri bilinmiyor (SYMBOL_FILTER_UNKNOWN)")
    elif not rules.tradable:
        rep.fail(f"Borsa durumu {rules.status} (işlem kapalı)")
    return rep
