"""Zeka motoru strateji tabanı.

Her strateji bağımsız bir modüldür ve yalnızca ADAY sinyal üretir. Nihai kararı
Strateji Yönlendirici + Uyum (confluence) Motoru + Risk Motoru verir.

Yön simetrisi: koşullar `s` (+1 LONG, -1 SHORT) ile yazılır; SHORT için ayrı
parametre seti (`short_params`) kullanılabilir — piyasa asimetrisi varsa farklı
eşikler tanımlanabilir.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..types import Regime

R = Regime
TREND_LONG = {R.STRONG_BULL, R.BULL}
NO_GO = {R.PANIC, R.UNKNOWN}

_MIRROR = {
    R.STRONG_BULL: R.STRONG_BEAR, R.BULL: R.BEAR, R.WEAK_BULL: R.BEAR,
    R.STRONG_BEAR: R.STRONG_BULL, R.BEAR: R.BULL,
}

STYLE_HOLD_MIN = {  # (beklenen, minimum, maksimum, sinyal ömrü) dakika — yalnızca başlangıç ayarı
    "scalp": (15, 5, 60, 10),
    "intraday": (180, 15, 720, 15),
    "swing": (720, 60, 4320, 60),
}

# Aynı bilgi kaynağını kullanan stratejiler tek onay sayılır (çifte sayım engeli)
INFO_GROUPS = {
    "ema": "trend", "macd": "trend", "supertrend": "trend", "adx": "trend",
    "momentum": "momentum",
    "mean_reversion": "mean_reversion", "statistical": "mean_reversion",
    "vwap": "vwap",
    "breakout": "breakout", "volatility": "breakout",
    "structure": "structure",
    "orderflow": "orderflow",
    "derivatives": "derivatives",
    "meta": "meta",
}


def mirror(regimes: set) -> set:
    return {_MIRROR.get(r, r) for r in regimes}


@dataclass
class StrategySpec:
    key: str
    name: str
    family: str
    style: str = "intraday"
    preferred: set = field(default_factory=set)      # LONG için; SHORT otomatik aynalanır
    forbidden: set = field(default_factory=set)
    requires: tuple = ()
    directions: tuple = ("LONG", "SHORT")
    risk_multiplier: float = 1.0
    stop_method: str = "atr"
    stop_mult: float = 2.0
    tp_rr: float = 2.0
    entry: str = ""
    confirmation: str = ""
    invalidation: str = ""
    exit: str = ""
    version: str = "1.0"

    @property
    def info_group(self) -> str:
        return INFO_GROUPS.get(self.family, self.family)

    def preferred_for(self, direction: str) -> set:
        return set(self.preferred) if direction == "LONG" else mirror(self.preferred)

    def forbidden_for(self, direction: str) -> set:
        base = set(self.forbidden) | NO_GO
        return base if direction == "LONG" else mirror(base)

    def holding_minutes(self) -> tuple[int, int, int, int]:
        return STYLE_HOLD_MIN[self.style]


# ---------------------------------------------------------------------- yardımcılar
def X(a, b, s: int) -> pd.Series:
    """Yöne göre 'a, b'nin üstünde' (LONG) / 'altında' (SHORT)."""
    return ((a - b) * s > 0).fillna(False) if isinstance(a - b, pd.Series) else (a - b) * s > 0


def cross(a: pd.Series, b, s: int) -> pd.Series:
    b_prev = b.shift() if isinstance(b, pd.Series) else b
    cur = (a - b) * s > 0
    prev = (a.shift() - b_prev) * s > 0
    valid = (a.shift() - b_prev).notna() & (a - b).notna()
    return (cur & ~prev & valid).fillna(False)


def osc(f: pd.DataFrame, col: str, s: int) -> pd.Series:
    """0-100 osilatörü yöne göre aynalar (SHORT için 100 - değer)."""
    return f[col] if s > 0 else 100 - f[col]


def edge(cond: pd.Series) -> pd.Series:
    cond = cond.fillna(False).astype(bool)
    return cond & ~cond.shift(fill_value=False)


def band_far(f: pd.DataFrame, s: int, lower: str = "bb_lower", upper: str = "bb_upper") -> pd.Series:
    return f[lower] if s > 0 else f[upper]


def band_near(f: pd.DataFrame, s: int, lower: str = "bb_lower", upper: str = "bb_upper") -> pd.Series:
    return f[upper] if s > 0 else f[lower]


def candle_dir(f: pd.DataFrame, s: int) -> pd.Series:
    return (f["close"] - f["open"]) * s > 0


def close_location(f: pd.DataFrame, s: int) -> pd.Series:
    """Kapanışın mum aralığındaki konumu, yöne göre (1 = en güçlü)."""
    rng = (f["high"] - f["low"]).replace(0, np.nan)
    loc = (f["close"] - f["low"]) / rng
    return loc if s > 0 else 1 - loc


class IntelStrategy:
    spec: StrategySpec
    default_params: dict = {}
    default_short_params: dict | None = None

    def __init__(self, params: dict | None = None, short_params: dict | None = None):
        self.params = {**self.default_params, **(params or {})}
        base_short = self.default_short_params if self.default_short_params is not None else self.default_params
        self.short_params = {**base_short, **(params or {}), **(short_params or {})}

    @property
    def key(self) -> str:
        return self.spec.key

    @property
    def name(self) -> str:
        return self.spec.name

    def missing_requirements(self, f: pd.DataFrame) -> list[str]:
        return [c for c in self.spec.requires if c not in f or f[c].notna().sum() == 0]

    # alt sınıflar uygular
    def setup(self, f: pd.DataFrame, p: dict, s: int) -> pd.Series:
        raise NotImplementedError

    def invalidation(self, f: pd.DataFrame, p: dict, s: int) -> pd.Series:
        """Varsayılan geçersizleşme: ters yönde kurulum."""
        return self.setup(f, p, -s)

    def strength(self, f: pd.DataFrame, p: dict, s: int) -> pd.Series:
        """0..1 kurulum gücü (varsayılan: ADX ve hacim)."""
        adx = (f["adx"] / 50).clip(0, 1).fillna(0)
        rv = (f["rvol"] / 2).clip(0, 1).fillna(0)
        return (0.6 * adx + 0.4 * rv).clip(0, 1)

    def candidates(self, f: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=f.index)
        missing = self.missing_requirements(f)
        for direction, s, p in (("LONG", 1, self.params), ("SHORT", -1, self.short_params)):
            col = direction.lower()
            if missing or direction not in self.spec.directions:
                out[col] = False
                out[f"{col}_strength"] = 0.0
                out[f"{col}_invalid"] = False
                continue
            sig = self.setup(f, p, s).fillna(False).astype(bool)
            req_ok = pd.Series(True, index=f.index)
            for c in self.spec.requires:
                req_ok &= f[c].notna()
            out[col] = sig & req_ok
            out[f"{col}_strength"] = self.strength(f, p, s).where(out[col], 0.0)
            out[f"{col}_invalid"] = self.invalidation(f, p, s).fillna(False).astype(bool)
        out.attrs["missing"] = missing
        return out
