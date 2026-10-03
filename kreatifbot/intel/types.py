"""Ortak tipler: yön, rejim, sinyal durumu, çıkış nedenleri, kaynaklı özellik değeri."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum


class StrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class MarketType(StrEnum):
    SPOT = "SPOT"
    FUTURES = "USDM_FUTURES"


class Direction(StrEnum):
    LONG = "LONG"
    SHORT = "SHORT"
    NONE = "NONE"


class Regime(StrEnum):
    STRONG_BULL = "STRONG_BULL"
    BULL = "BULL"
    WEAK_BULL = "WEAK_BULL"
    SIDEWAYS = "SIDEWAYS"
    CHOP = "CHOP"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    BEAR = "BEAR"
    STRONG_BEAR = "STRONG_BEAR"
    PANIC = "PANIC"
    UNKNOWN = "UNKNOWN"


BULL_REGIMES = {Regime.STRONG_BULL, Regime.BULL, Regime.WEAK_BULL}
BEAR_REGIMES = {Regime.STRONG_BEAR, Regime.BEAR}
RANGE_REGIMES = {Regime.SIDEWAYS, Regime.CHOP, Regime.LOW_VOLATILITY}


class SignalStatus(StrEnum):
    SCANNING = "SCANNING"
    CANDIDATE = "CANDIDATE"
    CONFIRMED = "CONFIRMED"
    ACTIVE = "ACTIVE"
    WEAKENING = "WEAKENING"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"
    EXECUTED = "EXECUTED"
    CLOSED = "CLOSED"
    NO_TRADE = "NO_TRADE"


class ExitReason(StrEnum):
    TP_HIT = "TP_HIT"
    SL_HIT = "SL_HIT"
    TRAILING_STOP = "TRAILING_STOP"
    TIME_EXPIRY = "TIME_EXPIRY"
    SIGNAL_REVERSAL = "SIGNAL_REVERSAL"
    SIGNAL_DECAY = "SIGNAL_DECAY"
    REGIME_CHANGE = "REGIME_CHANGE"
    RISK_LIMIT = "RISK_LIMIT"
    EMERGENCY_EXIT = "EMERGENCY_EXIT"
    MANUAL_EXIT = "MANUAL_EXIT"
    EXECUTION_FAILURE = "EXECUTION_FAILURE"
    END_OF_DATA = "END_OF_DATA"


class NoTradeReason(StrEnum):
    DATA_QUALITY_FAILURE = "DATA_QUALITY_FAILURE"
    DATA_UNAVAILABLE = "DATA_UNAVAILABLE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    STALE_DATA = "STALE_DATA"
    LOW_LIQUIDITY = "LOW_LIQUIDITY"
    SPREAD_TOO_HIGH = "SPREAD_TOO_HIGH"
    ABNORMAL_VOLATILITY = "ABNORMAL_VOLATILITY"
    CONFLICTING_TIMEFRAMES = "CONFLICTING_TIMEFRAMES"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    CONFLICT = "CONFLICT"
    POOR_RR = "POOR_RR"
    EDGE_BELOW_COSTS = "EDGE_BELOW_COSTS"
    NEGATIVE_EXPECTANCY = "NEGATIVE_EXPECTANCY"
    EXCESSIVE_PORTFOLIO_RISK = "EXCESSIVE_PORTFOLIO_RISK"
    CORRELATED_EXPOSURE = "CORRELATED_EXPOSURE"
    REGIME_INCOMPATIBLE = "REGIME_INCOMPATIBLE"
    NO_CANDIDATE = "NO_CANDIDATE"
    AI_UNCERTAIN = "AI_UNCERTAIN"
    META_MODEL_REJECT = "META_MODEL_REJECT"
    API_ERROR = "API_ERROR"
    CIRCUIT_BREAKER = "CIRCUIT_BREAKER"
    SHORT_NOT_SUPPORTED = "SHORT_NOT_SUPPORTED_ON_SPOT"
    POSITION_SIZE_FAILURE = "POSITION_SIZE_FAILURE"
    SYMBOL_FILTER_UNKNOWN = "SYMBOL_FILTER_UNKNOWN"
    LIQUIDATION_RISK = "LIQUIDATION_RISK"
    STRATEGY_NOT_LIVE = "STRATEGY_NOT_LIVE"
    STRATEGY_PAUSED = "STRATEGY_PAUSED"
    POSITION_EXISTS = "POSITION_EXISTS"
    DATABASE_FAILURE = "DATABASE_FAILURE"
    NEWS_RISK = "NEWS_RISK"


class StrategyHealth(StrEnum):
    ACTIVE = "ACTIVE"
    DEGRADED = "DEGRADED"
    PAUSED = "PAUSED"
    DISABLED = "DISABLED"


class LifecycleStage(StrEnum):
    RESEARCH = "RESEARCH"
    BACKTEST = "BACKTEST"
    WALK_FORWARD = "WALK_FORWARD"
    PAPER = "PAPER"
    SHADOW = "SHADOW"
    LIMITED_LIVE = "LIMITED_LIVE"
    FULL_LIVE = "FULL_LIVE"


STAGE_ORDER = list(LifecycleStage)
LIVE_STAGES = {LifecycleStage.LIMITED_LIVE, LifecycleStage.FULL_LIVE}
PAPER_STAGES = {LifecycleStage.PAPER, LifecycleStage.SHADOW} | LIVE_STAGES


@dataclass
class Feature:
    """Kaynağı belirtilmiş tek bir değer. Veri yoksa value=None ve available=False."""
    name: str
    value: object = None
    source: str = ""
    available: bool = True
    reason: str = ""
    timestamp: str = ""

    @classmethod
    def unavailable(cls, name: str, source: str, reason: str) -> "Feature":
        return cls(name=name, value=None, source=source, available=False, reason=reason)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FeatureSet:
    """Kaynak etiketli özellik koleksiyonu (spot ve futures ayrı alanlar)."""
    items: dict[str, Feature] = field(default_factory=dict)

    def add(self, f: Feature):
        self.items[f.name] = f

    def get(self, name: str, default=None):
        f = self.items.get(name)
        return f.value if f is not None and f.available else default

    def is_available(self, name: str) -> bool:
        f = self.items.get(name)
        return bool(f and f.available)

    def unavailable(self) -> dict[str, str]:
        return {k: f.reason for k, f in self.items.items() if not f.available}

    def to_rows(self) -> list[list]:
        return [[f.name, "UNAVAILABLE" if not f.available else f.value, f.source,
                 "OK" if f.available else f.reason] for f in self.items.values()]
