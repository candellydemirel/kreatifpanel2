"""Strateji yönlendirici: mevcut rejime göre stratejileri aktif/pasif yapar."""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import IntelConfig
from .strategies import IntelStrategy
from .types import LIVE_STAGES, LifecycleStage, Regime, StrategyHealth

GLOBAL_BLOCK = {Regime.PANIC: "PANIC: yeni işlemler durduruldu (savunma modu)",
                Regime.UNKNOWN: "UNKNOWN: rejim belirsiz, temkinli mod (işlem yok)"}


@dataclass
class RouteResult:
    regime: Regime
    active: dict = field(default_factory=dict)      # yön -> [strateji anahtarları]
    inactive: dict = field(default_factory=dict)    # anahtar -> neden
    blocked_reason: str = ""
    threshold_bonus: float = 0.0
    size_mult: float = 1.0


def route(regime: Regime, strategies: list[IntelStrategy], cfg: IntelConfig, market: str,
          health: dict | None = None, live: bool = False) -> RouteResult:
    res = RouteResult(regime=regime, active={"LONG": [], "SHORT": []})
    health = health or {}
    if regime in GLOBAL_BLOCK:
        res.blocked_reason = GLOBAL_BLOCK[regime]
    if regime == Regime.HIGH_VOLATILITY:
        res.threshold_bonus = 5.0       # Daha güçlü onay
        res.size_mult = cfg.risk.high_vol_size_mult
    for st in strategies:
        sc = cfg.strategy(st.key)
        h = health.get(st.key, StrategyHealth.ACTIVE)
        if not sc.enabled or h == StrategyHealth.DISABLED:
            res.inactive[st.key] = "Kapalı"
            continue
        if h == StrategyHealth.PAUSED:
            res.inactive[st.key] = "PAUSED (performans/drift)"
            continue
        if live and LifecycleStage(sc.stage) not in LIVE_STAGES:
            res.inactive[st.key] = f"Aşama {sc.stage}: canlı işlem izni yok"
            continue
        reasons = []
        for direction in ("LONG", "SHORT"):
            if direction == "SHORT" and market == "SPOT" and not cfg.backtest.allow_short_on_spot:
                reasons.append("SHORT: spot piyasada açığa satış yok")
                continue
            if direction not in st.spec.directions:
                continue
            if regime in st.spec.forbidden_for(direction):
                reasons.append(f"{direction}: {regime.value} rejiminde yasak")
                continue
            if regime not in st.spec.preferred_for(direction):
                reasons.append(f"{direction}: {regime.value} tercih edilen rejim değil")
                continue
            res.active[direction].append(st.key)
        if not any(st.key in v for v in res.active.values()):
            res.inactive[st.key] = "; ".join(reasons) or "Uygun yön yok"
    return res
