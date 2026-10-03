"""Modüler strateji motoru: her strateji bağımsız açılıp kapatılabilir."""

from __future__ import annotations

from ..config import IntelConfig
from .base import INFO_GROUPS, IntelStrategy, StrategySpec
from .library import ALL_STRATEGIES, REGISTRY


def build_strategies(cfg: IntelConfig | None = None, only_enabled: bool = True) -> list[IntelStrategy]:
    out = []
    for key, cls in REGISTRY.items():
        sc = cfg.strategy(key) if cfg is not None else None
        if only_enabled and sc is not None and not sc.enabled:
            continue
        out.append(cls(sc.params if sc else None, sc.short_params if sc else None))
    return out


__all__ = ["ALL_STRATEGIES", "REGISTRY", "INFO_GROUPS", "IntelStrategy", "StrategySpec", "build_strategies"]
