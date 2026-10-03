"""Otomatik pilot bakımı: bot çalışırken arka planda periyodik olarak

1. Her sembol için gerçek Binance geçmiş verisini yükler,
2. Backtest ile strateji istatistiklerini (EV için başarı olasılığı, tutma süresi) günceller,
3. Strateji sağlığını hesaplar (bozulan stratejiler DEGRADED / PAUSED olur),
4. Meta modeli yeniden eğitir ve SON verideki ayrı bir doğrulama penceresinde test eder.
   Örneklem dışı AUC eşiği geçilmezse model devreye ALINMAZ (deterministik mod).

Veri alınamazsa o sembol atlanır; sahte veri veya sahte sonuç kullanılmaz.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..utils import INTERVALS
from .backtest import run_intel_backtest, strategy_stats_from_trades
from .config import IntelConfig
from .decision import DecisionEngine
from .features import compute_features
from .market_data import load_history
from .ml import auc, brier
from .research import label_candidates, strategy_health, train_meta
from .types import StrategyHealth

logger = logging.getLogger("kreatifbot.intel.autopilot")


@dataclass
class MaintenanceResult:
    finished_at: str = ""
    metas: dict = field(default_factory=dict)        # sembol -> MetaModel (yalnızca doğrulamayı geçenler)
    stats: dict = field(default_factory=dict)        # strateji -> istatistik
    health: dict = field(default_factory=dict)       # strateji -> ACTIVE/DEGRADED/PAUSED
    lines: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    def summary(self) -> str:
        paused = [k for k, v in self.health.items() if v == StrategyHealth.PAUSED.value]
        degraded = [k for k, v in self.health.items() if v == StrategyHealth.DEGRADED.value]
        head = (f"Otomatik bakım tamamlandı ({self.finished_at[:16].replace('T', ' ')} UTC): "
                f"{len(self.metas)} sembolde meta model aktif, {len(self.stats)} strateji istatistiği güncellendi.")
        extra = []
        if paused:
            extra.append("Duraklatılan: " + ", ".join(paused))
        if degraded:
            extra.append("Zayıflayan: " + ", ".join(degraded))
        return "\n".join([head] + extra + self.lines + [f"Hata: {e}" for e in self.errors])


def run_maintenance(cfg: IntelConfig, symbols: list[str], spot=None, futures=None, progress=None,
                    strategies=None) -> MaintenanceResult:
    ap = cfg.autopilot
    res = MaintenanceResult()
    all_trades = []
    for sym in symbols:
        try:
            if progress:
                progress(f"Bakım: {sym} verisi yükleniyor")
            bundle = load_history(cfg, sym, ap.history_bars, spot, futures)
            if not bundle.ok:
                res.errors.append(f"{sym}: DATA_UNAVAILABLE ({'; '.join(bundle.errors)[:120]})")
                continue
            de = DecisionEngine(cfg, strategies)
            btc_f = compute_features(bundle.btc_trend, cfg.timeframes.trend, cfg.market) \
                if bundle.btc_trend is not None and not bundle.btc_trend.empty else None
            prep = de.prepare(bundle.entry, sym, cfg.timeframes.entry, bundle.htf, bundle.derivatives, btc_f)
            n = len(prep.f)
            r = run_intel_backtest(de, prep, 250, record_decisions=False)
            all_trades += r.trades
            res.lines.append(f"{sym}: {len(r.trades)} işlem (backtest, maliyet dahil), net "
                             f"%{r.metrics.get('Net getiri (%)', 0):+.2f}")
            # meta model: eğitim [250, split), doğrulama [split, n) — purged
            if cfg.ml.enabled:
                split = int(250 + (n - 250) * (1 - ap.holdout_frac))
                meta, _ = train_meta(prep, cfg, 250, split, purge_until=split, version=f"auto-{sym}-{n}")
                if not meta.ready:
                    res.lines.append(f"{sym}: meta model eğitilemedi ({meta.metrics.get('status')}) → deterministik")
                    continue
                max_bars = max(1, int(720 * 60 / INTERVALS[cfg.timeframes.entry]))
                hold = label_candidates(prep, cfg, split, n, max_bars)
                if prep.market == "SPOT":
                    hold = hold[hold["sign"] == 1]
                hold = hold[hold["outcome"] != 0]
                if len(hold) < ap.min_holdout_samples:
                    res.lines.append(f"{sym}: doğrulama örneği yetersiz ({len(hold)}) → model devreye alınmadı")
                    continue
                p = meta.predict(prep.meta_x, hold["idx"].to_numpy(), hold["sign"].to_numpy())
                y = (hold["outcome"].to_numpy() == 1).astype(float)
                a = auc(p, y)
                if np.isfinite(a) and a >= ap.min_auc:
                    res.metas[sym] = meta
                    res.lines.append(f"{sym}: meta model AKTİF (örneklem dışı AUC {a:.3f}, Brier {brier(p, y):.3f}, "
                                     f"n={len(hold)})")
                else:
                    res.lines.append(f"{sym}: meta model doğrulamayı geçemedi (AUC {a:.3f} < {ap.min_auc}) → "
                                     "deterministik mod")
        except Exception as exc:  # noqa: BLE001 - bir sembolün hatası bakımı durdurmasın
            logger.exception("bakım hatası")
            res.errors.append(f"{sym}: {exc}")
    if all_trades:
        interval = cfg.timeframes.entry
        res.stats = strategy_stats_from_trades(all_trades, interval)
        h = strategy_health(all_trades)
        if not h.empty:
            res.health = {r["strateji"]: r["durum"] for _, r in h.iterrows()}
    res.finished_at = datetime.now(timezone.utc).isoformat()
    _ = pd
    return res
