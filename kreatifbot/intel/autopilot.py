"""Otomatik pilot bakımı: bot çalışırken arka planda periyodik olarak

1. Her sembol için gerçek Binance geçmiş verisini yükler,
2. Backtest ile strateji istatistiklerini (EV için başarı olasılığı, tutma süresi) günceller,
3. Strateji sağlığını hesaplar (bozulan stratejiler DEGRADED / PAUSED olur),
4. Meta modeli yeniden eğitir ve SON verideki ayrı bir doğrulama penceresinde test eder.
   Örneklem dışı AUC eşiği geçilmezse model devreye ALINMAZ (deterministik mod).

Veri alınamazsa o sembol atlanır; sahte veri veya sahte sonuç kullanılmaz.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..config import data_dir
from ..utils import INTERVALS
from .backtest import run_intel_backtest, strategy_stats_from_trades
from .config import IntelConfig
from .decision import DecisionEngine
from .features import compute_features
from .market_data import load_history
from .ml import auc, brier
from .research import label_candidates, learned_risk, strategy_health, train_meta
from .types import StrategyHealth

logger = logging.getLogger("kreatifbot.intel.autopilot")


@dataclass
class MaintenanceResult:
    finished_at: str = ""
    metas: dict = field(default_factory=dict)        # sembol -> MetaModel (yalnızca doğrulamayı geçenler)
    stats: dict = field(default_factory=dict)        # strateji -> istatistik
    health: dict = field(default_factory=dict)       # strateji -> ACTIVE/DEGRADED/PAUSED
    learned: dict = field(default_factory=dict)      # strateji -> (öğrenilen risk çarpanı, gerekçe)
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
        if self.learned:
            full = sum(1 for m, _ in self.learned.values() if m >= 1.0)
            low = sum(1 for m, _ in self.learned.values() if m <= 0.25)
            extra.append(f"Öğrenilen risk: {full} strateji tam risk, {low} strateji çeyrek risk, "
                         f"{len(self.learned) - full - low} strateji arada")
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
    if len(res.errors) < len(symbols):               # en az bir sembolde gerçek veri işlendi
        keys = [s.key for s in (strategies if strategies is not None else DecisionEngine(cfg).strategies)]
        res.learned = learned_risk(all_trades, keys)
        save_learned(res.learned)
    if all_trades:
        interval = cfg.timeframes.entry
        res.stats = strategy_stats_from_trades(all_trades, interval)
        h = strategy_health(all_trades)
        if not h.empty:
            res.health = {r["strateji"]: r["durum"] for _, r in h.iterrows()}
    res.finished_at = datetime.now(timezone.utc).isoformat()
    _ = pd
    return res


def learned_path():
    return data_dir() / "learned_risk.json"


def save_learned(learned: dict) -> None:
    try:
        payload = {"updated_at": datetime.now(timezone.utc).isoformat(),
                   "risk": {k: {"mult": m, "why": w} for k, (m, w) in learned.items()}}
        learned_path().write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError as exc:
        logger.warning("Öğrenilen risk kaydedilemedi: %s", exc)


def load_learned() -> dict[str, tuple[float, str]]:
    """Son bakımda öğrenilen risk çarpanları (dosya yoksa boş → çarpan uygulanmaz)."""
    try:
        data = json.loads(learned_path().read_text(encoding="utf-8"))
        return {k: (min(1.0, max(0.0, float(v["mult"]))), str(v.get("why", ""))) for k, v in data["risk"].items()}
    except (OSError, ValueError, KeyError, TypeError):
        return {}
