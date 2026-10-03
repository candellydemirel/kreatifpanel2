"""Zeka motoru yapılandırması (yönetici ayarları).

Kritik değerlerin hiçbiri kod içinde sabit değildir; hepsi bu yapı üzerinden
değiştirilebilir ve JSON olarak saklanır. Varsayılanlar yalnızca başlangıç
noktasıdır, kârlı oldukları varsayılmaz; walk-forward ile ayarlanmalıdır.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import get_type_hints

from ..config import data_dir
from ..utils import INTERVALS
from .types import LifecycleStage

logger = logging.getLogger("kreatifbot.intel.config")


@dataclass
class TimeframeRoles:
    entry: str = "5m"
    confirmation: str = "15m"
    trend: str = "1h"
    major: str = "4h"
    macro: str = "1d"

    def as_dict(self) -> dict[str, str]:
        return asdict(self)

    def higher(self) -> dict[str, str]:
        """Giriş dışındaki bağlam zaman dilimleri."""
        d = self.as_dict()
        d.pop("entry")
        return d

    def validate(self) -> list[str]:
        errors = []
        for role, tf in self.as_dict().items():
            if tf not in INTERVALS:
                errors.append(f"{role}: Binance desteklemeyen zaman dilimi '{tf}'")
        secs = [INTERVALS.get(tf, 0) for tf in (self.entry, self.confirmation, self.trend, self.major, self.macro)]
        if secs != sorted(secs):
            errors.append("Zaman dilimleri küçükten büyüğe sıralı olmalı (giriş ≤ onay ≤ trend ≤ ana ≤ makro).")
        return errors


@dataclass
class ScoringConfig:
    weights: dict = field(default_factory=lambda: {
        "trend": 20.0, "momentum": 15.0, "volume": 10.0, "volatility": 0.0, "structure": 15.0,
        "mtf": 15.0, "orderflow": 10.0, "oi_funding": 5.0, "ai": 10.0,
    })
    min_trade_score: float = 65.0
    band_weak: float = 50.0
    band_moderate: float = 65.0
    band_strong: float = 75.0
    band_very_strong: float = 85.0
    conflict_margin: float = 10.0
    min_coverage: float = 0.6            # Mevcut verinin toplam ağırlığa oranı
    confluence_bonus_per_family: float = 2.0
    confluence_bonus_max: float = 6.0
    mtf_weights: dict = field(default_factory=lambda: {
        "major": 0.35, "trend": 0.30, "confirmation": 0.20, "macro": 0.15})
    mtf_conflict_threshold: float = -0.25  # Üst zaman dilimleri bu kadar ters ise NO TRADE

    def band(self, score: float) -> str:
        if score >= self.band_very_strong:
            return "VERY_STRONG"
        if score >= self.band_strong:
            return "STRONG"
        if score >= self.band_moderate:
            return "MODERATE"
        if score >= self.band_weak:
            return "WEAK"
        return "NO_TRADE"


@dataclass
class RiskConfig:
    risk_per_trade_pct: float = 0.5
    max_total_exposure_pct: float = 100.0
    max_correlated_exposure_pct: float = 60.0
    correlation_threshold: float = 0.7
    max_open_positions: int = 3
    max_daily_loss_pct: float = 3.0
    max_drawdown_pct: float = 15.0
    max_consecutive_losses: int = 4
    min_rr: float = 1.5
    leverage: int = 1
    max_leverage: int = 5
    margin_type: str = "ISOLATED"
    confidence_size_floor: float = 0.5   # Güven düşükken risk bu orana kadar azalır, asla 1'i aşmaz
    high_vol_size_mult: float = 0.5
    stop_method: str = "atr"             # fixed_pct | atr | swing | structure | volatility | chandelier | supertrend
    stop_fixed_pct: float = 1.5
    stop_atr_mult: float = 2.0
    stop_atr_mult_high_vol: float = 2.5
    stop_atr_mult_low_vol: float = 1.5
    tp_method: str = "rr"                # fixed_pct | atr | rr | structure | fibonacci | volatility | ai
    tp_fixed_pct: float = 3.0
    tp_atr_mult: float = 4.0
    tp_levels_r: list = field(default_factory=lambda: [1.0, 2.0, 3.0, 4.0])
    tp_fractions: list = field(default_factory=lambda: [0.25, 0.25, 0.25, 0.25])
    breakeven_at_r: float = 1.0
    profit_lock_at_r: float = 1.5
    profit_lock_r: float = 0.5
    trailing_at_r: float = 2.0
    trailing_method: str = "atr"         # percent | atr | chandelier | ema | supertrend | swing | volatility
    trailing_pct: float = 1.5
    trailing_atr_mult: float = 2.5
    aggressive_trailing_atr_mult: float = 1.5
    trailing_ema: int = 20
    decay_exit_confidence: float = 45.0
    decay_drop_points: float = 25.0
    liquidation_buffer_atr: float = 1.0
    default_maintenance_margin_rate: float = 0.005  # Binance bracket alınamazsa (tahmin olarak işaretlenir)
    require_probability_for_live: bool = True


@dataclass
class CostConfig:
    spot_maker_fee_pct: float = 0.10
    spot_taker_fee_pct: float = 0.10
    futures_maker_fee_pct: float = 0.02
    futures_taker_fee_pct: float = 0.05
    default_spread_pct: float = 0.02     # Order book yoksa (varsayım olarak işaretlenir)
    slippage_pct: float = 0.03
    latency_ms: float = 500.0
    safety_margin_pct: float = 0.05
    assumed_funding_rate_8h: float = 0.0001  # Funding verisi yoksa EV için temkinli varsayım
    limit_fill_ratio: float = 1.0        # Limit emir kısmi dolum varsayımı (backtest)


@dataclass
class DataQualityConfig:
    min_history_bars: int = 250
    max_missing_ratio: float = 0.01
    max_recent_gap_bars: int = 0         # Son 50 mumda izin verilen boşluk
    stale_multiplier: float = 2.0
    max_spread_pct: float = 0.25
    orderbook_max_age_s: float = 15.0
    abnormal_volume_z: float = 10.0
    abnormal_range_atr: float = 8.0


@dataclass
class CircuitBreakerConfig:
    enabled: bool = True
    flash_move_pct: float = 7.0          # Tek mumda
    extreme_vol_percentile: float = 0.99
    spread_explosion_pct: float = 0.5
    min_book_depth_quote: float = 20000.0
    api_error_limit: int = 5
    liquidation_spike_z: float = 6.0


@dataclass
class FundingConfig:
    neutral_abs: float = 0.0001          # %0.01 / 8s
    elevated_abs: float = 0.0003
    extreme_abs: float = 0.00075


@dataclass
class MLConfig:
    enabled: bool = True
    model: str = "logistic"              # logistic | random_forest (scikit-learn yüklüyse)
    min_train_samples: int = 120
    meta_threshold: float = 0.55
    uncertainty_band: float = 0.05       # |p-0.5| bu değerden küçükse belirsiz
    l2: float = 1.0
    drift_psi_degraded: float = 0.2
    drift_psi_paused: float = 0.35


@dataclass
class StrategyConfig:
    enabled: bool = True
    stage: str = "PAPER"
    risk_multiplier: float = 1.0
    params: dict = field(default_factory=dict)
    short_params: dict = field(default_factory=dict)
    expected_hold_bars: int = 0          # 0 = strateji varsayılanı
    max_hold_bars: int = 0
    expiry_bars: int = 0


@dataclass
class BacktestConfig:
    entry_mode: str = "next_open"        # next_open | limit_pullback | retest | vwap_reclaim
    allow_short_on_spot: bool = False
    initial_equity: float = 10000.0
    walk_forward_train: int = 3000
    walk_forward_validation: int = 1000
    walk_forward_test: int = 1000
    walk_forward_mode: str = "rolling"   # rolling | expanding
    monte_carlo_runs: int = 500


@dataclass
class NewsConfig:
    enabled: bool = True
    poll_seconds: int = 120
    use_binance_announcements: bool = True
    rss_coindesk: bool = True
    rss_cointelegraph: bool = True
    rss_decrypt: bool = True
    block_hours: float = 24.0            # Olumsuz haber sonrası yeni LONG yasağı süresi
    min_severity_block: int = 2
    delist_exit: bool = True             # Delist duyurusunda açık pozisyonu kapat
    notify_min_severity: int = 2


@dataclass
class ListingConfig:
    enabled: bool = True
    stage: str = "PAPER"                 # Canlı için LIMITED_LIVE / FULL_LIVE gerekir
    wait_minutes: int = 15               # Açılıştan sonra beklenecek süre (ilk dakikaların kaosu)
    range_minutes: int = 15              # Açılış aralığı (opening range)
    watch_hours: float = 6.0             # Listeleme sonrası izleme süresi
    max_chase_pct: float = 6.0           # Kırılım seviyesinin bu kadar üstündeyse girme (kovalamama)
    min_quote_volume: float = 2_000_000  # Açılıştan beri işlem hacmi (USDT)
    max_spread_pct: float = 0.3
    min_depth_quote: float = 20_000
    volume_mult: float = 1.5             # Kırılım mumunun hacmi / önceki mumların ortalaması
    max_stop_pct: float = 8.0
    stop_buffer_pct: float = 0.3
    tp_levels_r: list = field(default_factory=lambda: [1.0, 2.0, 3.0])
    tp_fractions: list = field(default_factory=lambda: [0.4, 0.3, 0.3])
    max_hold_minutes: int = 240
    risk_multiplier: float = 0.25        # Normal işlem riskinin çeyreği
    max_concurrent: int = 1
    slippage_mult: float = 3.0           # Listelemelerde kayma varsayımı (backtest)


@dataclass
class IntelConfig:
    market: str = "SPOT"                 # SPOT | USDM_FUTURES
    quote_asset: str = "USDT"
    allowed_symbols: list = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"])
    allowed_timeframes: list = field(default_factory=lambda: list(INTERVALS))
    timeframes: TimeframeRoles = field(default_factory=TimeframeRoles)
    use_futures_context: bool = True     # Spot işlemde futures OI/funding'i bağlam olarak kullan
    btc_context: bool = True
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    data_quality: DataQualityConfig = field(default_factory=DataQualityConfig)
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    funding: FundingConfig = field(default_factory=FundingConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    news: NewsConfig = field(default_factory=NewsConfig)
    listing: ListingConfig = field(default_factory=ListingConfig)
    strategies: dict = field(default_factory=dict)  # anahtar -> StrategyConfig

    def strategy(self, key: str) -> StrategyConfig:
        sc = self.strategies.get(key)
        if sc is None:
            sc = StrategyConfig()
            self.strategies[key] = sc
        return sc

    def validate(self) -> list[str]:
        errors = self.timeframes.validate()
        if self.market not in ("SPOT", "USDM_FUTURES"):
            errors.append("market SPOT veya USDM_FUTURES olmalı")
        r = self.risk
        if not 0 < r.risk_per_trade_pct <= 5:
            errors.append("risk_per_trade_pct 0-5 arasında olmalı")
        if r.leverage < 1 or r.leverage > r.max_leverage:
            errors.append("leverage 1 ile max_leverage arasında olmalı")
        if len(r.tp_levels_r) != len(r.tp_fractions) or abs(sum(r.tp_fractions) - 1) > 1e-6:
            errors.append("tp_levels_r ve tp_fractions aynı uzunlukta olmalı, oranların toplamı 1 olmalı")
        if any(w < 0 for w in self.scoring.weights.values()):
            errors.append("Skor ağırlıkları negatif olamaz")
        for key, sc in self.strategies.items():
            if sc.stage not in [s.value for s in LifecycleStage]:
                errors.append(f"{key}: geçersiz aşama {sc.stage}")
        lc = self.listing
        if len(lc.tp_levels_r) != len(lc.tp_fractions) or abs(sum(lc.tp_fractions) - 1) > 1e-6:
            errors.append("listing.tp_levels_r ve listing.tp_fractions uyumsuz")
        if lc.stage not in [s.value for s in LifecycleStage]:
            errors.append(f"listing: geçersiz aşama {lc.stage}")
        if not 0 < lc.risk_multiplier <= 1:
            errors.append("listing.risk_multiplier 0-1 arasında olmalı")
        for tf in self.allowed_timeframes:
            if tf not in INTERVALS:
                errors.append(f"İzin verilen zaman dilimi geçersiz: {tf}")
        return errors

    def to_dict(self) -> dict:
        return asdict(self)


def _build(cls, data):
    if not is_dataclass(cls) or not isinstance(data, dict):
        return data
    hints = get_type_hints(cls)
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        typ = hints.get(f.name)
        if is_dataclass(typ):
            kwargs[f.name] = _build(typ, value)
        elif cls is IntelConfig and f.name == "strategies":
            kwargs[f.name] = {k: _build(StrategyConfig, v) for k, v in (value or {}).items()}
        elif typ is int and value is not None:
            kwargs[f.name] = int(value)
        elif typ is float and value is not None:
            kwargs[f.name] = float(value)
        else:
            kwargs[f.name] = value
    obj = cls(**kwargs)
    # Sözlük alanlarında yeni anahtarlar eksikse varsayılanlarla tamamla
    if cls is ScoringConfig:
        obj.weights = {**ScoringConfig().weights, **obj.weights}
        obj.mtf_weights = {**ScoringConfig().mtf_weights, **obj.mtf_weights}
    return obj


def config_from_dict(data: dict) -> IntelConfig:
    return _build(IntelConfig, data or {})


def config_path() -> Path:
    return data_dir() / "intel_config.json"


def load_intel_config(path: Path | None = None) -> IntelConfig:
    path = path or config_path()
    if not path.exists():
        return IntelConfig()
    try:
        return config_from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError) as exc:
        logger.error("Zeka motoru ayarları okunamadı, varsayılanlar kullanılıyor: %s", exc)
        return IntelConfig()


def save_intel_config(cfg: IntelConfig, path: Path | None = None) -> None:
    errors = cfg.validate()
    if errors:
        raise ValueError("Geçersiz ayarlar:\n" + "\n".join(errors))
    path = path or config_path()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
