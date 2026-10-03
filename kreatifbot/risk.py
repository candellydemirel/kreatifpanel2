"""Risk yönetimi: pozisyon boyutu, stop-loss / kâr al, iz süren stop ve günlük zarar limiti."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields


@dataclass
class RiskSettings:
    risk_per_trade_pct: float = 1.0     # İşlem başına sermayenin riske edilecek yüzdesi
    stop_atr_mult: float = 2.0          # Stop mesafesi = ATR x bu değer (0 = stop yok)
    take_profit_rr: float = 2.0         # Kâr al = stop mesafesi x bu değer (0 = kapalı)
    trailing_atr_mult: float = 0.0      # İz süren stop = en yüksek - ATR x bu değer (0 = kapalı)
    max_position_pct: float = 25.0      # Tek pozisyon sermayenin en fazla bu yüzdesi
    max_open_positions: int = 3
    max_daily_loss_pct: float = 5.0     # Gün içi bu kadar zarar olursa yeni işlem açılmaz (0 = kapalı)
    fee_pct: float = 0.1                # İşlem ücreti (backtest / kağıt işlem)
    slippage_pct: float = 0.05          # Kayma (backtest / kağıt işlem)

    LABELS = {
        "risk_per_trade_pct": "İşlem başı risk (%)",
        "stop_atr_mult": "Stop-loss (ATR x)",
        "take_profit_rr": "Kâr al (risk/ödül)",
        "trailing_atr_mult": "İz süren stop (ATR x, 0=kapalı)",
        "max_position_pct": "Maks. pozisyon (% sermaye)",
        "max_open_positions": "Maks. açık pozisyon",
        "max_daily_loss_pct": "Günlük maks. zarar (%)",
        "fee_pct": "İşlem ücreti (%)",
        "slippage_pct": "Kayma (%)",
    }

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "RiskSettings":
        data = data or {}
        kwargs = {}
        for f in fields(cls):
            if f.name in data:
                kwargs[f.name] = int(data[f.name]) if f.type in ("int", int) else float(data[f.name])
        return cls(**kwargs)


class RiskManager:
    def __init__(self, settings: RiskSettings):
        self.s = settings

    def position_size(self, equity: float, price: float, atr_value: float | None) -> float:
        """Alınacak miktar (baz varlık cinsinden)."""
        if equity <= 0 or price <= 0:
            return 0.0
        max_qty = equity * self.s.max_position_pct / 100 / price
        if atr_value and math.isfinite(atr_value) and atr_value > 0 and self.s.stop_atr_mult > 0:
            risk_amount = equity * self.s.risk_per_trade_pct / 100
            qty = risk_amount / (atr_value * self.s.stop_atr_mult)
            return max(0.0, min(qty, max_qty))
        return max_qty

    def stops(self, entry: float, atr_value: float | None) -> tuple[float, float]:
        """(stop-loss, kâr al) fiyatları. 0 = tanımsız."""
        if not atr_value or not math.isfinite(atr_value) or atr_value <= 0 or self.s.stop_atr_mult <= 0:
            return 0.0, 0.0
        distance = atr_value * self.s.stop_atr_mult
        stop = entry - distance
        if stop <= 0:
            stop = entry * 0.5
            distance = entry - stop
        take = entry + distance * self.s.take_profit_rr if self.s.take_profit_rr > 0 else 0.0
        return stop, take

    def trail(self, current_stop: float, highest: float, atr_value: float | None) -> float:
        if self.s.trailing_atr_mult <= 0 or not atr_value or not math.isfinite(atr_value):
            return current_stop
        return max(current_stop, highest - atr_value * self.s.trailing_atr_mult)

    def daily_loss_hit(self, day_start_equity: float, equity: float) -> bool:
        if self.s.max_daily_loss_pct <= 0 or day_start_equity <= 0:
            return False
        return equity <= day_start_equity * (1 - self.s.max_daily_loss_pct / 100)
