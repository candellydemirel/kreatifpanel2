"""Pozisyon ve işlem kayıtları."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class Position:
    symbol: str
    qty: float
    entry_price: float
    cost: float            # Ücretler dahil harcanan karşı varlık (ör. USDT)
    stop_loss: float
    take_profit: float
    highest: float
    opened_at: str
    strategy: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Position":
        return cls(**d)

    def unrealized(self, price: float) -> tuple[float, float]:
        value = self.qty * price
        pnl = value - self.cost
        return pnl, (pnl / self.cost * 100 if self.cost else 0.0)


@dataclass
class ClosedTrade:
    symbol: str
    qty: float
    entry_price: float
    exit_price: float
    cost: float
    proceeds: float
    pnl: float
    pnl_pct: float
    opened_at: str
    closed_at: str
    reason: str
    strategy: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ClosedTrade":
        return cls(**d)
