"""Bot motoru: arka planda çalışır, sinyalleri izler, pozisyon açar/kapatır."""

from __future__ import annotations

import json
import logging
import math
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from .broker import BrokerError
from .indicators import atr
from .models import ClosedTrade, Position
from .risk import RiskManager, RiskSettings
from .strategies import BUY, SELL, Strategy

logger = logging.getLogger("kreatifbot.engine")

MIN_ORDER_QUOTE = 6.0  # Binance minimum işlem tutarı genelde 5 USDT


def _now() -> datetime:
    return datetime.now(timezone.utc)


class EngineCore:
    """Klasik ve Zeka motorlarının ortak altyapısı: iş parçacığı, olaylar ve kayıt."""

    poll_seconds: float = 30.0
    on_event: Callable[[str, object], None] | None = None

    def _init_core(self, poll_seconds: float, on_event):
        self.poll_seconds = max(5.0, float(poll_seconds))
        self.on_event = on_event
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ olaylar
    def _emit(self, kind: str, payload=None):
        if self.on_event:
            try:
                self.on_event(kind, payload)
            except Exception:  # GUI hatası motoru durdurmamalı
                logger.exception("Olay iletilemedi")

    def log(self, message: str, level: int = logging.INFO):
        logger.log(level, message)
        stamp = datetime.now().strftime("%H:%M:%S")
        self._emit("log", f"[{stamp}] {message}")
        if level >= logging.WARNING:
            self._emit("alert", {"level": level, "message": message})

    # ------------------------------------------------------------------ yaşam döngüsü
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=type(self).__name__, daemon=True)
        self._thread.start()

    def stop(self, wait: bool = False):
        self._stop.set()
        if wait and self._thread:
            self._thread.join(timeout=30)

    def start_message(self) -> str:
        return "Bot başlatıldı"

    def _run(self):
        self.log(self.start_message())
        self._emit("status", "running")
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception as exc:
                self.log(f"Beklenmeyen hata: {exc}", logging.ERROR)
                logger.exception("tick hatası")
            self._stop.wait(self.poll_seconds)
        self.log("Bot durduruldu.")
        self._emit("status", "stopped")

    def tick(self):
        raise NotImplementedError


class BotEngine(EngineCore):
    def __init__(self, client, broker, strategy: Strategy, risk: RiskSettings, symbols: list[str],
                 interval: str, quote_asset: str = "USDT", poll_seconds: float = 30,
                 state_path: str | Path | None = None,
                 on_event: Callable[[str, object], None] | None = None,
                 kline_limit: int = 300):
        self.client = client
        self.broker = broker
        self.strategy = strategy
        self.risk = risk
        self.rm = RiskManager(risk)
        self.symbols = [s.strip().upper() for s in symbols if s.strip()]
        self.interval = interval
        self.quote_asset = quote_asset.upper()
        self._init_core(poll_seconds, on_event)
        self.state_path = Path(state_path) if state_path else None
        self.kline_limit = max(kline_limit, strategy.min_bars() + 20)

        self.positions: dict[str, Position] = {}
        self.trades: list[ClosedTrade] = []
        self.last_prices: dict[str, float] = {}
        self.last_signals: dict[str, dict] = {}
        self._last_bar: dict[str, pd.Timestamp] = {}
        self._day = None
        self._day_start_equity = 0.0
        self.halted_today = False

        bad = [s for s in self.symbols if not s.endswith(self.quote_asset)]
        if bad:
            raise ValueError(f"Bu semboller {self.quote_asset} ile bitmiyor: {', '.join(bad)}")
        if not self.symbols:
            raise ValueError("En az bir sembol girin.")
        self._load_state()

    def start_message(self) -> str:
        mode = "CANLI" if getattr(self.broker, "is_live", False) else "KAĞIT (simülasyon)"
        return (f"Bot başlatıldı | Mod: {mode} | Strateji: {self.strategy.name} | "
                f"Semboller: {', '.join(self.symbols)} | Aralık: {self.interval}")

    # ------------------------------------------------------------------ ana döngü
    def tick(self):
        with self._lock:
            for symbol in self.symbols:
                if self._stop.is_set():
                    break
                try:
                    self._process(symbol)
                except (BrokerError, ValueError) as exc:
                    self.log(f"{symbol}: {exc}", logging.WARNING)
                except Exception as exc:
                    self.log(f"{symbol}: hata - {exc}", logging.ERROR)
            self._update_equity()

    def _process(self, symbol: str):
        df = self.client.klines(symbol, self.interval, limit=self.kline_limit)
        if df.empty:
            return
        price = float(df["close"].iloc[-1])
        self.last_prices[symbol] = price
        closed = df[df["close_time"] <= pd.Timestamp(_now())].reset_index(drop=True)
        if len(closed) < self.strategy.min_bars():
            self.log(f"{symbol}: yetersiz veri ({len(closed)} mum).", logging.WARNING)
            return
        atr_val = float(atr(closed, 14).iloc[-1])

        pos = self.positions.get(symbol)
        if pos is not None:
            pos.highest = max(pos.highest, price)
            new_stop = self.rm.trail(pos.stop_loss, pos.highest, atr_val)
            if new_stop > pos.stop_loss:
                pos.stop_loss = new_stop
                self._save_state()
            if pos.stop_loss > 0 and price <= pos.stop_loss:
                self._close(symbol, price, "Stop-loss")
                return
            if pos.take_profit > 0 and price >= pos.take_profit:
                self._close(symbol, price, "Kâr al")
                return

        bar_time = closed["open_time"].iloc[-1]
        if self._last_bar.get(symbol) == bar_time:
            return  # Bu mum zaten değerlendirildi.
        self._last_bar[symbol] = bar_time

        result = self.strategy.evaluate(closed)
        self.last_signals[symbol] = {"signal": result.text, "reason": result.reason,
                                     "time": bar_time.strftime("%Y-%m-%d %H:%M"), "price": price}
        self._emit("signal", {"symbol": symbol, **self.last_signals[symbol]})

        if pos is not None and result.signal == SELL:
            self._close(symbol, price, "Strateji sinyali")
        elif pos is None and result.signal == BUY:
            self.log(f"{symbol}: AL sinyali ({result.reason})")
            self._open(symbol, price, atr_val)

    # ------------------------------------------------------------------ işlemler
    def equity(self) -> float:
        total = self.broker.quote_balance()
        for sym, pos in self.positions.items():
            total += pos.qty * self.last_prices.get(sym, pos.entry_price)
        return total

    def _open(self, symbol: str, price: float, atr_val: float):
        if self.halted_today:
            self.log(f"{symbol}: günlük zarar limiti aşıldı, yeni işlem açılmıyor.", logging.WARNING)
            return
        if len(self.positions) >= self.risk.max_open_positions:
            self.log(f"{symbol}: maksimum açık pozisyon sayısına ulaşıldı.")
            return
        equity = self.equity()
        qty = self.rm.position_size(equity, price, atr_val)
        quote_amount = min(qty * price, self.broker.quote_balance() * 0.98)
        if quote_amount < MIN_ORDER_QUOTE:
            self.log(f"{symbol}: işlem tutarı çok düşük ({quote_amount:.2f} {self.quote_asset}), atlandı.",
                     logging.WARNING)
            return
        fill = self.broker.market_buy(symbol, quote_amount, price)
        stop, take = self.rm.stops(fill.price, atr_val)
        self.positions[symbol] = Position(
            symbol=symbol, qty=fill.qty, entry_price=fill.price, cost=fill.quote,
            stop_loss=stop, take_profit=take, highest=fill.price,
            opened_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), strategy=self.strategy.name,
        )
        self.log(f"ALIM {symbol}: {fill.qty:.8g} @ {fill.price:.8g} | Tutar {fill.quote:.2f} {self.quote_asset}"
                 f" | SL {stop:.8g} | TP {take:.8g}")
        self._save_state()
        self._emit("positions", None)
        self._emit("opened", self.positions[symbol])

    def _close(self, symbol: str, price: float, reason: str):
        pos = self.positions[symbol]
        fill = self.broker.market_sell(symbol, pos.qty, price)
        pnl = fill.quote - pos.cost
        trade = ClosedTrade(
            symbol=symbol, qty=pos.qty, entry_price=pos.entry_price, exit_price=fill.price,
            cost=pos.cost, proceeds=fill.quote, pnl=pnl, pnl_pct=pnl / pos.cost * 100 if pos.cost else 0.0,
            opened_at=pos.opened_at, closed_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            reason=reason, strategy=pos.strategy,
        )
        self.trades.append(trade)
        del self.positions[symbol]
        self.log(f"SATIŞ {symbol} ({reason}): {fill.qty:.8g} @ {fill.price:.8g} | "
                 f"K/Z {pnl:+.2f} {self.quote_asset} ({trade.pnl_pct:+.2f}%)")
        self._save_state()
        self._emit("positions", None)
        self._emit("trade", trade)

    def close_position(self, symbol: str, reason: str = "Elle kapatıldı"):
        """GUI'den elle kapatma (herhangi bir iş parçacığından çağrılabilir)."""
        with self._lock:
            if symbol not in self.positions:
                raise ValueError(f"{symbol} için açık pozisyon yok.")
            price = self.client.price(symbol)
            self.last_prices[symbol] = price
            self._close(symbol, price, reason)

    def forget_position(self, symbol: str):
        """Pozisyonu satmadan takipten çıkarır (ör. borsada elle satıldıysa)."""
        with self._lock:
            if self.positions.pop(symbol, None) is not None:
                self.log(f"{symbol} pozisyonu takipten çıkarıldı (satış yapılmadı).")
                self._save_state()
                self._emit("positions", None)

    def _update_equity(self):
        equity = self.equity()
        today = _now().date()
        if self._day != today:
            self._day = today
            self._day_start_equity = equity
            if self.halted_today:
                self.log("Yeni gün: işlem kısıtlaması kaldırıldı.")
            self.halted_today = False
        if not self.halted_today and self.rm.daily_loss_hit(self._day_start_equity, equity):
            self.halted_today = True
            self.log(f"Günlük zarar limiti (%{self.risk.max_daily_loss_pct}) aşıldı. "
                     "Bugün yeni pozisyon açılmayacak.")
            self._emit("halt", f"Gün başı: {self._day_start_equity:.2f}, şimdi: {equity:.2f} {self.quote_asset}")
        day_pct = (equity / self._day_start_equity - 1) * 100 if self._day_start_equity else 0.0
        self._emit("equity", {"equity": equity, "quote": self.broker.quote_balance(), "day_pct": day_pct})

    # ------------------------------------------------------------------ durum kaydı
    def _save_state(self):
        if not self.state_path:
            return
        data = {
            "positions": [p.to_dict() for p in self.positions.values()],
            "trades": [t.to_dict() for t in self.trades[-500:]],
        }
        if not getattr(self.broker, "is_live", False):
            data["paper_balance"] = self.broker.balance
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.state_path)
        except OSError as exc:
            logger.error("Durum kaydedilemedi: %s", exc)

    def _load_state(self):
        if not self.state_path or not self.state_path.exists():
            return
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.error("Durum okunamadı: %s", exc)
            return
        for p in data.get("positions", []):
            pos = Position.from_dict(p)
            if pos.symbol in self.symbols:
                self.positions[pos.symbol] = pos
        self.trades = [ClosedTrade.from_dict(t) for t in data.get("trades", [])]
        if not getattr(self.broker, "is_live", False) and "paper_balance" in data:
            balance = float(data["paper_balance"])
            if math.isfinite(balance):
                self.broker.balance = balance
