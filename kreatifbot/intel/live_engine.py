"""Zeka Motoru bot motoru (kağıt / canlı; Spot veya USDⓈ-M Futures).

Her döngüde: veri al → veri kalitesi → (yeni mumda) karar → sinyal veritabanı →
(onaylı ise) ön kontroller → emir → pozisyon yönetimi (her döngüde fiyat, her
yeni mumda süre/trailing/sinyal zayıflaması/rejim) → çıkış → performans kaydı.

Canlı güvenlik: piyasa verisi, sembol filtreleri, risk motoru, veritabanı veya
Binance API sorunu varsa yeni emir gönderilmez (NO TRADE). AI yoksa deterministik
moda düşülür (ayara göre olasılık zorunluysa NO TRADE).
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ..binance_client import BinanceAPIError
from ..engine import EngineCore
from ..models import ClosedTrade
from ..utils import INTERVALS
from .config import IntelConfig
from .data_quality import check_candles, check_orderbook
from .decision import DecisionEngine, DecisionObject
from .execution import ExecutionError, pretrade_check
from .market_data import closed_only
from .orderflow import orderbook_features
from .position_manager import ManagedPosition, evaluate_signal_decay, open_position, update_on_bar
from .risk_engine import OpenExposure, PortfolioState, return_correlation
from .signal_store import DatabaseError, SignalStore
from .types import ExitReason, Regime, SignalStatus

logger = logging.getLogger("kreatifbot.intel.live")


class IntelligentBotEngine(EngineCore):
    def __init__(self, cfg: IntelConfig, symbols: list[str], market_client, venue, store: SignalStore | None,
                 futures_client=None, decision_engine: DecisionEngine | None = None, poll_seconds: float = 20,
                 state_path: str | Path | None = None, on_event=None, kline_limit: int = 600,
                 btc_client=None):
        self.cfg = cfg
        self.symbols = [s.strip().upper() for s in symbols if s.strip()]
        if not self.symbols:
            raise ValueError("En az bir sembol girin.")
        bad = [s for s in self.symbols if not s.endswith(cfg.quote_asset)]
        if bad:
            raise ValueError(f"Bu semboller {cfg.quote_asset} ile bitmiyor: {', '.join(bad)}")
        not_allowed = [s for s in self.symbols if cfg.allowed_symbols and s not in cfg.allowed_symbols]
        if not_allowed:
            raise ValueError(f"İzin verilen sembol listesinde değil (Zeka Motoru ayarları): {', '.join(not_allowed)}")
        if cfg.timeframes.entry not in cfg.allowed_timeframes:
            raise ValueError(f"Giriş zaman dilimi izinli değil: {cfg.timeframes.entry}")
        self.market = cfg.market
        self.client = market_client
        self.futures = futures_client
        self.btc_client = btc_client or market_client
        self.venue = venue
        self.broker = venue  # GUI/Telegram uyumluluğu (is_live)
        self.store = store
        self.engine = decision_engine or DecisionEngine(cfg)
        self.interval = cfg.timeframes.entry
        self.quote_asset = cfg.quote_asset
        self.strategy = SimpleNamespace(name=f"Zeka Motoru ({'Futures' if self.market != 'SPOT' else 'Spot'})")
        self.kline_limit = kline_limit
        self.state_path = Path(state_path) if state_path else None
        self._init_core(poll_seconds, on_event)

        self.positions: dict[str, ManagedPosition] = {}
        self.trades: list[ClosedTrade] = []
        self.last_prices: dict[str, float] = {}
        self.last_signals: dict[str, dict] = {}
        self.last_decisions: dict[str, DecisionObject] = {}
        self.pending: dict[str, DecisionObject] = {}
        self._last_bar: dict[str, pd.Timestamp] = {}
        self._cache: dict[tuple, tuple[float, pd.DataFrame]] = {}
        self._rules: dict = {}
        self._api_errors: list[float] = []
        self.halted_today = False
        self._day = None
        self._day_start_equity = 0.0
        self._peak_equity = 0.0
        self._consecutive_losses = 0
        self._load_state()

    # ------------------------------------------------------------------ yardımcılar
    def start_message(self) -> str:
        mode = "CANLI" if self.venue.is_live else "KAĞIT (simülasyon)"
        tf = self.cfg.timeframes
        return (f"Zeka Motoru başlatıldı | {self.market} | Mod: {mode} | Semboller: {', '.join(self.symbols)} | "
                f"Giriş {tf.entry}, onay {tf.confirmation}, trend {tf.trend}, ana {tf.major}, makro {tf.macro}")

    def _api_error(self, exc):
        self._api_errors.append(time.time())
        self._api_errors = [t for t in self._api_errors if time.time() - t < 300]
        self.log(f"Binance API hatası: {exc}", logging.WARNING)

    def _klines(self, client, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        key = (id(client), symbol, interval)
        now = time.time()
        cached = self._cache.get(key)
        if cached is not None:
            ts, df = cached
            if not df.empty:
                next_close = pd.to_datetime(df["close_time"].iloc[-1], utc=True).timestamp() + INTERVALS[interval]
                if now < next_close + 1:
                    return df
        df = closed_only(client.klines(symbol, interval, limit=limit))
        self._cache[key] = (now, df)
        return df

    def _rules_for(self, symbol: str):
        if symbol not in self._rules:
            self._rules[symbol] = self.client.symbol_rules(symbol)
        return self._rules[symbol]

    def equity(self) -> float:
        cash = getattr(self.venue, "cash", None)
        if cash is None:
            cash = self.venue.available_balance()
            if self.market != "SPOT":
                cash += sum(p.entry_price * p.qty / max(p.leverage, 1) for p in self.positions.values())
        total = cash
        for sym, p in self.positions.items():
            price = self.last_prices.get(sym, p.entry_price)
            total += p.qty * price if self.market == "SPOT" else p.unrealized(price)[0]
        return total

    def _exposure(self) -> list[OpenExposure]:
        return [OpenExposure(s, p.direction, p.qty * self.last_prices.get(s, p.entry_price),
                             abs(p.entry_price - p.stop_loss) * p.qty) for s, p in self.positions.items()]

    def _derivatives(self, symbol: str) -> dict | None:
        if self.futures is None or (self.market == "SPOT" and not self.cfg.use_futures_context):
            return None
        key = ("deriv", symbol)
        cached = self._cache.get(key)
        if cached and time.time() - cached[0] < 300:
            return cached[1]
        d = {"period": self.interval if self.interval in ("5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d")
             else "5m"}
        try:
            d["funding"] = self.futures.funding_history(symbol, limit=200)
            d["oi"] = self.futures.open_interest_hist(symbol, d["period"], 500)
            d["long_short"] = self.futures.long_short_ratio(symbol, d["period"], 500)
            d["taker"] = self.futures.taker_volume(symbol, d["period"], 500)
        except (BinanceAPIError, ValueError) as exc:
            self._api_error(exc)
        self._cache[key] = (time.time(), d)
        return d

    # ------------------------------------------------------------------ ana döngü
    def tick(self):
        with self._lock:
            for symbol in self.symbols:
                if self._stop.is_set():
                    break
                try:
                    self._process(symbol)
                except BinanceAPIError as exc:
                    self._api_error(exc)
                except (DatabaseError, ExecutionError, ValueError) as exc:
                    self.log(f"{symbol}: {exc}", logging.WARNING)
                except Exception as exc:  # noqa: BLE001
                    self.log(f"{symbol}: beklenmeyen hata - {exc}", logging.ERROR)
                    logger.exception("intel tick")
            self._update_equity()

    def _process(self, symbol: str):
        cfg = self.cfg
        tf = cfg.timeframes
        entry = self._klines(self.client, symbol, tf.entry, self.kline_limit)
        price = None
        book = None
        try:
            book = self.client.depth(symbol, 100)
            book["_fetched_ms"] = int(time.time() * 1000)
            ob = orderbook_features(book)
            price = ob.get("mid") if ob.get("available") else None
        except BinanceAPIError as exc:
            self._api_error(exc)
        if price is None and not entry.empty:
            try:
                price = self.client.price(symbol)
            except BinanceAPIError as exc:
                self._api_error(exc)
        if price is None:
            self.log(f"{symbol}: fiyat alınamadı → yeni işlem yok", logging.WARNING)
            return
        self.last_prices[symbol] = price

        # 1) Açık pozisyon: anlık fiyat kontrolü
        pos = self.positions.get(symbol)
        if pos is not None:
            tick_row = pd.Series({"open": price, "high": price, "low": price, "close": price, "atr": np.nan})
            acts = update_on_bar(pos, tick_row, cfg.risk, count_bar=False)
            self._apply_actions(symbol, pos, acts, price)

        # 2) Yeni kapanmış mum?
        if entry.empty:
            return
        bar_time = pd.to_datetime(entry["open_time"].iloc[-1], utc=True)
        if self._last_bar.get(symbol) == bar_time:
            self._try_pending(symbol, price, book)
            return
        self._last_bar[symbol] = bar_time

        dq = check_candles(entry, tf.entry, cfg.data_quality, min_history=min(cfg.data_quality.min_history_bars,
                                                                             self.kline_limit - 10))
        dq.merge(check_orderbook(book, cfg.data_quality, int(time.time() * 1000), book.get("_fetched_ms") if book else None),
                 "orderbook: ")
        htf = {}
        for role, interval in tf.higher().items():
            try:
                htf[role] = self._klines(self.client, symbol, interval, 400)
            except BinanceAPIError as exc:
                self._api_error(exc)
        btc = None
        if cfg.btc_context and symbol != "BTCUSDT":
            try:
                btc = self._klines(self.btc_client, "BTCUSDT", tf.trend, 400)
            except BinanceAPIError as exc:
                self._api_error(exc)
        deriv = self._derivatives(symbol)
        prep = self.engine.prepare(entry, symbol, tf.entry, htf, deriv,
                                   None if btc is None or btc.empty else _features(btc, tf.trend, cfg.market))
        i = len(prep.f) - 1
        self._funding_paper(symbol)

        # 3) Açık pozisyonu yeni mumla yönet (süre, trailing, zayıflama, rejim)
        pos = self.positions.get(symbol)
        if pos is not None:
            row = prep.f.iloc[i]
            opened = pd.Timestamp(pos.opened_at)
            if opened.tzinfo is None:
                opened = opened.tz_localize("UTC")
            if pd.to_datetime(row["open_time"], utc=True) >= opened:
                acts = update_on_bar(pos, row, cfg.risk)
                self._apply_actions(symbol, pos, acts, price)
            pos = self.positions.get(symbol)
            if pos is not None:
                d_s = pos.direction.lower()
                o_s = "short" if d_s == "long" else "long"
                conf = float(prep.totals[f"score_{d_s}"].iat[i])
                opp_c = any(prep.candidates[k][o_s][i] for k in prep.candidates)
                opp = float(prep.totals[f"score_{o_s}"].iat[i]) if opp_c else None
                st = self.engine.by_key.get(pos.strategy)
                forbidden = st is not None and Regime(prep.f["regime"].iat[i]) in st.spec.forbidden_for(pos.direction)
                act = evaluate_signal_decay(pos, conf, cfg.risk, opp, forbidden, price)
                if act is not None:
                    self._apply_actions(symbol, pos, [act], price)
                elif pos.status == "WEAKENING":
                    self.log(f"{symbol}: sinyal zayıflıyor (güven {pos.initial_confidence:.0f} → {conf:.0f})")

        # 4) Karar
        ps = PortfolioState(equity=self.equity(), peak_equity=max(self._peak_equity, self.equity()),
                            day_start_equity=self._day_start_equity or self.equity(),
                            consecutive_losses=self._consecutive_losses, api_errors_recent=len(self._api_errors))
        try:
            rules = self._rules_for(symbol)
        except BinanceAPIError as exc:
            self._api_error(exc)
            rules = None
        corr = None
        if len(self.positions) and len(self.symbols) > 1:
            closes = {symbol: prep.f["close"]}
            for s2 in self.positions:
                cached = self._cache.get((id(self.client), s2, tf.entry))
                if cached is not None and not cached[1].empty:
                    closes[s2] = cached[1]["close"].reset_index(drop=True)
            corr = return_correlation(closes)
        ob = orderbook_features(book) if book else None
        live = bool(self.venue.is_live)
        d = self.engine.decide(prep, i, equity=ps.equity, open_exposure=self._exposure(), portfolio=ps,
                               rules=rules, book_stats=ob, live=live, dq=dq, has_position=symbol in self.positions,
                               available_balance=self.venue.available_balance(), correlations=corr)
        if self.store is not None:
            try:
                self.store.save_decision(d, source="live" if live else "paper")
            except DatabaseError as exc:
                self.log(f"Veritabanı hatası: {exc} → yeni emir gönderilmeyecek", logging.ERROR)
                return
        self.last_decisions[symbol] = d
        self._emit("decision", d)
        sig_text = {"LONG": "AL", "SHORT": "SAT"}.get(d.direction, "BEKLE") if d.is_trade else "BEKLE"
        self.last_signals[symbol] = {
            "signal": sig_text, "time": bar_time.strftime("%Y-%m-%d %H:%M"), "price": price,
            "reason": (f"{d.direction} güven {d.confidence:.0f}, {d.market_regime}, {d.strategy}" if d.is_trade else
                       f"NO TRADE: {', '.join(d.no_trade_reasons) or d.signal_status} ({d.market_regime})"),
        }
        self._emit("signal", {"symbol": symbol, **self.last_signals[symbol]})
        if d.is_trade:
            self.log(f"{symbol}: {d.direction} sinyali (güven {d.confidence:.0f}, {d.strategy}, R:R {d.risk_reward})")
            self.pending[symbol] = d
            self._try_pending(symbol, price, book)

    def _funding_paper(self, symbol: str):
        """Kağıt futures: 00/08/16 UTC funding kesimlerinde son funding oranıyla ödeme/tahsilat."""
        pos = self.positions.get(symbol)
        if pos is None or self.market == "SPOT" or self.venue.is_live or self.futures is None:
            return
        now = pd.Timestamp.now(tz="UTC")
        last = getattr(pos, "_last_funding", None) or pd.Timestamp(pos.opened_at)
        last = last if last.tzinfo else last.tz_localize("UTC")
        cuts = [now.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=h) for h in (0, 8, 16, 24, 32, 40)]
        crossed = [c for c in cuts if last < c <= now]
        if not crossed:
            return
        try:
            rate = float(self.futures.premium_index(symbol)["lastFundingRate"])
        except (BinanceAPIError, KeyError) as exc:
            self._api_error(exc)
            return
        amount = len(crossed) * rate * pos.qty * self.last_prices.get(symbol, pos.entry_price) * pos.sign
        pos.funding_paid += amount
        self.venue.pay_funding(amount)
        object.__setattr__(pos, "_last_funding", now)
        self.log(f"{symbol}: funding {'ödendi' if amount > 0 else 'alındı'} {abs(amount):.4f} {self.quote_asset} "
                 f"(oran %{rate * 100:.4f})")

    # ------------------------------------------------------------------ giriş
    def _try_pending(self, symbol: str, price: float, book: dict | None):
        d = self.pending.get(symbol)
        if d is None or symbol in self.positions:
            return
        now = datetime.now(timezone.utc)
        if d.signal_expiry and now > pd.Timestamp(d.signal_expiry).to_pydatetime():
            self.pending.pop(symbol, None)
            self.log(f"{symbol}: sinyal süresi doldu (SIGNAL_EXPIRED)")
            self._status(d, SignalStatus.EXPIRED)
            return
        s = 1 if d.direction == "LONG" else -1
        if (price - d.stop_loss) * s <= 0:
            self.pending.pop(symbol, None)
            self._status(d, SignalStatus.CANCELLED)
            self.log(f"{symbol}: fiyat stop'un ötesinde, sinyal iptal")
            return
        if self.store is not None and not self.store.healthy():
            self.log("Veritabanı erişilemiyor → yeni emir yok", logging.ERROR)
            return
        if len(self._api_errors) >= self.cfg.circuit_breaker.api_error_limit:
            self.log("API kararsız (devre kesici) → yeni emir yok", logging.WARNING)
            return
        rules = self._rules.get(symbol)
        risk_amount = abs(d.entry - d.stop_loss) * d.position_size
        qty = min(risk_amount / abs(price - d.stop_loss), d.notional / price * 1.05)
        chk = pretrade_check(rules, qty, price, d.direction, self.cfg, book, self.venue.available_balance(),
                             self.market, d.leverage)
        if not chk.ok:
            self.pending.pop(symbol, None)
            self._status(d, SignalStatus.CANCELLED)
            self.log(f"{symbol}: emir ön kontrolü başarısız → {'; '.join(chk.errors)}", logging.WARNING)
            return
        try:
            fill = self.venue.open(symbol, d.direction, float(chk.qty), price, rules, d.leverage, d.stop_loss)
        except (ExecutionError, BinanceAPIError) as exc:
            self.pending.pop(symbol, None)
            self._status(d, SignalStatus.CANCELLED)
            self.log(f"{symbol}: EXECUTION_FAILURE — {exc}", logging.ERROR)
            return
        self.pending.pop(symbol, None)
        pos = open_position(symbol, self.market, d.direction, fill.price, fill.qty, d.stop_loss, d.take_profit_levels,
                            self.cfg.risk.tp_fractions[:len(d.take_profit_levels)], d.strategy,
                            datetime.now(timezone.utc).isoformat(), 0, d.max_hold_bars, d.expected_hold_bars,
                            d.confidence, d.market_regime, d.signal_id, d.leverage, d.liquidation_price)
        pos.fees_paid = fill.fee
        self.positions[symbol] = pos
        self._status(d, SignalStatus.EXECUTED)
        self.log(f"{'ALIM' if d.direction == 'LONG' else 'SHORT'} {symbol}: {fill.qty:.8g} @ {fill.price:.8g} | "
                 f"SL {d.stop_loss:.8g} | TP {', '.join(f'{t:.6g}' for t in d.take_profit_levels)}")
        self._save_state()
        self._emit("positions", None)
        self._emit("opened", pos)

    def _status(self, d: DecisionObject, status: SignalStatus):
        if self.store is not None:
            try:
                self.store.update_status(d.signal_id, status.value)
            except DatabaseError as exc:
                self.log(str(exc), logging.ERROR)

    # ------------------------------------------------------------------ çıkış
    def _apply_actions(self, symbol: str, pos: ManagedPosition, actions, price: float):
        prev_stop = pos.stop_loss
        for a in actions:
            try:
                fill = self.venue.close(symbol, pos.direction, a.qty, a.price if not self.venue.is_live else price,
                                        pos.entry_price, self._rules.get(symbol), full=a.full)
            except (ExecutionError, BinanceAPIError) as exc:
                # Emir gerçekleşmedi: miktarı geri ekle. Tam çıkışlar (SL/süre vb.) sonraki döngüde yeniden tetiklenir;
                # kısmi TP reddedilirse (ör. minimum tutar altı) pozisyon sonraki hedef/stop ile yönetilmeye devam eder.
                pos.qty += a.qty
                self.log(f"{symbol}: çıkış emri başarısız ({a.reason}) — {exc}. Pozisyon korunuyor, tekrar denenecek.",
                         logging.ERROR)
                return
            gross = (fill.price - pos.entry_price) * pos.sign * fill.qty
            pos.realized_pnl += gross
            pos.fees_paid += fill.fee
            pos.confidence_history.append(f"{a.reason}:{a.note}")
            self.log(f"{symbol}: {a.reason} {a.note} — {fill.qty:.8g} @ {fill.price:.8g} (brüt {gross:+.2f})")
            if a.full or pos.qty <= 1e-12:
                self._finalize(symbol, pos, a.reason)
                return
        if self.positions.get(symbol) is pos and pos.stop_loss != prev_stop:
            try:
                self.venue.update_stop(symbol, pos.direction, pos.stop_loss, self._rules.get(symbol))
            except (ExecutionError, BinanceAPIError) as exc:
                self.log(f"{symbol}: koruyucu stop güncellenemedi — {exc}", logging.WARNING)
            self._save_state()
            self._emit("positions", None)

    def _finalize(self, symbol: str, pos: ManagedPosition, reason):
        net = pos.realized_pnl - pos.fees_paid - pos.funding_paid
        base = pos.entry_price * pos.initial_qty / (pos.leverage if self.market != "SPOT" else 1)
        risk = pos.risk_per_unit * pos.initial_qty
        opened = pd.Timestamp(pos.opened_at)
        hold_min = (pd.Timestamp.now(tz="UTC") - (opened if opened.tzinfo else opened.tz_localize("UTC"))) \
            .total_seconds() / 60
        trade = ClosedTrade(
            symbol=symbol, qty=pos.initial_qty, entry_price=pos.entry_price,
            exit_price=pos.entry_price + pos.realized_pnl / (pos.initial_qty * pos.sign) if pos.initial_qty else 0.0,
            cost=base, proceeds=base + net, pnl=net, pnl_pct=net / base * 100 if base else 0.0,
            opened_at=pos.opened_at[:19].replace("T", " "), closed_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            reason=str(reason), strategy=pos.strategy, side=pos.direction, exit_reason=str(reason),
            r_multiple=net / risk if risk else 0.0, gross_pnl=pos.realized_pnl, fees=pos.fees_paid,
            signal_id=pos.signal_id)
        self.trades.append(trade)
        self._consecutive_losses = self._consecutive_losses + 1 if net <= 0 else 0
        self.positions.pop(symbol, None)
        if self.store is not None and pos.signal_id:
            try:
                self.store.record_outcome(pos.signal_id, "WIN" if net > 0 else "LOSS", str(reason), net,
                                          pos.realized_pnl, pos.fees_paid, trade.r_multiple, hold_min)
            except DatabaseError as exc:
                self.log(str(exc), logging.ERROR)
        self.log(f"KAPANDI {symbol} ({reason}): net {net:+.2f} {self.quote_asset}, R {trade.r_multiple:+.2f}")
        self._save_state()
        self._emit("positions", None)
        self._emit("trade", trade)

    def close_position(self, symbol: str, reason: str = "MANUAL_EXIT"):
        with self._lock:
            pos = self.positions.get(symbol)
            if pos is None:
                raise ValueError(f"{symbol} için açık pozisyon yok.")
            price = self.client.price(symbol)
            self.last_prices[symbol] = price
            from .position_manager import ExitAction
            self._apply_actions(symbol, pos, [ExitAction(ExitReason.MANUAL_EXIT, pos.qty, price, True, "Elle")], price)

    def forget_position(self, symbol: str):
        with self._lock:
            if self.positions.pop(symbol, None) is not None:
                self.log(f"{symbol} pozisyonu takipten çıkarıldı (emir gönderilmedi).")
                self._save_state()
                self._emit("positions", None)

    def _update_equity(self):
        eq = self.equity()
        today = datetime.now(timezone.utc).date()
        if self._day != today:
            self._day, self._day_start_equity = today, eq
            self.halted_today = False
        self._peak_equity = max(self._peak_equity, eq)
        day_pct = (eq / self._day_start_equity - 1) * 100 if self._day_start_equity else 0.0
        if not self.halted_today and day_pct <= -self.cfg.risk.max_daily_loss_pct:
            self.halted_today = True
            self._emit("halt", f"Günlük değişim %{day_pct:.2f}")
        self._emit("equity", {"equity": eq, "quote": self.venue.available_balance(), "day_pct": day_pct})

    # ------------------------------------------------------------------ durum
    def _save_state(self):
        if not self.state_path:
            return
        data = {"positions": [p.to_dict() for p in self.positions.values()],
                "trades": [t.to_dict() for t in self.trades[-500:]]}
        if not self.venue.is_live:
            data["paper_cash"] = self.venue.cash
            data["margin_used"] = getattr(self.venue, "margin_used", {})
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            tmp.replace(self.state_path)
        except OSError as exc:
            logger.error("Durum kaydedilemedi: %s", exc)

    def _load_state(self):
        if not self.state_path or not self.state_path.exists():
            return
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        for p in data.get("positions", []):
            pos = ManagedPosition.from_dict(p)
            if pos.symbol in self.symbols and pos.market == self.market:
                self.positions[pos.symbol] = pos
        self.trades = [ClosedTrade.from_dict(t) for t in data.get("trades", [])]
        if not self.venue.is_live and "paper_cash" in data:
            self.venue.cash = float(data["paper_cash"])
            if hasattr(self.venue, "margin_used"):
                self.venue.margin_used = {k: float(v) for k, v in data.get("margin_used", {}).items()}


def _features(df: pd.DataFrame, interval: str, market: str) -> pd.DataFrame:
    from .features import compute_features
    return compute_features(df, interval, market)
