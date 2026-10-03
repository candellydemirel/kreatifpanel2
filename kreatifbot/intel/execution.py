"""Emir yürütme (Binance Spot / USDⓈ-M Futures; kağıt ve canlı).

Emir göndermeden önce: bakiye, mevcut pozisyon, miktar/fiyat hassasiyeti,
minimum notional, spread, tahmini kayma ve order book likiditesi kontrol edilir.
Herhangi biri başarısızsa emir gönderilmez (NO TRADE / EXECUTION_FAILURE).

Futures canlıda, pozisyon açıldıktan sonra borsa tarafında STOP_MARKET
(closePosition) koruyucu stop yerleştirilir; trailing ilerledikçe güncellenir.
Spot canlıda stop ve hedefler motor tarafından izlenir (bot kapalıyken koruma yok).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..binance_client import BinanceAPIError, BinanceClient, SymbolRules
from .config import IntelConfig
from .futures_client import BinanceFuturesClient
from .orderflow import estimate_slippage_pct, orderbook_features


class ExecutionError(Exception):
    pass


@dataclass
class ExecFill:
    qty: float
    price: float
    fee: float
    notional: float
    order_id: str = ""


@dataclass
class PreTradeResult:
    ok: bool
    qty: str = "0"
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    est_slippage_pct: float | None = None


def pretrade_check(rules: SymbolRules | None, qty: float, price: float, direction: str, cfg: IntelConfig,
                   book: dict | None = None, available_balance: float | None = None, market: str = "SPOT",
                   leverage: int = 1, reduce_only: bool = False) -> PreTradeResult:
    res = PreTradeResult(ok=False)
    if rules is None:
        res.errors.append("Sembol filtreleri bilinmiyor")
        return res
    chk = rules.check_order(qty, price, market_order=True)
    if not chk.ok:
        res.errors += chk.errors
        return res
    res.qty = chk.quantity
    notional = float(chk.quantity) * price
    if not reduce_only and available_balance is not None:
        need = notional / (leverage if market != "SPOT" else 1) * 1.01
        if need > available_balance:
            res.errors.append(f"Yetersiz bakiye: gerekli {need:.2f}, mevcut {available_balance:.2f}")
    if book is None and not reduce_only:
        res.errors.append("Order book yok (giriş için spread/likidite kontrolü zorunlu)")
    if book is not None:
        ob = orderbook_features(book)
        if ob.get("available"):
            if ob["spread_pct"] > cfg.data_quality.max_spread_pct:
                res.errors.append(f"Spread çok yüksek: %{ob['spread_pct']:.3f}")
            side = ("BUY" if direction == "LONG" else "SELL") if not reduce_only else \
                ("SELL" if direction == "LONG" else "BUY")
            slip = estimate_slippage_pct(book, side, notional)
            res.est_slippage_pct = slip
            if slip is None:
                res.errors.append("Order book derinliği emir için yetersiz (likidite)")
            elif slip > cfg.costs.slippage_pct * 3:
                res.errors.append(f"Tahmini kayma %{slip:.3f} > izin verilen %{cfg.costs.slippage_pct * 3:.3f}")
        else:
            res.warnings.append("Order book alınamadı: spread/likidite kontrolü yapılamadı")
            if not reduce_only:
                res.errors.append("Order book yok (canlı girişte zorunlu)")
    res.ok = not res.errors
    return res


class PaperVenue:
    """Gerçek emir göndermeden yürütme (spot: nakit; futures: cüzdan + marj)."""

    is_live = False

    def __init__(self, market: str, balance: float, cfg: IntelConfig, quote_asset: str = "USDT"):
        self.market = market
        self.cash = float(balance)
        self.quote_asset = quote_asset
        c = cfg.costs
        self.fee = (c.spot_taker_fee_pct if market == "SPOT" else c.futures_taker_fee_pct) / 100
        self.slip = c.slippage_pct / 100
        self.margin_used: dict[str, float] = {}
        self.leverage: dict[str, int] = {}

    @property
    def balance(self) -> float:  # klasik motorla uyumluluk
        return self.cash

    @balance.setter
    def balance(self, v: float):
        self.cash = float(v)

    def quote_balance(self) -> float:
        return self.available_balance()

    def available_balance(self) -> float:
        return self.cash - sum(self.margin_used.values()) if self.market != "SPOT" else self.cash

    def open(self, symbol, direction, qty, price, rules=None, leverage=1, stop=None) -> ExecFill:
        if self.market == "SPOT" and direction != "LONG":
            raise ExecutionError("Spot piyasada SHORT açılamaz")
        s = 1 if direction == "LONG" else -1
        px = price * (1 + s * self.slip)
        if rules is not None:
            qty = float(rules.floor_qty(qty, market_order=True))
        if qty <= 0:
            raise ExecutionError("Miktar sıfır")
        notional = qty * px
        fee = notional * self.fee
        if self.market == "SPOT":
            if notional + fee > self.cash + 1e-9:
                raise ExecutionError(f"Yetersiz bakiye ({self.cash:.2f})")
            self.cash -= notional + fee
        else:
            margin = notional / max(leverage, 1)
            if margin + fee > self.available_balance() + 1e-9:
                raise ExecutionError(f"Yetersiz marj ({self.available_balance():.2f})")
            self.cash -= fee
            self.margin_used[symbol] = self.margin_used.get(symbol, 0.0) + margin
            self.leverage[symbol] = max(int(leverage), 1)
        return ExecFill(qty, px, fee, notional)

    def close(self, symbol, direction, qty, price, entry_price, rules=None, full=True) -> ExecFill:
        s = 1 if direction == "LONG" else -1
        px = price * (1 - s * self.slip)
        notional = qty * px
        fee = notional * self.fee
        if self.market == "SPOT":
            self.cash += notional - fee
        else:
            self.cash += (px - entry_price) * s * qty - fee
            used = self.margin_used.get(symbol, 0.0)
            if full:
                self.margin_used.pop(symbol, None)
            else:
                self.margin_used[symbol] = max(0.0, used - qty * entry_price / self.leverage.get(symbol, 1))
        return ExecFill(qty, px, fee, notional)

    def update_stop(self, symbol, direction, stop, rules=None):
        return None

    def pay_funding(self, amount: float):
        self.cash -= amount


class SpotLiveVenue:
    is_live = True
    market = "SPOT"

    def __init__(self, client: BinanceClient, cfg: IntelConfig, quote_asset: str = "USDT"):
        if not client.has_keys:
            raise ExecutionError("Canlı işlem için API anahtarı gerekli")
        self.client, self.cfg, self.quote_asset = client, cfg, quote_asset
        self.fee = cfg.costs.spot_taker_fee_pct / 100

    def quote_balance(self) -> float:
        return self.available_balance()

    def available_balance(self) -> float:
        return self.client.free_balance(self.quote_asset)

    def _parse(self, resp, rules, side) -> ExecFill:
        executed = float(resp.get("executedQty", 0))
        quote = float(resp.get("cummulativeQuoteQty", 0))
        if executed <= 0:
            raise ExecutionError(f"Emir gerçekleşmedi (durum {resp.get('status')})")
        fee_base = sum(float(f["commission"]) for f in resp.get("fills", []) if f.get("commissionAsset") == rules.base_asset)
        fee_quote = sum(float(f["commission"]) for f in resp.get("fills", []) if f.get("commissionAsset") == self.quote_asset)
        price = quote / executed
        qty = executed - fee_base if side == "BUY" else executed
        fee = fee_quote + fee_base * price or quote * self.fee
        return ExecFill(qty, price, fee, quote, str(resp.get("orderId", "")))

    def open(self, symbol, direction, qty, price, rules=None, leverage=1, stop=None) -> ExecFill:
        if direction != "LONG":
            raise ExecutionError("Spot piyasada SHORT açılamaz")
        rules = rules or self.client.symbol_rules(symbol)
        chk = rules.check_order(qty, price, market_order=True)
        if not chk.ok:
            raise ExecutionError("; ".join(chk.errors))
        return self._parse(self.client.order_market(symbol, "BUY", quantity=chk.quantity), rules, "BUY")

    def close(self, symbol, direction, qty, price, entry_price, rules=None, full=True) -> ExecFill:
        rules = rules or self.client.symbol_rules(symbol)
        free = self.client.free_balance(rules.base_asset)
        q = rules.floor_qty(min(qty, free), market_order=True)
        if float(q) <= 0:
            raise ExecutionError(f"Satılacak {rules.base_asset} yok (serbest {free})")
        return self._parse(self.client.order_market(symbol, "SELL", quantity=q), rules, "SELL")

    def update_stop(self, symbol, direction, stop, rules=None):
        return None  # Spot: motor tarafından izlenir


class FuturesLiveVenue:
    is_live = True
    market = "USDM_FUTURES"

    def __init__(self, client: BinanceFuturesClient, cfg: IntelConfig, quote_asset: str = "USDT"):
        if not client.has_keys:
            raise ExecutionError("Canlı işlem için API anahtarı gerekli")
        self.client, self.cfg, self.quote_asset = client, cfg, quote_asset
        self.fee = cfg.costs.futures_taker_fee_pct / 100
        self._prepared: set[str] = set()

    def quote_balance(self) -> float:
        return self.available_balance()

    def available_balance(self) -> float:
        return self.client.balances().get(self.quote_asset, {}).get("free", 0.0)

    def _prepare(self, symbol, leverage):
        if symbol in self._prepared:
            return
        self.client.change_margin_type(symbol, self.cfg.risk.margin_type)
        self.client.change_leverage(symbol, leverage)
        self._prepared.add(symbol)

    def _fill(self, resp) -> ExecFill:
        qty = float(resp.get("executedQty", 0) or 0)
        price = float(resp.get("avgPrice", 0) or 0)
        if qty <= 0 or price <= 0:
            raise ExecutionError(f"Futures emri gerçekleşmedi (durum {resp.get('status')})")
        notional = float(resp.get("cumQuote", qty * price) or qty * price)
        return ExecFill(qty, price, notional * self.fee, notional, str(resp.get("orderId", "")))

    def open(self, symbol, direction, qty, price, rules=None, leverage=1, stop=None) -> ExecFill:
        rules = rules or self.client.symbol_rules(symbol)
        self._prepare(symbol, leverage)
        chk = rules.check_order(qty, price, market_order=True)
        if not chk.ok:
            raise ExecutionError("; ".join(chk.errors))
        side = "BUY" if direction == "LONG" else "SELL"
        fill = self._fill(self.client.new_order(symbol, side, "MARKET", quantity=chk.quantity))
        if stop is not None:
            try:
                self.update_stop(symbol, direction, stop, rules)
            except BinanceAPIError as exc:
                # Koruyucu stop konulamazsa pozisyonu hemen kapat (güvenlik)
                self.client.new_order(symbol, "SELL" if side == "BUY" else "BUY", "MARKET",
                                      quantity=rules.floor_qty(fill.qty, True), reduce_only=True)
                raise ExecutionError(f"Koruyucu stop yerleştirilemedi, pozisyon kapatıldı: {exc}") from exc
        return fill

    def close(self, symbol, direction, qty, price, entry_price, rules=None, full=True) -> ExecFill:
        rules = rules or self.client.symbol_rules(symbol)
        side = "SELL" if direction == "LONG" else "BUY"
        fill = self._fill(self.client.new_order(symbol, side, "MARKET", quantity=rules.floor_qty(qty, True),
                                                reduce_only=True))
        if full:
            try:
                self.client.cancel_all_orders(symbol)
            except BinanceAPIError:
                pass
        return fill

    def update_stop(self, symbol, direction, stop, rules=None):
        rules = rules or self.client.symbol_rules(symbol)
        side = "SELL" if direction == "LONG" else "BUY"
        try:
            self.client.cancel_all_orders(symbol)
        except BinanceAPIError:
            pass
        return self.client.new_order(symbol, side, "STOP_MARKET", stop_price=rules.round_price(stop, side),
                                     close_position=True)
