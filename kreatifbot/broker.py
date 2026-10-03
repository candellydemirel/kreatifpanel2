"""Emir yürütme: kağıt (simülasyon) ve gerçek Binance aracı."""

from __future__ import annotations

from dataclasses import dataclass

from .binance_client import BinanceClient


class BrokerError(Exception):
    pass


@dataclass
class Fill:
    qty: float      # Alımda eline geçen net baz miktar / satımda satılan miktar
    price: float    # Ortalama gerçekleşme fiyatı
    quote: float    # Alımda harcanan, satımda eline geçen net karşı varlık
    fee: float      # Karşı varlık cinsinden tahmini ücret


class PaperBroker:
    """Gerçek emir göndermeden, ücret ve kayma ekleyerek işlemleri simüle eder."""

    is_live = False

    def __init__(self, quote_asset: str = "USDT", balance: float = 1000.0,
                 fee_pct: float = 0.1, slippage_pct: float = 0.05):
        self.quote_asset = quote_asset
        self.balance = float(balance)
        self.fee = fee_pct / 100
        self.slippage = slippage_pct / 100

    def quote_balance(self) -> float:
        return self.balance

    def market_buy(self, symbol: str, quote_amount: float, price: float) -> Fill:
        if quote_amount <= 0:
            raise BrokerError("Alım tutarı sıfır.")
        if quote_amount > self.balance + 1e-9:
            raise BrokerError(f"Yetersiz bakiye: {self.balance:.2f} {self.quote_asset}")
        fill_price = price * (1 + self.slippage)
        fee = quote_amount * self.fee
        qty = (quote_amount - fee) / fill_price
        self.balance -= quote_amount
        return Fill(qty=qty, price=fill_price, quote=quote_amount, fee=fee)

    def market_sell(self, symbol: str, qty: float, price: float) -> Fill:
        if qty <= 0:
            raise BrokerError("Satış miktarı sıfır.")
        fill_price = price * (1 - self.slippage)
        gross = qty * fill_price
        fee = gross * self.fee
        self.balance += gross - fee
        return Fill(qty=qty, price=fill_price, quote=gross - fee, fee=fee)


class LiveBroker:
    """Binance üzerinde gerçek piyasa emirleri gönderir."""

    is_live = True

    def __init__(self, client: BinanceClient, quote_asset: str = "USDT", fee_pct: float = 0.1):
        if not client.has_keys:
            raise BrokerError("Canlı işlem için API anahtarı gerekli.")
        self.client = client
        self.quote_asset = quote_asset
        self.fee = fee_pct / 100

    def quote_balance(self) -> float:
        return self.client.free_balance(self.quote_asset)

    def _parse(self, resp: dict, base_asset: str, side: str) -> Fill:
        executed = float(resp.get("executedQty", 0))
        quote = float(resp.get("cummulativeQuoteQty", 0))
        if executed <= 0:
            raise BrokerError(f"Emir gerçekleşmedi (durum: {resp.get('status')}).")
        price = quote / executed
        fee_quote, fee_base, other_fee = 0.0, 0.0, False
        for f in resp.get("fills", []):
            commission = float(f.get("commission", 0))
            asset = f.get("commissionAsset")
            if asset == base_asset:
                fee_base += commission
            elif asset == self.quote_asset:
                fee_quote += commission
            elif commission > 0:
                other_fee = True  # ör. BNB ile ödenen ücret
        if other_fee:
            fee_quote += quote * self.fee * 0.75
        if side == "BUY":
            return Fill(qty=executed - fee_base, price=price, quote=quote,
                        fee=fee_quote + fee_base * price)
        return Fill(qty=executed, price=price, quote=quote - fee_quote - fee_base * price,
                    fee=fee_quote + fee_base * price)

    def market_buy(self, symbol: str, quote_amount: float, price: float) -> Fill:
        rules = self.client.symbol_rules(symbol)
        amount = rules.floor_quote(quote_amount)
        if float(amount) < rules.min_notional:
            raise BrokerError(
                f"{symbol}: işlem tutarı {amount} {rules.quote_asset}, minimum {rules.min_notional} gerekli."
            )
        resp = self.client.order_market(symbol, "BUY", quote_qty=amount)
        return self._parse(resp, rules.base_asset, "BUY")

    def market_sell(self, symbol: str, qty: float, price: float) -> Fill:
        rules = self.client.symbol_rules(symbol)
        free = self.client.free_balance(rules.base_asset)
        sell_qty = rules.floor_qty(min(qty, free))
        if float(sell_qty) <= 0 or float(sell_qty) < rules.min_qty:
            raise BrokerError(f"{symbol}: satılacak yeterli {rules.base_asset} yok (serbest: {free}).")
        if float(sell_qty) * price < rules.min_notional:
            raise BrokerError(f"{symbol}: satış tutarı minimum işlem tutarının altında.")
        resp = self.client.order_market(symbol, "SELL", quantity=sell_qty)
        return self._parse(resp, rules.base_asset, "SELL")
