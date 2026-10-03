"""Binance Spot REST API istemcisi (imzalı istekler dahil).

Yalnızca ihtiyaç duyulan uç noktalar uygulanmıştır. Para çekme (withdraw)
uç noktaları bilerek yoktur.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from urllib.parse import urlencode

import pandas as pd
import requests

MAINNET_URL = "https://api.binance.com"
TESTNET_URL = "https://testnet.binance.vision"

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_base", "taker_quote", "ignore",
]


class BinanceAPIError(Exception):
    def __init__(self, status: int, code: int, message: str):
        self.status = status
        self.code = code
        self.message = message
        super().__init__(f"Binance hatası ({code}): {message}")


@dataclass
class SymbolRules:
    symbol: str
    base_asset: str
    quote_asset: str
    step_size: str
    min_qty: float
    tick_size: str
    min_notional: float
    quote_precision: int

    def floor_qty(self, qty: float) -> str:
        return floor_to_step(qty, self.step_size)

    def floor_quote(self, amount: float) -> str:
        step = "1" if self.quote_precision <= 0 else "0." + "0" * (self.quote_precision - 1) + "1"
        return floor_to_step(amount, step)


def floor_to_step(value: float, step: str) -> str:
    """Değeri adım büyüklüğüne aşağı yuvarlar ve Binance'in kabul ettiği metne çevirir."""
    step_d = Decimal(str(step)).normalize()
    if step_d <= 0:
        return format(Decimal(str(value)).normalize(), "f")
    units = (Decimal(str(value)) / step_d).to_integral_value(rounding=ROUND_DOWN)
    text = format((units * step_d).quantize(step_d), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def sign_query(secret: str, query: str) -> str:
    return hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()


class BinanceClient:
    def __init__(self, api_key: str = "", api_secret: str = "", testnet: bool = True,
                 timeout: float = 15.0, session: requests.Session | None = None):
        self.api_key = (api_key or "").strip()
        self.api_secret = (api_secret or "").strip()
        self.testnet = testnet
        self.base_url = TESTNET_URL if testnet else MAINNET_URL
        self.timeout = timeout
        self.session = session or requests.Session()
        self.time_offset_ms = 0
        self._rules_cache: dict[str, SymbolRules] = {}

    @property
    def has_keys(self) -> bool:
        return bool(self.api_key and self.api_secret)

    # ------------------------------------------------------------------ çekirdek
    def _request(self, method: str, path: str, params: dict | None = None,
                 signed: bool = False, _retry: bool = True):
        params = {k: v for k, v in (params or {}).items() if v is not None}
        headers = {}
        if self.api_key:
            headers["X-MBX-APIKEY"] = self.api_key
        if signed:
            if not self.has_keys:
                raise BinanceAPIError(0, -1, "API anahtarı ve gizli anahtar girilmemiş (Ayarlar sekmesi).")
            params["timestamp"] = int(time.time() * 1000) + self.time_offset_ms
            params.setdefault("recvWindow", 5000)
            query = urlencode(params)
            query += "&signature=" + sign_query(self.api_secret, query)
        else:
            query = urlencode(params)

        url = self.base_url + path + ("?" + query if query else "")
        try:
            resp = self.session.request(method, url, headers=headers, timeout=self.timeout)
        except requests.RequestException as exc:
            raise BinanceAPIError(0, -2, f"Bağlantı hatası: {exc}") from exc

        try:
            data = resp.json()
        except ValueError:
            data = None

        if resp.status_code >= 400:
            code = data.get("code", resp.status_code) if isinstance(data, dict) else resp.status_code
            msg = data.get("msg", resp.text) if isinstance(data, dict) else resp.text
            # Saat farkı hatası: sunucu saatiyle eşitle ve bir kez daha dene.
            if signed and code == -1021 and _retry:
                self.sync_time()
                params.pop("timestamp", None)
                return self._request(method, path, params, signed, _retry=False)
            raise BinanceAPIError(resp.status_code, int(code), str(msg))
        return data

    # ------------------------------------------------------------------ genel
    def ping(self) -> bool:
        self._request("GET", "/api/v3/ping")
        return True

    def server_time(self) -> int:
        return int(self._request("GET", "/api/v3/time")["serverTime"])

    def sync_time(self) -> int:
        local = int(time.time() * 1000)
        self.time_offset_ms = self.server_time() - local
        return self.time_offset_ms

    def exchange_info(self, symbol: str | None = None) -> dict:
        return self._request("GET", "/api/v3/exchangeInfo", {"symbol": symbol})

    def symbol_rules(self, symbol: str) -> SymbolRules:
        symbol = symbol.upper()
        if symbol in self._rules_cache:
            return self._rules_cache[symbol]
        info = self.exchange_info(symbol)
        sym = info["symbols"][0]
        filters = {f["filterType"]: f for f in sym.get("filters", [])}
        lot = filters.get("LOT_SIZE", {})
        price = filters.get("PRICE_FILTER", {})
        notional = filters.get("NOTIONAL") or filters.get("MIN_NOTIONAL") or {}
        rules = SymbolRules(
            symbol=symbol,
            base_asset=sym["baseAsset"],
            quote_asset=sym["quoteAsset"],
            step_size=lot.get("stepSize", "0.00000001"),
            min_qty=float(lot.get("minQty", 0)),
            tick_size=price.get("tickSize", "0.00000001"),
            min_notional=float(notional.get("minNotional", 0)),
            quote_precision=int(sym.get("quoteAssetPrecision", sym.get("quotePrecision", 8))),
        )
        self._rules_cache[symbol] = rules
        return rules

    def klines(self, symbol: str, interval: str, limit: int = 500,
               end_time: int | None = None, closed_only: bool = False) -> pd.DataFrame:
        raw = self._request("GET", "/api/v3/klines", {
            "symbol": symbol.upper(), "interval": interval,
            "limit": max(1, min(int(limit), 1000)), "endTime": end_time,
        })
        if closed_only and raw:
            now_ms = int(time.time() * 1000) + self.time_offset_ms
            raw = [row for row in raw if int(row[6]) < now_ms]
        return klines_to_frame(raw)

    def klines_history(self, symbol: str, interval: str, total: int) -> pd.DataFrame:
        """1000'den fazla mum gerektiğinde geriye doğru sayfalayarak indirir."""
        frames = []
        remaining = int(total)
        end_time = None
        while remaining > 0:
            df = self.klines(symbol, interval, limit=min(remaining, 1000), end_time=end_time)
            if df.empty:
                break
            frames.insert(0, df)
            remaining -= len(df)
            end_time = int(df["open_time"].iloc[0].timestamp() * 1000) - 1
            if len(df) < 1000 and remaining > 0:
                break
        if not frames:
            return klines_to_frame([])
        out = pd.concat(frames, ignore_index=True)
        return out.drop_duplicates("open_time").reset_index(drop=True)

    def ticker_24h(self, symbol: str | None = None):
        return self._request("GET", "/api/v3/ticker/24hr", {"symbol": symbol})

    def price(self, symbol: str) -> float:
        return float(self._request("GET", "/api/v3/ticker/price", {"symbol": symbol.upper()})["price"])

    # ------------------------------------------------------------------ hesap
    def account(self) -> dict:
        return self._request("GET", "/api/v3/account", signed=True)

    def balances(self) -> dict[str, dict[str, float]]:
        out = {}
        for b in self.account().get("balances", []):
            free, locked = float(b["free"]), float(b["locked"])
            if free > 0 or locked > 0:
                out[b["asset"]] = {"free": free, "locked": locked}
        return out

    def free_balance(self, asset: str) -> float:
        return self.balances().get(asset, {}).get("free", 0.0)

    def order_market(self, symbol: str, side: str, quantity: str | None = None,
                     quote_qty: str | None = None, test: bool = False) -> dict:
        if (quantity is None) == (quote_qty is None):
            raise ValueError("quantity veya quote_qty'den yalnızca biri verilmelidir.")
        path = "/api/v3/order/test" if test else "/api/v3/order"
        return self._request("POST", path, {
            "symbol": symbol.upper(), "side": side.upper(), "type": "MARKET",
            "quantity": quantity, "quoteOrderQty": quote_qty,
            "newOrderRespType": "FULL",
        }, signed=True)

    def my_trades(self, symbol: str, limit: int = 50) -> list:
        return self._request("GET", "/api/v3/myTrades", {"symbol": symbol.upper(), "limit": limit}, signed=True)


def klines_to_frame(raw: list) -> pd.DataFrame:
    df = pd.DataFrame(raw, columns=KLINE_COLUMNS)
    for col in ("open", "high", "low", "close", "volume", "quote_volume"):
        df[col] = df[col].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"].astype("int64"), unit="ms", utc=True)
    return df[["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume"]]
