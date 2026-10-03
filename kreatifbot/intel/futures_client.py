"""Binance USDⓈ-M Futures REST istemcisi.

Spot istemcisinin imzalama/saat eşitleme altyapısını kullanır; yalnızca futures
uç noktalarını içerir. Spot ve futures verileri asla aynı alana yazılmaz.

Not: Binance, geçmiş tasfiye (liquidation) verisi için herkese açık REST uç
noktasını kapatmıştır (yalnızca websocket `forceOrder` akışı vardır). Bu yüzden
`liquidations()` veri üretmez; LiquidationUnavailable fırlatır.
"""

from __future__ import annotations

import pandas as pd

from ..binance_client import BinanceAPIError, BinanceClient, SymbolRules, klines_to_frame, parse_symbol_rules

FUTURES_MAINNET_URL = "https://fapi.binance.com"
FUTURES_TESTNET_URL = "https://testnet.binancefuture.com"
DATA_PERIODS = ("5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d")


class LiquidationUnavailable(Exception):
    pass


class BinanceFuturesClient(BinanceClient):
    market = "USDM_FUTURES"

    def __init__(self, api_key: str = "", api_secret: str = "", testnet: bool = True, **kwargs):
        super().__init__(api_key, api_secret, testnet=testnet, **kwargs)
        self.base_url = FUTURES_TESTNET_URL if testnet else FUTURES_MAINNET_URL

    # ------------------------------------------------------------------ genel
    def ping(self) -> bool:
        self._request("GET", "/fapi/v1/ping")
        return True

    def server_time(self) -> int:
        return int(self._request("GET", "/fapi/v1/time")["serverTime"])

    def exchange_info(self, symbol: str | None = None) -> dict:
        info = self._request("GET", "/fapi/v1/exchangeInfo")
        if symbol:
            info = dict(info)
            info["symbols"] = [s for s in info.get("symbols", []) if s.get("symbol") == symbol.upper()]
            if not info["symbols"]:
                raise BinanceAPIError(400, -1121, f"Futures sembolü bulunamadı: {symbol}")
        return info

    def symbol_rules(self, symbol: str) -> SymbolRules:
        symbol = symbol.upper()
        if symbol not in self._rules_cache:
            self.all_symbol_rules(trading_only=False)
        if symbol not in self._rules_cache:
            raise BinanceAPIError(400, -1121, f"Futures sembolü bulunamadı: {symbol}")
        return self._rules_cache[symbol]

    def all_symbol_rules(self, quote_asset: str | None = None, trading_only: bool = True) -> dict[str, SymbolRules]:
        out = {}
        for sym in self.exchange_info().get("symbols", []):
            if sym.get("contractType", "PERPETUAL") != "PERPETUAL":
                continue
            rules = parse_symbol_rules(sym, "USDM_FUTURES")
            self._rules_cache[rules.symbol] = rules
            if quote_asset and rules.quote_asset != quote_asset:
                continue
            if trading_only and rules.status != "TRADING":
                continue
            out[rules.symbol] = rules
        return out

    # ------------------------------------------------------------------ piyasa verisi
    def klines(self, symbol: str, interval: str, limit: int = 500, end_time: int | None = None,
               closed_only: bool = False) -> pd.DataFrame:
        raw = self._request("GET", "/fapi/v1/klines", {
            "symbol": symbol.upper(), "interval": interval, "limit": max(1, min(int(limit), 1500)),
            "endTime": end_time})
        if closed_only and raw:
            import time
            now_ms = int(time.time() * 1000) + self.time_offset_ms
            raw = [r for r in raw if int(r[6]) < now_ms]
        return klines_to_frame(raw)

    def mark_price_klines(self, symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
        raw = self._request("GET", "/fapi/v1/markPriceKlines",
                            {"symbol": symbol.upper(), "interval": interval, "limit": min(limit, 1500)})
        return _price_klines(raw)

    def index_price_klines(self, pair: str, interval: str, limit: int = 500) -> pd.DataFrame:
        raw = self._request("GET", "/fapi/v1/indexPriceKlines",
                            {"pair": pair.upper(), "interval": interval, "limit": min(limit, 1500)})
        return _price_klines(raw)

    def premium_index(self, symbol: str) -> dict:
        """markPrice, indexPrice, lastFundingRate, nextFundingTime."""
        return self._request("GET", "/fapi/v1/premiumIndex", {"symbol": symbol.upper()})

    def funding_history(self, symbol: str, start_time: int | None = None, end_time: int | None = None,
                        limit: int = 1000) -> pd.DataFrame:
        raw = self._request("GET", "/fapi/v1/fundingRate", {
            "symbol": symbol.upper(), "startTime": start_time, "endTime": end_time, "limit": min(limit, 1000)})
        df = pd.DataFrame(raw or [], columns=["symbol", "fundingTime", "fundingRate", "markPrice"])
        df["fundingTime"] = pd.to_datetime(df["fundingTime"].astype("int64"), unit="ms", utc=True)
        df["fundingRate"] = df["fundingRate"].astype(float)
        return df[["fundingTime", "fundingRate"]]

    def funding_history_range(self, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        frames, cursor = [], start_ms
        while cursor < end_ms:
            df = self.funding_history(symbol, start_time=cursor, end_time=end_ms)
            if df.empty:
                break
            frames.append(df)
            cursor = int(df["fundingTime"].iloc[-1].timestamp() * 1000) + 1
            if len(df) < 1000:
                break
        if not frames:
            return pd.DataFrame(columns=["fundingTime", "fundingRate"])
        return pd.concat(frames, ignore_index=True).drop_duplicates("fundingTime")

    def open_interest(self, symbol: str) -> dict:
        return self._request("GET", "/fapi/v1/openInterest", {"symbol": symbol.upper()})

    def _data_endpoint(self, path: str, symbol: str, period: str, limit: int, start_time=None, end_time=None):
        if period not in DATA_PERIODS:
            raise ValueError(f"Futures veri periyodu desteklenmiyor: {period} (izinli: {', '.join(DATA_PERIODS)})")
        return self._request("GET", path, {"symbol": symbol.upper(), "period": period, "limit": min(limit, 500),
                                           "startTime": start_time, "endTime": end_time})

    def open_interest_hist(self, symbol: str, period: str = "5m", limit: int = 500, **kw) -> pd.DataFrame:
        """Binance yalnızca son ~30 günü sağlar."""
        raw = self._data_endpoint("/futures/data/openInterestHist", symbol, period, limit, **kw)
        df = pd.DataFrame(raw or [], columns=["symbol", "sumOpenInterest", "sumOpenInterestValue", "timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
        df["open_interest"] = df["sumOpenInterest"].astype(float)
        df["open_interest_value"] = df["sumOpenInterestValue"].astype(float)
        return df[["timestamp", "open_interest", "open_interest_value"]]

    def long_short_ratio(self, symbol: str, period: str = "5m", limit: int = 500, top_traders: bool = False,
                         **kw) -> pd.DataFrame:
        path = "/futures/data/topLongShortPositionRatio" if top_traders else "/futures/data/globalLongShortAccountRatio"
        raw = self._data_endpoint(path, symbol, period, limit, **kw)
        df = pd.DataFrame(raw or [], columns=["symbol", "longShortRatio", "longAccount", "shortAccount", "timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
        df["long_short_ratio"] = df["longShortRatio"].astype(float)
        return df[["timestamp", "long_short_ratio"]]

    def taker_volume(self, symbol: str, period: str = "5m", limit: int = 500, **kw) -> pd.DataFrame:
        raw = self._data_endpoint("/futures/data/takerlongshortRatio", symbol, period, limit, **kw)
        df = pd.DataFrame(raw or [], columns=["buySellRatio", "buyVol", "sellVol", "timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
        for c in ("buySellRatio", "buyVol", "sellVol"):
            df[c] = df[c].astype(float)
        return df.rename(columns={"buySellRatio": "taker_buy_sell_ratio", "buyVol": "taker_buy_volume",
                                  "sellVol": "taker_sell_volume"})

    def depth(self, symbol: str, limit: int = 100) -> dict:
        return self._request("GET", "/fapi/v1/depth", {"symbol": symbol.upper(), "limit": limit})

    def book_ticker(self, symbol: str | None = None):
        return self._request("GET", "/fapi/v1/ticker/bookTicker", {"symbol": symbol})

    def ticker_24h(self, symbol: str | None = None):
        return self._request("GET", "/fapi/v1/ticker/24hr", {"symbol": symbol})

    def price(self, symbol: str) -> float:
        return float(self._request("GET", "/fapi/v1/ticker/price", {"symbol": symbol.upper()})["price"])

    def agg_trades(self, symbol: str, limit: int = 500, start_time=None, end_time=None) -> list:
        return self._request("GET", "/fapi/v1/aggTrades", {
            "symbol": symbol.upper(), "limit": min(limit, 1000), "startTime": start_time, "endTime": end_time})

    def trades(self, symbol: str, limit: int = 500) -> list:
        return self._request("GET", "/fapi/v1/trades", {"symbol": symbol.upper(), "limit": min(limit, 1000)})

    def liquidations(self, symbol: str):
        raise LiquidationUnavailable(
            "Binance geçmiş tasfiye verisi için herkese açık REST uç noktası sunmuyor (yalnızca websocket "
            "forceOrder akışı). Tasfiye özellikleri UNAVAILABLE olarak işaretlenir.")

    # ------------------------------------------------------------------ hesap (imzalı)
    def account(self) -> dict:
        return self._request("GET", "/fapi/v2/account", signed=True)

    def balances(self) -> dict[str, dict[str, float]]:
        out = {}
        for b in self._request("GET", "/fapi/v2/balance", signed=True):
            bal = float(b.get("balance", 0))
            avail = float(b.get("availableBalance", 0))
            if bal or avail:
                out[b["asset"]] = {"free": avail, "locked": max(0.0, bal - avail), "wallet": bal}
        return out

    def position_risk(self, symbol: str | None = None) -> list:
        return self._request("GET", "/fapi/v2/positionRisk", {"symbol": symbol}, signed=True)

    def leverage_bracket(self, symbol: str) -> list:
        data = self._request("GET", "/fapi/v1/leverageBracket", {"symbol": symbol.upper()}, signed=True)
        if isinstance(data, list) and data and "brackets" in data[0]:
            return data[0]["brackets"]
        return data.get("brackets", []) if isinstance(data, dict) else []

    def change_leverage(self, symbol: str, leverage: int) -> dict:
        return self._request("POST", "/fapi/v1/leverage", {"symbol": symbol.upper(), "leverage": int(leverage)},
                             signed=True)

    def change_margin_type(self, symbol: str, margin_type: str) -> None:
        try:
            self._request("POST", "/fapi/v1/marginType",
                          {"symbol": symbol.upper(), "marginType": margin_type.upper()}, signed=True)
        except BinanceAPIError as exc:
            if exc.code != -4046:  # "No need to change margin type."
                raise

    def new_order(self, symbol: str, side: str, order_type: str, quantity: str | None = None,
                  price: str | None = None, stop_price: str | None = None, reduce_only: bool = False,
                  close_position: bool = False, callback_rate: float | None = None,
                  time_in_force: str | None = None, working_type: str = "MARK_PRICE") -> dict:
        params = {
            "symbol": symbol.upper(), "side": side.upper(), "type": order_type.upper(),
            "quantity": quantity, "price": price, "stopPrice": stop_price,
            "newOrderRespType": "RESULT",
        }
        if reduce_only and not close_position:
            params["reduceOnly"] = "true"
        if close_position:
            params["closePosition"] = "true"
            params.pop("quantity")
        if callback_rate is not None:
            params["callbackRate"] = callback_rate
        if order_type.upper() in ("LIMIT",):
            params["timeInForce"] = time_in_force or "GTC"
        if order_type.upper() in ("STOP_MARKET", "TAKE_PROFIT_MARKET", "STOP", "TAKE_PROFIT",
                                  "TRAILING_STOP_MARKET"):
            params["workingType"] = working_type
        return self._request("POST", "/fapi/v1/order", params, signed=True)

    def cancel_all_orders(self, symbol: str) -> dict:
        return self._request("DELETE", "/fapi/v1/allOpenOrders", {"symbol": symbol.upper()}, signed=True)

    def cancel_order(self, symbol: str, order_id: int) -> dict:
        return self._request("DELETE", "/fapi/v1/order", {"symbol": symbol.upper(), "orderId": order_id},
                             signed=True)

    def open_orders(self, symbol: str | None = None) -> list:
        return self._request("GET", "/fapi/v1/openOrders", {"symbol": symbol}, signed=True)


def _price_klines(raw) -> pd.DataFrame:
    df = pd.DataFrame(raw or [], columns=["open_time", "open", "high", "low", "close", "_v", "close_time",
                                          "_q", "_n", "_tb", "_tq", "_i"])
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"].astype("int64"), unit="ms", utc=True)
    return df[["open_time", "open", "high", "low", "close", "close_time"]]
