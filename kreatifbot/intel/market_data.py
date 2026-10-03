"""Binance piyasa verisi servisi.

- Geçmiş veri: giriş + üst zaman dilimleri (spot veya futures klines) ve futures
  türevleri (funding geçmişi, OI geçmişi, long/short oranı, taker hacmi).
- Canlı anlık görüntü: kaynak etiketli FeatureSet (spot_price, futures_mark_price,
  futures_index_price, funding_rate, open_interest, orderbook, ...).

Spot ve futures alanları ayrıdır. Bir kaynak alınamazsa değer UNAVAILABLE olarak
işaretlenir; asla tahmin edilip doldurulmaz.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import pandas as pd

from ..binance_client import BinanceAPIError, BinanceClient
from ..utils import INTERVALS
from .config import IntelConfig
from .futures_client import DATA_PERIODS, BinanceFuturesClient, LiquidationUnavailable
from .orderflow import aggressive_flow, orderbook_features
from .types import Feature, FeatureSet


@dataclass
class HistoricalBundle:
    symbol: str
    market: str
    entry: pd.DataFrame
    htf: dict
    derivatives: dict | None
    btc_trend: pd.DataFrame | None
    errors: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.entry is not None and not self.entry.empty


def _history(client, symbol: str, interval: str, total: int) -> pd.DataFrame:
    frames, remaining, end_time = [], int(total), None
    page = 1500 if isinstance(client, BinanceFuturesClient) else 1000
    while remaining > 0:
        df = client.klines(symbol, interval, limit=min(remaining, page), end_time=end_time)
        if df.empty:
            break
        frames.insert(0, df)
        remaining -= len(df)
        end_time = int(df["open_time"].iloc[0].timestamp() * 1000) - 1
        if len(df) < min(page, remaining + len(df)):
            break
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    return out.reset_index(drop=True)


def _data_history(fn, symbol: str, period: str, start_ms: int, end_ms: int, max_pages: int = 40) -> pd.DataFrame:
    """futures/data uç noktalarını (500 kayıt/sayfa) geriye doğru sayfalar."""
    frames, end = [], end_ms
    for _ in range(max_pages):
        df = fn(symbol, period=period, limit=500, end_time=end)
        if df.empty:
            break
        frames.insert(0, df)
        first = int(df["timestamp"].iloc[0].timestamp() * 1000)
        if first <= start_ms or len(df) < 500:
            break
        end = first - 1
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates("timestamp").sort_values("timestamp") \
        .reset_index(drop=True)


def closed_only(df: pd.DataFrame, now_ms: int | None = None) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    now = pd.Timestamp(now_ms or int(time.time() * 1000), unit="ms", tz="UTC")
    return df[pd.to_datetime(df["close_time"], utc=True) < now].reset_index(drop=True)


def load_history(cfg: IntelConfig, symbol: str, bars: int, spot: BinanceClient | None = None,
                 futures: BinanceFuturesClient | None = None, include_derivatives: bool = True) -> HistoricalBundle:
    """Gerçek Binance geçmiş verisini yükler. Hata olursa bundle.errors doldurulur."""
    spot = spot or BinanceClient(testnet=False)
    futures = futures or BinanceFuturesClient(testnet=False)
    price_client = spot if cfg.market == "SPOT" else futures
    tf = cfg.timeframes
    errors = []
    try:
        entry = closed_only(_history(price_client, symbol, tf.entry, bars))
    except BinanceAPIError as exc:
        return HistoricalBundle(symbol, cfg.market, pd.DataFrame(), {}, None, None, [f"entry: {exc}"])
    if entry.empty:
        return HistoricalBundle(symbol, cfg.market, entry, {}, None, None, ["DATA_UNAVAILABLE: mum verisi yok"])
    span_s = (entry["open_time"].iloc[-1] - entry["open_time"].iloc[0]).total_seconds()
    htf = {}
    for role, interval in tf.higher().items():
        need = int(span_s / INTERVALS[interval]) + 260
        try:
            htf[role] = closed_only(_history(price_client, symbol, interval, min(need, 5000)))
        except BinanceAPIError as exc:
            errors.append(f"{role} ({interval}): {exc}")
    deriv = None
    if include_derivatives and (cfg.market != "SPOT" or cfg.use_futures_context):
        deriv = {}
        start_ms = int(entry["open_time"].iloc[0].timestamp() * 1000)
        end_ms = int(entry["close_time"].iloc[-1].timestamp() * 1000)
        try:
            deriv["funding"] = futures.funding_history_range(symbol, start_ms, end_ms)
        except BinanceAPIError as exc:
            errors.append(f"funding: {exc}")
        period = tf.entry if tf.entry in DATA_PERIODS else "5m"
        deriv["period"] = period
        for key, fn in (("oi", futures.open_interest_hist), ("long_short", futures.long_short_ratio),
                        ("taker", futures.taker_volume)):
            try:
                deriv[key] = _data_history(fn, symbol, period, start_ms, end_ms)
            except (BinanceAPIError, ValueError) as exc:
                errors.append(f"{key}: {exc}")
        if deriv.get("oi") is not None and not deriv["oi"].empty:
            first = deriv["oi"]["timestamp"].iloc[0]
            if first > entry["open_time"].iloc[0]:
                errors.append(f"OI/long-short/taker geçmişi {first:%Y-%m-%d} öncesini kapsamıyor (Binance yalnızca "
                              "son ~30 günü sağlar); önceki mumlarda bu özellikler UNAVAILABLE")
    btc = None
    if cfg.btc_context and symbol != "BTCUSDT":
        try:
            btc = closed_only(_history(price_client, "BTCUSDT", tf.trend, min(int(span_s / INTERVALS[tf.trend]) + 260,
                                                                                    5000)))
        except BinanceAPIError as exc:
            errors.append(f"BTC bağlamı: {exc}")
    return HistoricalBundle(symbol, cfg.market, entry, htf, deriv, btc, errors)


def live_snapshot(symbol: str, spot: BinanceClient | None, futures: BinanceFuturesClient | None,
                  want_spot: bool = True, want_futures: bool = True) -> tuple[FeatureSet, dict]:
    """Kaynak etiketli canlı özellikler ve ham order book."""
    fs = FeatureSet()
    raw = {"spot_book": None, "futures_book": None}
    now = int(time.time() * 1000)
    if want_spot and spot is not None:
        try:
            bt = spot.book_ticker(symbol)
            fs.add(Feature("spot_best_bid", float(bt["bidPrice"]), "binance_spot_bookTicker"))
            fs.add(Feature("spot_best_ask", float(bt["askPrice"]), "binance_spot_bookTicker"))
            fs.add(Feature("spot_price", (float(bt["bidPrice"]) + float(bt["askPrice"])) / 2, "binance_spot_bookTicker"))
        except (BinanceAPIError, KeyError) as exc:
            fs.add(Feature.unavailable("spot_price", "binance_spot_bookTicker", str(exc)))
        try:
            book = spot.depth(symbol, 100)
            book["_fetched_ms"] = now
            raw["spot_book"] = book
            ob = orderbook_features(book)
            for k in ("spread_pct", "imbalance", "bid_depth_quote", "ask_depth_quote"):
                fs.add(Feature(f"spot_orderbook_{k}", ob.get(k), "binance_spot_depth"))
        except BinanceAPIError as exc:
            fs.add(Feature.unavailable("spot_orderbook", "binance_spot_depth", str(exc)))
        try:
            t24 = spot.ticker_24h(symbol)
            fs.add(Feature("spot_volume_24h_quote", float(t24["quoteVolume"]), "binance_spot_ticker24h"))
            fs.add(Feature("spot_change_24h_pct", float(t24["priceChangePercent"]), "binance_spot_ticker24h"))
        except (BinanceAPIError, KeyError) as exc:
            fs.add(Feature.unavailable("spot_volume_24h_quote", "binance_spot_ticker24h", str(exc)))
        try:
            flow = aggressive_flow(spot.agg_trades(symbol, 1000))
            if flow.get("available"):
                fs.add(Feature("spot_aggressive_delta", flow["delta"], "binance_spot_aggTrades"))
                fs.add(Feature("spot_aggressive_buy_ratio", flow["buy_ratio"], "binance_spot_aggTrades"))
        except BinanceAPIError as exc:
            fs.add(Feature.unavailable("spot_aggressive_delta", "binance_spot_aggTrades", str(exc)))
    if want_futures and futures is not None:
        try:
            pi = futures.premium_index(symbol)
            fs.add(Feature("futures_mark_price", float(pi["markPrice"]), "binance_futures_premiumIndex"))
            fs.add(Feature("futures_index_price", float(pi["indexPrice"]), "binance_futures_premiumIndex"))
            fs.add(Feature("funding_rate", float(pi["lastFundingRate"]), "binance_futures_premiumIndex"))
            fs.add(Feature("next_funding_time", int(pi.get("nextFundingTime", 0)), "binance_futures_premiumIndex"))
        except (BinanceAPIError, KeyError) as exc:
            for name in ("futures_mark_price", "futures_index_price", "funding_rate"):
                fs.add(Feature.unavailable(name, "binance_futures_premiumIndex", str(exc)))
        try:
            oi = futures.open_interest(symbol)
            fs.add(Feature("open_interest", float(oi["openInterest"]), "binance_futures_openInterest"))
        except (BinanceAPIError, KeyError) as exc:
            fs.add(Feature.unavailable("open_interest", "binance_futures_openInterest", str(exc)))
        try:
            ls = futures.long_short_ratio(symbol, "5m", 2)
            if not ls.empty:
                fs.add(Feature("long_short_ratio", float(ls["long_short_ratio"].iloc[-1]),
                               "binance_futures_globalLongShortAccountRatio"))
        except (BinanceAPIError, ValueError) as exc:
            fs.add(Feature.unavailable("long_short_ratio", "binance_futures_globalLongShortAccountRatio", str(exc)))
        try:
            tk = futures.taker_volume(symbol, "5m", 2)
            if not tk.empty:
                fs.add(Feature("futures_taker_buy_volume", float(tk["taker_buy_volume"].iloc[-1]),
                               "binance_futures_takerlongshortRatio"))
                fs.add(Feature("futures_taker_sell_volume", float(tk["taker_sell_volume"].iloc[-1]),
                               "binance_futures_takerlongshortRatio"))
        except (BinanceAPIError, ValueError) as exc:
            fs.add(Feature.unavailable("futures_taker_buy_volume", "binance_futures_takerlongshortRatio", str(exc)))
        try:
            book = futures.depth(symbol, 100)
            book["_fetched_ms"] = now
            raw["futures_book"] = book
            ob = orderbook_features(book)
            for k in ("spread_pct", "imbalance", "bid_depth_quote", "ask_depth_quote"):
                fs.add(Feature(f"futures_orderbook_{k}", ob.get(k), "binance_futures_depth"))
        except BinanceAPIError as exc:
            fs.add(Feature.unavailable("futures_orderbook", "binance_futures_depth", str(exc)))
        try:
            futures.liquidations(symbol)
        except LiquidationUnavailable as exc:
            fs.add(Feature.unavailable("liquidations", "binance_futures", str(exc)))
    return fs, raw
