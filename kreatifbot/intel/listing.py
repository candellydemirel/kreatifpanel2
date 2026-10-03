"""Yeni listeleme stratejisi ve listeleme backtest'i.

Yeni coinlerin geçmiş verisi olmadığı için normal stratejiler NO TRADE der. Bu
modül açılış aralığı kırılımı (opening range breakout) kullanır:

1. Açılıştan sonra `wait_minutes` beklenir (ilk dakikaların fiyat keşfi kaosu).
2. İlk `range_minutes` mumun yüksek/düşük aralığı belirlenir.
3. Kapanış aralık tepesini hacimle YENİ kırarsa ve fiyat açılış VWAP'ının üstündeyse LONG adayı.
4. Fiyat kırılım seviyesinin çok üstündeyse girilmez (kovalamama).
5. Stop aralık dibi (en fazla %max_stop_pct), hedefler R katları, maksimum süre sınırlı.

Uyarı: Listelemelerin ilk saatleri aşırı oynaktır; pek çok coin ilk yükselişten sonra
sert düşer. Strateji varsayılan olarak PAPER aşamasındadır ve küçük risk kullanır.
Gerçek performansı `listing_backtest` ile geçmiş listelemelerde ölçün.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .config import CostConfig, ListingConfig, RiskConfig
from .position_manager import open_position, update_on_bar
from .types import ExitReason


@dataclass
class ListingSignal:
    symbol: str
    ok: bool
    reasons: list = field(default_factory=list)
    entry: float = float("nan")
    stop: float = float("nan")
    targets: list = field(default_factory=list)
    range_high: float = float("nan")
    range_low: float = float("nan")
    vwap: float = float("nan")
    minutes_since_open: float = 0.0
    quote_volume: float = 0.0

    def explain(self) -> str:
        head = f"{self.symbol} — Yeni listeleme analizi: {'LONG ADAYI' if self.ok else 'İŞLEM YOK'}"
        lines = [head, f"Açılıştan beri {self.minutes_since_open:.0f} dk, hacim {self.quote_volume:,.0f} USDT",
                 f"Açılış aralığı: {self.range_low:.6g} – {self.range_high:.6g}, VWAP {self.vwap:.6g}"]
        if self.ok:
            lines += [f"Giriş ~{self.entry:.6g}, stop {self.stop:.6g}, hedefler " +
                      ", ".join(f"{t:.6g}" for t in self.targets)]
        lines += [f"  • {r}" for r in self.reasons]
        lines.append("Not: Listeleme işlemleri yüksek risklidir; kâr garantisi yoktur.")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_listing(df: pd.DataFrame, symbol: str, cfg: ListingConfig, i: int | None = None,
                     book_stats: dict | None = None) -> ListingSignal:
    """df: listeleme açılışından itibaren KAPANMIŞ 1 dakikalık mumlar. i: değerlendirilecek mum (varsayılan son)."""
    n = len(df) if i is None else i + 1
    sig = ListingSignal(symbol, False)
    if n < cfg.range_minutes + 2:
        sig.reasons.append(f"Yetersiz veri: {n} dk (en az {cfg.range_minutes + 2} dk gerekli)")
        return sig
    d = df.iloc[:n]
    t0 = pd.to_datetime(d["open_time"].iloc[0], utc=True)
    t_now = pd.to_datetime(d["close_time"].iloc[-1], utc=True)
    sig.minutes_since_open = (t_now - t0).total_seconds() / 60
    rng = d.iloc[:cfg.range_minutes]
    sig.range_high, sig.range_low = float(rng["high"].max()), float(rng["low"].min())
    tp = (d["high"] + d["low"] + d["close"]) / 3
    vol = d["volume"].replace(0, np.nan)
    sig.vwap = float((tp * d["volume"]).sum() / vol.sum()) if vol.sum() > 0 else float("nan")
    sig.quote_volume = float(d["quote_volume"].sum()) if "quote_volume" in d else float((d["close"] * d["volume"]).sum())
    close, prev_close = float(d["close"].iloc[-1]), float(d["close"].iloc[-2])
    last_vol = float(d["volume"].iloc[-1])
    base_vol = float(d["volume"].iloc[cfg.range_minutes:-1].mean()) if n - 1 > cfg.range_minutes else \
        float(rng["volume"].mean())

    checks = [
        (sig.minutes_since_open >= cfg.wait_minutes,
         f"Bekleme süresi {cfg.wait_minutes} dk dolmadı ({sig.minutes_since_open:.0f} dk)"),
        (sig.quote_volume >= cfg.min_quote_volume,
         f"Hacim yetersiz ({sig.quote_volume:,.0f} < {cfg.min_quote_volume:,.0f} USDT)"),
        (close > sig.range_high and prev_close <= sig.range_high,
         "Açılış aralığının taze kırılımı yok (kapanış tepeyi yeni aşmadı)"),
        (close <= sig.range_high * (1 + cfg.max_chase_pct / 100),
         f"Fiyat kırılım seviyesinin %{cfg.max_chase_pct} üstünde (kovalanmaz)"),
        (np.isfinite(sig.vwap) and close > sig.vwap, "Fiyat açılış VWAP'ının altında"),
        (base_vol > 0 and last_vol >= cfg.volume_mult * base_vol,
         f"Kırılım mumunda hacim onayı yok ({last_vol / base_vol if base_vol else 0:.1f}x < {cfg.volume_mult}x)"),
    ]
    if book_stats is not None:
        if book_stats.get("available"):
            depth = book_stats.get("bid_depth_quote", 0) + book_stats.get("ask_depth_quote", 0)
            checks.append((book_stats.get("spread_pct", 99) <= cfg.max_spread_pct,
                           f"Spread yüksek (%{book_stats.get('spread_pct', 0):.3f})"))
            checks.append((depth >= cfg.min_depth_quote, f"Order book derinliği düşük ({depth:,.0f} USDT)"))
        else:
            checks.append((False, "Order book alınamadı"))
    failed = [msg for ok, msg in checks if not ok]
    if failed:
        sig.reasons = failed
        return sig
    stop = max(sig.range_low, close * (1 - cfg.max_stop_pct / 100)) * (1 - cfg.stop_buffer_pct / 100)
    risk = close - stop
    if risk <= 0 or risk / close * 100 > cfg.max_stop_pct + cfg.stop_buffer_pct:
        sig.reasons = [f"Stop mesafesi çok geniş (%{risk / close * 100:.1f})"]
        return sig
    sig.ok, sig.entry, sig.stop = True, close, stop
    sig.targets = [close + r * risk for r in cfg.tp_levels_r]
    sig.reasons = [f"Açılış aralığı tepesi {sig.range_high:.6g} hacimle kırıldı ({last_vol / base_vol:.1f}x)",
                   "Fiyat açılış VWAP'ının üstünde", f"Hacim {sig.quote_volume:,.0f} USDT"]
    return sig


# ---------------------------------------------------------------------- backtest
@dataclass
class ListingTrade:
    symbol: str
    entry_time: str
    entry: float
    exit: float
    exit_reason: str
    minutes: float
    net_ret_pct: float
    gross_ret_pct: float
    r_multiple: float


def simulate_listing(df: pd.DataFrame, symbol: str, cfg: ListingConfig, costs: CostConfig,
                     risk: RiskConfig | None = None) -> tuple[ListingTrade | None, str]:
    """Tek listeleme: ilk geçerli sinyalde sonraki mumun açılışından gir, 1 dk mumlarla yönet."""
    if df is None or len(df) < cfg.range_minutes + 5:
        return None, "DATA_UNAVAILABLE: yetersiz 1 dk veri"
    risk = risk or RiskConfig()
    rc = RiskConfig(**{**risk.__dict__, "tp_levels_r": cfg.tp_levels_r, "tp_fractions": cfg.tp_fractions,
                       "trailing_method": "percent"})
    fee = costs.spot_taker_fee_pct / 100
    slip = costs.slippage_pct * cfg.slippage_mult / 100
    t0 = pd.to_datetime(df["open_time"].iloc[0], utc=True)
    last_i = len(df) - 2
    for i in range(cfg.range_minutes + 1, last_i):
        if (pd.to_datetime(df["close_time"].iat[i], utc=True) - t0).total_seconds() / 3600 > cfg.watch_hours:
            return None, "İzleme süresinde sinyal yok"
        sig = evaluate_listing(df, symbol, cfg, i)
        if not sig.ok:
            continue
        e = i + 1
        entry = float(df["open"].iat[e]) * (1 + slip)
        if entry <= sig.stop:
            continue
        pos = open_position(symbol, "SPOT", "LONG", entry, 1.0, sig.stop,
                            [entry + (t - sig.entry) for t in sig.targets], cfg.tp_fractions, "new_listing",
                            str(df["open_time"].iat[e]), e, cfg.max_hold_minutes, cfg.max_hold_minutes, 0, "LISTING")
        realized, exit_px_w, reason = 0.0, 0.0, ExitReason.END_OF_DATA
        for j in range(e, len(df)):
            row = df.iloc[j]
            acts = update_on_bar(pos, pd.Series({"open": row["open"], "high": row["high"], "low": row["low"],
                                                 "close": row["close"], "atr": np.nan}), rc,
                                 bar_open=float(row["open"]))
            for a in acts:
                px = a.price * (1 - slip)
                realized += (px - entry) * a.qty
                exit_px_w += px * a.qty
                reason = a.reason
            if pos.qty <= 1e-12:
                end = j
                break
        else:
            end = len(df) - 1
            px = float(df["close"].iat[end]) * (1 - slip)
            realized += (px - entry) * pos.qty
            exit_px_w += px * pos.qty
        gross = realized / entry * 100
        net = gross - 2 * fee * 100
        r_mult = (realized - 2 * fee * entry) / (entry - sig.stop)
        minutes = (end - e + 1)
        return ListingTrade(symbol, str(df["open_time"].iat[e]), entry, exit_px_w, str(reason), minutes,
                            net, gross, r_mult), "OK"
    return None, "Sinyal oluşmadı"


def listing_backtest(frames: dict[str, pd.DataFrame], cfg: ListingConfig, costs: CostConfig) -> dict:
    rows, notes = [], {}
    for sym, df in frames.items():
        trade, msg = simulate_listing(df, sym, cfg, costs)
        notes[sym] = msg
        if trade is not None:
            rows.append(asdict(trade))
    trades = pd.DataFrame(rows)
    if trades.empty:
        summary = {"listeleme": len(frames), "işlem": 0}
    else:
        r = trades["net_ret_pct"]
        wins, losses = r[r > 0], r[r <= 0]
        summary = {
            "listeleme": len(frames), "işlem": len(trades), "kazanma_%": float((r > 0).mean() * 100),
            "ort_net_getiri_%": float(r.mean()), "medyan_net_getiri_%": float(r.median()),
            "en_iyi_%": float(r.max()), "en_kötü_%": float(r.min()), "ort_R": float(trades["r_multiple"].mean()),
            "kâr_faktörü": float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() < 0 else float("inf"),
            "ort_süre_dk": float(trades["minutes"].mean()),
        }
        if len(trades) < 20:
            summary["uyarı"] = "20'den az işlem: istatistiksel olarak güvenilir değil"
    return {"summary": summary, "trades": trades, "notes": notes}


def fetch_listing_history(client, symbol: str, hours: float = 6.0) -> pd.DataFrame:
    """Sembolün ilk işlem anından itibaren 1 dk mumlar (gerçek Binance verisi)."""
    frames, start, need = [], 0, int(hours * 60)
    while need > 0:
        df = client.klines(symbol, "1m", limit=min(1000, need), start_time=start)
        if df.empty:
            break
        frames.append(df)
        need -= len(df)
        start = int(pd.to_datetime(df["open_time"].iloc[-1], utc=True).timestamp() * 1000) + 60_000
        if len(df) < 1000:
            break
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True).drop_duplicates("open_time")
    now = pd.Timestamp.now(tz="UTC")
    return out[pd.to_datetime(out["close_time"], utc=True) < now].reset_index(drop=True)
