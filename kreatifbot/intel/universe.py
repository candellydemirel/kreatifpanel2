"""Otomatik coin seçimi (işlem evreni).

Bot yalnızca elle yazılan birkaç ana coine bakmak yerine Binance'teki USDT çiftlerinden
likit ve hareketli olanları kendisi seçer:
- Yarısı: son 24 saatin en yüksek işlem hacimli coinleri (likidite).
- Yarısı: yeterli hacmi olan coinler arasında en çok hareket edenler (fırsat).
Stablecoinler, kaldıraçlı tokenlar (UP/DOWN/BULL/BEAR) ve işlemi kapalı çiftler elenir.
Gerçek Binance verisi alınamazsa seçim yapılmaz; elle girilen semboller kullanılır.
"""

from __future__ import annotations

import logging
import re
import time

logger = logging.getLogger("kreatifbot.intel.universe")

STABLES = {"USDC", "FDUSD", "TUSD", "BUSD", "DAI", "USDP", "USDD", "PYUSD", "USDE", "EUR", "EURI", "AEUR", "GBP",
           "TRY", "BRL", "XUSD", "USD1", "RLUSD", "PAXG", "WBTC", "WBETH", "BFUSD"}
LEVERAGED = re.compile(r"(UP|DOWN|BULL|BEAR)$")


def select_universe(tickers: list[dict], quote: str = "USDT", n: int = 15, min_quote_volume: float = 20_000_000,
                    tradable: set | None = None, always: list[str] | None = None) -> tuple[list[str], dict]:
    """(semboller, sembol -> seçilme nedeni). `always` her zaman listede kalır (ör. açık pozisyonlar)."""
    rows = []
    for t in tickers or []:
        sym = str(t.get("symbol", ""))
        if not sym.endswith(quote):
            continue
        base = sym[:-len(quote)]
        if not base or base in STABLES or LEVERAGED.search(base):
            continue
        if tradable is not None and sym not in tradable:
            continue
        try:
            qv = float(t.get("quoteVolume") or 0)
            chg = float(t.get("priceChangePercent") or 0)
        except (TypeError, ValueError):
            continue
        if qv < min_quote_volume:
            continue
        rows.append((sym, qv, chg))
    reasons: dict[str, str] = {}
    out: list[str] = []
    for sym in always or []:
        if sym not in out:
            out.append(sym)
            reasons[sym] = "Sizin listeniz / açık pozisyon"
    by_vol = sorted(rows, key=lambda r: -r[1])
    by_move = sorted(rows, key=lambda r: -abs(r[2]))
    n_vol = max(1, n // 2)
    for sym, qv, chg in by_vol:
        if len([s for s in out if reasons.get(s, "").startswith("Hacim")]) >= n_vol or len(out) >= n:
            break
        if sym not in out:
            out.append(sym)
            reasons[sym] = f"Hacim: {qv / 1e6:.0f}M {quote}"
    for sym, qv, chg in by_move:
        if len(out) >= n:
            break
        if sym not in out:
            out.append(sym)
            reasons[sym] = f"Hareket: %{chg:+.1f} (24s), hacim {qv / 1e6:.0f}M"
    return out, reasons


class UniverseSelector:
    """Coin listesini belirli aralıklarla Binance'ten yeniler."""

    def __init__(self, client, quote: str = "USDT", n: int = 15, min_quote_volume: float = 20_000_000,
                 refresh_s: float = 3600, base_symbols: list[str] | None = None):
        self.client = client
        self.quote = quote
        self.n = n
        self.min_quote_volume = min_quote_volume
        self.refresh_s = refresh_s
        self.base_symbols = list(base_symbols or [])
        self.last_refresh = 0.0
        self.reasons: dict[str, str] = {}
        self.status = "Henüz seçilmedi"

    def due(self) -> bool:
        return time.time() - self.last_refresh >= self.refresh_s

    def refresh(self, keep: list[str] | None = None) -> list[str] | None:
        """Yeni liste; veri alınamazsa None (mevcut liste korunur)."""
        self.last_refresh = time.time()
        try:
            tickers = self.client.ticker_24h()
            tradable = None
            try:
                info = self.client.exchange_info()
                tradable = {s["symbol"] for s in info.get("symbols", [])
                            if s.get("status") == "TRADING" and s.get("quoteAsset") == self.quote
                            and s.get("isSpotTradingAllowed", True)
                            and s.get("contractType", "PERPETUAL") == "PERPETUAL"}
            except Exception as exc:  # noqa: BLE001 - durum filtresi olmadan da hacim filtresi çalışır
                logger.info("exchangeInfo alınamadı: %s", exc)
        except Exception as exc:  # noqa: BLE001
            self.status = f"Coin listesi alınamadı ({exc}); mevcut liste korunuyor"
            return None
        always = list(dict.fromkeys(self.base_symbols + list(keep or [])))
        syms, reasons = select_universe(tickers if isinstance(tickers, list) else [], self.quote,
                                        max(self.n, len(always)), self.min_quote_volume, tradable, always)
        if len(syms) <= len(always) and not reasons:
            self.status = "Uygun coin bulunamadı; mevcut liste korunuyor"
            return None
        self.reasons = reasons
        self.status = f"{len(syms)} coin seçildi"
        return syms
