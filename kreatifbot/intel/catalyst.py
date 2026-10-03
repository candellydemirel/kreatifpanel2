"""Öngörü motoru: haber katalizörleri + temel veriler + teknik onay → İZLE / AL sinyali.

Katalizör: ortaklık/anlaşma, mainnet/yükseltme, ETF, kurumsal benimseme, büyük borsa
listelemesi, yakım/geri alım, yatırım turu (olumlu); token kilidi açılması, hack, dava,
delist (olumsuz). Kaynak güvenilirliği, bağımsız kaynak sayısı, yenilik (yarılanma)
ve 'söylenti' ifadeleri hesaba katılır.

Temel skor (whitepaper'ın işlevselliği için ÖLÇÜLEBİLİR vekil): DeFiLlama TVL ve
değişimi, piyasa değeri/TVL; CoinGecko geliştirici aktivitesi, piyasa değeri sırası,
proje yaşı. Whitepaper metni otomatik değerlendirilmez (yalnızca link gösterilir).
Veri yoksa ilgili bileşen UNAVAILABLE olur; sahte değer üretilmez.

Teknik onay Binance verisiyle yapılır; haberden sonra fiyat zaten çok yükseldiyse
"fiyatlanmış" sayılır ve AL verilmez. Öngörüler kaydedilip 4s/24s/72s sonraki gerçek
getirileri ölçülür; hangi katalizörün işe yaradığı zamanla buradan görülür.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ..config import data_dir
from .config import CatalystConfig
from ..i18n import tr
from .news import NewsItem

DEFILLAMA_PROTOCOLS = "https://api.llama.fi/protocols"
COINGECKO_SEARCH = "https://api.coingecko.com/api/v3/search"
COINGECKO_COIN = "https://api.coingecko.com/api/v3/coins/{id}"

# tür → (desenler, ağırlık)
CATALYST_RULES = [
    ("DELISTING", [r"\bdelist", r"will (cease|end) trading"], -1.0),
    ("HACK", [r"\bhack(ed|er|s)?\b", r"\bexploit", r"drain(ed)?", r"stolen", r"rug ?pull", r"breach",
              r"saldırı", r"hackl", r"çalın", r"istismar", r"güvenlik açığı"], -1.0),
    ("REGULATION_NEG", [r"\bsec (sues|charges)", r"lawsuit", r"\bsued\b", r"\bban(s|ned)?\b", r"crackdown",
                        r"dava", r"yasak", r"soruşturma", r"suçlama"], -0.8),
    ("TOKEN_UNLOCK", [r"token unlock", r"\bunlocks?\b", r"vesting (cliff|release)", r"kilit (açılımı|açılması)"],
     -0.6),
    ("ETF", [r"\betf\b.*(approv|filing|files|launch|inflow)", r"spot etf", r"etf.*(onay|başvuru|giriş)"], 1.0),
    ("PARTNERSHIP", [r"partner(s|ed|ship)? with", r"partnership", r"teams? up", r"collaborat",
                     r"signs? (a |an )?(deal|agreement|mou|contract)", r"strategic (deal|alliance|investment)",
                     r"integrat(es|ed|ion) (with|into)", r"anlaşma", r"ortaklık", r"iş ?birliği", r"ortak oldu",
                     r"entegre (etti|ediyor|edildi)", r"entegrasyon"], 1.0),
    ("INSTITUTIONAL", [r"(treasury|reserve)s? (buys?|adds?|purchases?)", r"institutional (adoption|demand|investors?)",
                       r"\b(blackrock|fidelity|visa|mastercard|paypal|stripe|google|microsoft|amazon|nvidia|jpmorgan)\b"],
     0.9),
    ("MAINNET_UPGRADE", [r"mainnet", r"upgrade (goes live|activated|completed|is live)", r"hard ?fork", r"güncellemesi (yayında|aktif|tamamlandı)",
                         r"launch(es|ed)? (its |the )?(v\d|protocol|network|chain|layer)", r"ana ağ"], 0.8),
    ("EXCHANGE_LISTING", [r"(coinbase|upbit|robinhood|kraken|okx|bybit) (will )?(list|adds?)",
                          r"listing on (coinbase|upbit|robinhood|kraken)", r"binance will list",
                          r"(coinbase|upbit|robinhood|kraken|okx|bybit).*listeled"], 0.7),
    ("ADOPTION", [r"adopts?\b", r"adoption", r"accepts? .*payments?", r"payments? (with|in) \w+", r"benimse"], 0.6),
    ("FUNDING", [r"raises? \$", r"funding round", r"series [abc]\b", r"seed round", r"yatırım turu", r"fon topladı", r"yatırım aldı"], 0.5),
    ("BURN_BUYBACK", [r"\bburn(s|ed)?\b", r"buy-?back", r"yakım", r"geri alım"], 0.5),
]
RUMOR = [r"\brumou?r", r"reportedly", r"\bcould\b", r"\bmay\b", r"speculat", r"unconfirmed", r"söylenti", r"iddia"]
SOURCE_WEIGHT = {"Binance duyuru": 1.0, "CoinDesk": 0.9, "Cointelegraph": 0.8, "Decrypt": 0.8, "CryptoPanic": 0.6,
                 "Cointürk": 0.7, "Koinmedya": 0.7, "BTC Haber": 0.7}

CATALYST_TR = {
    "PARTNERSHIP": "ortaklık/anlaşma", "ETF": "ETF", "INSTITUTIONAL": "kurumsal ilgi", "MAINNET_UPGRADE":
    "mainnet/yükseltme", "EXCHANGE_LISTING": "borsa listelemesi", "ADOPTION": "benimseme", "FUNDING": "yatırım turu",
    "BURN_BUYBACK": "yakım/geri alım", "TOKEN_UNLOCK": "token kilidi açılımı", "HACK": "hack/exploit",
    "REGULATION_NEG": "düzenleyici risk", "DELISTING": "delist",
}


def detect_catalysts(text: str) -> list[tuple[str, float]]:
    t = text.lower()
    out = []
    for name, patterns, w in CATALYST_RULES:
        if any(re.search(p, t) for p in patterns):
            out.append((name, w))
    return out


def is_rumor(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in RUMOR)


@dataclass
class CatalystResult:
    base: str
    score: float = 0.0                 # 0-100 olumlu katalizör gücü
    negative: float = 0.0              # 0-100 olumsuz baskı
    types: list = field(default_factory=list)
    sources: list = field(default_factory=list)
    headlines: list = field(default_factory=list)
    first_seen: str = ""
    rumor_only: bool = False


def score_catalysts(base: str, items: list[NewsItem], cfg: CatalystConfig) -> CatalystResult:
    res = CatalystResult(base)
    pos_raw, neg_raw, rumors, total = 0.0, 0.0, 0, 0
    sources, first = set(), None
    for it in items:
        if base not in it.symbols or it.age_hours > cfg.lookback_hours:
            continue
        cats = detect_catalysts(f"{it.title} {it.summary}")
        if not cats:
            continue
        total += 1
        decay = 0.5 ** (max(it.age_hours, 0) / cfg.half_life_hours)
        sw = SOURCE_WEIGHT.get(it.source, 0.6)
        rumor = is_rumor(it.title)
        rumors += rumor
        rf = 0.5 if rumor else 1.0
        for name, w in cats:
            contrib = abs(w) * sw * decay * rf
            if w > 0:
                pos_raw += contrib
            else:
                neg_raw += contrib
            if name not in res.types:
                res.types.append(name)
        if any(w > 0 for _, w in cats):
            sources.add(it.source)
            if first is None or it.published_at < first:
                first = it.published_at
        res.headlines.append(f"[{it.source}] {it.display_title}")
    if len(sources) >= 2:
        pos_raw *= 1.0 + 0.15 * (len(sources) - 1)  # bağımsız kaynak teyidi
    res.score = round(100 * (1 - math.exp(-pos_raw)), 1)
    res.negative = round(100 * (1 - math.exp(-neg_raw)), 1)
    res.sources = sorted(sources)
    res.first_seen = first or ""
    res.rumor_only = total > 0 and rumors == total
    return res


# ---------------------------------------------------------------------- temel veriler
@dataclass
class Fundamentals:
    base: str
    score: float | None = None          # 0-100, veri yoksa None
    coverage: float = 0.0
    tvl: float | None = None
    tvl_change_7d: float | None = None
    mcap: float | None = None
    mcap_tvl: float | None = None
    dev_commits_4w: float | None = None
    rank: int | None = None
    age_years: float | None = None
    whitepaper: str = ""
    categories: list = field(default_factory=list)
    notes: list = field(default_factory=list)


class FundamentalsProvider:
    """DeFiLlama + CoinGecko (ücretsiz, anahtarsız). Sonuçlar önbelleklenir (6 saat)."""

    def __init__(self, session=None, use_defillama: bool = True, use_coingecko: bool = True, ttl_s: float = 21600):
        self.session = session or requests.Session()
        self.use_defillama, self.use_coingecko, self.ttl = use_defillama, use_coingecko, ttl_s
        self._llama: tuple[float, dict] | None = None
        self._cg: dict[str, tuple[float, dict | None]] = {}
        self.status: dict[str, str] = {}

    def _get(self, url, params=None):
        r = self.session.get(url, params=params, timeout=15, headers={"User-Agent": "KreatifBot/1.0"})
        if r.status_code != 200:
            raise requests.HTTPError(f"HTTP {r.status_code}")
        return r.json()

    def llama_by_symbol(self) -> dict:
        if not self.use_defillama:
            return {}
        if self._llama and time.time() - self._llama[0] < self.ttl:
            return self._llama[1]
        try:
            data = self._get(DEFILLAMA_PROTOCOLS)
            out = {}
            for p in data:
                sym = (p.get("symbol") or "").upper()
                if not sym or sym == "-":
                    continue
                tvl = p.get("tvl") or 0
                if sym not in out or tvl > (out[sym].get("tvl") or 0):
                    out[sym] = p
            self._llama = (time.time(), out)
            self.status["DeFiLlama"] = "OK"
            return out
        except (requests.RequestException, ValueError, TypeError) as exc:
            self.status["DeFiLlama"] = f"ERİŞİLEMİYOR: {exc}"
            return {}

    def coingecko(self, base: str) -> dict | None:
        if not self.use_coingecko:
            return None
        cached = self._cg.get(base)
        if cached and time.time() - cached[0] < self.ttl:
            return cached[1]
        try:
            found = self._get(COINGECKO_SEARCH, {"query": base}).get("coins", [])
            cands = [c for c in found if (c.get("symbol") or "").upper() == base]
            cands.sort(key=lambda c: c.get("market_cap_rank") or 10**9)
            if not cands:
                self._cg[base] = (time.time(), None)
                return None
            coin = self._get(COINGECKO_COIN.format(id=cands[0]["id"]),
                             {"localization": "false", "tickers": "false", "market_data": "true",
                              "community_data": "false", "developer_data": "true", "sparkline": "false"})
            self._cg[base] = (time.time(), coin)
            self.status["CoinGecko"] = "OK"
            return coin
        except (requests.RequestException, ValueError, TypeError, KeyError) as exc:
            self.status["CoinGecko"] = f"ERİŞİLEMİYOR: {exc}"
            return None

    def get(self, base: str) -> Fundamentals:
        fd = Fundamentals(base)
        parts = []
        p = self.llama_by_symbol().get(base)
        if p:
            fd.tvl = float(p.get("tvl") or 0)
            fd.tvl_change_7d = p.get("change_7d")
            fd.mcap = p.get("mcap") or None
            if fd.tvl and fd.tvl > 0:
                parts.append(min(1.0, max(0.0, (math.log10(fd.tvl) - 6) / 3)))  # 1M→0, 1B→1
                if fd.tvl_change_7d is not None:
                    parts.append(min(1.0, max(0.0, 0.5 + fd.tvl_change_7d / 40)))
                if fd.mcap:
                    fd.mcap_tvl = fd.mcap / fd.tvl
                    parts.append(1.0 if fd.mcap_tvl < 1 else 0.7 if fd.mcap_tvl < 3 else 0.4 if fd.mcap_tvl < 10
                                 else 0.2)
            fd.categories.append(p.get("category", ""))
        else:
            fd.notes.append("DeFiLlama: TVL verisi yok (DeFi protokolü olmayabilir)")
        cg = self.coingecko(base)
        if cg:
            dev = cg.get("developer_data") or {}
            commits = dev.get("commit_count_4_weeks")
            if commits is not None:
                fd.dev_commits_4w = float(commits)
                parts.append(min(1.0, float(commits) / 150))
            fd.rank = cg.get("market_cap_rank")
            if fd.rank:
                parts.append(1.0 if fd.rank <= 100 else 0.7 if fd.rank <= 300 else 0.4 if fd.rank <= 1000 else 0.2)
            gd = cg.get("genesis_date")
            if gd:
                try:
                    fd.age_years = (datetime.now(timezone.utc) - datetime.fromisoformat(gd).replace(
                        tzinfo=timezone.utc)).days / 365
                    parts.append(min(1.0, fd.age_years / 3))
                except ValueError:
                    pass
            wp = ((cg.get("links") or {}).get("whitepaper")) or ""
            fd.whitepaper = wp if isinstance(wp, str) else ""
            fd.categories += [c for c in (cg.get("categories") or []) if c]
            if fd.mcap is None:
                fd.mcap = ((cg.get("market_data") or {}).get("market_cap") or {}).get("usd")
        else:
            fd.notes.append("CoinGecko: proje verisi alınamadı")
        fd.coverage = len(parts) / 7
        fd.score = round(100 * float(np.mean(parts)), 1) if parts else None
        fd.notes.append("Whitepaper içeriği otomatik değerlendirilmez; işlevsellik kullanım verileriyle (TVL, "
                        "geliştirici aktivitesi, yaş) ölçülür.")
        return fd


# ---------------------------------------------------------------------- teknik onay
@dataclass
class TechnicalCheck:
    ok: bool = False
    score: float | None = None
    regime: str = "UNKNOWN"
    price: float = float("nan")
    move_since_news_pct: float | None = None
    quote_volume_24h: float = 0.0
    rvol_24h: float | None = None
    atr: float = float("nan")
    notes: list = field(default_factory=list)


def technical_check(df_1h: pd.DataFrame, df_4h: pd.DataFrame | None, news_time: str, cfg: CatalystConfig
                    ) -> TechnicalCheck:
    from .features import compute_features, htf_bias
    from .regime import classify_regime
    tc = TechnicalCheck()
    if df_1h is None or len(df_1h) < 120:
        tc.notes.append("Yetersiz 1s veri")
        return tc
    f = compute_features(df_1h, "1h")
    reg = classify_regime(f)
    row = f.iloc[-1]
    tc.regime = str(reg["regime"].iat[-1])
    tc.price = float(row["close"])
    tc.atr = float(row["atr"])
    qv = df_1h["quote_volume"] if "quote_volume" in df_1h else df_1h["volume"] * df_1h["close"]
    tc.quote_volume_24h = float(qv.tail(24).sum())
    prev = float(qv.iloc[-48:-24].sum()) if len(qv) >= 48 else 0
    tc.rvol_24h = tc.quote_volume_24h / prev if prev > 0 else None
    if news_time:
        t = pd.Timestamp(news_time)
        t = t if t.tzinfo else t.tz_localize("UTC")
        before = df_1h[pd.to_datetime(df_1h["open_time"], utc=True) <= t]
        if not before.empty:
            tc.move_since_news_pct = (tc.price / float(before["close"].iloc[-1]) - 1) * 100
    bias4 = None
    if df_4h is not None and len(df_4h) >= 60:
        bias4 = float(htf_bias(compute_features(df_4h, "4h")).iat[-1])
    parts = [
        1.0 if row["close"] > row["ema50"] else 0.0,
        1.0 if row["ema20"] > row["ema50"] else 0.3,
        min(1.0, (tc.rvol_24h or 1.0) / 2),
        0.0 if tc.regime in ("BEAR", "STRONG_BEAR", "PANIC") else 1.0,
    ]
    if bias4 is not None and np.isfinite(bias4):
        parts.append((bias4 + 1) / 2)
    tc.score = round(100 * float(np.mean(parts)), 1)
    reasons = []
    if tc.regime in ("BEAR", "STRONG_BEAR", "PANIC"):
        reasons.append(f"Rejim {tc.regime}: düşen trendde haberle alım yapılmaz")
    if row["close"] <= row["ema50"]:
        reasons.append("Fiyat 1s EMA50 altında")
    if bias4 is not None and bias4 < -0.25:
        reasons.append(f"4s trend aşağı ({bias4:+.2f})")
    if tc.move_since_news_pct is not None and tc.move_since_news_pct > cfg.max_chase_pct:
        reasons.append(f"Haberden beri %{tc.move_since_news_pct:.1f} yükselmiş (fiyatlanmış, kovalanmaz)")
    if tc.quote_volume_24h < cfg.min_quote_volume_24h:
        reasons.append(f"24s hacim düşük ({tc.quote_volume_24h:,.0f} USDT)")
    tc.notes = reasons
    tc.ok = not reasons
    return tc


# ---------------------------------------------------------------------- öngörü
@dataclass
class Insight:
    insight_id: str
    symbol: str
    created_at: str
    signal: str                  # AL | İZLE | YOK
    potential: float
    catalyst: CatalystResult
    fundamentals: Fundamentals
    technical: TechnicalCheck
    entry: float = float("nan")
    stop: float = float("nan")
    targets: list = field(default_factory=list)
    reasons: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def explain(self) -> str:
        c, fd, tc = self.catalyst, self.fundamentals, self.technical
        icon = {"AL": "🟢", "İZLE": "💡"}.get(self.signal, "⚪")
        lines = [f"{icon} {self.symbol} — {self.signal}  (potansiyel {self.potential:.0f}/100)", "",
                 f"Katalizör: {c.score:.0f}/100 — " + (", ".join(CATALYST_TR.get(t, t) for t in c.types) or "yok"),
                 f"  Kaynaklar: {', '.join(c.sources) or '-'}" + ("  (yalnızca söylenti)" if c.rumor_only else "")]
        lines += [f"  • {h}" for h in c.headlines[:5]]
        if c.negative:
            lines.append(f"Olumsuz baskı: {c.negative:.0f}/100")
        lines.append(f"Temel skor: {'Veri yok' if fd.score is None else f'{fd.score:.0f}/100'} "
                     f"(kapsam %{fd.coverage * 100:.0f})")
        if fd.tvl:
            lines.append(f"  TVL {fd.tvl:,.0f} USD, 7g değişim {fd.tvl_change_7d if fd.tvl_change_7d is not None else '-'}%"
                         + (f", piyasa değeri/TVL {fd.mcap_tvl:.2f}" if fd.mcap_tvl else ""))
        if fd.dev_commits_4w is not None:
            lines.append(f"  Geliştirici: son 4 haftada {fd.dev_commits_4w:.0f} commit")
        if fd.rank:
            lines.append(f"  Piyasa değeri sırası #{fd.rank}" + (f", yaş {fd.age_years:.1f} yıl" if fd.age_years else ""))
        if fd.whitepaper:
            lines.append(f"  Whitepaper: {fd.whitepaper}")
        lines.append(f"Teknik: {'Veri yok' if tc.score is None else f'{tc.score:.0f}/100'}, rejim {tr(tc.regime)}, "
                     f"24s hacim {tc.quote_volume_24h:,.0f} USDT"
                     + (f", haberden beri %{tc.move_since_news_pct:+.1f}" if tc.move_since_news_pct is not None else ""))
        if self.signal == "AL":
            lines += ["", f"Giriş ~{self.entry:.6g}, stop {self.stop:.6g}, hedefler "
                      + ", ".join(f"{t:.6g}" for t in self.targets)]
        if self.reasons:
            lines += ["", "Gerekçe:"] + [f"  • {r}" for r in self.reasons]
        if self.warnings:
            lines += ["Uyarılar:"] + [f"  ⚠ {w}" for w in self.warnings]
        lines.append("Not: Haber tabanlı öngörüdür; kâr garantisi yoktur. Gerçek isabeti 'Öngörü performansı'nda ölçülür.")
        return "\n".join(lines)


def build_insight(symbol: str, base: str, items: list[NewsItem], fundamentals: Fundamentals, df_1h, df_4h,
                  cfg: CatalystConfig) -> Insight:
    cat = score_catalysts(base, items, cfg)
    tc = technical_check(df_1h, df_4h, cat.first_seen, cfg)
    w = {"c": cfg.weight_catalyst, "f": cfg.weight_fundamental, "t": cfg.weight_technical}
    num, den = w["c"] * cat.score, w["c"]
    if fundamentals.score is not None:
        num += w["f"] * fundamentals.score
        den += w["f"]
    if tc.score is not None:
        num += w["t"] * tc.score
        den += w["t"]
    potential = num / den if den else 0.0
    potential = max(0.0, potential - 0.5 * cat.negative)
    ins = Insight(uuid.uuid4().hex[:16], symbol, datetime.now(timezone.utc).isoformat(), "YOK",
                  round(potential, 1), cat, fundamentals, tc)
    if cat.types:
        ins.reasons.append("Katalizör: " + ", ".join(CATALYST_TR.get(t, t) for t in cat.types))
    if cat.negative >= 30:
        ins.warnings.append(f"Olumsuz haber baskısı {cat.negative:.0f}/100 (AL verilmez)")
    if cat.rumor_only:
        ins.warnings.append("Haberler yalnızca söylenti/iddia içeriyor (güvenilirlik düşük)")
    if fundamentals.score is None:
        ins.warnings.append("Temel veri yok: potansiyel yalnızca haber + teknik ile hesaplandı")
    ins.warnings += tc.notes
    can_buy = (cat.score >= cfg.min_catalyst and potential >= cfg.min_potential and tc.ok and cat.negative < 30
               and not cat.rumor_only and np.isfinite(tc.atr) and tc.atr > 0)
    if can_buy:
        ins.signal = "AL"
        ins.entry = tc.price
        ins.stop = tc.price - cfg.stop_atr_mult * tc.atr
        risk = ins.entry - ins.stop
        ins.targets = [ins.entry + r * risk for r in cfg.tp_levels_r]
        ins.reasons.append("Güçlü katalizör + teknik onay + yeterli likidite, fiyat henüz haberi fiyatlamamış")
    elif potential >= cfg.watch_potential or cat.score >= 40:
        ins.signal = "İZLE"
        ins.reasons.append("Potansiyel var ancak AL koşulları tam değil (teknik onay / eşik / risk)")
    return ins


def candidate_bases(items: list[NewsItem], cfg: CatalystConfig, tradable_bases: set) -> list[str]:
    """Son dönemde olumlu katalizör içeren ve Binance'te işlem gören coinler."""
    out = []
    for it in items:
        if it.age_hours > cfg.lookback_hours:
            continue
        if not any(w > 0 for _, w in detect_catalysts(f"{it.title} {it.summary}")):
            continue
        for b in it.symbols:
            if b in tradable_bases and b not in out and b not in ("USDT", "USDC", "FDUSD"):
                out.append(b)
    return out


class InsightEngine:
    def __init__(self, cfg: CatalystConfig, market_client, fundamentals: FundamentalsProvider | None = None,
                 quote: str = "USDT"):
        self.cfg, self.client, self.quote = cfg, market_client, quote
        self.fund = fundamentals or FundamentalsProvider(use_defillama=cfg.use_defillama,
                                                         use_coingecko=cfg.use_coingecko)

    def scan(self, items: list[NewsItem], tradable_bases: set, limit: int = 15) -> list[Insight]:
        from .market_data import closed_only
        out = []
        for base in candidate_bases(items, self.cfg, tradable_bases)[:limit]:
            symbol = f"{base}{self.quote}"
            try:
                df1 = closed_only(self.client.klines(symbol, "1h", limit=300))
                df4 = closed_only(self.client.klines(symbol, "4h", limit=200))
            except Exception:  # noqa: BLE001 - tek coin hatası taramayı durdurmasın
                df1 = df4 = None
            out.append(build_insight(symbol, base, items, self.fund.get(base), df1, df4, self.cfg))
        out.sort(key=lambda x: ({"AL": 0, "İZLE": 1}.get(x.signal, 2), -x.potential))
        return out


# ---------------------------------------------------------------------- öngörü kaydı ve performans
class InsightStore:
    SCHEMA = """CREATE TABLE IF NOT EXISTS insights (insight_id TEXT PRIMARY KEY, symbol TEXT, created_at TEXT,
        signal TEXT, potential REAL, catalyst REAL, fundamental REAL, technical REAL, types TEXT, price REAL,
        ret_4h REAL, ret_24h REAL, ret_72h REAL, max_up REAL, max_down REAL, explanation TEXT);"""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else data_dir() / "insights.sqlite3"
        self._lock = threading.Lock()
        with self._conn() as c:
            c.executescript(self.SCHEMA)

    def _conn(self):
        return sqlite3.connect(self.path, timeout=10)

    def save(self, ins: Insight, dedupe_hours: float = 12.0) -> bool:
        """Aynı sembol ve sinyal son X saatte kaydedildiyse tekrar kaydetmez."""
        with self._lock, self._conn() as c:
            last = c.execute("SELECT created_at FROM insights WHERE symbol = ? AND signal = ? ORDER BY created_at "
                             "DESC LIMIT 1", (ins.symbol, ins.signal)).fetchone()
            if last:
                age = (datetime.now(timezone.utc) - datetime.fromisoformat(last[0])).total_seconds() / 3600
                if age < dedupe_hours:
                    return False
            cur = c.execute("INSERT OR IGNORE INTO insights VALUES (?,?,?,?,?,?,?,?,?,?,NULL,NULL,NULL,NULL,NULL,?)",
                      (ins.insight_id, ins.symbol, ins.created_at, ins.signal, ins.potential, ins.catalyst.score,
                       ins.fundamentals.score, ins.technical.score, json.dumps(ins.catalyst.types),
                       ins.technical.price if np.isfinite(ins.technical.price) else None, ins.explain()))
            return bool(cur.rowcount)

    def frame(self) -> pd.DataFrame:
        with self._lock, self._conn() as c:
            return pd.read_sql_query("SELECT * FROM insights ORDER BY created_at DESC", c)

    def update_outcomes(self, client) -> int:
        """4s/24s/72s sonraki gerçek getirileri Binance 1s mumlarından hesaplar."""
        df = self.frame()
        updated = 0
        now = pd.Timestamp.now(tz="UTC")
        for _, r in df[df["ret_72h"].isna() & df["price"].notna()].iterrows():
            t0 = pd.Timestamp(r["created_at"])
            t0 = t0 if t0.tzinfo else t0.tz_localize("UTC")
            hours = (now - t0).total_seconds() / 3600
            if hours < 4:
                continue
            try:
                k = client.klines(r["symbol"], "1h", limit=80, start_time=int(t0.timestamp() * 1000))
            except Exception:  # noqa: BLE001
                continue
            if k.empty:
                continue
            p0 = float(r["price"])
            closes = k["close"].to_numpy(dtype=float)
            vals = {}
            for h, col in ((4, "ret_4h"), (24, "ret_24h"), (72, "ret_72h")):
                if hours >= h and len(closes) >= h:
                    vals[col] = (closes[h - 1] / p0 - 1) * 100
            window = k.iloc[:min(len(k), 72)]
            vals["max_up"] = (float(window["high"].max()) / p0 - 1) * 100
            vals["max_down"] = (float(window["low"].min()) / p0 - 1) * 100
            sets = ", ".join(f"{k2} = ?" for k2 in vals)
            with self._lock, self._conn() as c:
                c.execute(f"UPDATE insights SET {sets} WHERE insight_id = ?", [*vals.values(), r["insight_id"]])
            updated += 1
        return updated

    def performance(self) -> pd.DataFrame:
        """Sinyal ve katalizör türüne göre gerçekleşen getiriler (yalnızca ölçülmüş öngörüler)."""
        df = self.frame()
        df = df[df["ret_24h"].notna()]
        if df.empty:
            return pd.DataFrame()
        rows = []
        for (sig,), g in df.groupby(["signal"]):
            rows.append(_perf_row(f"Sinyal: {sig}", g))
        exploded = df.assign(t=df["types"].map(lambda s: json.loads(s or "[]"))).explode("t")
        for t, g in exploded.groupby("t"):
            if isinstance(t, str):
                rows.append(_perf_row(f"Katalizör: {CATALYST_TR.get(t, t)}", g))
        return pd.DataFrame(rows)


def _perf_row(name: str, g: pd.DataFrame) -> dict:
    return {"grup": name, "n": len(g), "ort_4s_%": g["ret_4h"].mean(), "ort_24s_%": g["ret_24h"].mean(),
            "ort_72s_%": g["ret_72h"].mean(), "24s_isabet_%": float((g["ret_24h"] > 0).mean() * 100),
            "ort_maks_yükseliş_%": g["max_up"].mean(), "ort_maks_düşüş_%": g["max_down"].mean(),
            "not": "n<20: güvenilir değil" if len(g) < 20 else ""}


def insight_to_dict(ins: Insight) -> dict:
    d = asdict(ins)
    return d
