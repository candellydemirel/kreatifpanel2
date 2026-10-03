"""Kripto haber ve Binance listeleme motoru.

Kaynaklar:
- Binance duyuruları (CMS, herkese açık ama resmi belgelenmemiş uç nokta; biçim değişirse
  kaynak 'erişilemiyor' olarak işaretlenir): yeni listeleme, delist, futures listeleme.
- Binance exchangeInfo farkı (resmi API): yeni açılan / durumu değişen USDT çiftleri.
- RSS: CoinDesk, Cointelegraph, Decrypt (anahtar gerekmez).
- CryptoPanic (isteğe bağlı, ücretsiz API anahtarı ile).

Haberler deterministik anahtar kelime kurallarıyla sınıflandırılır. Haber tek başına
işlem açtırmaz: olumsuz/yüksek riskli haber yeni LONG'ları engeller, delist duyurusu
açık pozisyonu kapattırır, olumlu haber skoru yükseltmez (hype kovalanmaz).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import requests

from ..config import data_dir

logger = logging.getLogger("kreatifbot.intel.news")

BINANCE_CMS_URL = "https://www.binance.com/bapi/composite/v1/public/cms/article/list/query"
BINANCE_ARTICLE_URL = "https://www.binance.com/en/support/announcement/{code}"
BINANCE_CATALOGS = {48: "Yeni listeleme", 161: "Delist", 49: "Binance haberleri"}

RSS_FEEDS = {
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "Cointelegraph": "https://cointelegraph.com/rss",
    "Decrypt": "https://decrypt.co/feed",
}
CRYPTOPANIC_URL = "https://cryptopanic.com/api/v1/posts/"

# Kategori → (anahtar kelimeler, önem 0-3, yön -1/0/+1)
CATEGORIES = [
    ("DELISTING", [r"\bdelist", r"will (cease|end) trading", r"remove .* trading pairs?", r"işlemden kald",
                   r"monitoring tag"], 3, -1),
    ("HACK", [r"\bhack(ed|er|s)?\b", r"\bexploit", r"drain(ed)?", r"stolen", r"breach", r"rug ?pull",
              r"saldırı", r"çalın"], 3, -1),
    ("REGULATION_NEG", [r"\bsec (sues|charges|lawsuit)", r"lawsuit", r"\bban(s|ned)?\b", r"crackdown",
                        r"investigation", r"\bsued\b", r"yasak", r"dava"], 2, -1),
    ("FUTURES_LISTING", [r"futures will launch", r"perpetual contract", r"usdⓢ-m .*perpetual",
                         r"launch .*usd.?-margined"], 1, 1),
    ("LISTING", [r"binance will list", r"will list\b", r"\blisting\b", r"adds? .* (spot )?trading pairs?",
                 r"listeleyecek", r"listeleme"], 2, 1),
    ("LAUNCHPOOL", [r"launchpool", r"launchpad", r"hodler airdrop", r"megadrop"], 1, 1),
    ("PARTNERSHIP", [r"partnership", r"integrat", r"etf (approval|approved|inflow)", r"adopt", r"ortaklık"], 1, 1),
    ("MACRO", [r"\bfed\b", r"interest rate", r"\bcpi\b", r"inflation", r"faiz", r"enflasyon"], 1, 0),
]
POS_WORDS = [r"surge", r"soar", r"rall(y|ies)", r"record high", r"approv", r"bullish", r"gain", r"yüksel",
             r"rekor"]
NEG_WORDS = [r"plunge", r"crash", r"slump", r"bearish", r"liquidat", r"outflow", r"fraud", r"collapse",
             r"düş", r"çöküş"]

NAME_TO_TICKER = {
    "bitcoin": "BTC", "ethereum": "ETH", "ether": "ETH", "solana": "SOL", "ripple": "XRP", "xrp": "XRP",
    "cardano": "ADA", "dogecoin": "DOGE", "bnb": "BNB", "avalanche": "AVAX", "chainlink": "LINK",
    "polkadot": "DOT", "litecoin": "LTC", "tron": "TRX", "toncoin": "TON", "shiba inu": "SHIB",
    "polygon": "POL", "near protocol": "NEAR", "cosmos": "ATOM", "arbitrum": "ARB", "optimism": "OP",
    "aptos": "APT", "sui": "SUI", "pepe": "PEPE", "uniswap": "UNI", "aave": "AAVE", "filecoin": "FIL",
}
TICKER_STOP = {"USD", "USDT", "THE", "AND", "FOR", "NEW", "CEO", "SEC", "ETF", "NFT", "API", "ALL", "ONE", "TOP",
               "NOW", "OUT", "BIG", "CAN", "WILL", "LIST", "BUY", "GAS", "KEY", "TAG", "FUN", "EUR", "TRY",
               "AI", "IT", "ON", "UP", "GO", "AT", "OR", "IN", "AN", "BY"}


@dataclass
class NewsItem:
    item_id: str
    source: str
    title: str
    url: str = ""
    published_at: str = ""
    symbols: list = field(default_factory=list)
    category: str = "GENERAL"
    severity: int = 0
    sentiment: float = 0.0
    summary: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def age_hours(self) -> float:
        try:
            t = datetime.fromisoformat(self.published_at)
        except ValueError:
            return 1e9
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - t).total_seconds() / 3600


def _id(*parts) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def classify(text: str) -> tuple[str, int, float]:
    """(kategori, önem, duyarlılık -1..1) — deterministik kurallar."""
    t = text.lower()
    category, severity, direction = "GENERAL", 0, 0
    for name, patterns, sev, d in CATEGORIES:
        if any(re.search(p, t) for p in patterns):
            category, severity, direction = name, sev, d
            break
    pos = sum(bool(re.search(p, t)) for p in POS_WORDS)
    neg = sum(bool(re.search(p, t)) for p in NEG_WORDS)
    tone = (pos - neg) / max(1, pos + neg)
    sentiment = max(-1.0, min(1.0, 0.6 * direction + 0.4 * tone)) if (direction or pos or neg) else 0.0
    return category, severity, round(sentiment, 3)


def extract_symbols(text: str, known_bases: set | None = None) -> list[str]:
    """Başlıktaki coin sembolleri. Parantez içi semboller (ör. 'XYZ (XYZ)') en güvenilir olandır."""
    found: list[str] = []
    for m in re.findall(r"\(([A-Z0-9]{2,12})\)", text):
        if m not in TICKER_STOP:
            found.append(m)
    low = text.lower()
    for name, tick in NAME_TO_TICKER.items():
        if re.search(rf"\b{re.escape(name)}\b", low):
            found.append(tick)
    if known_bases:
        for tok in re.findall(r"\b[A-Z][A-Z0-9]{1,9}\b", text):
            if tok in known_bases and tok not in TICKER_STOP:
                found.append(tok)
    out = []
    for s in found:
        if s not in out:
            out.append(s)
    return out


# ---------------------------------------------------------------------- kaynaklar
class NewsSourceError(Exception):
    pass


def _get(session, url, params=None, timeout=10):
    try:
        r = session.get(url, params=params, timeout=timeout, headers={"User-Agent": "KreatifBot/1.0"})
    except requests.RequestException as exc:
        raise NewsSourceError(f"bağlantı hatası: {exc.__class__.__name__}") from exc
    if r.status_code != 200:
        raise NewsSourceError(f"HTTP {r.status_code}")
    return r


def fetch_binance_announcements(session, catalog_id: int, page_size: int = 20,
                                known_bases: set | None = None) -> list[NewsItem]:
    r = _get(session, BINANCE_CMS_URL, {"type": 1, "catalogId": catalog_id, "pageNo": 1, "pageSize": page_size})
    try:
        data = r.json()
        catalogs = data["data"]["catalogs"]
    except (ValueError, KeyError, TypeError) as exc:
        raise NewsSourceError("Binance duyuru biçimi beklenenden farklı (uç nokta değişmiş olabilir)") from exc
    items = []
    for cat in catalogs:
        for a in cat.get("articles", []):
            title = a.get("title", "")
            ts = a.get("releaseDate") or a.get("publishDate")
            pub = datetime.fromtimestamp(int(ts) / 1000, timezone.utc).isoformat() if ts else ""
            category, sev, sent = classify(title)
            if catalog_id == 161 and category not in ("DELISTING",):
                category, sev, sent = "DELISTING", 3, -0.8
            items.append(NewsItem(
                item_id=_id("binance", a.get("id") or a.get("code") or title), source="Binance duyuru",
                title=title, url=BINANCE_ARTICLE_URL.format(code=a.get("code", "")), published_at=pub,
                symbols=extract_symbols(title, known_bases), category=category, severity=sev, sentiment=sent))
    return items


def parse_rss(xml_text: str, source: str, known_bases: set | None = None) -> list[NewsItem]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise NewsSourceError(f"{source}: RSS çözümlenemedi") from exc
    items = []
    entries = root.findall(".//item")
    atom = "{http://www.w3.org/2005/Atom}"
    if not entries:
        entries = root.findall(f".//{atom}entry")
    for e in entries:
        def txt(tag):
            el = e.find(tag)
            if el is None:
                el = e.find(atom + tag)
            return (el.text or "").strip() if el is not None and el.text else ""
        title = txt("title")
        if not title:
            continue
        link = txt("link")
        if not link:
            el = e.find(atom + "link")
            link = el.get("href", "") if el is not None else ""
        pub_raw = txt("pubDate") or txt("published") or txt("updated")
        try:
            pub = parsedate_to_datetime(pub_raw).astimezone(timezone.utc).isoformat()
        except (TypeError, ValueError):
            try:
                pub = datetime.fromisoformat(pub_raw.replace("Z", "+00:00")).astimezone(timezone.utc).isoformat()
            except ValueError:
                pub = ""
        desc = re.sub(r"<[^>]+>", " ", txt("description"))[:400]
        category, sev, sent = classify(f"{title} {desc}")
        items.append(NewsItem(item_id=_id(source, txt("guid") or link or title), source=source, title=title,
                              url=link, published_at=pub, symbols=extract_symbols(title, known_bases),
                              category=category, severity=sev, sentiment=sent, summary=desc.strip()))
    return items


def fetch_rss(session, source: str, url: str, known_bases: set | None = None) -> list[NewsItem]:
    return parse_rss(_get(session, url).text, source, known_bases)


def fetch_cryptopanic(session, token: str, known_bases: set | None = None) -> list[NewsItem]:
    r = _get(session, CRYPTOPANIC_URL, {"auth_token": token, "public": "true"})
    try:
        results = r.json()["results"]
    except (ValueError, KeyError, TypeError) as exc:
        raise NewsSourceError("CryptoPanic biçimi beklenenden farklı") from exc
    items = []
    for p in results:
        title = p.get("title", "")
        cats, sev, sent = classify(title)
        votes = p.get("votes") or {}
        vp, vn = votes.get("positive", 0) or 0, votes.get("negative", 0) or 0
        if vp + vn >= 5:
            sent = round(0.5 * sent + 0.5 * (vp - vn) / (vp + vn), 3)
        syms = [c.get("code") for c in p.get("currencies") or [] if c.get("code")] or extract_symbols(title, known_bases)
        items.append(NewsItem(item_id=_id("cryptopanic", p.get("id") or title), source="CryptoPanic", title=title,
                              url=p.get("url", ""), published_at=p.get("published_at", ""), symbols=syms,
                              category=cats, severity=sev, sentiment=sent))
    return items


# ---------------------------------------------------------------------- listeleme izleyici (resmi API)
@dataclass
class ListingEvent:
    symbol: str
    base: str
    detected_at: str
    kind: str              # NEW_SYMBOL | NOW_TRADING | HALTED | ANNOUNCED
    status: str
    source: str
    open_time: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def diff_exchange_info(previous: dict[str, str], symbols: list[dict], quote: str = "USDT") -> tuple[dict, list]:
    """previous: sembol → durum. Yeni semboller ve durum değişiklikleri (ör. PRE_TRADING → TRADING)."""
    now = datetime.now(timezone.utc).isoformat()
    current, events = {}, []
    for s in symbols:
        if s.get("quoteAsset") != quote:
            continue
        sym, status = s["symbol"], s.get("status", "")
        current[sym] = status
        if not previous:
            continue
        old = previous.get(sym)
        if old is None:
            events.append(ListingEvent(sym, s.get("baseAsset", ""), now, "NEW_SYMBOL", status, "exchangeInfo"))
        elif old != status:
            kind = "NOW_TRADING" if status == "TRADING" else "HALTED" if status in ("BREAK", "HALT") else "STATUS"
            events.append(ListingEvent(sym, s.get("baseAsset", ""), now, kind, status, "exchangeInfo"))
    return current, events


# ---------------------------------------------------------------------- servis
class NewsStore:
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS news (item_id TEXT PRIMARY KEY, source TEXT, title TEXT, url TEXT,
        published_at TEXT, symbols TEXT, category TEXT, severity INTEGER, sentiment REAL, summary TEXT,
        fetched_at TEXT);
    CREATE INDEX IF NOT EXISTS ix_news_pub ON news(published_at);
    CREATE TABLE IF NOT EXISTS listings (symbol TEXT, kind TEXT, detected_at TEXT, status TEXT, source TEXT,
        base TEXT, open_time TEXT, PRIMARY KEY(symbol, kind));
    CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT);
    """

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else data_dir() / "news.sqlite3"
        self._lock = threading.Lock()
        with self._conn() as c:
            c.executescript(self.SCHEMA)

    def _conn(self):
        return sqlite3.connect(self.path, timeout=10)

    def add(self, items: list[NewsItem]) -> list[NewsItem]:
        """Yeni (daha önce görülmemiş) haberleri kaydeder ve döndürür."""
        new = []
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._conn() as c:
            for it in items:
                cur = c.execute("INSERT OR IGNORE INTO news VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                                (it.item_id, it.source, it.title, it.url, it.published_at, json.dumps(it.symbols),
                                 it.category, it.severity, it.sentiment, it.summary, now))
                if cur.rowcount:
                    new.append(it)
        return new

    def recent(self, hours: float = 48, symbol_base: str | None = None, limit: int = 500) -> list[NewsItem]:
        with self._lock, self._conn() as c:
            rows = c.execute("SELECT item_id, source, title, url, published_at, symbols, category, severity, "
                             "sentiment, summary FROM news ORDER BY published_at DESC LIMIT ?", (limit,)).fetchall()
        out = []
        for r in rows:
            it = NewsItem(r[0], r[1], r[2], r[3], r[4], json.loads(r[5] or "[]"), r[6], r[7], r[8], r[9] or "")
            if it.age_hours > hours:
                continue
            if symbol_base and symbol_base not in it.symbols:
                continue
            out.append(it)
        return out

    def add_listing(self, ev: ListingEvent) -> bool:
        with self._lock, self._conn() as c:
            cur = c.execute("INSERT OR IGNORE INTO listings VALUES (?,?,?,?,?,?,?)",
                            (ev.symbol, ev.kind, ev.detected_at, ev.status, ev.source, ev.base, ev.open_time))
            return bool(cur.rowcount)

    def listings(self, limit: int = 200) -> list[ListingEvent]:
        with self._lock, self._conn() as c:
            rows = c.execute("SELECT symbol, base, detected_at, kind, status, source, open_time FROM listings "
                             "ORDER BY detected_at DESC LIMIT ?", (limit,)).fetchall()
        return [ListingEvent(*r) for r in rows]

    def get(self, key: str, default=None):
        with self._lock, self._conn() as c:
            r = c.execute("SELECT v FROM kv WHERE k = ?", (key,)).fetchone()
        return json.loads(r[0]) if r else default

    def put(self, key: str, value) -> None:
        with self._lock, self._conn() as c:
            c.execute("INSERT OR REPLACE INTO kv VALUES (?, ?)", (key, json.dumps(value)))


@dataclass
class NewsContext:
    """Bir sembol için karar motoruna verilen haber bağlamı."""
    symbol: str
    items: list = field(default_factory=list)
    block_long: bool = False
    block_short: bool = False
    force_exit: bool = False
    sentiment: float = 0.0
    reasons: list = field(default_factory=list)


class NewsMonitor:
    def __init__(self, store: NewsStore | None = None, session=None, rss_feeds: dict | None = None,
                 use_binance: bool = True, cryptopanic_token: str = "", quote: str = "USDT",
                 block_hours: float = 24.0, min_severity_block: int = 2):
        self.store = store or NewsStore()
        self.session = session or requests.Session()
        self.rss_feeds = RSS_FEEDS if rss_feeds is None else rss_feeds
        self.use_binance = use_binance
        self.cryptopanic_token = cryptopanic_token
        self.quote = quote
        self.block_hours = block_hours
        self.min_severity_block = min_severity_block
        self.source_status: dict[str, str] = {}
        self.known_bases: set = set()
        self.last_poll = 0.0

    def poll(self, exchange_symbols: list[dict] | None = None) -> tuple[list[NewsItem], list[ListingEvent]]:
        """Tüm kaynakları tarar. Hata veren kaynak işareti konur, diğerleri devam eder."""
        self.last_poll = time.time()
        if exchange_symbols:
            self.known_bases = {s.get("baseAsset") for s in exchange_symbols if s.get("quoteAsset") == self.quote}
        items: list[NewsItem] = []
        if self.use_binance:
            for cid, name in BINANCE_CATALOGS.items():
                key = f"Binance: {name}"
                try:
                    items += fetch_binance_announcements(self.session, cid, known_bases=self.known_bases)
                    self.source_status[key] = "OK"
                except NewsSourceError as exc:
                    self.source_status[key] = f"ERİŞİLEMİYOR: {exc}"
        for name, url in self.rss_feeds.items():
            try:
                items += fetch_rss(self.session, name, url, self.known_bases)
                self.source_status[name] = "OK"
            except NewsSourceError as exc:
                self.source_status[name] = f"ERİŞİLEMİYOR: {exc}"
        if self.cryptopanic_token:
            try:
                items += fetch_cryptopanic(self.session, self.cryptopanic_token, self.known_bases)
                self.source_status["CryptoPanic"] = "OK"
            except NewsSourceError as exc:
                self.source_status["CryptoPanic"] = f"ERİŞİLEMİYOR: {exc}"
        new_items = self.store.add(items)
        events: list[ListingEvent] = []
        if exchange_symbols:
            prev = self.store.get("exchange_status", {})
            current, events = diff_exchange_info(prev, exchange_symbols, self.quote)
            self.store.put("exchange_status", current)
            events = [e for e in events if self.store.add_listing(e)]
        for it in new_items:
            if it.category == "LISTING" and it.source == "Binance duyuru":
                for base in it.symbols:
                    ev = ListingEvent(f"{base}{self.quote}", base, it.published_at or datetime.now(timezone.utc)
                                      .isoformat(), "ANNOUNCED", "", "Binance duyuru")
                    if self.store.add_listing(ev):
                        events.append(ev)
        return new_items, events

    def context(self, symbol: str) -> NewsContext:
        base = symbol[:-len(self.quote)] if symbol.endswith(self.quote) else symbol
        ctx = NewsContext(symbol)
        items = self.store.recent(self.block_hours, base)
        ctx.items = items
        if items:
            ctx.sentiment = round(sum(i.sentiment for i in items) / len(items), 3)
        for it in items:
            if it.category == "DELISTING":
                ctx.block_long = True
                ctx.force_exit = True
                ctx.reasons.append(f"Delist duyurusu: {it.title}")
            elif it.sentiment < 0 and it.severity >= self.min_severity_block:
                ctx.block_long = True
                ctx.reasons.append(f"Olumsuz haber ({it.category}): {it.title}")
        return ctx
