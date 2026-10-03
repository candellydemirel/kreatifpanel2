"""Arayüzsüz (headless) KreatifBot: başka uygulamalara (ör. Jarvis sesli asistanı) gömmek için.

Masaüstü uygulamasındaki Zeka Motoru'nun aynısını Qt olmadan çalıştırır: aynı ayar dosyaları,
aynı strateji/risk ayarları, aynı öğrenilen risk çarpanları, aynı Telegram bildirimleri.

Güvenlik:
- Aynı bilgisayarda aynı anda yalnızca BİR bot motoru çalışabilir (masaüstü uygulaması veya
  gömülü bot). Aksi halde aynı hesapta çift emir açılabilirdi → EngineLock.
- Canlı (gerçek para) başlatma için `confirm_live=True` açıkça verilmelidir.

Kullanım:
    from kreatifbot.headless import HeadlessBot, analyze_symbol
    bot = HeadlessBot()
    print(bot.start(live=False))     # kağıt işlem
    print(bot.status_text())
    print(analyze_symbol("SOLUSDT"))
    bot.stop()
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime

from .config import Settings, data_dir, load_settings, save_settings
from .i18n import tr, tr_reason

logger = logging.getLogger("kreatifbot.headless")


# ---------------------------------------------------------------------- tek motor kilidi
def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class EngineLock:
    """Bilgisayar başına tek çalışan bot motoru (masaüstü uygulaması + gömülü kullanım ortak)."""

    def __init__(self, owner: str):
        self.owner = owner
        self.path = data_dir() / "engine.lock"
        self.held = False

    def holder(self) -> dict | None:
        try:
            info = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not _pid_alive(int(info.get("pid", 0))):
            return None
        if int(info.get("pid", 0)) == os.getpid() and info.get("owner") == self.owner and not self.held:
            return None  # aynı süreçte önceki kilidimiz
        return info

    def acquire(self) -> tuple[bool, str]:
        other = self.holder()
        if other is not None and not (other.get("pid") == os.getpid() and other.get("owner") == self.owner):
            return False, (f"Başka bir bot zaten çalışıyor ({other.get('owner')}, {other.get('started', '')[:16]}). "
                           "Aynı hesapta çift işlem olmaması için önce onu durdurun.")
        self.path.write_text(json.dumps({"pid": os.getpid(), "owner": self.owner,
                                         "started": datetime.now().isoformat()}), encoding="utf-8")
        self.held = True
        return True, ""

    def release(self):
        if not self.held:
            return
        try:
            info = json.loads(self.path.read_text(encoding="utf-8"))
            if info.get("pid") == os.getpid() and info.get("owner") == self.owner:
                self.path.unlink(missing_ok=True)
        except (OSError, ValueError):
            pass
        self.held = False


# ---------------------------------------------------------------------- tek seferlik analiz
def analyze_symbol(symbol: str, market: str | None = None, bars: int = 600) -> str:
    """Bir coin için Zeka Motoru kararı ve Türkçe açıklaması (gerçek Binance verisi, API anahtarı gerekmez)."""
    from .binance_client import BinanceClient
    from .intel.autopilot import load_learned
    from .intel.config import load_intel_config
    from .intel.data_quality import check_candles, check_orderbook
    from .intel.decision import DecisionEngine
    from .intel.features import compute_features
    from .intel.futures_client import BinanceFuturesClient
    from .intel.market_data import live_snapshot, load_history
    from .intel.orderflow import orderbook_features

    symbol = (symbol or "BTCUSDT").upper().replace("/", "").replace("-", "").replace("_", "")
    cfg = load_intel_config()
    if market:
        cfg.market = market
    if not symbol.endswith(cfg.quote_asset):
        symbol += cfg.quote_asset
    spot, fut = BinanceClient(testnet=False), BinanceFuturesClient(testnet=False)
    bundle = load_history(cfg, symbol, bars, spot, fut)
    if not bundle.ok:
        return f"{symbol}: veri alınamadı (VERİ YOK) — " + "; ".join(bundle.errors)[:300]
    client = spot if cfg.market == "SPOT" else fut
    try:
        rules = client.symbol_rules(symbol)
    except Exception:  # noqa: BLE001
        rules = None
    fs, raw = live_snapshot(symbol, spot, fut, want_spot=True,
                            want_futures=cfg.market != "SPOT" or cfg.use_futures_context)
    book = raw["futures_book"] if cfg.market != "SPOT" else raw["spot_book"]
    dq = check_candles(bundle.entry, cfg.timeframes.entry, cfg.data_quality,
                       min_history=min(cfg.data_quality.min_history_bars, bars - 10))
    dq.merge(check_orderbook(book, cfg.data_quality), "orderbook: ")
    de = DecisionEngine(cfg, learned=load_learned())
    btc_f = None
    if bundle.btc_trend is not None and not bundle.btc_trend.empty:
        btc_f = compute_features(bundle.btc_trend, cfg.timeframes.trend, cfg.market)
    prep = de.prepare(bundle.entry, symbol, cfg.timeframes.entry, bundle.htf, bundle.derivatives, btc_f)
    d = de.decide(prep, -1, dq=dq, rules=rules, book_stats=orderbook_features(book) if book else None,
                  funding_now=fs.get("funding_rate"))
    return d.explanation


def scan_market(top: int = 10, min_quote_volume: float = 20_000_000) -> str:
    """Binance USDT çiftlerinde hacim ve 24 saatlik hareket taraması (hızlı ön tarama)."""
    from .binance_client import BinanceClient
    from .intel.universe import select_universe
    c = BinanceClient(testnet=False)
    tickers = c.ticker_24h()
    syms, why = select_universe(tickers, "USDT", top, min_quote_volume)
    return "Öne çıkan coinler:\n" + "\n".join(f"• {s}: {why.get(s, '')}" for s in syms)


# ---------------------------------------------------------------------- arayüzsüz bot
class HeadlessBot:
    """Masaüstü uygulamasındaki 'Zeka Motoru'nu arayüzsüz çalıştırır (thread-safe)."""

    OWNER = "gömülü bot"

    def __init__(self, settings: Settings | None = None, owner: str | None = None, on_event=None):
        self.settings = settings or load_settings()
        self.engine = None
        self.notifier = None
        self.lock = EngineLock(owner or self.OWNER)
        self.on_event = on_event
        self.logs: deque = deque(maxlen=300)
        self.approvals_waiting: dict = {}
        self._mx = threading.Lock()

    # ---------------------------------------------------------------- olaylar
    def _event(self, kind, payload):
        if kind == "log":
            self.logs.append(f"[{time.strftime('%H:%M:%S')}] {payload}")
        elif kind == "approval_request":
            self.approvals_waiting[payload.request_id] = payload
        elif kind == "approval_resolved":
            self.approvals_waiting.pop(payload.request_id, None)
        n = self.notifier
        if n is not None:
            n.handle_event(kind, payload)
        if self.on_event is not None:
            try:
                self.on_event(kind, payload)
            except Exception:  # noqa: BLE001 - dış dinleyici hatası botu durdurmasın
                logger.exception("on_event")

    # ---------------------------------------------------------------- başlat / durdur
    @property
    def running(self) -> bool:
        return self.engine is not None and self.engine.running

    def start(self, live: bool | None = None, confirm_live: bool = False, market: str | None = None) -> str:
        """Botu başlatır. live=None → ayarlardaki mod. Canlı için confirm_live=True zorunlu."""
        with self._mx:
            if self.running:
                return "Bot zaten çalışıyor."
            s = self.settings = load_settings()
            live = s.live_mode if live is None else bool(live)
            if live and not confirm_live:
                return ("Canlı (gerçek para) mod için açık onay gerekli. Gerçek parayla başlatmak istediğinizi "
                        "onaylarsanız tekrar 'onaylıyorum' diyerek başlatın.")
            if live and (not s.api_key or not s.api_secret):
                return "Canlı işlem için Binance API anahtarı gerekli."
            ok, msg = self.lock.acquire()
            if not ok:
                return msg
            try:
                engine = self._build(s, live, market or s.intel_market)
            except Exception as exc:  # noqa: BLE001
                self.lock.release()
                return f"Bot başlatılamadı: {exc}"
            self.engine = engine
            self.notifier = self._make_notifier(s)
            if self.notifier is not None:
                self.notifier.attach(engine)
                self.notifier.start()
            engine.start()
            s.live_mode = live
            try:
                save_settings(s)
            except OSError:
                pass
            return engine.start_message()

    def stop(self, wait: bool = True) -> str:
        with self._mx:
            if self.engine is None or not self.engine.running:
                self.lock.release()
                return "Bot zaten durmuş."
            self.engine.stop(wait=wait)
            if self.notifier is not None:
                self.notifier.stop(timeout=5)
                self.notifier = None
            self.lock.release()
            return "Bot durduruldu. Açık pozisyonlar satılmadı; stop-loss bot kapalıyken izlenmez."

    def _make_notifier(self, s: Settings):
        if not (s.telegram_enabled and s.telegram_token and s.telegram_chat_id):
            return None
        from .telegram import TelegramClient, TelegramNotifier
        return TelegramNotifier(TelegramClient(s.telegram_token), s.telegram_chat_id, s.telegram_notify,
                                s.quote_asset, s.telegram_commands, s.telegram_summary_hour)

    def _build(self, s: Settings, live: bool, market: str):
        from .binance_client import BinanceClient
        from .intel.autopilot import load_learned
        from .intel.config import load_intel_config
        from .intel.decision import DecisionEngine
        from .intel.execution import FuturesLiveVenue, PaperVenue, SpotLiveVenue
        from .intel.futures_client import BinanceFuturesClient
        from .intel.live_engine import IntelligentBotEngine
        from .intel.signal_store import SignalStore

        cfg = load_intel_config()
        cfg.market = market
        cfg.quote_asset = s.quote_asset
        cfg.risk.leverage = min(s.intel_leverage, cfg.risk.max_leverage) if cfg.market != "SPOT" else 1
        cfg.risk.hard_take_profit_pct = s.hard_take_profit_pct
        errors = cfg.validate()
        if errors:
            raise ValueError("; ".join(errors))
        futures_mkt = cfg.market != "SPOT"
        if live:
            spot = BinanceClient(s.api_key, s.api_secret, testnet=s.testnet)
            fut = BinanceFuturesClient(s.api_key, s.api_secret, testnet=s.testnet)
        else:
            spot = BinanceClient(testnet=False) if s.mainnet_data else BinanceClient(s.api_key, s.api_secret,
                                                                                     testnet=s.testnet)
            fut = BinanceFuturesClient(testnet=not s.mainnet_data and s.testnet)
        market_client = fut if futures_mkt else spot
        market_client.sync_time()
        for sym in s.symbols:
            market_client.symbol_rules(sym)
        if live:
            venue = FuturesLiveVenue(fut, cfg, s.quote_asset) if futures_mkt else SpotLiveVenue(spot, cfg,
                                                                                                s.quote_asset)
            venue.available_balance()
        else:
            venue = PaperVenue(cfg.market, s.paper_balance, cfg, s.quote_asset)
        news = insight_engine = insight_store = None
        if cfg.news.enabled or cfg.listing.enabled or cfg.catalyst.enabled:
            from .intel.news import RSS_FEEDS, RSS_FEEDS_TR, NewsMonitor
            from .intel.translate import Translator
            nc = cfg.news
            feeds = {k: v for k, v in RSS_FEEDS.items() if {"CoinDesk": nc.rss_coindesk,
                     "Cointelegraph": nc.rss_cointelegraph, "Decrypt": nc.rss_decrypt}[k]}
            if nc.rss_turkish:
                feeds.update(RSS_FEEDS_TR)
            news = NewsMonitor(rss_feeds=feeds, use_binance=nc.use_binance_announcements,
                               cryptopanic_token=s.cryptopanic_token, quote=s.quote_asset,
                               block_hours=nc.block_hours, min_severity_block=nc.min_severity_block,
                               translator=Translator(email=s.translate_email) if nc.translate_titles else None)
            if cfg.catalyst.enabled:
                from .intel.catalyst import InsightEngine, InsightStore
                insight_engine = InsightEngine(cfg.catalyst, market_client, quote=cfg.quote_asset)
                insight_store = InsightStore()
        universe = None
        if s.universe_mode in ("top", "all"):
            from .intel.universe import UniverseSelector
            all_mode = s.universe_mode == "all"
            universe = UniverseSelector(market_client, cfg.quote_asset, s.universe_size,
                                        min_quote_volume=2_000_000 if all_mode else 20_000_000,
                                        base_symbols=s.symbols, all_coins=all_mode)
        approvals = None
        if s.trade_approval == "manual":
            from .intel.approval import ApprovalBook
            approvals = ApprovalBook(s.approval_timeout_min)
        net = "paper" if not live else ("testnet" if s.testnet else "live")
        state = data_dir() / f"state_intel_{net}_{cfg.market.lower()}.json"
        return IntelligentBotEngine(cfg, s.symbols, market_client, venue, SignalStore(), futures_client=fut,
                                    decision_engine=DecisionEngine(cfg, learned=load_learned()),
                                    poll_seconds=s.poll_seconds, state_path=state, on_event=self._event,
                                    news_monitor=news, insight_engine=insight_engine, insight_store=insight_store,
                                    universe=universe, approvals=approvals)

    # ---------------------------------------------------------------- bilgi
    def status_text(self) -> str:
        e = self.engine
        if e is None:
            return "Bot çalışmıyor."
        mode = "GERÇEK PARA" if e.venue.is_live else "KAĞIT (simülasyon)"
        try:
            eq = e.equity()
            free = e.venue.available_balance()
        except Exception:  # noqa: BLE001
            eq = free = float("nan")
        lines = [f"Durum: {'çalışıyor' if e.running else 'durdu'} · Mod: {mode} · Piyasa: {tr(e.market)}",
                 f"Toplam değer: {eq:.2f} {e.quote_asset} · Serbest: {free:.2f} {e.quote_asset}",
                 f"Taranan coin: {len(e.symbols)} · Açık pozisyon: {len(e.positions)} · "
                 f"Kapanan işlem: {len(e.trades)}"]
        if self.approvals_waiting:
            lines.append(f"Onay bekleyen işlem: {len(self.approvals_waiting)} (onay_bekleyenler ile görün)")
        summary = [x for x in self.logs if "Özet" in x]
        if summary:
            lines.append(summary[-1])
        return "\n".join(lines)

    def positions_text(self) -> str:
        e = self.engine
        if e is None or not e.positions:
            return "Açık pozisyon yok."
        out = []
        for sym, p in e.positions.items():
            px = e.last_prices.get(sym, p.entry_price)
            pnl_pct = (px / p.entry_price - 1) * 100 * p.sign
            out.append(f"{sym} {tr(p.direction)}: giriş {p.entry_price:.6g}, şimdi {px:.6g} (%{pnl_pct:+.2f}), "
                       f"stop {p.stop_loss:.6g}, strateji {p.strategy}")
        return "\n".join(out)

    def trades_text(self, n: int = 10) -> str:
        e = self.engine
        trades = list(e.trades)[-n:] if e is not None else []
        if not trades:
            return "Henüz kapanmış işlem yok."
        return "\n".join(f"{t.symbol}: {t.pnl:+.2f} ({t.pnl_pct:+.2f}%) — {tr_reason(t.reason)} — {t.closed_at}"
                         for t in reversed(trades))

    def recent_log(self, n: int = 15) -> str:
        return "\n".join(list(self.logs)[-n:]) or "Kayıt yok."

    def pending_approvals_text(self) -> str:
        if not self.approvals_waiting:
            return "Onay bekleyen işlem yok."
        return "\n\n".join(f"[{rid}] {r.summary()}" for rid, r in self.approvals_waiting.items())

    def resolve_approval(self, request_id: str = "", approve: bool = True, by: str = "Jarvis") -> str:
        e = self.engine
        if e is None or e.approvals is None:
            return "Manuel onay modu kapalı ya da bot çalışmıyor."
        if not request_id:
            if len(self.approvals_waiting) != 1:
                return "Hangi işlem? " + self.pending_approvals_text()
            request_id = next(iter(self.approvals_waiting))
        return e.resolve_approval(request_id, approve, by)

    def close_position(self, symbol: str) -> str:
        e = self.engine
        if e is None:
            return "Bot çalışmıyor."
        symbol = symbol.upper()
        if symbol not in e.positions:
            return f"{symbol} için açık pozisyon yok."
        try:
            e.close_position(symbol)
        except Exception as exc:  # noqa: BLE001
            return f"{symbol} kapatılamadı: {exc}"
        return f"{symbol} pozisyonu piyasa fiyatından kapatıldı."
