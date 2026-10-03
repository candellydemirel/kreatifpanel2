# KreatifBot → Jarvis entegrasyonu (jarvis2/actions/kreatifbot_actions.py olarak kopyalanır)
"""Jarvis sesli asistanından KreatifBot trading botunu yönetmek için araçlar.

Bot, masaüstü KreatifBot uygulamasıyla AYNI ayarları kullanır (%APPDATA%\\KreatifBot):
API anahtarları, Telegram, strateji aşamaları, %15 kesin kâr al, işlem onayı modu, öğrenilen risk.
Aynı anda yalnızca bir bot çalışabilir (masaüstü uygulaması veya Jarvis) — çift emir olmaz.

Tüm fonksiyonlar Türkçe metin döndürür; main.py bunları run_in_executor ile çağırır.
"""
from __future__ import annotations

import atexit
import functools
import threading

_BOT = None
_LOCK = threading.Lock()


def _bot():
    global _BOT
    with _LOCK:
        if _BOT is None:
            from kreatifbot.config import setup_logging
            from kreatifbot.headless import HeadlessBot
            setup_logging()
            _BOT = HeadlessBot(owner="Jarvis")
            atexit.register(lambda: _BOT.stop(wait=False) if _BOT and _BOT.running else None)
        return _BOT


def _safe(fn):
    @functools.wraps(fn)                     # imza korunur (araç çağırıcı parametreleri buna göre süzer)
    def wrapper(*a, **kw):
        try:
            return fn(*a, **kw)
        except Exception as exc:  # noqa: BLE001 - asistan çökmesin, hatayı söylesin
            return f"KreatifBot hatası: {exc}"
    return wrapper


# ---------------------------------------------------------------------- bot yönetimi
@_safe
def kreatif_bot_start(mode: str = "paper", confirm: bool = False) -> str:
    live = str(mode or "").strip().lower() in {"live", "canli", "canlı", "gercek", "gerçek", "real"}
    return _bot().start(live=live, confirm_live=bool(confirm))


@_safe
def kreatif_bot_stop() -> str:
    return _bot().stop()


@_safe
def kreatif_bot_status() -> str:
    return _bot().status_text()


@_safe
def kreatif_bot_positions() -> str:
    return _bot().positions_text()


@_safe
def kreatif_bot_trades() -> str:
    return _bot().trades_text()


@_safe
def kreatif_bot_log(lines: int = 15) -> str:
    return _bot().recent_log(int(lines or 15))


@_safe
def kreatif_bot_pending_approvals() -> str:
    return _bot().pending_approvals_text()


@_safe
def kreatif_bot_approve(request_id: str = "", approve: bool = True) -> str:
    return _bot().resolve_approval(str(request_id or ""), bool(approve), "Jarvis")


@_safe
def kreatif_bot_close_position(symbol: str, confirm: bool = False) -> str:
    if not confirm:
        return (f"{symbol} pozisyonu piyasa fiyatından GERÇEKTEN satılsın mı? Onaylarsanız aynı isteği "
                "confirm=true ile tekrarlayın.")
    return _bot().close_position(symbol)


@_safe
def kreatif_bot_settings(take_profit_pct: float | None = None, approval_mode: str = "",
                         coin_mode: str = "", coin_count: int | None = None) -> str:
    from kreatifbot.config import load_settings, save_settings
    s = load_settings()
    changed = []
    if take_profit_pct is not None and float(take_profit_pct) >= 0:
        s.hard_take_profit_pct = float(take_profit_pct)
        changed.append(f"kesin kâr al %{s.hard_take_profit_pct:g}" if s.hard_take_profit_pct else
                       "kesin kâr al kapalı")
    am = str(approval_mode or "").strip().lower()
    if am in {"auto", "otomatik", "manual", "manuel"}:
        s.trade_approval = "manual" if am.startswith("manu") else "auto"
        changed.append("işlem onayı " + ("manuel" if s.trade_approval == "manual" else "otomatik"))
    cm = str(coin_mode or "").strip().lower()
    if cm in {"all", "hepsi", "tum", "tüm", "top", "hacimli", "manual", "liste"}:
        s.universe_mode = "all" if cm in {"all", "hepsi", "tum", "tüm"} else "top" if cm in {"top", "hacimli"} \
            else "manual"
        changed.append(f"coin taraması {s.universe_mode}")
    if coin_count:
        s.universe_size = max(2, min(40, int(coin_count)))
        changed.append(f"coin sayısı {s.universe_size}")
    if not changed:
        return (f"Mevcut ayarlar: kesin kâr al %{s.hard_take_profit_pct:g}, işlem onayı {s.trade_approval}, "
                f"coin taraması {s.universe_mode} ({s.universe_size}), mod "
                f"{'CANLI' if s.live_mode else 'kağıt'}.")
    save_settings(s)
    return "Kaydedildi: " + ", ".join(changed) + ". Bot çalışıyorsa yeniden başlatınca uygulanır."


# ---------------------------------------------------------------------- analiz
@_safe
def kreatif_analyze(symbol: str = "BTCUSDT") -> str:
    from kreatifbot.headless import analyze_symbol
    return analyze_symbol(symbol)


@_safe
def kreatif_market_scan(top: int = 10) -> str:
    from kreatifbot.headless import scan_market
    return scan_market(int(top or 10))


@_safe
def kreatif_news(symbol: str = "", hours: int = 24) -> str:
    from kreatifbot.i18n import tr
    from kreatifbot.intel.news import NewsStore
    base = str(symbol or "").upper().replace("USDT", "") or None
    items = NewsStore().recent(float(hours or 24), base, limit=200)[:8]
    if not items:
        return ("Kayıtlı haber yok. Haberleri bot çalışırken otomatik tarar; botu başlatın veya KreatifBot "
                "uygulamasında Haberler sekmesini açın.")
    return "\n".join(f"• [{it.source}] {tr(it.category)}" + (f" ({', '.join(it.symbols)})" if it.symbols else "")
                     + f": {it.display_title}" for it in items)
