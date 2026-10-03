"""Telegram bildirimleri ve uzaktan komutlar.

Bot olayları (başlatma/durdurma, sinyaller, alım/satım, hatalar, günlük özet)
Telegram sohbetinize gönderilir. Komutlar yalnızca ayarlarda tanımlı sohbetten
kabul edilir.
"""

from __future__ import annotations

import html
import logging
import queue
import threading
import time
from datetime import datetime

import requests

from .i18n import LISTING_KIND_TR, tr, tr_reason
from .intel.translate import display_title

logger = logging.getLogger("kreatifbot.telegram")

API_URL = "https://api.telegram.org/bot{token}/{method}"
MAX_LEN = 4000

NOTIFY_LABELS = {
    "status": "Bot başlatıldı / durduruldu",
    "signals": "AL / SAT sinyalleri",
    "trades": "Alım ve satım işlemleri (kâr/zarar)",
    "risk": "Risk uyarıları (günlük zarar limiti)",
    "errors": "Hatalar ve uyarılar",
    "daily_summary": "Günlük özet",
    "news": "Önemli haberler ve yeni listelemeler (Zeka Motoru)",
}
NOTIFY_DEFAULTS = {k: True for k in NOTIFY_LABELS}

HELP_TEXT = (
    "<b>KreatifBot komutları</b>\n"
    "/durum – bot durumu, bakiye, günlük K/Z\n"
    "/pozisyonlar – açık pozisyonlar\n"
    "/islemler – son 10 işlem\n"
    "/ozet – bugünün özeti\n"
    "/durdur – botu durdurur (açık pozisyonlar satılmaz)\n"
    "/yardim – bu mesaj\n\n"
    "Manuel onay modunda (Ayarlar → İşlem onayı) her yeni işlem için Onayla / Reddet düğmeli mesaj gelir."
)


class TelegramError(Exception):
    pass


def esc(value) -> str:
    return html.escape(str(value), quote=False)


def _num(value, digits: int = 8) -> str:
    try:
        return f"{float(value):.{digits}g}"
    except (TypeError, ValueError):
        return str(value)


class TelegramClient:
    def __init__(self, token: str, session: requests.Session | None = None, timeout: float = 15):
        self.token = (token or "").strip()
        self.session = session or requests.Session()
        self.timeout = timeout

    def call(self, method: str, params: dict | None = None, timeout: float | None = None):
        if not self.token:
            raise TelegramError("Telegram bot token'ı girilmemiş.")
        url = API_URL.format(token=self.token, method=method)
        try:
            resp = self.session.post(url, json=params or {}, timeout=timeout or self.timeout)
        except requests.RequestException as exc:
            raise TelegramError(f"Telegram bağlantı hatası: {exc.__class__.__name__}") from exc
        try:
            data = resp.json()
        except ValueError as exc:
            raise TelegramError(f"Telegram geçersiz yanıt (HTTP {resp.status_code})") from exc
        if not data.get("ok"):
            desc = data.get("description", f"HTTP {resp.status_code}")
            if resp.status_code == 401:
                desc = "Token geçersiz (BotFather'dan aldığınız token'ı kontrol edin)."
            elif resp.status_code == 400 and "chat not found" in desc.lower():
                desc = "Sohbet bulunamadı: önce botunuza Telegram'dan /start yazın ve Chat ID'yi kontrol edin."
            raise TelegramError(desc)
        return data["result"]

    def get_me(self) -> dict:
        return self.call("getMe")

    def send_message(self, chat_id: str | int, text: str, html_mode: bool = True,
                     reply_markup: dict | None = None) -> dict:
        if len(text) > MAX_LEN:
            text = text[: MAX_LEN - 20] + "\n…(kısaltıldı)"
        params = {"chat_id": chat_id, "text": text, "disable_web_page_preview": True}
        if html_mode:
            params["parse_mode"] = "HTML"
        if reply_markup:
            params["reply_markup"] = reply_markup
        return self.call("sendMessage", params)

    def answer_callback(self, callback_id: str, text: str = "") -> None:
        self.call("answerCallbackQuery", {"callback_query_id": callback_id, "text": text[:190]})

    def remove_buttons(self, chat_id, message_id) -> None:
        self.call("editMessageReplyMarkup", {"chat_id": chat_id, "message_id": message_id,
                                             "reply_markup": {"inline_keyboard": []}})

    def get_updates(self, offset: int | None = None, timeout: int = 0) -> list:
        params = {"timeout": timeout, "allowed_updates": ["message", "callback_query"]}
        if offset is not None:
            params["offset"] = offset
        return self.call("getUpdates", params, timeout=timeout + 15)

    def find_chat_id(self) -> tuple[str, str]:
        """Bota en son mesaj atan sohbetin kimliğini döndürür."""
        updates = self.get_updates()
        for upd in reversed(updates):
            msg = upd.get("message") or upd.get("edited_message")
            if msg and "chat" in msg:
                chat = msg["chat"]
                name = chat.get("title") or " ".join(
                    x for x in (chat.get("first_name"), chat.get("last_name")) if x) or chat.get("username", "")
                return str(chat["id"]), name
        raise TelegramError("Mesaj bulunamadı. Telegram'da botunuzu açıp /start yazın, sonra tekrar deneyin.")


class TelegramNotifier:
    """Bot olaylarını kuyruğa alıp arka planda Telegram'a gönderir; komutları dinler."""

    def __init__(self, client: TelegramClient, chat_id: str, notify: dict | None = None,
                 quote_asset: str = "USDT", commands: bool = True, summary_hour: int = 21):
        self.client = client
        self.chat_id = str(chat_id).strip()
        self.notify = {**NOTIFY_DEFAULTS, **(notify or {})}
        self.quote = quote_asset
        self.commands = commands
        self.summary_hour = int(summary_hour)
        self.engine = None
        self.last_equity: dict = {}
        self._queue: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._closing = False
        self._summary_sent_for = datetime.now().date() if datetime.now().hour >= self.summary_hour else None
        self._threads: list[threading.Thread] = []
        self.sent_count = 0

    # ---------------------------------------------------------------- yaşam döngüsü
    def attach(self, engine):
        self.engine = engine

    def start(self):
        self._stop.clear()
        self._threads = [threading.Thread(target=self._sender, name="TelegramSender", daemon=True)]
        if self.commands or getattr(self.engine, "approvals", None) is not None:
            self._threads.append(threading.Thread(target=self._poller, name="TelegramPoller", daemon=True))
        for t in self._threads:
            t.start()

    def stop(self, timeout: float = 10):
        self._closing = True
        end = time.time() + timeout
        while not self._queue.empty() and time.time() < end:
            time.sleep(0.1)
        self._stop.set()

    @property
    def running(self) -> bool:
        return any(t.is_alive() for t in self._threads)

    def send(self, text: str, reply_markup: dict | None = None):
        self._queue.put((text, reply_markup) if reply_markup else text)

    # ---------------------------------------------------------------- olaylar
    def handle_event(self, kind: str, payload=None):
        """Bot motorundan gelen olay (motor iş parçacığından çağrılır, bloklamaz)."""
        try:
            text = self.format_event(kind, payload)
        except Exception:  # biçimlendirme hatası motoru etkilemesin
            logger.exception("Telegram olayı biçimlendirilemedi")
            return
        if kind == "approval_request" and text:
            self.send(text, {"inline_keyboard": [[
                {"text": "✅ Onayla", "callback_data": f"ap:{payload.request_id}:1"},
                {"text": "❌ Reddet", "callback_data": f"ap:{payload.request_id}:0"}]]})
            return
        if text:
            self.send(text)
        if kind == "status" and payload == "stopped":
            self._closing = True

    def format_event(self, kind: str, payload) -> str | None:
        e = self.engine
        if kind == "equity" and isinstance(payload, dict):
            self.last_equity = payload
            return None
        if kind == "status" and self.notify["status"]:
            if payload == "running":
                mode = "CANLI" if e is not None and getattr(e.broker, "is_live", False) else "KAĞIT (simülasyon)"
                lines = ["🟢 <b>KreatifBot başlatıldı</b>", f"Mod: <b>{mode}</b>"]
                if e is not None:
                    lines += [f"Strateji: {esc(e.strategy.name)}",
                              f"Semboller: {esc(', '.join(e.symbols))} ({esc(e.interval)})",
                              f"Açık pozisyon: {len(e.positions)}"]
                if self.commands:
                    lines.append("Komutlar için /yardim")
                return "\n".join(lines)
            if payload == "stopped":
                text = "🔴 <b>KreatifBot durduruldu</b>"
                if e is not None and e.positions:
                    text += (f"\n⚠️ {len(e.positions)} açık pozisyon var; bot kapalıyken stop-loss izlenmez: "
                             + esc(", ".join(e.positions)))
                return text
        if kind == "signal" and self.notify["signals"] and isinstance(payload, dict):
            sig = payload.get("signal")
            if sig not in ("AL", "SAT"):
                return None
            icon = "📈" if sig == "AL" else "📉"
            return (f"{icon} <b>{sig} sinyali — {esc(payload['symbol'])}</b>\n"
                    f"Fiyat: {_num(payload.get('price'))}\nMum: {esc(payload.get('time', ''))}\n"
                    f"{esc(payload.get('reason', ''))}")
        if kind == "opened" and self.notify["trades"]:
            p = payload
            side = getattr(p, "direction", "LONG")
            title = "ALIM" if side == "LONG" else "SHORT AÇILDI"
            lines = [f"✅ <b>{title} — {esc(p.symbol)}</b>",
                     f"Miktar: {_num(p.qty)} @ {_num(p.entry_price)}",
                     f"Tutar: {p.cost:.2f} {self.quote}"]
            if p.stop_loss:
                lines.append(f"Stop-loss: {_num(p.stop_loss)}")
            if p.take_profit:
                lines.append(f"Kâr al: {_num(p.take_profit)}")
            return "\n".join(lines)
        if kind == "trade" and self.notify["trades"]:
            t = payload
            icon = "💰" if t.pnl > 0 else "🔻"
            title = "SATIŞ" if getattr(t, "side", "LONG") == "LONG" else "SHORT KAPANDI"
            return (f"{icon} <b>{title} — {esc(t.symbol)}</b> ({esc(tr_reason(t.reason))})\n"
                    f"Giriş {_num(t.entry_price)} → Çıkış {_num(t.exit_price)}\n"
                    f"K/Z: <b>{t.pnl:+.2f} {self.quote} ({t.pnl_pct:+.2f}%)</b>")
        if kind == "news" and self.notify.get("news", True):
            it = payload
            icon = {"DELISTING": "🚫", "HACK": "🚨", "REGULATION_NEG": "⚖️", "LISTING": "🆕",
                    "FUTURES_LISTING": "🆕", "LAUNCHPOOL": "🎁"}.get(it.category, "📰")
            coins = f" [{', '.join(it.symbols)}]" if it.symbols else ""
            return (f"{icon} <b>{esc(tr(it.category))}</b>{esc(coins)} — {esc(it.source)}\n{esc(display_title(it))}\n"
                    f"Duyarlılık {it.sentiment:+.2f}" + (f"\n{esc(it.url)}" if it.url else ""))
        if kind == "listing" and self.notify.get("news", True):
            ev = payload
            label = LISTING_KIND_TR.get(ev.kind, ev.kind)
            return f"🆕 <b>Listeleme: {esc(ev.symbol)}</b> — {label} ({esc(ev.source)}, durum {esc(ev.status or '-')})"
        if kind == "insight" and self.notify.get("news", True):
            ins = payload
            icon = "🟢" if ins.signal == "AL" else "💡"
            from .intel.catalyst import CATALYST_TR
            cats = ", ".join(CATALYST_TR.get(t, t) for t in ins.catalyst.types) or "-"
            fd = ins.fundamentals
            text = (f"{icon} <b>Öngörü: {esc(ins.symbol)} — {esc(ins.signal)}</b> (potansiyel {ins.potential:.0f}/100)\n"
                    f"Katalizör: {esc(cats)} ({esc(', '.join(ins.catalyst.sources))})\n"
                    f"Temel skor: {'-' if fd.score is None else f'{fd.score:.0f}'} · Teknik: "
                    f"{'-' if ins.technical.score is None else f'{ins.technical.score:.0f}'}")
            if ins.signal == "AL":
                text += f"\nGiriş ~{ins.entry:.6g} · Stop {ins.stop:.6g}"
            if ins.catalyst.headlines:
                text += f"\n{esc(ins.catalyst.headlines[0][:200])}"
            return text + "\n<i>Yatırım tavsiyesi değildir.</i>"
        if kind == "maintenance" and self.notify["status"]:
            return "🛠 " + esc(payload.summary()[:3000])
        if kind == "approval_request":
            return f"🟡 <b>İŞLEM ONAYI GEREKLİ</b>\n{esc(payload.summary())}\n\nOnaylamazsanız emir gönderilmez."
        if kind == "approval_resolved":
            from .intel.approval import STATE_TR
            return f"ℹ️ {esc(payload.symbol)} onay isteği: {esc(STATE_TR.get(payload.state, payload.state))}"
        if kind == "halt" and self.notify["risk"]:
            return f"⛔ <b>Günlük zarar limiti aşıldı</b>\n{esc(payload)}\nBugün yeni pozisyon açılmayacak."
        if kind == "alert" and self.notify["errors"] and isinstance(payload, dict):
            icon = "❗" if payload.get("level", 0) >= logging.ERROR else "⚠️"
            return f"{icon} {esc(payload.get('message', ''))}"
        return None

    # ---------------------------------------------------------------- raporlar
    def status_text(self) -> str:
        e = self.engine
        if e is None:
            return "Bot bağlı değil."
        mode = "CANLI" if getattr(e.broker, "is_live", False) else "KAĞIT"
        eq = self.last_equity
        lines = [f"<b>Durum:</b> {'🟢 çalışıyor' if e.running else '🔴 durdu'} ({mode})",
                 f"Strateji: {esc(e.strategy.name)}",
                 f"Semboller: {esc(', '.join(e.symbols))} ({esc(e.interval)})"]
        if eq:
            lines += [f"Toplam değer: <b>{eq['equity']:.2f} {self.quote}</b>",
                      f"Serbest bakiye: {eq['quote']:.2f} {self.quote}",
                      f"Günlük K/Z: {eq['day_pct']:+.2f}%"]
        if getattr(e, "halted_today", False):
            lines.append("⛔ Günlük zarar limiti nedeniyle yeni işlem açılmıyor.")
        lines.append(f"Açık pozisyon: {len(e.positions)}")
        return "\n".join(lines)

    def positions_text(self) -> str:
        e = self.engine
        if e is None or not e.positions:
            return "Açık pozisyon yok."
        lines = ["<b>Açık pozisyonlar</b>"]
        for sym, p in list(e.positions.items()):
            price = e.last_prices.get(sym, p.entry_price)
            pnl, pct = p.unrealized(price)
            lines.append(f"• <b>{esc(sym)}</b> {_num(p.qty)} @ {_num(p.entry_price)} → {_num(price)} "
                         f"| {pnl:+.2f} {self.quote} ({pct:+.2f}%)"
                         + (f" | SL {_num(p.stop_loss)}" if p.stop_loss else ""))
        return "\n".join(lines)

    def trades_text(self, n: int = 10) -> str:
        e = self.engine
        trades = list(e.trades)[-n:] if e is not None else []
        if not trades:
            return "Henüz kapanmış işlem yok."
        lines = [f"<b>Son {len(trades)} işlem</b>"]
        for t in reversed(trades):
            lines.append(f"{'💰' if t.pnl > 0 else '🔻'} {esc(t.symbol)} {t.pnl:+.2f} ({t.pnl_pct:+.2f}%) "
                         f"– {esc(tr_reason(t.reason))} – {esc(t.closed_at)}")
        return "\n".join(lines)

    def summary_text(self) -> str:
        e = self.engine
        today = datetime.now().strftime("%Y-%m-%d")
        trades = [t for t in (e.trades if e is not None else []) if str(t.closed_at).startswith(today)]
        total = sum(t.pnl for t in trades)
        wins = sum(1 for t in trades if t.pnl > 0)
        lines = [f"📊 <b>Günlük özet — {today}</b>",
                 f"Kapanan işlem: {len(trades)} (kazanan {wins}, kaybeden {len(trades) - wins})",
                 f"Gerçekleşen K/Z: <b>{total:+.2f} {self.quote}</b>"]
        if self.last_equity:
            lines += [f"Toplam değer: {self.last_equity['equity']:.2f} {self.quote}",
                      f"Günlük değişim: {self.last_equity['day_pct']:+.2f}%"]
        lines.append(f"Açık pozisyon: {len(e.positions) if e is not None else 0}")
        return "\n".join(lines)

    def handle_command(self, text: str) -> str:
        cmd = text.strip().split()[0].split("@")[0].lower() if text.strip() else ""
        if cmd in ("/start", "/yardim", "/help"):
            return HELP_TEXT
        if cmd == "/durum":
            return self.status_text()
        if cmd == "/pozisyonlar":
            return self.positions_text()
        if cmd == "/islemler":
            return self.trades_text()
        if cmd == "/ozet":
            return self.summary_text()
        if cmd == "/durdur":
            if self.engine is None or not self.engine.running:
                return "Bot zaten çalışmıyor."
            self.engine.stop()
            return "⏹ Bot durduruluyor. (Açık pozisyonlar satılmadı; yeniden başlatmak için uygulamayı kullanın.)"
        return "Bilinmeyen komut. /yardim yazın."

    # ---------------------------------------------------------------- iş parçacıkları
    def _deliver(self, item) -> bool:
        text, markup = item if isinstance(item, tuple) else (item, None)
        for attempt in range(3):
            try:
                self.client.send_message(self.chat_id, text, reply_markup=markup)
                self.sent_count += 1
                return True
            except TelegramError as exc:
                logger.warning("Telegram mesajı gönderilemedi (deneme %d): %s", attempt + 1, exc)
                if self._stop.wait(2 * (attempt + 1)):
                    break
        return False

    def _handle_callback(self, cq: dict):
        """Telegram'daki Onayla / Reddet düğmeleri (yalnızca yetkili sohbetten)."""
        msg = cq.get("message") or {}
        chat_id = str(msg.get("chat", {}).get("id"))
        data = str(cq.get("data") or "")
        if chat_id != self.chat_id or not data.startswith("ap:"):
            try:
                self.client.answer_callback(cq.get("id", ""), "Yetkisiz")
            except TelegramError:
                pass
            return
        try:
            _, req_id, flag = data.split(":")
        except ValueError:
            return
        e = self.engine
        if e is None or not hasattr(e, "resolve_approval"):
            reply = "Bot çalışmıyor; istek geçersiz."
        else:
            reply = e.resolve_approval(req_id, flag == "1", "Telegram")
        try:
            self.client.answer_callback(cq.get("id", ""), reply)
            if msg.get("message_id"):
                self.client.remove_buttons(chat_id, msg["message_id"])
        except TelegramError as exc:
            logger.warning("Telegram onay yanıtı gönderilemedi: %s", exc)

    def _check_summary(self):
        if not self.notify["daily_summary"]:
            return
        now = datetime.now()
        if now.hour >= self.summary_hour and self._summary_sent_for != now.date():
            self._summary_sent_for = now.date()
            self.send(self.summary_text())

    def _sender(self):
        while not self._stop.is_set():
            try:
                text = self._queue.get(timeout=1)
            except queue.Empty:
                if self._closing:
                    self._stop.set()
                    break
                self._check_summary()
                continue
            self._deliver(text)
            time.sleep(0.05)  # Telegram hız sınırı: sohbet başına ~1 mesaj/sn ortalama

    def _poller(self):
        offset = None
        try:  # Bot kapalıyken gelen eski komutları atla.
            old = self.client.get_updates()
            if old:
                offset = old[-1]["update_id"] + 1
        except TelegramError as exc:
            logger.warning("Telegram komutları dinlenemiyor: %s", exc)
        while not self._stop.is_set():
            try:
                updates = self.client.get_updates(offset, timeout=20)
            except TelegramError as exc:
                logger.warning("Telegram getUpdates hatası: %s", exc)
                self._stop.wait(5)
                continue
            for upd in updates:
                offset = upd["update_id"] + 1
                if upd.get("callback_query"):
                    self._handle_callback(upd["callback_query"])
                    continue
                msg = upd.get("message") or {}
                text = msg.get("text") or ""
                if str(msg.get("chat", {}).get("id")) != self.chat_id or not text.startswith("/"):
                    continue  # Yalnızca yetkili sohbetten gelen komutlar.
                try:
                    reply = self.handle_command(text)
                except Exception as exc:  # noqa: BLE001
                    reply = f"Komut hatası: {esc(exc)}"
                self.send(reply)
