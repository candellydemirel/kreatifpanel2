"""Haber başlıklarını Türkçeye çevirme.

Ücretsiz ve anahtarsız iki servis sırayla denenir:
1. Google Translate'in herkese açık (resmi belgelenmemiş) uç noktası.
2. MyMemory (günlük ücretsiz kota; isteğe bağlı e-posta girilirse kota artar).

Her başlık bir kez çevrilir ve haber veritabanında saklanır. Çeviri yalnızca gösterim
içindir: sınıflandırma ve katalizör tespiti her zaman orijinal metin üzerinde yapılır.
Çeviri başarısız olursa orijinal başlık gösterilir ("(çevrilemedi)" işaretiyle).
"""

from __future__ import annotations

import logging
import re
import threading
import time

import requests

logger = logging.getLogger("kreatifbot.intel.translate")

GOOGLE_URL = "https://translate.googleapis.com/translate_a/single"
MYMEMORY_URL = "https://api.mymemory.translated.net/get"
UNTRANSLATED_MARK = "(çevrilemedi)"

_TR_CHARS = re.compile(r"[çğıöşüÇĞİÖŞÜ]")
_TR_WORDS = re.compile(r"\b(ve|bir|ile|için|bu|da|de|olarak|yeni|fiyat|yükseliş|düşüş|haber|kripto)\b", re.I)


def looks_turkish(text: str) -> bool:
    if not text:
        return False
    return bool(_TR_CHARS.search(text)) or len(_TR_WORDS.findall(text)) >= 2


class Translator:
    def __init__(self, session=None, email: str = "", timeout: float = 8.0, enabled: bool = True):
        self.session = session or requests.Session()
        self.email = email.strip()
        self.timeout = timeout
        self.enabled = enabled
        self._cache: dict[str, str] = {}
        self._lock = threading.Lock()
        self._disabled_until: dict[str, float] = {}   # kota/erişim hatasında servisi bir süre atla
        self.status = "Hazır"

    # ---------------------------------------------------------------- servisler
    def _google(self, text: str) -> str:
        r = self.session.get(GOOGLE_URL, params={"client": "gtx", "sl": "auto", "tl": "tr", "dt": "t", "q": text},
                             timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        out = "".join(part[0] for part in (data[0] or []) if part and part[0])
        if not out.strip():
            raise ValueError("boş çeviri")
        return out.strip()

    def _mymemory(self, text: str) -> str:
        params = {"q": text[:480], "langpair": "en|tr"}
        if self.email:
            params["de"] = self.email
        r = self.session.get(MYMEMORY_URL, params=params, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        status = int(data.get("responseStatus") or 0)
        out = (data.get("responseData") or {}).get("translatedText") or ""
        if status != 200 or not out or "MYMEMORY WARNING" in out.upper():
            raise ValueError(f"MyMemory: {data.get('responseDetails') or status}")
        return out.strip()

    # ---------------------------------------------------------------- arayüz
    def translate(self, text: str) -> str | None:
        """Türkçe çeviri; zaten Türkçeyse aynen, başarısızsa None döner."""
        text = (text or "").strip()
        if not text:
            return ""
        if looks_turkish(text):
            return text
        if not self.enabled:
            return None
        with self._lock:
            if text in self._cache:
                return self._cache[text]
        now = time.time()
        for name, fn in (("google", self._google), ("mymemory", self._mymemory)):
            if self._disabled_until.get(name, 0) > now:
                continue
            try:
                out = fn(text)
            except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as exc:
                logger.info("Çeviri servisi %s başarısız: %s", name, exc)
                self._disabled_until[name] = now + 600   # 10 dk sonra tekrar dene
                continue
            with self._lock:
                if len(self._cache) > 5000:
                    self._cache.clear()
                self._cache[text] = out
            self.status = f"OK ({name})"
            return out
        self.status = "Çeviri servislerine ulaşılamıyor"
        return None


def display_title(item) -> str:
    """Kullanıcıya gösterilecek başlık: Türkçe çeviri varsa o, yoksa işaretli orijinal."""
    tr_title = getattr(item, "title_tr", "") or ""
    if tr_title:
        return tr_title
    title = getattr(item, "title", "") or ""
    return title if looks_turkish(title) else f"{title} {UNTRANSLATED_MARK}"
