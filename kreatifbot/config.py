"""Ayarların kaydedilmesi. API gizli anahtarı Windows'ta DPAPI ile şifrelenir."""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .risk import RiskSettings

APP_NAME = "KreatifBot"
GUIDE_PDF = "docs/KreatifBot_Strateji_Rehberi.pdf"
logger = logging.getLogger("kreatifbot.config")


def data_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        path = base / APP_NAME
    else:
        path = Path.home() / ".kreatifbot"
    override = os.environ.get("KREATIFBOT_HOME")
    if override:
        path = Path(override)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resource_path(relative: str) -> Path:
    """Uygulamayla birlikte gelen dosyanın yolu (PyInstaller EXE içinde de çalışır)."""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return base / relative


# ---------------------------------------------------------------------- şifreleme
if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    class _DataBlob(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

    def _dpapi(data: bytes, encrypt: bool) -> bytes:
        crypt32 = ctypes.windll.crypt32
        kernel32 = ctypes.windll.kernel32
        buf = ctypes.create_string_buffer(data, len(data))
        blob_in = _DataBlob(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)))
        blob_out = _DataBlob()
        func = crypt32.CryptProtectData if encrypt else crypt32.CryptUnprotectData
        ok = func(ctypes.byref(blob_in), None, None, None, None, 0x01, ctypes.byref(blob_out))
        if not ok:
            raise ctypes.WinError()
        try:
            return ctypes.string_at(blob_out.pbData, blob_out.cbData)
        finally:
            kernel32.LocalFree(blob_out.pbData)
else:
    _dpapi = None


def protect(secret: str) -> str:
    if not secret:
        return ""
    raw = secret.encode("utf-8")
    if _dpapi is not None:
        try:
            return "dpapi:" + base64.b64encode(_dpapi(raw, True)).decode()
        except OSError as exc:
            logger.warning("DPAPI şifreleme başarısız: %s", exc)
    return "plain:" + base64.b64encode(raw).decode()


def unprotect(value: str) -> str:
    if not value:
        return ""
    try:
        kind, payload = value.split(":", 1)
        raw = base64.b64decode(payload)
        if kind == "dpapi":
            if _dpapi is None:
                return ""
            return _dpapi(raw, False).decode("utf-8")
        return raw.decode("utf-8")
    except (ValueError, OSError) as exc:
        logger.warning("Gizli anahtar çözülemedi: %s", exc)
        return ""


# ---------------------------------------------------------------------- ayarlar
@dataclass
class Settings:
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = True
    mainnet_data: bool = True            # Analiz/backtest için gerçek piyasa verisi
    quote_asset: str = "USDT"
    symbols: list = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    interval: str = "15m"
    strategy: str = "ensemble"
    strategy_params: dict = field(default_factory=dict)
    risk: dict = field(default_factory=lambda: RiskSettings().to_dict())
    paper_balance: float = 1000.0
    poll_seconds: int = 30
    live_mode: bool = False

    @property
    def risk_settings(self) -> RiskSettings:
        return RiskSettings.from_dict(self.risk)


def settings_path() -> Path:
    return data_dir() / "settings.json"


def load_settings() -> Settings:
    path = settings_path()
    if not path.exists():
        return Settings()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.error("Ayarlar okunamadı: %s", exc)
        return Settings()
    data["api_secret"] = unprotect(data.get("api_secret", ""))
    known = {k: v for k, v in data.items() if k in Settings.__dataclass_fields__}
    settings = Settings(**known)
    settings.risk = {**RiskSettings().to_dict(), **(settings.risk or {})}
    return settings


def save_settings(settings: Settings) -> None:
    data = asdict(settings)
    data["api_secret"] = protect(settings.api_secret)
    path = settings_path()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def setup_logging() -> Path:
    log_dir = data_dir() / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "kreatifbot.log"
    from logging.handlers import RotatingFileHandler

    handler = RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not any(isinstance(h, RotatingFileHandler) for h in root.handlers):
        root.addHandler(handler)
    return log_file
