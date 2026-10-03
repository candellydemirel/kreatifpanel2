"""Windows sistem entegrasyonu: uyku engeli ve Windows ile otomatik başlatma.

Windows dışındaki sistemlerde işlevler güvenle hiçbir şey yapmaz (False döner).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger("kreatifbot.system")

RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
APP_VALUE = "KreatifBot"

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def is_windows() -> bool:
    return sys.platform == "win32"


def prevent_sleep(enable: bool) -> bool:
    """Bot çalışırken sistemin uykuya geçmesini engeller (ekran yine kapanabilir)."""
    if not is_windows():
        return False
    try:
        import ctypes
        flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if enable else 0)
        return bool(ctypes.windll.kernel32.SetThreadExecutionState(flags))
    except (AttributeError, OSError) as exc:
        logger.warning("Uyku ayarı değiştirilemedi: %s", exc)
        return False


def launch_command(minimized: bool = True) -> str:
    """Windows açılışında çalıştırılacak komut (EXE veya kaynak koddan)."""
    extra = " --minimized" if minimized else ""
    if getattr(sys, "frozen", False):
        return f'"{sys.executable}"{extra}'
    exe = Path(sys.executable)
    pythonw = exe.with_name("pythonw.exe")
    runner = pythonw if pythonw.exists() else exe
    main_py = Path(__file__).resolve().parent.parent / "main.py"
    return f'"{runner}" "{main_py}"{extra}'


def autostart_enabled() -> bool:
    if not is_windows():
        return False
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as key:
            winreg.QueryValueEx(key, APP_VALUE)
        return True
    except OSError:
        return False


def set_autostart(enable: bool) -> bool:
    """Kullanıcı oturumu açıldığında uygulamayı (tepside) başlatır. Başarılıysa True."""
    if not is_windows():
        return False
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
            if enable:
                winreg.SetValueEx(key, APP_VALUE, 0, winreg.REG_SZ, launch_command(True))
            else:
                try:
                    winreg.DeleteValue(key, APP_VALUE)
                except FileNotFoundError:
                    pass
        return True
    except OSError as exc:
        logger.warning("Otomatik başlatma ayarlanamadı: %s", exc)
        return False
