"""KreatifBot'u Jarvis (jarvis2) sesli asistanına kurar.

Çalıştırma (Windows):  kur.bat  (veya)  jarvis2\\venv\\Scripts\\python.exe jarvis2_kur.py [jarvis2_klasörü]

Yaptıkları (her adım tekrar çalıştırılabilir, ikinci kez çalıştırmak bir şeyi bozmaz):
1. kreatifbot paketini jarvis2 klasörüne kopyalar (eski kopyanın yerine).
2. actions/kreatifbot_actions.py ve kreatifbot_tools.py dosyalarını kopyalar.
3. Gerekli Python paketlerini (pandas, numpy, requests) jarvis2'nin sanal ortamına kurar.
4. tool_defs.py'nin SONUNA KreatifBot araçlarını ekler.
5. main.py'deki araç dağıtımına tek bir 'elif' satırı ekler (önce .bak yedeği alınır).
6. Her şeyi derleyip kontrol eder.
"""
from __future__ import annotations

import os
import py_compile
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MARK = "# >>> KreatifBot entegrasyonu"
END = "# <<< KreatifBot entegrasyonu"


def say(msg: str):
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:                 # eski Windows konsolu (cp1252/cp857)
        print(msg.encode("ascii", "replace").decode(), flush=True)


def find_jarvis(arg: str | None) -> Path:
    cands = [Path(arg)] if arg else []
    home = Path(os.environ.get("USERPROFILE", Path.home()))
    cands += [home / "Desktop" / "jarvis2", home / "OneDrive" / "Desktop" / "jarvis2",
              home / "Masaüstü" / "jarvis2", HERE.parent / "jarvis2"]
    for c in cands:
        if (c / "main.py").exists() and (c / "tool_defs.py").exists():
            return c.resolve()
    raise SystemExit("jarvis2 klasörü bulunamadı. Klasör yolunu verin:  python jarvis2_kur.py C:\\yol\\jarvis2")


def copy_files(j: Path):
    src_pkg = HERE / "kreatifbot"
    if not src_pkg.exists():
        src_pkg = HERE.parent.parent / "kreatifbot"          # depo içinden çalıştırılıyorsa
    if not (src_pkg / "headless.py").exists():
        raise SystemExit("kreatifbot paketi bulunamadı (zip'in tamamını çıkardığınızdan emin olun).")
    dst = j / "kreatifbot"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src_pkg, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "gui"))
    (j / "actions").mkdir(exist_ok=True)
    shutil.copy2(HERE / "kreatifbot_actions.py", j / "actions" / "kreatifbot_actions.py")
    shutil.copy2(HERE / "kreatifbot_tools.py", j / "kreatifbot_tools.py")
    say(f"✓ Dosyalar kopyalandı → {j}")


def install_deps():
    reqs = ["requests>=2.31", "numpy>=1.26", "pandas>=2.1"]
    say("… Python paketleri kuruluyor (pandas, numpy, requests) — birkaç dakika sürebilir")
    r = subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", *reqs])
    if r.returncode != 0:
        raise SystemExit("Paket kurulumu başarısız. İnternet bağlantısını kontrol edip tekrar deneyin.")
    say("✓ Paketler hazır")


def patch_tool_defs(j: Path):
    p = j / "tool_defs.py"
    s = p.read_text(encoding="utf-8")
    if MARK in s:
        say("✓ tool_defs.py zaten bağlı")
        return
    if "TOOL_DECLARATIONS" not in s:
        raise SystemExit("tool_defs.py içinde TOOL_DECLARATIONS bulunamadı; README'deki elle kurulum adımlarını izleyin.")
    shutil.copy2(p, p.with_suffix(".py.bak"))
    s = s.rstrip() + (f"\n\n\n{MARK}\nfrom kreatifbot_tools import KREATIFBOT_TOOLS as _KREATIF_TOOLS\n"
                      "_mevcut = {t.get('name') for t in TOOL_DECLARATIONS}\n"
                      "TOOL_DECLARATIONS += [t for t in _KREATIF_TOOLS if t['name'] not in _mevcut]\n"
                      f"{END}\n")
    p.write_text(s, encoding="utf-8")
    say("✓ tool_defs.py: KreatifBot araçları eklendi (yedek: tool_defs.py.bak)")


def patch_main(j: Path):
    p = j / "main.py"
    s = p.read_text(encoding="utf-8")
    if MARK in s:
        say("✓ main.py zaten bağlı")
        return
    m = re.search(r'^(?P<ind>[ \t]+)elif name == "get_crypto_price":', s, re.M) or \
        re.search(r'^(?P<ind>[ \t]+)elif name == "[a-z_]+":', s, re.M)
    if not m:
        raise SystemExit("main.py'de araç dağıtım bloğu bulunamadı; README'deki elle kurulum adımlarını izleyin.")
    ind = m.group("ind")
    block = (f"{ind}{MARK}\n"
             f"{ind}elif name in KREATIFBOT_TOOL_NAMES:\n"
             f"{ind}    r = await loop.run_in_executor(None, lambda: handle_kreatifbot_tool(name, args))\n"
             f"{ind}    result = r or \"Tamam.\"\n"
             f"{ind}{END}\n\n")
    s = s[:m.start()] + block + s[m.start():]
    imp = f"{MARK}\nfrom kreatifbot_tools import KREATIFBOT_TOOL_NAMES, handle_kreatifbot_tool\n{END}\n"
    lines = s.splitlines(keepends=True)
    at = 0
    for i, line in enumerate(lines[:200]):
        if line.startswith(("import ", "from ")) and not line.startswith("from __future__"):
            at = i
            break
        if line.startswith("from __future__"):
            at = i + 1
    lines.insert(at, imp)
    shutil.copy2(p, p.with_suffix(".py.bak"))
    p.write_text("".join(lines), encoding="utf-8")
    say("✓ main.py: araç dağıtımı eklendi (yedek: main.py.bak)")


def verify(j: Path):
    for f in ("main.py", "tool_defs.py", "kreatifbot_tools.py", "actions/kreatifbot_actions.py"):
        py_compile.compile(str(j / f), doraise=True)
    code = ("import sys; sys.path.insert(0, '.');"
            "from tool_defs import TOOL_DECLARATIONS as T;"
            "n=[t['name'] for t in T]; assert len(n)==len(set(n)), 'çift araç adı';"
            "assert 'kreatif_bot_start' in n;"
            "import kreatifbot.headless, actions.kreatifbot_actions;"
            "print(len(n), 'araç hazır')")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    r = subprocess.run([sys.executable, "-c", code], cwd=j, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env)
    if r.returncode != 0:
        raise SystemExit("Kontrol başarısız:\n" + (r.stderr or r.stdout)[-1500:] +
                         "\nGeri almak için .bak dosyalarını eski adlarına döndürün.")
    say("✓ Kontrol tamam: " + r.stdout.strip())


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    j = find_jarvis(sys.argv[1] if len(sys.argv) > 1 else None)
    say(f"jarvis2 klasörü: {j}")
    copy_files(j)
    install_deps()
    patch_tool_defs(j)
    patch_main(j)
    verify(j)
    say("\nKurulum bitti. Jarvis'i yeniden başlatın ve deneyin:\n"
        "  • 'Trading botun durumu ne?'\n  • 'Solana'yı analiz et'\n  • 'Botu kağıt modda başlat'\n"
        "  • 'Onay bekleyen işlem var mı?'\n"
        "Not: Masaüstü KreatifBot uygulamasında bot çalışıyorsa Jarvis botu başlatmaz (çift işlem olmaması için).")


if __name__ == "__main__":
    main()
