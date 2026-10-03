"""Jarvis (jarvis2) entegrasyon paketi: kurulum betiği gerçek jarvis2 yapısına benzer sahte klasörde denenir."""

import asyncio
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INTEG = ROOT / "integrations" / "jarvis2"

MAIN_PY = '''# Can Demirel tarafından yapılmıştır
from __future__ import annotations
import asyncio
from tool_defs import TOOL_DECLARATIONS


def get_crypto_price(symbol, exchange="global"):
    return f"{symbol} fiyat"


async def dispatch(name, args):
    loop = asyncio.get_running_loop()
    result = None
    if name == "open_app":
        result = "app"
    elif name == "get_crypto_price":
        r = await loop.run_in_executor(None, lambda: get_crypto_price(args.get("symbol", "BTCUSDT")))
        result = r or "Fiyat bilgisi alindi."
    return result
'''

TOOL_DEFS = '''TOOL_DECLARATIONS = [
    {"name": "open_app", "description": "x", "parameters": {"type": "OBJECT", "properties": {}}},
    {"name": "get_crypto_price", "description": "y", "parameters": {"type": "OBJECT", "properties": {}}},
]
'''


def _load_kur():
    spec = importlib.util.spec_from_file_location("jarvis2_kur", INTEG / "jarvis2_kur.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_installer_patches_and_tools_work(tmp_path, monkeypatch):
    j = tmp_path / "jarvis2"
    (j / "actions").mkdir(parents=True)
    (j / "main.py").write_text(MAIN_PY, encoding="utf-8")
    (j / "tool_defs.py").write_text(TOOL_DEFS, encoding="utf-8")
    kur = _load_kur()
    monkeypatch.setattr(kur, "install_deps", lambda: None)
    monkeypatch.setattr(sys, "argv", ["jarvis2_kur.py", str(j)])
    kur.main()
    kur.main()                                           # ikinci çalıştırma bir şeyi bozmamalı
    main_src = (j / "main.py").read_text(encoding="utf-8")
    assert main_src.count("elif name in KREATIFBOT_TOOL_NAMES") == 1
    assert (j / "main.py.bak").exists() and (j / "tool_defs.py.bak").exists()
    assert (j / "kreatifbot" / "headless.py").exists() and not (j / "kreatifbot" / "gui").exists()

    monkeypatch.syspath_prepend(str(j))
    for m in [m for m in list(sys.modules) if m in ("main", "tool_defs", "kreatifbot_tools")
              or m.startswith("actions")]:
        sys.modules.pop(m)
    tool_defs = importlib.import_module("tool_defs")
    names = [t["name"] for t in tool_defs.TOOL_DECLARATIONS]
    assert "kreatif_bot_start" in names and len(names) == len(set(names))
    main = importlib.import_module("main")
    # eski araçlar çalışmaya devam eder
    assert asyncio.run(main.dispatch("get_crypto_price", {"symbol": "ETHUSDT"})) == "ETHUSDT fiyat"
    # KreatifBot araçları (bot çalışmıyorken)
    assert asyncio.run(main.dispatch("kreatif_bot_status", {})) == "Bot çalışmıyor."
    out = asyncio.run(main.dispatch("kreatif_bot_start", {"mode": "live", "confirm": False}))
    assert "onay" in out.lower()
    s = asyncio.run(main.dispatch("kreatif_bot_settings", {"take_profit_pct": 12, "approval_mode": "manuel",
                                                           "bogus": 1}))
    assert "Kaydedildi" in s
    from kreatifbot.config import load_settings
    st = load_settings()
    assert st.hard_take_profit_pct == 12 and st.trade_approval == "manual"
    assert "Bilinmeyen" in __import__("kreatifbot_tools").handle_kreatifbot_tool("yok", {})


def test_engine_lock_blocks_second_engine():
    from kreatifbot.headless import EngineLock
    a, b = EngineLock("KreatifBot masaüstü"), EngineLock("Jarvis")
    assert a.acquire()[0]
    ok, msg = b.acquire()
    assert not ok and "masaüstü" in msg
    a.release()
    assert b.acquire()[0]
    b.release()


def test_headless_bot_paper_run(monkeypatch):
    import kreatifbot.binance_client as bc
    import kreatifbot.intel.futures_client as fc
    from kreatifbot.config import load_settings, save_settings
    from kreatifbot.headless import HeadlessBot

    from .conftest import make_ohlcv
    from .test_intel import FakeMarket, with_taker

    market = FakeMarket(with_taker(make_ohlcv(700, seed=9)))
    monkeypatch.setattr(bc, "BinanceClient", lambda *a, **k: market)
    monkeypatch.setattr(fc, "BinanceFuturesClient", lambda *a, **k: market)
    s = load_settings()
    s.universe_mode, s.symbols, s.poll_seconds, s.paper_balance = "manual", ["BTCUSDT"], 3600, 20
    save_settings(s)
    bot = HeadlessBot(owner="test")
    msg = bot.start(live=False)
    try:
        assert "Zeka Motoru başlatıldı" in msg and "KAĞIT" in msg, msg
        assert bot.running and "Mod: KAĞIT" in bot.status_text()
        assert "Başka bir bot" in HeadlessBot(owner="ikinci").start(live=False)   # tek motor kilidi
        assert bot.positions_text() in ("Açık pozisyon yok.",) or "BTCUSDT" in bot.positions_text()
        assert "Onay bekleyen işlem yok." == bot.pending_approvals_text()
    finally:
        assert "durduruldu" in bot.stop()
    assert not bot.running and "zaten" in bot.stop()
