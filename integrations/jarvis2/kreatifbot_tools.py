# KreatifBot → Jarvis entegrasyonu (jarvis2/kreatifbot_tools.py olarak kopyalanır)
"""Gemini araç tanımları ve tek noktadan çağırıcı.

tool_defs.py sonuna:   from kreatifbot_tools import KREATIFBOT_TOOLS; TOOL_DECLARATIONS += KREATIFBOT_TOOLS
main.py dispatch'ine:  elif name in KREATIFBOT_TOOL_NAMES:
                           r = await loop.run_in_executor(None, lambda: handle_kreatifbot_tool(name, args))
                           result = r or "Tamam."
"""
from __future__ import annotations


def _obj(props: dict | None = None, required: list | None = None) -> dict:
    d = {"type": "OBJECT", "properties": props or {}}
    if required:
        d["required"] = required
    return d


KREATIFBOT_TOOLS = [
    {
        "name": "kreatif_bot_start",
        "description": (
            "KreatifBot otomatik trading botunu başlatır (Binance, bütün coinleri tarar, stratejilerle işlem açar). "
            "mode='paper' kağıt işlemdir (gerçek para yok). mode='live' GERÇEK PARA ile işlem yapar: kullanıcı açıkça "
            "'gerçek parayla başlat, onaylıyorum' demeden confirm=true KULLANMA; önce confirm=false ile çağır, "
            "dönen uyarıyı kullanıcıya oku, onaylarsa confirm=true ile tekrar çağır."),
        "parameters": _obj({
            "mode": {"type": "STRING", "description": "paper (varsayılan) | live"},
            "confirm": {"type": "BOOLEAN", "description": "Canlı mod için kullanıcının açık onayı"},
        }),
    },
    {"name": "kreatif_bot_stop", "description": "KreatifBot trading botunu durdurur (açık pozisyonları satmaz).",
     "parameters": _obj()},
    {"name": "kreatif_bot_status",
     "description": "KreatifBot'un durumu: çalışıyor mu, mod, bakiye, taranan coin sayısı, açık pozisyon, son özet.",
     "parameters": _obj()},
    {"name": "kreatif_bot_positions", "description": "KreatifBot'un açık pozisyonları ve anlık kâr/zararı.",
     "parameters": _obj()},
    {"name": "kreatif_bot_trades", "description": "KreatifBot'un son kapanan işlemleri ve sonuçları.",
     "parameters": _obj()},
    {"name": "kreatif_bot_log", "description": "KreatifBot'un son kayıt satırları (ne yaptığı, neden işlem açmadığı).",
     "parameters": _obj({"lines": {"type": "NUMBER", "description": "Satır sayısı (varsayılan 15)"}})},
    {"name": "kreatif_bot_pending_approvals",
     "description": "Manuel onay modunda onay bekleyen işlem önerilerini listeler.", "parameters": _obj()},
    {
        "name": "kreatif_bot_approve",
        "description": (
            "Onay bekleyen bir işlemi onaylar veya reddeder. Kullanıcı 'onayla', 'al', 'reddet', 'açma' dediğinde kullan. "
            "Tek bekleyen istek varsa request_id boş bırakılabilir."),
        "parameters": _obj({
            "request_id": {"type": "STRING", "description": "Onay isteği numarası (köşeli parantez içindeki)"},
            "approve": {"type": "BOOLEAN", "description": "true = onayla, false = reddet"},
        }),
    },
    {
        "name": "kreatif_bot_close_position",
        "description": ("KreatifBot'un açık bir pozisyonunu piyasa fiyatından satar. Önce confirm=false ile sor, "
                        "kullanıcı onaylarsa confirm=true ile tekrar çağır."),
        "parameters": _obj({"symbol": {"type": "STRING", "description": "Örn. SOLUSDT"},
                            "confirm": {"type": "BOOLEAN", "description": "Kullanıcının açık onayı"}}, ["symbol"]),
    },
    {
        "name": "kreatif_bot_settings",
        "description": ("KreatifBot ayarlarını gösterir veya değiştirir: kesin kâr al yüzdesi, işlem onayı "
                        "(otomatik/manuel), coin taraması (all=bütün coinler, top=hacimli, manual=liste). "
                        "Parametresiz çağrılırsa mevcut ayarları söyler."),
        "parameters": _obj({
            "take_profit_pct": {"type": "NUMBER", "description": "Kesin kâr al yüzdesi (0 = kapalı)"},
            "approval_mode": {"type": "STRING", "description": "otomatik | manuel"},
            "coin_mode": {"type": "STRING", "description": "all | top | manual"},
            "coin_count": {"type": "NUMBER", "description": "top modunda coin sayısı (2-40)"},
        }),
    },
    {
        "name": "kreatif_analyze",
        "description": ("Bir coin için KreatifBot Zeka Motoru analizi: piyasa rejimi, strateji sinyalleri, güven skoru, "
                        "giriş/stop/hedef veya neden işlem olmadığı (gerçek Binance verisi). Kullanıcı bir coinin "
                        "detaylı analizini, 'alınır mı', 'ne düşünüyorsun' diye sorduğunda kullan."),
        "parameters": _obj({"symbol": {"type": "STRING", "description": "Örn. BTCUSDT, SOL"}}),
    },
    {"name": "kreatif_market_scan",
     "description": "Binance'te şu an en hacimli ve en çok hareket eden coinleri listeler.",
     "parameters": _obj({"top": {"type": "NUMBER", "description": "Kaç coin (varsayılan 10)"}})},
    {"name": "kreatif_news",
     "description": "Kripto haberleri (Türkçe başlıklar, kategori, ilgili coinler). Coin verilirse yalnız onunla ilgili.",
     "parameters": _obj({"symbol": {"type": "STRING", "description": "Opsiyonel coin, örn. SOL"},
                         "hours": {"type": "NUMBER", "description": "Son kaç saat (varsayılan 24)"}})},
]

KREATIFBOT_TOOL_NAMES = {t["name"] for t in KREATIFBOT_TOOLS}


def handle_kreatifbot_tool(name: str, args: dict | None) -> str:
    from actions import kreatifbot_actions as kb
    args = dict(args or {})
    fn = getattr(kb, name, None)
    if fn is None:
        return f"Bilinmeyen KreatifBot aracı: {name}"
    import inspect
    allowed = set(inspect.signature(fn).parameters)
    return fn(**{k: v for k, v in args.items() if k in allowed})
