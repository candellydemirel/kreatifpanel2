import requests

from kreatifbot.i18n import tr, tr_reason
from kreatifbot.intel.news import NewsItem, NewsMonitor, NewsStore
from kreatifbot.intel.translate import UNTRANSLATED_MARK, Translator, display_title, looks_turkish


class _Resp:
    def __init__(self, data, status=200):
        self._data, self.status_code = data, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def json(self):
        return self._data


class FakeSession:
    def __init__(self, google_ok=True, mymemory_ok=True):
        self.google_ok, self.mymemory_ok, self.calls = google_ok, mymemory_ok, []

    def get(self, url, params=None, timeout=None):
        self.calls.append(url)
        if "googleapis" in url:
            if not self.google_ok:
                return _Resp({}, 429)
            return _Resp([[["TR: " + params["q"], params["q"], None]], None, "en"])
        if not self.mymemory_ok:
            return _Resp({"responseStatus": 429, "responseData": {"translatedText": "MYMEMORY WARNING: quota"}})
        return _Resp({"responseStatus": 200, "responseData": {"translatedText": "MM: " + params["q"]}})


def test_tr_labels():
    assert tr("STRONG_BULL") == "Güçlü yükseliş"
    assert tr("HACK") == "Hack/Saldırı"
    assert tr("NEWS_RISK") == "Olumsuz haber riski"
    assert tr("bilinmeyen") == "bilinmeyen"
    assert tr_reason("TP_HIT tp1") == "Kâr al tp1"


def test_translator_fallback_and_cache():
    s = FakeSession(google_ok=False)
    t = Translator(session=s)
    assert t.translate("Crypto lost money") == "MM: Crypto lost money"
    n = len(s.calls)
    assert t.translate("Crypto lost money") == "MM: Crypto lost money"
    assert len(s.calls) == n  # önbellek
    assert t.translate("Bitcoin yükselişe geçti") == "Bitcoin yükselişe geçti"  # zaten Türkçe
    assert looks_turkish("Kripto piyasasında düşüş")


def test_translator_all_fail_returns_none():
    t = Translator(session=FakeSession(google_ok=False, mymemory_ok=False))
    assert t.translate("Hack hits exchange") is None
    it = NewsItem("x", "CoinDesk", "Hack hits exchange")
    assert display_title(it).endswith(UNTRANSLATED_MARK)


def test_monitor_translates_and_persists(tmp_path):
    store = NewsStore(tmp_path / "n.sqlite3")
    mon = NewsMonitor(store=store, session=FakeSession(), rss_feeds={}, use_binance=False,
                      translator=Translator(session=FakeSession()))
    new = store.add([NewsItem("a", "CoinDesk", "Exchange hacked", published_at="2099-01-01T00:00:00+00:00")])
    assert mon.translate_pending(new) == 1
    got = store.recent(hours=1e9)[0]
    assert got.title_tr == "TR: Exchange hacked" and got.display_title == "TR: Exchange hacked"
    assert got.title == "Exchange hacked"  # orijinal başlık korunur
