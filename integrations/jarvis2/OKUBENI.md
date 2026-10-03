# KreatifBot → Jarvis (jarvis2) entegrasyonu

Bu paket, KreatifBot trading botunun bütün özelliklerini Jarvis sesli asistanına ekler. Jarvis'e sesle şunları söyleyebilirsin:

- "Trading botu kağıt modda başlat" / "botu durdur"
- "Botun durumu ne?", "açık pozisyonlar neler?", "son işlemler?"
- "Solana'yı analiz et", "Bitcoin alınır mı?"
- "Piyasada en çok hareket eden coinler hangileri?"
- "Kripto haberlerini oku", "SOL ile ilgili haber var mı?"
- "Onay bekleyen işlem var mı?" / "onayla" / "reddet" (manuel onay modunda)
- "Kesin kâr alı yüzde 12 yap", "işlem onayını manuel yap"
- "SOL pozisyonunu sat" (Jarvis önce onay ister)

## Nasıl çalışır

- Jarvis, masaüstü KreatifBot uygulamasıyla **aynı ayarları** kullanır (`%APPDATA%\KreatifBot`). Bunlar: API anahtarları, Telegram, strateji aşamaları, %15 kesin kâr al, işlem onayı, öğrenilen riskler, kağıt hesabı.
- **Aynı anda tek bot çalışır.** Masaüstü uygulamasında bot açıksa Jarvis başlatmaz (tersi de geçerli). Böylece aynı hesapta çift emir açılmaz.
- **Gerçek para** ile başlatmak için Jarvis önce uyarı okur. "Onaylıyorum" demeden canlı başlamaz.
- Telegram bildirimleri, Onayla/Reddet düğmeleri ve `/durdur` komutu Jarvis'ten başlatılan botta da çalışır.

## Kurulum (otomatik)

1. GitHub Actions'tan **KreatifBot-jarvis2-paketi** zip'ini indir ve bir klasöre çıkar.
2. Jarvis'i kapat.
3. `kur.bat` dosyasına çift tıkla.
   - jarvis2 klasörünü kendisi arar. Önce `Masaüstü\jarvis2` konumuna bakar; bulamazsa yolunu sorar.
   - jarvis2'nin kendi `venv` ortamını kullanır ve gerekli paketleri (pandas, numpy, requests) kurar.
   - `tool_defs.py` ve `main.py` dosyalarına KreatifBot bağlantısını ekler. Değiştirmeden önce `.bak` yedeklerini alır.
   - Sonunda her şeyi derleyip kontrol eder ve "✓ Kontrol tamam" yazar.
4. Jarvis'i yeniden başlat ve "trading botun durumu ne?" diye sor.

İkinci kez çalıştırmak bir şeyi bozmaz. Yeni KreatifBot sürümünde aynı adımlarla güncellersin.

**Geri almak için:** `main.py.bak` ve `tool_defs.py.bak` dosyalarını eski adlarına döndür. Ardından `kreatifbot` klasörünü, `kreatifbot_tools.py` dosyasını ve `actions\kreatifbot_actions.py` dosyasını sil.

## Elle kurulum (betik main.py'yi tanıyamazsa)

1. Kopyala:
   - `kreatifbot\` → `jarvis2\kreatifbot\`
   - `kreatifbot_tools.py` → `jarvis2\`
   - `kreatifbot_actions.py` → `jarvis2\actions\`
2. `jarvis2\venv\Scripts\python.exe -m pip install pandas numpy requests`
3. `tool_defs.py` dosyasının en sonuna şunu ekle:
   ```python
   from kreatifbot_tools import KREATIFBOT_TOOLS
   TOOL_DECLARATIONS += KREATIFBOT_TOOLS
   ```
4. `main.py` dosyasının üstüne şunu ekle:
   `from kreatifbot_tools import KREATIFBOT_TOOL_NAMES, handle_kreatifbot_tool`
   Araç dağıtımındaki `elif name == "get_crypto_price":` satırının hemen önüne de şunu ekle:
   ```python
   elif name in KREATIFBOT_TOOL_NAMES:
       r = await loop.run_in_executor(None, lambda: handle_kreatifbot_tool(name, args))
       result = r or "Tamam."
   ```

## Bilgisayarındaki Claude Code ile kurulum

"Jarvis2 klasörü düzelt" oturumuna şunu yapıştırabilirsin:

> jarvis2 klasörüne KreatifBot entegrasyon paketini kur. Paket şu klasörde: <zip'i çıkardığın yol>. Önce `OKUBENI.md` dosyasını oku. Sonra jarvis2'nin venv Python'u ile `jarvis2_kur.py` betiğini çalıştır. Betik main.py'yi tanıyamazsa OKUBENI'deki elle kurulum adımlarını uygula. Mevcut araçları (get_crypto_price, place_crypto_order, Binance TR vb.) bozma. Bitince Jarvis'i başlatıp `kreatif_bot_status` aracının "Bot çalışmıyor." döndürdüğünü doğrula.

## Araç listesi (Gemini)

| Araç | Ne yapar |
|---|---|
| `kreatif_bot_start` | Botu başlatır (`paper` / `live`, canlı için `confirm`) |
| `kreatif_bot_stop` | Botu durdurur (pozisyonları satmaz) |
| `kreatif_bot_status` | Durum, mod, bakiye, coin sayısı, son özet |
| `kreatif_bot_positions` / `kreatif_bot_trades` | Açık pozisyonlar / kapanan işlemler |
| `kreatif_bot_log` | Son kayıt satırları |
| `kreatif_bot_pending_approvals` / `kreatif_bot_approve` | Manuel onay listesi / onayla-reddet |
| `kreatif_bot_close_position` | Pozisyonu sat (önce onay sorar) |
| `kreatif_bot_settings` | Kesin kâr al %, onay modu, coin taraması |
| `kreatif_analyze` | Coin için Zeka Motoru analizi |
| `kreatif_market_scan` | En hacimli ve en hareketli coinler |
| `kreatif_news` | Türkçe kripto haberleri |

Kâr garantisi yoktur. Bot, kurallarına uyan güvenli fırsat yokken işlem açmaz.
