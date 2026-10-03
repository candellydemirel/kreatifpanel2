"""KreatifBot Trading Intelligence Engine.

Binance Spot ve USDⓈ-M Futures verileriyle çalışan karar motoru:
veri kalitesi → özellikler → rejim → strateji yönlendirici → aday sinyaller →
çoklu zaman dilimi → güven skoru → meta-model → seviyeler → maliyet/EV →
risk & portföy & devre kesici → nihai karar nesnesi (açıklamalı).

Sistem kâr garantisi vermez; tüm stratejiler walk-forward ve örneklem dışı
testlerle doğrulanmalıdır. Veri yoksa sahte değer üretilmez (UNAVAILABLE).
"""

ENGINE_VERSION = "1.0.0"
