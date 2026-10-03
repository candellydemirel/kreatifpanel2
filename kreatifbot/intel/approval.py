"""Manuel işlem onayı.

Onay modunda bot yeni pozisyon açmadan önce bir onay isteği oluşturur (uygulamada pencere,
Telegram'da Onayla/Reddet düğmeleri). Onaylanana kadar emir gönderilmez; onaylandıktan sonra
fiyat, stop ve emir kuralları güncel verilerle YENİDEN kontrol edilir. Süresi dolan istek iptal olur.
Çıkışlar (stop-loss, kâr al, iz süren stop) güvenlik için her zaman otomatiktir.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field

PENDING, APPROVED, REJECTED, EXPIRED, DONE = "PENDING", "APPROVED", "REJECTED", "EXPIRED", "DONE"
STATE_TR = {PENDING: "Onay bekliyor", APPROVED: "Onaylandı", REJECTED: "Reddedildi", EXPIRED: "Süresi doldu",
            DONE: "Uygulandı"}


@dataclass
class ApprovalRequest:
    request_id: str
    key: str                      # aynı sinyal için tek istek (decision: signal_id, harici: strateji+sembol)
    symbol: str
    direction: str
    strategy: str
    price: float
    qty: float
    notional: float
    stop: float
    targets: list = field(default_factory=list)
    confidence: float = 0.0
    reason: str = ""
    live: bool = False
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    state: str = PENDING
    resolved_by: str = ""
    payload: object = None        # harici sinyaller için yeniden çalıştırma bilgisi

    @property
    def expired(self) -> bool:
        return self.state == PENDING and time.time() > self.expires_at

    @property
    def minutes_left(self) -> float:
        return max(0.0, (self.expires_at - time.time()) / 60)

    def summary(self) -> str:
        risk = abs(self.price - self.stop) * self.qty
        tps = ", ".join(f"{t:.6g}" for t in self.targets[:3])
        return (f"{'GERÇEK' if self.live else 'KAĞIT'} {self.direction} {self.symbol} — {self.strategy}\n"
                f"Fiyat ~{self.price:.6g} · Tutar ~{self.notional:.2f} · Stop {self.stop:.6g} "
                f"(en fazla kayıp ~{risk:.2f})" + (f"\nHedefler: {tps}" if tps else "")
                + (f"\nGüven {self.confidence:.0f}/100" if self.confidence else "")
                + (f"\n{self.reason}" if self.reason else "")
                + f"\nSüre: {self.minutes_left:.0f} dk")


class ApprovalBook:
    """Onay istekleri (motor, arayüz ve Telegram iş parçacıklarından güvenle kullanılır)."""

    def __init__(self, timeout_min: float = 10.0):
        self.timeout_s = max(60.0, timeout_min * 60)
        self._lock = threading.Lock()
        self._by_key: dict[str, ApprovalRequest] = {}
        self._by_id: dict[str, ApprovalRequest] = {}

    def gate(self, key: str, make) -> tuple[str, ApprovalRequest | None, bool]:
        """(durum, istek, yeni_mi). `make()` yeni ApprovalRequest alanlarını (dict) döndürür."""
        with self._lock:
            req = self._by_key.get(key)
            if req is not None and req.expired:
                req.state = EXPIRED
            if req is None or req.state in (EXPIRED, DONE):
                if req is not None and req.state == EXPIRED:
                    self._by_key.pop(key, None)
                    return EXPIRED, req, False
                fields = make()
                req = ApprovalRequest(request_id=uuid.uuid4().hex[:10], key=key,
                                      expires_at=time.time() + self.timeout_s, **fields)
                self._by_key[key] = req
                self._by_id[req.request_id] = req
                return PENDING, req, True
            return req.state, req, False

    def resolve(self, request_id: str, approve: bool, by: str = "") -> tuple[bool, str, ApprovalRequest | None]:
        with self._lock:
            req = self._by_id.get(request_id)
            if req is None:
                return False, "İstek bulunamadı (bot yeniden başlatılmış olabilir).", None
            if req.expired:
                req.state = EXPIRED
            if req.state != PENDING:
                return False, f"Bu istek zaten sonuçlandı: {STATE_TR.get(req.state, req.state)}.", req
            req.state = APPROVED if approve else REJECTED
            req.resolved_by = by
            return True, ("✅ Onaylandı — emir güncel fiyatla yeniden kontrol edilip gönderilecek."
                          if approve else "❌ Reddedildi — bu işlem açılmayacak."), req

    def finish(self, key: str):
        with self._lock:
            req = self._by_key.pop(key, None)
            if req is not None and req.state == APPROVED:
                req.state = DONE

    def approved_external(self) -> list[ApprovalRequest]:
        with self._lock:
            return [r for r in self._by_key.values() if r.state == APPROVED and r.payload is not None]

    def pending(self) -> list[ApprovalRequest]:
        with self._lock:
            return [r for r in self._by_key.values() if r.state == PENDING and not r.expired]
