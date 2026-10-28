"""Hệ luật mờ lưới Shape × Margin với làm trơn phân cấp — viết mới cho Bước 1.

Mỗi luật r = (s, m):  IF shape is s AND margin is m THEN malignant với độ c_{s,m}.
  - Độ kích hoạt α_r(x) = T(μ_s(x), μ_m(x)), T là t-norm product hoặc min (Gödel).
  - Hệ quả c_{s,m} ước lượng bằng hỗ trợ mờ trên train, làm trơn về luật cha c_m (chỉ margin),
    và c_m làm trơn về prior π. Đây là làm trơn Bayes phân cấp theo ontology: cell → margin → prior.
  - Suy luận: p(x) = Σ_r α_r c_r / Σ_r α_r ; nếu không luật nào kích hoạt thì p = π.

Mô hình minh bạch hoàn toàn: điểm của một mẫu là trung bình có trọng số của các luật kích hoạt,
nên mọi dự đoán được phân rã chính xác thành đóng góp của từng luật.
"""
from __future__ import annotations

import numpy as np


class FuzzyGridRules:
    def __init__(self, smoothing: float = 5.0, tnorm: str = "product"):
        self.a = smoothing
        self.tnorm = tnorm

    def _fire(self, Ms: np.ndarray, Mm: np.ndarray) -> np.ndarray:
        if self.tnorm == "product":
            return Ms[:, :, None] * Mm[:, None, :]
        if self.tnorm == "min":
            return np.minimum(Ms[:, :, None], Mm[:, None, :])
        raise ValueError(self.tnorm)

    def fit(self, Ms: np.ndarray, Mm: np.ndarray, y: np.ndarray):
        y = y.astype(float)
        self.prior_ = y.mean()
        wm = Mm.sum(0)
        self.c_margin_ = (Mm.T @ y + self.a * self.prior_) / (wm + self.a)
        alpha = self._fire(Ms, Mm)                      # n × S × M
        self.support_ = alpha.sum(0)                     # S × M (hỗ trợ mờ)
        pos = np.einsum("nsm,n->sm", alpha, y)
        self.c_ = (pos + self.a * self.c_margin_[None, :]) / (self.support_ + self.a)
        return self

    def predict_proba(self, Ms: np.ndarray, Mm: np.ndarray) -> np.ndarray:
        alpha = self._fire(Ms, Mm)
        num = np.einsum("nsm,sm->n", alpha, self.c_)
        den = alpha.sum((1, 2))
        out = np.full(len(Ms), self.prior_)
        ok = den > 1e-12
        out[ok] = num[ok] / den[ok]
        return out

    def explain(self, ms: np.ndarray, mm: np.ndarray, shape_names, margin_names, top: int = 5):
        """Phân rã một dự đoán: danh sách (luật, α, c, phần đóng góp α·c/Σα)."""
        alpha = self._fire(ms[None], mm[None])[0]
        den = alpha.sum()
        rows = []
        for s in range(alpha.shape[0]):
            for m in range(alpha.shape[1]):
                if alpha[s, m] > 0:
                    rows.append((shape_names[s], margin_names[m], float(alpha[s, m]), float(self.c_[s, m]),
                                 float(alpha[s, m] * self.c_[s, m] / den)))
        rows.sort(key=lambda r: -r[2])
        return rows[:top]

    def rule_table(self, shape_names, margin_names, min_support: float = 1.0):
        rows = []
        for s, sn in enumerate(shape_names):
            for m, mn in enumerate(margin_names):
                if self.support_[s, m] >= min_support:
                    rows.append({"shape": sn, "margin": mn, "support": float(self.support_[s, m]),
                                 "p_malignant": float(self.c_[s, m])})
        return sorted(rows, key=lambda r: -r["support"])
