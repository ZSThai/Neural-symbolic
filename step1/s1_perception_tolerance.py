"""Bước 1c — Dung sai perception: lớp luật cần khái niệm chính xác đến đâu để còn giá trị?

Hai thí nghiệm, chỉ trên train chính thức, CV theo bệnh nhân, luật luôn học trên chú giải sạch của fold train:

(1) Mô phỏng bộ dự đoán khái niệm: với mỗi nhóm (shape, margin), logits = κ·onehot(v) + N(0,1),
    q = softmax(logits). Đầu vào luật ở fold test là:
      - fuzzy/soft : mức thuộc = q (giữ nguyên bất định)
      - crisp      : onehot(argmax q) (quyết định cứng trước khi suy luận)
    Quét κ để vẽ AUROC theo độ chính xác khái niệm. Đây là mô hình nhiễu lý tưởng (độc lập, đối xứng),
    không phải sai số thật của CNN.

(2) Nhiễu 'kiểu con người': với tổn thương có cả CC và MLO, thay chú giải của view này bằng
    chú giải của view kia (một lần đọc khác của cùng tổn thương). Đo AUROC trên cùng các dòng.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from s1_common import RESULTS, load_mass, load_ontology, memberships, patient_folds
from s1_fuzzy_rules import FuzzyGridRules

OUT = RESULTS / "1c_perception_tolerance"
KAPPAS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0]
SEEDS = [0, 1, 2]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
B_REF = 0.82  # CNN ROI ở lượt phát triển trước; chỉ để định hướng, Bước 3 sẽ đo lại công bằng


def simulate(M_true: np.ndarray, kappa: float, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trả về (q soft, onehot argmax, đúng/sai) cho một nhóm khái niệm."""
    n, k = M_true.shape
    t = M_true / M_true.sum(1, keepdims=True)
    v = np.array([rng.choice(k, p=row) for row in t])          # giá trị 'thật' được lấy mẫu từ chú giải
    logits = kappa * np.eye(k)[v] + rng.standard_normal((n, k))
    q = np.exp(logits - logits.max(1, keepdims=True))
    q /= q.sum(1, keepdims=True)
    hard = np.eye(k)[q.argmax(1)]
    correct = M_true[np.arange(n), q.argmax(1)] > 0             # argmax nằm trong tập giá trị đã chú giải
    return q, hard, correct


def exp_simulation(df, onto):
    Ms, Mm = memberships(df, onto, "shape"), memberships(df, onto, "margin")
    y = df.y.to_numpy()
    rows = []
    for seed in SEEDS:
        fold = patient_folds(df, 5, seed=seed)
        clean = np.zeros(len(df))
        for k in range(5):
            tr, te = fold != k, fold == k
            clean[te] = FuzzyGridRules().fit(Ms[tr], Mm[tr], y[tr]).predict_proba(Ms[te], Mm[te])
        a_clean = roc_auc_score(y, clean)
        for kappa in KAPPAS:
            rng = np.random.default_rng(1000 * seed + int(kappa * 10))
            qs, hs, cs = simulate(Ms, kappa, rng)
            qm, hm, cm = simulate(Mm, kappa, rng)
            p_soft, p_hard, p_soft_min = np.zeros(len(df)), np.zeros(len(df)), np.zeros(len(df))
            for k in range(5):
                tr, te = fold != k, fold == k
                m = FuzzyGridRules(tnorm="product").fit(Ms[tr], Mm[tr], y[tr])
                p_soft[te] = m.predict_proba(qs[te], qm[te])
                p_hard[te] = m.predict_proba(hs[te], hm[te])
                m2 = FuzzyGridRules(tnorm="min").fit(Ms[tr], Mm[tr], y[tr])
                p_soft_min[te] = m2.predict_proba(qs[te], qm[te])
            rows.append({"seed": seed, "kappa": kappa, "shape_acc": cs.mean(), "margin_acc": cm.mean(),
                         "concept_acc_mean": (cs.mean() + cm.mean()) / 2, "auroc_clean": a_clean,
                         "auroc_fuzzy_soft_product": roc_auc_score(y, p_soft),
                         "auroc_fuzzy_soft_min": roc_auc_score(y, p_soft_min),
                         "auroc_crisp_argmax": roc_auc_score(y, p_hard)})
    return pd.DataFrame(rows)


def exp_cross_view(df, onto):
    """Thay chú giải bằng chú giải của view còn lại cùng tổn thương."""
    Ms, Mm = memberships(df, onto, "shape"), memberships(df, onto, "margin")
    y = df.y.to_numpy()
    partner = np.full(len(df), -1)
    for _, idx in df.groupby("lesion_key").indices.items():
        if len(idx) == 2 and set(df.view.iloc[idx]) == {"CC", "MLO"}:
            partner[idx[0]], partner[idx[1]] = idx[1], idx[0]
    has = partner >= 0
    chg_s = (Ms[has] != Ms[partner[has]]).any(1)
    chg_m = (Mm[has] != Mm[partner[has]]).any(1)
    changed = chg_s | chg_m
    rows = []
    for seed in range(5):
        fold = patient_folds(df, 5, seed=seed)
        rng = np.random.default_rng(seed)
        # Đối chứng: đổi ngẫu nhiên đúng cùng tỉ lệ dòng như con người, sang giá trị khác theo tần suất train
        Ms_r, Mm_r = random_relabel(Ms, chg_s.mean(), rng), random_relabel(Mm, chg_m.mean(), rng)
        p_own, p_swap, p_rand = np.zeros(len(df)), np.zeros(len(df)), np.zeros(len(df))
        for k in range(5):
            tr, te = fold != k, fold == k
            m = FuzzyGridRules().fit(Ms[tr], Mm[tr], y[tr])
            p_own[te] = m.predict_proba(Ms[te], Mm[te])
            te_p = te & has
            p_swap[te_p] = m.predict_proba(Ms[partner[te_p]], Mm[partner[te_p]])
            p_rand[te] = m.predict_proba(Ms_r[te], Mm_r[te])
        rows.append({"seed": seed, "rows_with_partner": int(has.sum()), "rows_concept_changed": int(changed.sum()),
                     "shape_change_rate": float(chg_s.mean()), "margin_change_rate": float(chg_m.mean()),
                     "auroc_own_annotation": roc_auc_score(y[has], p_own[has]),
                     "auroc_other_view_annotation": roc_auc_score(y[has], p_swap[has]),
                     "auroc_random_relabel_same_rate": roc_auc_score(y[has], p_rand[has])})
    return pd.DataFrame(rows)


def random_relabel(M, rate, rng):
    """Đổi đúng tỉ lệ `rate` số dòng sang một giá trị KHÁC, chọn ngẫu nhiên theo tần suất train."""
    M = M.copy()
    freq = M.sum(0) / M.sum()
    for i in np.flatnonzero(rng.random(len(M)) < rate):
        w = freq * (M[i] == 0)
        M[i] = 0
        M[i, rng.choice(len(w), p=w / w.sum())] = 1
    return M


def plot(sim, path, human_acc):
    g = sim.groupby("kappa").mean(numeric_only=True).reset_index()
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.plot(g.concept_acc_mean, g.auroc_fuzzy_soft_product, color=C1, lw=2, marker="o", ms=5, label="Luật mờ, đầu vào soft (product)")
    ax.plot(g.concept_acc_mean, g.auroc_crisp_argmax, color=C2, lw=2, marker="s", ms=5, label="Cùng luật, đầu vào crisp (argmax)")
    ax.axhline(g.auroc_clean.mean(), color=INK, lw=1, ls=":")
    ax.text(g.concept_acc_mean.min() + 0.01, g.auroc_clean.mean() + 0.004, f"A: luật trên chú giải sạch = {g.auroc_clean.mean():.3f}", fontsize=8, color=INK)
    ax.axhline(B_REF, color=INK2, lw=1, ls="--")
    ax.text(g.concept_acc_mean.min() + 0.01, B_REF + 0.004, "B tham khảo ≈0,82 (CNN ROI, lượt trước; Bước 3 đo lại)", fontsize=8, color=INK2)
    ax.axvspan(human_acc[0], human_acc[1], color=C3, alpha=0.15, lw=0)
    ax.text(human_acc[0] - 0.006, 0.60, "đồng thuận CC↔MLO\ncủa chú giải chuyên gia", fontsize=7.5, color=INK2, ha="right")
    ax.set_xlabel("Độ chính xác khái niệm mô phỏng (trung bình shape & margin)", fontsize=9, color=INK2)
    ax.set_ylabel("AUROC OOF", fontsize=9, color=INK2)
    ax.set_ylim(0.5, 0.92)
    ax.grid(color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0.01, 0.70))
    ax.set_title("Dung sai perception của lớp luật (Z_morph, train, 5-fold theo bệnh nhân × 3 seeds)", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def required_accuracy(sim, col, target):
    g = sim.groupby("kappa").mean(numeric_only=True).sort_values("concept_acc_mean")
    x, y = g.concept_acc_mean.to_numpy(), g[col].to_numpy()
    above = np.flatnonzero(y >= target)
    if len(above) == 0:
        return None
    j = above[0]
    if j == 0:
        return float(x[0])
    return float(x[j - 1] + (target - y[j - 1]) * (x[j] - x[j - 1]) / (y[j] - y[j - 1]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    onto = load_ontology()
    df = load_mass("train").reset_index(drop=True)
    sim = exp_simulation(df, onto)
    cv = exp_cross_view(df, onto)
    sim.to_csv(OUT / "simulation.csv", index=False)
    cv.to_csv(OUT / "cross_view_substitution.csv", index=False)
    agree = pd.read_csv(RESULTS / "1a_audit" / "cross_view_agreement.csv").set_index("field").agreement
    human = (float(agree["shape_set"]), float(agree["margin_set"]))
    plot(sim, OUT / "perception_tolerance.png", (min(human), max(human)))
    summary = {
        "auroc_clean_mean": float(sim.auroc_clean.mean()),
        "B_reference_used": B_REF,
        "concept_acc_needed_for_B_fuzzy_soft": required_accuracy(sim, "auroc_fuzzy_soft_product", B_REF),
        "concept_acc_needed_for_B_crisp": required_accuracy(sim, "auroc_crisp_argmax", B_REF),
        "cross_view": cv.mean(numeric_only=True).to_dict(),
        "human_cross_view_agreement_shape_margin": human,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(sim.groupby("kappa").mean(numeric_only=True).round(4).to_string())
    print(cv.round(4).to_string(index=False))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
