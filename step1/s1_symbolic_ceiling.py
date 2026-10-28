"""Bước 1b — Trần biểu tượng A: Y dự đoán được bao nhiêu từ các khái niệm chuyên gia Z.

Chỉ partition TRAIN chính thức. CV 5-fold theo bệnh nhân × 5 lần lặp seed.
So sánh: không học (prior), chỉ tri thức BI-RADS (không học), LR, GBM, cây crisp sâu 3, luật mờ lưới.
Kèm các bộ Z rò rỉ (assessment) để định lượng vì sao phải loại chúng.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mutual_info_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier, export_text

from s1_common import (RESULTS, expert_score, feature_matrix, load_mass, load_ontology, memberships, metrics,
                       patient_bootstrap_ci, patient_folds, vocab)
from s1_fuzzy_rules import FuzzyGridRules

OUT = RESULTS / "1b_symbolic_ceiling"
N_REPEATS, N_SPLITS = 5, 5
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C1, C2 = "#2a78d6", "#eb6834"


def oof_predict(df, onto, model_name, zset, fold):
    p = np.zeros(len(df))
    y = df.y.to_numpy()
    if model_name == "Expert_BIRADS_rank (không học)":
        s = expert_score(df, onto)
        return (s - s.min()) / (s.max() - s.min())  # điểm thứ hạng; chỉ AUROC có nghĩa
    if model_name.startswith("FuzzyGridRules"):
        sem = "probabilistic" if "prob" in model_name else "possibilistic"
        Ms, Mm = memberships(df, onto, "shape", sem), memberships(df, onto, "margin", sem)
        for k in np.unique(fold):
            tr, te = fold != k, fold == k
            m = FuzzyGridRules(smoothing=5.0).fit(Ms[tr], Mm[tr], y[tr])
            p[te] = m.predict_proba(Ms[te], Mm[te])
        return p
    X = feature_matrix(df, onto, zset).to_numpy()
    for k in np.unique(fold):
        tr, te = fold != k, fold == k
        if model_name == "Prior (đa số)":
            p[te] = y[tr].mean()
            continue
        m = {
            "LogReg": lambda: LogisticRegression(C=1.0, max_iter=2000),
            "GBM": lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=0),
            "CrispTree_depth3": lambda: DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=0),
        }[model_name]()
        m.fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


PLAN = [
    ("Prior (đa số)", "Z_morph"),
    ("Expert_BIRADS_rank (không học)", "Z_morph"),
    ("CrispTree_depth3", "Z_morph"),
    ("FuzzyGridRules (possibilistic)", "Z_morph"),
    ("FuzzyGridRules (probabilistic)", "Z_morph"),
    ("LogReg", "Z_morph"),
    ("GBM", "Z_morph"),
    ("LogReg", "Z_morph_den"),
    ("GBM", "Z_morph_den"),
    ("LogReg", "Z_morph_den_sub"),
    ("LogReg", "Z_assess"),
    ("LogReg", "Z_morph_assess"),
    ("GBM", "Z_morph_assess"),
]


def run_plan(df, onto, tag):
    rows, keep = [], {}
    for r in range(N_REPEATS):
        fold = patient_folds(df, N_SPLITS, seed=r)
        for model, z in PLAN:
            p = oof_predict(df, onto, model, z, fold)
            m = metrics(df.y.to_numpy(), p)
            if model.startswith("Expert") or model.startswith("Prior"):
                m["accuracy"] = m["balanced_accuracy"] = m["brier"] = np.nan
            rows.append({"cohort": tag, "model": model, "zset": z, "repeat": r, **m})
            if r == 0:
                keep[(model, z)] = p
    res = pd.DataFrame(rows)
    agg = res.groupby(["cohort", "model", "zset"], sort=False).agg(
        auroc_mean=("auroc", "mean"), auroc_sd=("auroc", "std"),
        acc_mean=("accuracy", "mean"), acc_sd=("accuracy", "std"),
        bacc_mean=("balanced_accuracy", "mean"), brier_mean=("brier", "mean")).reset_index()
    cis = []
    for (model, z), p in keep.items():
        lo, hi = patient_bootstrap_ci(df, p, "auroc")
        cis.append({"model": model, "zset": z, "auroc_ci95_low_rep0": lo, "auroc_ci95_high_rep0": hi})
    agg = agg.merge(pd.DataFrame(cis), on=["model", "zset"], how="left")
    return res, agg


def information(df, onto):
    def key(s):
        return "+".join(s) if s else "MISSING"
    y = df.y.to_numpy()
    sh, mg = df.shape_set.apply(key), df.margin_set.apply(key)
    both = sh + "|" + mg
    h_y = mutual_info_score(y, y)
    out = {"H(Y)_nats": h_y}
    for name, z in (("shape", sh), ("margin", mg), ("shape×margin", both), ("assessment", df.assessment.astype(str))):
        out[f"I(Y;{name})_nats"] = mutual_info_score(y, z)
        out[f"I/H(Y) {name}"] = mutual_info_score(y, z) / h_y
    # trần lạc quan in-sample: phân lớp theo đa số trong từng ô shape×margin
    cell = pd.DataFrame({"c": both, "y": y}).groupby("c").y.agg(["mean", "size"])
    acc = (np.maximum(cell["mean"], 1 - cell["mean"]) * cell["size"]).sum() / len(y)
    out["optimistic_in_sample_cell_majority_accuracy"] = acc
    out["n_distinct_cells"] = int(len(cell))
    return out


def export_rules(df, onto):
    Ms, Mm = memberships(df, onto, "shape"), memberships(df, onto, "margin")
    sn = vocab(onto, "shape") + ["MISSING"]
    mn = vocab(onto, "margin") + ["MISSING"]
    m = FuzzyGridRules(smoothing=5.0).fit(Ms, Mm, df.y.to_numpy())
    rt = pd.DataFrame(m.rule_table(sn, mn, min_support=1.0))
    X = feature_matrix(df, onto, "Z_morph")
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=0).fit(X, df.y)
    txt = export_text(tree, feature_names=list(X.columns), show_weights=True)
    # ví dụ phân rã một dự đoán đa trị
    i = int(np.flatnonzero((df.margin_set.apply(len) > 1).to_numpy())[0])
    ex = m.explain(Ms[i], Mm[i], sn, mn)
    expl = {"row": int(i), "shape": df.shape_raw.iloc[i], "margin": df.margin_raw.iloc[i], "y": int(df.y.iloc[i]),
            "p_malignant": float(m.predict_proba(Ms[i:i+1], Mm[i:i+1])[0]),
            "rules": [{"shape": a, "margin": b, "alpha": c, "consequent": d, "contribution": e} for a, b, c, d, e in ex]}
    return rt, txt, expl


def plot(agg, path):
    d = agg[agg.cohort == "all_mass_train"].copy()
    d["label"] = d.model + "  ·  " + d.zset
    d = d.iloc[::-1]
    leak = d.zset.str.contains("assess")
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    y = np.arange(len(d))
    ax.barh(y, d.auroc_mean - 0.5, left=0.5, height=0.55, color=np.where(leak, C2, C1), zorder=2)
    for yi, (m, lo, hi) in enumerate(zip(d.auroc_mean, d.auroc_ci95_low_rep0, d.auroc_ci95_high_rep0)):
        ax.plot([lo, hi], [yi, yi], color=INK2, lw=1.2, zorder=3)
        ax.text(max(hi, m) + 0.006, yi, f"{m:.3f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, d.label, fontsize=8, color=INK)
    ax.set_xlim(0.5, 1.0)
    ax.axvline(0.82, color=INK2, lw=1, ls="--", zorder=1)
    ax.text(0.826, len(d) - 1, "← CNN ROI ≈0,82 (lượt trước, chỉ định hướng)", fontsize=7.5, color=INK2, va="center", ha="left")
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("AUROC OOF (5×5 CV theo bệnh nhân, train chính thức); vạch = CI95 bootstrap bệnh nhân", fontsize=8.5, color=INK2)
    ax.set_title("Trần biểu tượng A: xanh = Z hợp lệ, cam = có assessment (rò rỉ, chỉ tham chiếu)", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    onto = load_ontology()
    df = load_mass("train").reset_index(drop=True)
    res_all, agg_all = run_plan(df, onto, "all_mass_train")
    # độ nhạy 1: bỏ BENIGN_WITHOUT_CALLBACK (nhãn lành không qua sinh thiết)
    d2 = df[df.bwc == 0].reset_index(drop=True)
    res2, agg2 = run_plan(d2, onto, "exclude_BWC")
    # độ nhạy 2: chỉ 'mass thật' theo BI-RADS 5 (loại các dòng chỉ có OtherFinding)
    other = {k for k, v in onto["groups"]["shape"]["values"].items() if v["parent"] == "OtherFinding"}
    d3 = df[df.shape_set.apply(lambda s: len(s) > 0 and not set(s) <= other)].reset_index(drop=True)
    res3, agg3 = run_plan(d3, onto, "true_masses_only")
    agg = pd.concat([agg_all, agg2, agg3])
    pd.concat([res_all, res2, res3]).to_csv(OUT / "ceiling_per_repeat.csv", index=False)
    agg.to_csv(OUT / "ceiling_summary.csv", index=False)
    info = information(df, onto)
    (OUT / "information.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    rt, txt, expl = export_rules(df, onto)
    rt.to_csv(OUT / "fuzzy_grid_rules_full_train.csv", index=False)
    (OUT / "crisp_tree_depth3.txt").write_text(txt, encoding="utf-8")
    (OUT / "example_explanation.json").write_text(json.dumps(expl, indent=2, ensure_ascii=False), encoding="utf-8")
    plot(agg, OUT / "symbolic_ceiling.png")
    pd.set_option("display.width", 200)
    print(agg.round(4).to_string(index=False))
    print(json.dumps(info, indent=1))
    print(txt)
    print(json.dumps(expl, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
