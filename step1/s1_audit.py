"""Bước 1a — Audit dữ liệu từ 4 CSV chính thức (chỉ thống kê mô tả; test chỉ đếm, không mô hình hóa)."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from s1_common import DATA, RESULTS, load_mass, load_ontology, sha256

OUT = RESULTS / "1a_audit"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C1, C2 = "#2a78d6", "#eb6834"


def cohort_table(tr, te):
    rows = []
    for name, d in (("train", tr), ("test", te)):
        nviews = d.groupby("lesion_key")["view"].nunique()
        rows.append({
            "split": name, "rows_lesion_view": len(d), "patients": d.patient_id.nunique(),
            "images": d["image file path"].nunique(), "putative_lesions": d.lesion_key.nunique(),
            "lesions_with_CC_and_MLO": int((nviews == 2).sum()), "lesions_single_view": int((nviews == 1).sum()),
            "malignant_rows": int(d.y.sum()), "benign_rows": int(((d.y == 0) & (d.bwc == 0)).sum()),
            "benign_without_callback_rows": int(d.bwc.sum()),
            "malignant_rate": round(d.y.mean(), 4),
        })
    return pd.DataFrame(rows)


def vocab_table(tr, onto):
    rows = []
    for group in ("shape", "margin"):
        spec = onto["groups"][group]["values"]
        exploded = tr[[f"{group}_set", "y"]].explode(f"{group}_set").dropna()
        for v, d in exploded.groupby(f"{group}_set"):
            rows.append({"group": group, "cbis_value": v, "birads5": spec[v]["birads5"], "parent": spec[v]["parent"],
                         "suspicion_rank_prior": spec[v]["suspicion_rank"], "n_rows": len(d),
                         "malignant_rate": round(d.y.mean(), 4)})
        n_multi = int((tr[f"{group}_set"].apply(len) > 1).sum())
        n_miss = int((tr[f"{group}_set"].apply(len) == 0).sum())
        rows.append({"group": group, "cbis_value": "(đa trị)", "n_rows": n_multi})
        rows.append({"group": group, "cbis_value": "(thiếu)", "n_rows": n_miss,
                     "malignant_rate": round(tr.loc[tr[f"{group}_set"].apply(len) == 0, "y"].mean(), 4) if n_miss else None})
    return pd.DataFrame(rows)


def suspicion_level(s, onto, group):
    """Mức nghi ngờ cao nhất của một ô (nguyên tắc 'đặc điểm nghi ngờ nhất'); None nếu thiếu.

    margin: 0 = benign-appearing (circumscribed), 1 = indeterminate (obscured), 2 = suspicious (micro/indistinct/spiculated)
    shape : 0 = rank 0-1 (round/oval/lobulated, asymmetry, lymph node), 1 = rank 2 (irregular, architectural distortion)
    """
    ranks = [onto["groups"][group]["values"][v]["suspicion_rank"] for v in s]
    if not ranks:
        return None
    r = max(ranks)
    return min(r, 2) if group == "margin" else int(r >= 2)


def cross_view(tr, onto):
    both = tr.groupby("lesion_key").filter(lambda d: set(d.view) >= {"CC", "MLO"} and len(d) == 2).copy()
    both["margin_level"] = both.margin_set.apply(lambda s: suspicion_level(s, onto, "margin"))
    both["shape_level"] = both.shape_set.apply(lambda s: suspicion_level(s, onto, "shape"))
    rows = []
    for col in ("pathology", "shape_set", "margin_set", "shape_level", "margin_level", "assessment", "subtlety", "density"):
        agree = both.groupby("lesion_key")[col].agg(lambda s: s.iloc[0] == s.iloc[1])
        rows.append({"field": col, "lesions": len(agree), "agreement": round(agree.mean(), 4)})
    # ma trận nhầm lẫn margin giữa hai view (chỉ khi mỗi view đúng một giá trị)
    piv = both.pivot_table(index="lesion_key", columns="view", values="margin_raw", aggfunc="first").dropna()
    piv = piv[~piv.CC.str.contains("-") & ~piv.MLO.str.contains("-")]
    conf = pd.crosstab(piv.CC, piv.MLO)
    piv_s = both.pivot_table(index="lesion_key", columns="view", values="shape_raw", aggfunc="first").dropna()
    piv_s = piv_s[~piv_s.CC.str.contains("-") & ~piv_s.MLO.str.contains("-")]
    conf_s = pd.crosstab(piv_s.CC, piv_s.MLO)
    return pd.DataFrame(rows), conf, conf_s


def plot_rates(vt, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [8, 5]})
    for ax, group, title in ((axes[0], "shape", "Shape (CBIS-DDSM)"), (axes[1], "margin", "Margins (CBIS-DDSM)")):
        d = vt[(vt.group == group) & ~vt.cbis_value.str.startswith("(")].sort_values("suspicion_rank_prior")
        x = np.arange(len(d))
        ax.bar(x, d.malignant_rate, width=0.6, color=C1, zorder=2)
        for xi, (r, n) in enumerate(zip(d.malignant_rate, d.n_rows)):
            ax.text(xi, r + 0.02, f"{r:.0%}\nn={n}", ha="center", va="bottom", fontsize=8, color=INK2)
        ax.set_xticks(x, [v.replace("_", " ").title() for v in d.cbis_value], fontsize=7.5, color=INK, rotation=30, ha="right", rotation_mode="anchor")
        ax.set_ylim(0, 1.18)
        ax.set_yticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"], fontsize=8, color=INK2)
        ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
    axes[0].set_ylabel("Tỉ lệ ác tính (train)", fontsize=9, color=INK2)
    fig.suptitle("Tỉ lệ ác tính theo khái niệm, sắp theo thứ hạng nghi ngờ BI-RADS (prior)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    onto = load_ontology()
    tr, te = load_mass("train"), load_mass("test")
    hashes = {p.name: sha256(p) for p in sorted(DATA.glob("*.csv"))}
    ct = cohort_table(tr, te)
    vt = vocab_table(tr, onto)
    cv, conf_margin, conf_shape = cross_view(tr, onto)
    leak = pd.crosstab(tr.assessment, tr.pathology)
    calc = {s: len(pd.read_csv(DATA / f"calc_case_description_{s}_set.csv")) for s in ("train", "test")}
    summary = {
        "input_sha256": hashes,
        "patient_overlap_train_test": len(set(tr.patient_id) & set(te.patient_id)),
        "calc_rows_not_used": calc,
        "multi_valued_shape_rows_train": int((tr.shape_set.apply(len) > 1).sum()),
        "multi_valued_margin_rows_train": int((tr.margin_set.apply(len) > 1).sum()),
        "missing_shape_rows_train": int((tr.shape_set.apply(len) == 0).sum()),
        "missing_margin_rows_train": int((tr.margin_set.apply(len) == 0).sum()),
        "rows_otherfinding_only_train": int(tr.shape_set.apply(
            lambda s: len(s) > 0 and all(onto["groups"]["shape"]["values"][v]["parent"] == "OtherFinding" for v in s)).sum()),
    }
    ct.to_csv(OUT / "cohort.csv", index=False)
    vt.to_csv(OUT / "vocabulary_vs_birads.csv", index=False)
    cv.to_csv(OUT / "cross_view_agreement.csv", index=False)
    conf_margin.to_csv(OUT / "cross_view_margin_confusion.csv")
    conf_shape.to_csv(OUT / "cross_view_shape_confusion.csv")
    leak.to_csv(OUT / "assessment_vs_pathology_train.csv")
    (OUT / "audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_rates(vt, OUT / "malignancy_by_concept.png")
    print(ct.to_string(index=False)); print(cv.to_string(index=False)); print(json.dumps(summary, indent=1))
    print(conf_margin); print(conf_shape)


if __name__ == "__main__":
    main()
