"""Bước 1 — tiện ích chung: đọc CSV chính thức, chuẩn hóa từ vựng, mã hóa Z, chia CV theo bệnh nhân.

Viết mới hoàn toàn cho claude_research; không dùng mã của các phiên bản trước.
Chỉ đọc 4 CSV chính thức của TCIA. Mọi mô hình hóa trong Bước 1 chỉ dùng partition TRAIN chính thức.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data" / "official_csv"
ONTOLOGY = HERE / "ontology" / "birads_mass_lexicon.json"
RESULTS = HERE / "results"

SEP = "-"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_ontology() -> dict:
    return json.loads(ONTOLOGY.read_text(encoding="utf-8"))


def load_mass(split: str = "train") -> pd.DataFrame:
    """Đọc CSV mass chính thức và chuẩn hóa tên cột. split ∈ {train, test}."""
    path = DATA / f"mass_case_description_{split}_set.csv"
    df = pd.read_csv(path)
    df = df.rename(columns={
        "breast_density": "density", "breast density": "density",
        "left or right breast": "side", "image view": "view",
        "abnormality id": "abn_id", "mass shape": "shape_raw", "mass margins": "margin_raw",
    })
    df["split"] = split
    df["y"] = (df["pathology"] == "MALIGNANT").astype(int)
    df["bwc"] = (df["pathology"] == "BENIGN_WITHOUT_CALLBACK").astype(int)
    # Khóa tổn thương giả định: cùng bệnh nhân, cùng bên vú, cùng abnormality id.
    # Đây là giả thuyết ghép CC/MLO, được kiểm chứng gián tiếp qua độ khớp pathology.
    df["lesion_key"] = df["patient_id"] + "|" + df["side"] + "|" + df["abn_id"].astype(str)
    df["shape_set"] = df["shape_raw"].apply(_split_values)
    df["margin_set"] = df["margin_raw"].apply(_split_values)
    return df


def _split_values(v) -> tuple:
    if pd.isna(v):
        return tuple()
    return tuple(sorted(set(str(v).strip().split(SEP))))


def vocab(onto: dict, group: str) -> list[str]:
    return list(onto["groups"][group]["values"].keys())


def multi_hot(df: pd.DataFrame, onto: dict, group: str, add_missing: bool = True) -> pd.DataFrame:
    """Mã hóa đa trị: mỗi giá trị được liệt kê có mức thuộc 1 (ngữ nghĩa khả năng/possibilistic)."""
    col = f"{group}_set"
    vals = vocab(onto, group)
    out = pd.DataFrame(0.0, index=df.index, columns=[f"{group}={v}" for v in vals])
    unknown = set()
    for i, s in df[col].items():
        for v in s:
            if v in vals:
                out.at[i, f"{group}={v}"] = 1.0
            else:
                unknown.add(v)
    if unknown:
        raise ValueError(f"Giá trị ngoài ontology trong {group}: {sorted(unknown)}")
    if add_missing:
        out[f"{group}=MISSING"] = (df[col].apply(len) == 0).astype(float)
    return out


def memberships(df: pd.DataFrame, onto: dict, group: str, semantics: str = "possibilistic") -> np.ndarray:
    """Ma trận mức thuộc n × (|V|+1) cho một nhóm khái niệm (cột cuối là MISSING).

    possibilistic: μ=1 cho mọi giá trị được liệt kê (chú giải 'vừa A vừa B').
    probabilistic: μ=1/k, tổng bằng 1 (chú giải là sự phân vân giữa k giá trị).
    """
    m = multi_hot(df, onto, group, add_missing=True).to_numpy()
    if semantics == "probabilistic":
        m = m / m.sum(axis=1, keepdims=True)
    elif semantics != "possibilistic":
        raise ValueError(semantics)
    return m


def expert_score(df: pd.DataFrame, onto: dict) -> np.ndarray:
    """Điểm nghi ngờ thuần tri thức (không học): max rank margin + max rank shape.

    Nguyên tắc 'đặc điểm nghi ngờ nhất quyết định'. Giá trị thiếu nhận rank trung vị của nhóm.
    """
    score = np.zeros(len(df))
    for group in ("shape", "margin"):
        ranks = {k: v["suspicion_rank"] for k, v in onto["groups"][group]["values"].items()}
        fill = float(np.median(list(ranks.values())))
        score += df[f"{group}_set"].apply(lambda s: max((ranks[v] for v in s), default=fill)).to_numpy(float)
    return score


def feature_matrix(df: pd.DataFrame, onto: dict, zset: str) -> pd.DataFrame:
    """Các bộ Z dùng cho trần biểu tượng.

    Z_morph      : shape + margin (đa trị, có MISSING)  — Z chính, nhìn thấy được trên tổn thương
    Z_morph_den  : + breast density (one-hot)           — ngữ cảnh vú
    Z_morph_den_sub : + subtlety (one-hot)              — chỉ để phân tích, mô tả người đọc
    Z_assess     : chỉ assessment (one-hot)             — tham chiếu RÒ RỈ, không phải Z hợp lệ
    Z_morph_assess : Z_morph + assessment               — tham chiếu RÒ RỈ
    """
    parts = []
    if zset.startswith("Z_morph"):
        parts += [multi_hot(df, onto, "shape"), multi_hot(df, onto, "margin")]
    if zset in ("Z_morph_den", "Z_morph_den_sub"):
        parts.append(pd.get_dummies(df["density"].astype(int), prefix="density").astype(float))
    if zset == "Z_morph_den_sub":
        parts.append(pd.get_dummies(df["subtlety"].astype(int), prefix="subtlety").astype(float))
    if zset in ("Z_assess", "Z_morph_assess"):
        parts.append(pd.get_dummies(df["assessment"].astype(int), prefix="assess").astype(float))
    if not parts:
        raise ValueError(zset)
    return pd.concat(parts, axis=1)


def patient_folds(df: pd.DataFrame, n_splits: int = 5, seed: int = 0) -> np.ndarray:
    """Gán fold theo bệnh nhân, phân tầng theo y. Mọi view/tổn thương của một người cùng fold."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = np.full(len(df), -1)
    for k, (_, te) in enumerate(sgkf.split(np.zeros(len(df)), df["y"], groups=df["patient_id"])):
        fold[te] = k
    assert (fold >= 0).all()
    return fold


def metrics(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> dict:
    yhat = (p >= thr).astype(int)
    return {
        "auroc": roc_auc_score(y, p),
        "accuracy": accuracy_score(y, yhat),
        "balanced_accuracy": balanced_accuracy_score(y, yhat),
        "brier": brier_score_loss(y, np.clip(p, 0, 1)),
    }


def patient_bootstrap_ci(df: pd.DataFrame, p: np.ndarray, metric: str = "auroc", n_boot: int = 2000, seed: int = 20261028):
    """CI 95% bằng bootstrap theo bệnh nhân trên dự đoán OOF cố định."""
    rng = np.random.default_rng(seed)
    pats = df["patient_id"].to_numpy()
    uniq = np.unique(pats)
    idx_by_pat = {u: np.flatnonzero(pats == u) for u in uniq}
    y = df["y"].to_numpy()
    vals = []
    for _ in range(n_boot):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_pat[u] for u in draw])
        if y[idx].min() == y[idx].max():
            continue
        vals.append(metrics(y[idx], p[idx])[metric])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
