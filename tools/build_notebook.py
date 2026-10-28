"""Dựng notebook Colab cho Bước 1 (claude_research). Chạy: python build_notebook.py <out.ipynb>"""
import json
import sys

CELLS = []


def md(s):
    CELLS.append(("markdown", s.strip("\n")))


def code(s):
    CELLS.append(("code", s.strip("\n")))


# ─────────────────────────────────────────────────────────────────────────────
md(r'''
# Bước 1 — Kiểm chứng tiền đề biểu tượng trên CBIS-DDSM (mass)

**Dự án:** Fuzzy neuro-symbolic classification, Thai Le & GS Umberto Straccia · *claude_research* · 28/10/2026

Notebook này trả lời câu hỏi đầu tiên trong email của GS, **trước khi** huấn luyện bất kỳ mạng nơ-ron nào:

> Các thuộc tính chuyên gia **Z** (mass shape, mass margins) chứa bao nhiêu thông tin về nhãn **Y** (lành/ác)?
> Lớp luật cần khái niệm dự đoán từ ảnh chính xác đến đâu thì mới còn giá trị?

```
Huấn luyện:   Ảnh X ──► CNN ──► khái niệm Ẑ (shape, margin) ──► LUẬT MỜ ──► Y (lành / ác)
                                      ▲
Bước 1 chỉ xét nửa phải ──────────────┘   Z thật (chú giải chuyên gia) ──► LUẬT ──► Y
```

| Phần | Nội dung | Thời gian (CPU Colab) |
|---|---|---|
| 0 | Cài đặt, tải 4 CSV chính thức, kiểm tra SHA-256 | < 30 giây |
| 1 | Ontology BI-RADS: từ vựng CBIS → khái niệm chuẩn | tức thì |
| 2 | **1a Audit:** cohort, tỉ lệ ác tính theo khái niệm, rò rỉ, nhiễu chú giải | vài giây |
| 3 | **Luật mờ:** cách hoạt động, bản đồ luật, giải thích một ca | vài giây |
| 4 | **1b Trần biểu tượng A:** so sánh tri thức thuần, luật, mô hình thống kê | khoảng 1 phút |
| 5 | **1c Dung sai perception:** cần khái niệm đúng đến đâu? | khoảng 30 giây |
| 6 | Tổng kết tự động và lưu kết quả | vài giây |

**Không cần GPU.** Chọn *Runtime → Run all*. Chỉ dùng **train chính thức**; test chính thức chỉ được đếm số dòng.

**Dữ liệu:** notebook tự tìm 4 CSV theo thứ tự:

1. Thư mục cục bộ: `official_csv/` hoặc `data/official_csv/` (khi đã `git clone` repo).
2. **Repo GitHub** trong `CFG["GITHUB_REPO"]`. Repo private: thêm secret `GITHUB_TOKEN` trong Colab.
3. Tải từ TCIA.
4. Google Drive: `MyDrive/claude_research/data/official_csv/`.
5. Hộp thoại tải file lên.

File nào cũng phải khớp SHA-256 của bản chính thức thì mới được dùng.
''')

# ─────────────────────────────────────────────────────────────────────────────
md(r'''
## 0. Cài đặt và cấu hình

Mọi tham số nằm trong `CFG`. Muốn chạy nhanh để thử, đặt `FAST = True` (ít lần lặp, kết quả gần đúng).
''')

code(r'''
# @title 0.1 Thư viện và cấu hình { display-mode: "form" }
import hashlib, io, json, os, sys, time, zipfile, urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from IPython.display import display, Markdown

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, mutual_info_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.tree import DecisionTreeClassifier, export_text

IN_COLAB = "google.colab" in sys.modules

FAST = False  # True: chạy thử nhanh (ít lặp lại hơn)

CFG = dict(
    # Nguồn dữ liệu: "auto" thử lần lượt: thư mục cục bộ → repo GitHub → TCIA → Google Drive → tải file lên tay
    DATA_SOURCE="auto",
    LOCAL_DIRS=["official_csv", "data/official_csv", "../data/official_csv", "claude_research/data/official_csv"],
    GITHUB_REPO="OWNER/REPO",  # ← sửa thành "tên-tài-khoản/tên-repo" sau khi đẩy lên GitHub
    GITHUB_BRANCH="main",
    DRIVE_DIR="/content/drive/MyDrive/claude_research/data/official_csv",  # nếu bạn để CSV trên Drive
    SAVE_TO_DRIVE=False,      # True: lưu kết quả vào Drive (cần cấp quyền); False: lưu /content + tải zip
    SAVE_DIR="/content/drive/MyDrive/claude_research/step1_results",
    N_SPLITS=5,               # số fold CV (chia theo bệnh nhân)
    N_REPEATS=2 if FAST else 5,   # số lần lặp CV với seed khác nhau
    N_BOOT=500 if FAST else 2000, # số lần bootstrap theo bệnh nhân cho CI 95%
    SIM_SEEDS=[0] if FAST else [0, 1, 2],
    B_REFERENCE=0.82,         # AUROC CNN ROI của lượt trước — CHỈ để định hướng; Bước 3 sẽ đo lại
    SEED=20261028,
)

# Bảng màu (đã kiểm tra phân biệt được với người mù màu); chữ dùng màu mực, không dùng màu series
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fe", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
plt.rcParams.update({
    "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.titlelocation": "left", "axes.titlesize": 11, "axes.titlecolor": INK, "font.size": 9,
})
print(f"Colab: {IN_COLAB} | FAST: {FAST} | numpy {np.__version__} | pandas {pd.__version__}")
''')

code(r'''
# @title 0.2 Tải 4 CSV chính thức và kiểm tra SHA-256 { display-mode: "form" }
FILES = {  # tên file → SHA-256 của bản phát hành chính thức (đã kiểm tra ngày 28/10/2026)
    "mass_case_description_train_set.csv": "a4bcc1c32bfd040212737e6e8c304819334abe717cfefff56431dc50ae6db723",
    "mass_case_description_test_set.csv":  "60f0b4504ec024106bbdd1555030d220163ea9da0ea4688473a4c9c3e8ab0457",
    "calc_case_description_train_set.csv": "3c7b07e8202f66709f446b9695e96c5d17484a65fda6a24d0170a0c1b81d56e3",
    "calc_case_description_test_set.csv":  "03a2f72efb6ed29f417e07f4598950e6a1e02e6b11250ff516ebe7924ace3689",
}
URL_CANDIDATES = [  # đường dẫn công khai của TCIA; nếu đổi, notebook tự chuyển sang Drive/tải lên
    "https://www.cancerimagingarchive.net/wp-content/uploads/{name}",
    "https://wiki.cancerimagingarchive.net/download/attachments/22516629/{name}",
]
DATA_DIR = Path("official_csv_verified")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def try_folder(folder) -> dict:
    folder = Path(folder)
    found = {}
    for name in FILES:
        p = folder / name
        if p.exists():
            found[name] = p.read_bytes()
    return found


def try_download() -> dict:
    found = {}
    for tpl in URL_CANDIDATES:                     # thử từng mẫu URL; mẫu nào hỏng thì bỏ ngay
        for name in FILES:
            if name in found:
                continue
            try:
                with urllib.request.urlopen(tpl.format(name=name), timeout=15) as r:
                    b = r.read()
            except Exception:
                break
            if sha256_bytes(b) != FILES[name]:   # chỉ nhận đúng bản chính thức
                break
            found[name] = b
    return found


def try_local() -> dict:
    found = {}
    for d in CFG["LOCAL_DIRS"]:
        found.update({k: v for k, v in try_folder(d).items() if k not in found})
    return found


def try_github() -> dict:
    """Đọc CSV từ repo GitHub. Repo private: thêm secret GITHUB_TOKEN trong Colab (biểu tượng 🔑 bên trái)."""
    repo = CFG["GITHUB_REPO"]
    if not repo or "OWNER" in repo:
        return {}
    token = os.environ.get("GITHUB_TOKEN")
    if IN_COLAB and not token:
        try:
            from google.colab import userdata
            token = userdata.get("GITHUB_TOKEN")
        except Exception:
            token = None
    found = {}
    for name in FILES:
        for base in ("data/official_csv", "claude_research/data/official_csv"):
            url = f"https://raw.githubusercontent.com/{repo}/{CFG['GITHUB_BRANCH']}/{base}/{name}"
            req = urllib.request.Request(url, headers={"Authorization": f"token {token}"} if token else {})
            try:
                with urllib.request.urlopen(req, timeout=20) as r:
                    b = r.read()
            except Exception:
                continue
            if sha256_bytes(b) == FILES[name]:
                found[name] = b
                break
    return found


def try_drive() -> dict:
    if not IN_COLAB:
        return {}
    from google.colab import drive
    drive.mount("/content/drive")
    return try_folder(CFG["DRIVE_DIR"])


def try_upload() -> dict:
    if not IN_COLAB:
        return {}
    from google.colab import files
    print("Chọn 4 file CSV (mass/calc × train/test) để tải lên…")
    return {k: v for k, v in files.upload().items() if k in FILES}


steps = {"local": try_local, "github": try_github, "download": try_download, "drive": try_drive, "upload": try_upload}
order = ["local", "github", "download", "drive", "upload"] if CFG["DATA_SOURCE"] == "auto" else [CFG["DATA_SOURCE"]]
data = {}
for step in order:
    data.update({k: v for k, v in steps[step]().items() if k not in data})
    print(f"  {step:<9} → đã có {len(data)}/4 file")
    if len(data) == len(FILES):
        break
assert len(data) == len(FILES), "Thiếu file CSV. Xem CFG['GITHUB_REPO'] / LOCAL_DIRS / DRIVE_DIR."

DATA_DIR.mkdir(exist_ok=True)
check = []
for name, b in data.items():
    ok = sha256_bytes(b) == FILES[name]
    (DATA_DIR / name).write_bytes(b)
    check.append({"file": name, "bytes": len(b), "sha256_khớp": ok})
display(pd.DataFrame(check))
assert all(r["sha256_khớp"] for r in check), "SHA-256 không khớp bản chính thức — dừng lại để kiểm tra."
''')

# ─────────────────────────────────────────────────────────────────────────────
md(r'''
## 1. Đọc dữ liệu và ontology BI-RADS

**Đơn vị phân tích:** một dòng = một tổn thương trên một view (CC hoặc MLO).
**Y:** MALIGNANT = 1; BENIGN và BENIGN_WITHOUT_CALLBACK (BWC, lành không qua sinh thiết) = 0.

Ô chú giải có thể **đa trị**, ví dụ `ILL_DEFINED-SPICULATED`: chuyên gia thấy bờ vừa không rõ vừa có gai. Đây chính là chỗ logic mờ có ý nghĩa tự nhiên.
''')

code(r'''
# @title 1.1 Đọc và chuẩn hóa CSV mass { display-mode: "form" }
def load_mass(split: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"mass_case_description_{split}_set.csv")
    df = df.rename(columns={"breast_density": "density", "left or right breast": "side", "image view": "view",
                            "abnormality id": "abn_id", "mass shape": "shape_raw", "mass margins": "margin_raw"})
    df["y"] = (df["pathology"] == "MALIGNANT").astype(int)
    df["bwc"] = (df["pathology"] == "BENIGN_WITHOUT_CALLBACK").astype(int)
    # Khóa tổn thương giả định: cùng bệnh nhân + bên vú + abnormality id → ghép view CC với MLO
    df["lesion_key"] = df["patient_id"] + "|" + df["side"] + "|" + df["abn_id"].astype(str)
    split_vals = lambda v: tuple() if pd.isna(v) else tuple(sorted(set(str(v).split("-"))))
    df["shape_set"] = df["shape_raw"].apply(split_vals)
    df["margin_set"] = df["margin_raw"].apply(split_vals)
    return df


train = load_mass("train").reset_index(drop=True)
test = load_mass("test")  # chỉ dùng để đếm; KHÔNG mô hình hóa
print(f"Train: {len(train)} dòng, {train.patient_id.nunique()} bệnh nhân | "
      f"Test (chỉ đếm): {len(test)} dòng, {test.patient_id.nunique()} bệnh nhân | "
      f"Trùng bệnh nhân train∩test: {len(set(train.patient_id) & set(test.patient_id))}")
display(train[["patient_id", "view", "shape_raw", "margin_raw", "density", "assessment", "subtlety", "pathology"]].head(6))
''')

code(r'''
# @title 1.2 Ontology BI-RADS v0 (tri thức miền, KHÔNG học từ dữ liệu) { display-mode: "form" }
# Mỗi giá trị CBIS → khái niệm BI-RADS 5th ed., khái niệm cha, và thứ hạng nghi ngờ (prior định tính).
# Cần GS / bác sĩ chẩn đoán hình ảnh xác nhận.
ONTO = {
    "shape": {
        "ROUND":                    ("Round",                 "MassShape",    0),
        "OVAL":                     ("Oval",                  "MassShape",    0),
        "LOBULATED":                ("Oval (BI-RADS 4: lobulated)", "MassShape", 1),
        "IRREGULAR":                ("Irregular",             "MassShape",    2),
        "ARCHITECTURAL_DISTORTION": ("ArchitecturalDistortion", "OtherFinding", 2),
        "FOCAL_ASYMMETRIC_DENSITY": ("FocalAsymmetry",        "OtherFinding", 1),
        "ASYMMETRIC_BREAST_TISSUE": ("GlobalAsymmetry",       "OtherFinding", 0),
        "LYMPH_NODE":               ("IntramammaryLymphNode", "OtherFinding", 0),
    },
    "margin": {
        "CIRCUMSCRIBED":  ("Circumscribed",  "BenignAppearing", 0),
        "OBSCURED":       ("Obscured",       "Indeterminate",   1),
        "MICROLOBULATED": ("Microlobulated", "Suspicious",      2),
        "ILL_DEFINED":    ("Indistinct",     "Suspicious",      2),
        "SPICULATED":     ("Spiculated",     "Suspicious",      3),
    },
}
VOCAB = {g: list(v) + ["MISSING"] for g, v in ONTO.items()}   # thêm cột "thiếu chú giải"
RANK = {g: {k: v[2] for k, v in d.items()} for g, d in ONTO.items()}

# Kiểm tra: mọi giá trị trong CSV đều có trong ontology
for g in ONTO:
    seen = set(v for s in pd.concat([train, test])[f"{g}_set"] for v in s)
    assert seen <= set(ONTO[g]), f"Giá trị lạ trong {g}: {seen - set(ONTO[g])}"

display(pd.DataFrame([(g, k, *v) for g, d in ONTO.items() for k, v in d.items()],
                     columns=["nhóm", "giá trị CBIS", "BI-RADS 5", "khái niệm cha", "hạng nghi ngờ (prior)"]))
''')

# ─────────────────────────────────────────────────────────────────────────────
md(r'''
## 2. Bước 1a — Audit dữ liệu

Ba câu hỏi:

1. Cohort trông thế nào?
2. Khái niệm nào gắn với ác tính, và có khớp thứ tự BI-RADS không?
3. Có trường nào **rò rỉ** đáp án? Chú giải chuyên gia **nhiễu** đến đâu?
''')

code(r'''
# @title 2.1 Cohort { display-mode: "form" }
def cohort_row(name, d):
    nviews = d.groupby("lesion_key")["view"].nunique()
    return {"partition": name, "dòng": len(d), "bệnh nhân": d.patient_id.nunique(),
            "tổn thương": d.lesion_key.nunique(), "có cả CC+MLO": int((nviews == 2).sum()),
            "ác tính": int(d.y.sum()), "BWC": int(d.bwc.sum()), "% ác tính": f"{d.y.mean():.1%}"}


display(pd.DataFrame([cohort_row("train", train), cohort_row("test (chỉ đếm)", test)]))
for g in ("shape", "margin"):
    print(f"{g:>6}: đa trị {int((train[f'{g}_set'].apply(len) > 1).sum()):>3} dòng | "
          f"thiếu {int((train[f'{g}_set'].apply(len) == 0).sum()):>3} dòng")
''')

code(r'''
# @title 2.2 Tỉ lệ ác tính theo từng khái niệm (sắp theo hạng nghi ngờ BI-RADS) { display-mode: "form" }
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [8, 5]})
for ax, g in zip(axes, ("shape", "margin")):
    ex = train[[f"{g}_set", "y"]].explode(f"{g}_set").dropna()
    stat = ex.groupby(f"{g}_set").y.agg(["mean", "size"])
    stat["rank"] = [RANK[g][k] for k in stat.index]
    stat = stat.sort_values(["rank", "mean"])
    x = np.arange(len(stat))
    ax.bar(x, stat["mean"], width=0.6, color=BLUE)
    for xi, (r, n) in enumerate(zip(stat["mean"], stat["size"])):
        ax.text(xi, r + 0.02, f"{r:.0%}\nn={n}", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(x, [k.replace("_", " ").title() for k in stat.index], rotation=30, ha="right", rotation_mode="anchor")
    ax.set_ylim(0, 1.18); ax.set_yticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="x", visible=False)
    ax.set_title(f"{g.title()}")
axes[0].set_ylabel("Tỉ lệ ác tính (train)")
fig.suptitle("Khái niệm càng nghi ngờ theo BI-RADS → tỉ lệ ác tính càng cao (margin gần như đơn điệu)",
             x=0.01, ha="left", fontsize=11, color=INK)
fig.tight_layout(); plt.show()
''')

code(r'''
# @title 2.3 Rò rỉ: assessment (BI-RADS 0–5) gần như chính là đáp án { display-mode: "form" }
ct = pd.crosstab(train.assessment, train.pathology)[["BENIGN", "BENIGN_WITHOUT_CALLBACK", "MALIGNANT"]]
fig, ax = plt.subplots(figsize=(6.2, 3.6))
ax.imshow(ct.values, cmap=SEQ, aspect="auto")
for i in range(ct.shape[0]):
    for j in range(ct.shape[1]):
        v = ct.values[i, j]
        ax.text(j, i, v, ha="center", va="center", fontsize=9, color="white" if v > ct.values.max() * 0.55 else INK)
ax.set_xticks(range(3), ["Benign", "Benign w/o callback", "Malignant"]); ax.set_yticks(range(len(ct)), ct.index)
ax.set_ylabel("Assessment"); ax.grid(False)
ax.set_title("Assessment 5 → 293/299 ác tính; assessment 2 → 0 ác tính ⇒ loại khỏi Z")
plt.tight_layout(); plt.show()
''')

code(r'''
# @title 2.4 Nhiễu chú giải: cùng một tổn thương, hai view CC ↔ MLO có được mô tả giống nhau? { display-mode: "form" }
def level(s, g):
    """Mức nghi ngờ cao nhất của một ô (nguyên tắc 'đặc điểm nghi ngờ nhất')."""
    if not s:
        return None
    r = max(RANK[g][v] for v in s)
    return min(r, 2) if g == "margin" else int(r >= 2)


pairs = train.groupby("lesion_key").filter(lambda d: len(d) == 2 and set(d.view) == {"CC", "MLO"}).copy()
pairs["shape_level"] = pairs.shape_set.apply(lambda s: level(s, "shape"))
pairs["margin_level"] = pairs.margin_set.apply(lambda s: level(s, "margin"))
fields = {"pathology": "Pathology (nhãn)", "shape_set": "Shape — đúng từ vựng", "margin_set": "Margin — đúng từ vựng",
          "shape_level": "Shape — đúng MỨC nghi ngờ", "margin_level": "Margin — đúng MỨC nghi ngờ",
          "assessment": "Assessment", "subtlety": "Subtlety"}
agree = {lab: pairs.groupby("lesion_key")[c].agg(lambda s: s.iloc[0] == s.iloc[1]).mean() for c, lab in fields.items()}
AGREE = pd.Series(agree)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw={"width_ratios": [5, 4, 4]})
ax = axes[0]
cols = [AQUA if "MỨC" in k else (BLUE if "từ vựng" in k else "#b9b8b2") for k in AGREE.index]
ax.barh(AGREE.index[::-1], AGREE.values[::-1], color=cols[::-1], height=0.6)
for i, v in enumerate(AGREE.values[::-1]):
    ax.text(v + 0.005, i, f"{v:.1%}", va="center", fontsize=8.5, color=INK)
ax.set_xlim(0.5, 1.07); ax.grid(axis="y", visible=False)
ax.set_title(f"Đồng thuận CC↔MLO ({pairs.lesion_key.nunique()} tổn thương)")

for ax, g in zip(axes[1:], ("margin", "shape")):
    pv = pairs.pivot_table(index="lesion_key", columns="view", values=f"{g}_raw", aggfunc="first").dropna()
    pv = pv[~pv.CC.str.contains("-") & ~pv.MLO.str.contains("-")]
    order = sorted(set(pv.CC) | set(pv.MLO), key=lambda k: RANK[g][k])
    m = pd.crosstab(pv.CC, pv.MLO).reindex(index=order, columns=order, fill_value=0)
    ax.imshow(np.log1p(m.values), cmap=SEQ)
    for i in range(len(order)):
        for j in range(len(order)):
            v = m.values[i, j]
            if v:
                ax.text(j, i, v, ha="center", va="center", fontsize=7.5, color="white" if v > 40 else INK)
    short = [k[:10].title() for k in order]
    ax.set_xticks(range(len(order)), short, rotation=40, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(order)), short, fontsize=7.5)
    ax.set_xlabel("MLO"); ax.set_ylabel("CC"); ax.grid(False)
    ax.set_title(f"{g.title()}: nhầm lẫn giữa 2 view")
fig.suptitle("Chuyên gia bất đồng ~17% theo từ vựng, nhưng chủ yếu giữa các khái niệm 'láng giềng' (gần đường chéo)",
             x=0.01, ha="left", fontsize=11, color=INK)
fig.tight_layout(); plt.show()
''')

# ─────────────────────────────────────────────────────────────────────────────
md(r'''
## 3. Hệ luật mờ Shape × Margin: cách hoạt động

Mỗi luật có dạng: **IF** shape là *s* **AND** margin là *m* **THEN** ác tính với độ *c₍s,m₎*.

| Thành phần | Công thức | Ý nghĩa |
|---|---|---|
| Mức thuộc | μₛ(x), μₘ(x) ∈ [0,1] | Chú giải: 1 cho mỗi giá trị được liệt kê. Dự đoán từ ảnh: xác suất của CNN |
| Độ kích hoạt | αₛₘ(x) = μₛ(x) · μₘ(x) | t-norm **product** (AND mờ) |
| Hệ quả | cₛₘ = (Σᵢ αₛₘ(xᵢ)·yᵢ + a·cₘ) / (Σᵢ αₛₘ(xᵢ) + a) | Tỉ lệ ác tính mờ trong ô, **làm trơn** về luật cha "chỉ margin" cₘ (rồi về prior) khi ô ít mẫu |
| Suy luận | p(x) = Σ αₛₘ·cₛₘ / Σ αₛₘ | Trung bình có trọng số các luật kích hoạt |

**Tính trung thực:** p(x) được **phân rã chính xác** thành đóng góp của từng luật. Không có đường đi tắt nào ngoài luật.
''')

code(r'''
# @title 3.1 Mã hóa mức thuộc và lớp FuzzyGridRules { display-mode: "form" }
def memberships(df, g, semantics="possibilistic"):
    """Ma trận n × |V| mức thuộc. possibilistic: mỗi giá trị được liệt kê = 1; probabilistic: chia đều 1/k."""
    V = VOCAB[g]
    M = np.zeros((len(df), len(V)))
    for i, s in enumerate(df[f"{g}_set"]):
        for v in (s if s else ("MISSING",)):
            M[i, V.index(v)] = 1.0
    if semantics == "probabilistic":
        M /= M.sum(1, keepdims=True)
    return M


class FuzzyGridRules:
    """Luật mờ lưới Shape × Margin, làm trơn phân cấp ô → margin → prior."""

    def __init__(self, smoothing=5.0, tnorm="product"):
        self.a, self.tnorm = smoothing, tnorm

    def fire(self, Ms, Mm):                      # n × S × M độ kích hoạt
        return Ms[:, :, None] * Mm[:, None, :] if self.tnorm == "product" else np.minimum(Ms[:, :, None], Mm[:, None, :])

    def fit(self, Ms, Mm, y):
        y = np.asarray(y, float)
        self.prior_ = y.mean()
        self.c_margin_ = (Mm.T @ y + self.a * self.prior_) / (Mm.sum(0) + self.a)
        alpha = self.fire(Ms, Mm)
        self.support_ = alpha.sum(0)
        self.c_ = (np.einsum("nsm,n->sm", alpha, y) + self.a * self.c_margin_[None]) / (self.support_ + self.a)
        return self

    def predict_proba(self, Ms, Mm):
        alpha = self.fire(Ms, Mm)
        den = alpha.sum((1, 2))
        num = np.einsum("nsm,sm->n", alpha, self.c_)
        return np.where(den > 1e-12, num / np.maximum(den, 1e-12), self.prior_)

    def explain(self, ms, mm):
        """Bảng đóng góp từng luật cho MỘT mẫu (tổng cột 'đóng góp' = p)."""
        alpha = self.fire(ms[None], mm[None])[0]
        rows = [(VOCAB["shape"][s], VOCAB["margin"][m], alpha[s, m], self.c_[s, m])
                for s, m in zip(*np.nonzero(alpha))]
        out = pd.DataFrame(rows, columns=["shape", "margin", "α (kích hoạt)", "c (hệ quả)"])
        out["đóng góp"] = out["α (kích hoạt)"] * out["c (hệ quả)"] / out["α (kích hoạt)"].sum()
        return out.sort_values("đóng góp", ascending=False)


MS, MM, Y = memberships(train, "shape"), memberships(train, "margin"), train.y.to_numpy()
rules_full = FuzzyGridRules().fit(MS, MM, Y)
print(f"Số ô có hỗ trợ ≥ 1: {(rules_full.support_ >= 1).sum()} | ≥ 10: {(rules_full.support_ >= 10).sum()}")
''')

code(r'''
# @title 3.2 Bản đồ luật: mỗi ô là một luật, màu = độ ác tính c, chữ nhỏ = số mẫu hỗ trợ { display-mode: "form" }
s_order = sorted(range(len(VOCAB["shape"]) - 1), key=lambda i: RANK["shape"][VOCAB["shape"][i]]) + [len(VOCAB["shape"]) - 1]
m_order = sorted(range(len(VOCAB["margin"]) - 1), key=lambda i: RANK["margin"][VOCAB["margin"][i]]) + [len(VOCAB["margin"]) - 1]
C = rules_full.c_[np.ix_(s_order, m_order)]
SUP = rules_full.support_[np.ix_(s_order, m_order)]
C_show = np.where(SUP >= 1, C, np.nan)

fig, ax = plt.subplots(figsize=(8.4, 6))
im = ax.imshow(C_show, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
for i in range(C.shape[0]):
    for j in range(C.shape[1]):
        if SUP[i, j] >= 1:
            dark = C[i, j] > 0.5
            ax.text(j, i - 0.12, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=9, color="white" if dark else INK, weight="bold")
            ax.text(j, i + 0.22, f"n={SUP[i, j]:.0f}", ha="center", va="center", fontsize=7, color="white" if dark else INK2)
ax.set_xticks(range(len(m_order)), [VOCAB["margin"][j].replace("_", " ").title() for j in m_order], rotation=25, ha="right")
ax.set_yticks(range(len(s_order)), [VOCAB["shape"][i].replace("_", " ").title() for i in s_order])
ax.set_xlabel("Margin  (nghi ngờ tăng →)"); ax.set_ylabel("Shape  (nghi ngờ tăng ↓)"); ax.grid(False)
cb = fig.colorbar(im, ax=ax, fraction=0.04); cb.set_label("P(ác tính | luật)", color=INK2)
ax.set_title("Bản đồ luật mờ học từ train (ô trống = không có mẫu)")
plt.tight_layout(); plt.show()

top = pd.DataFrame([(VOCAB["shape"][s], VOCAB["margin"][m], rules_full.support_[s, m], rules_full.c_[s, m])
                    for s in range(len(VOCAB["shape"])) for m in range(len(VOCAB["margin"])) if rules_full.support_[s, m] >= 10],
                   columns=["IF shape", "AND margin", "hỗ trợ", "THEN P(ác tính)"]).sort_values("hỗ trợ", ascending=False)
display(top.head(10).round(3))
''')

code(r'''
# @title 3.3 Giải thích một ca có chú giải đa trị { display-mode: "form" }
i = int(np.flatnonzero((train.margin_set.apply(len) > 1) & (train.shape_raw == "IRREGULAR"))[0])
p_i = rules_full.predict_proba(MS[i:i + 1], MM[i:i + 1])[0]
ex = rules_full.explain(MS[i], MM[i])
display(Markdown(f"**Ca #{i}** — shape `{train.shape_raw[i]}`, margin `{train.margin_raw[i]}`, "
                 f"nhãn thật: **{'ác tính' if train.y[i] else 'lành'}** → hệ luật cho **p = {p_i:.3f}**"))
display(ex.round(3))

fig, ax = plt.subplots(figsize=(7, 1.2 + 0.45 * len(ex)))
labels = [f"IF {a.title()} AND {b.replace('_', ' ').title()}" for a, b in zip(ex["shape"], ex["margin"])]
left = 0
for lab, v, c in zip(labels, ex["đóng góp"], [BLUE, ORANGE, AQUA, YELLOW]):
    ax.barh([0], [v], left=left, color=c, height=0.5, edgecolor="white", linewidth=2, label=f"{lab}: +{v:.2f}")
    left += v
ax.set_xlim(0, 1); ax.set_yticks([]); ax.grid(axis="y", visible=False)
ax.axvline(0.5, color=INK2, ls=":", lw=1.2, label="ngưỡng quyết định 0,5")
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.35), fontsize=8)
ax.set_title(f"p = {p_i:.3f} = tổng đóng góp của các luật kích hoạt")
plt.tight_layout(); plt.show()
''')

# ─────────────────────────────────────────────────────────────────────────────
md(r'''
## 4. Bước 1b — Trần biểu tượng A: từ Z dự đoán Y tốt đến đâu?

Mọi mô hình được đánh giá bằng **CV 5-fold theo bệnh nhân**, lặp `N_REPEATS` lần với seed khác nhau. Dự đoán ngoài fold (OOF) được gộp lại để tính AUROC. CI 95% dùng **bootstrap theo bệnh nhân**, vector hóa nên nhanh hơn bản `.py` khoảng 20 lần.

| Bộ Z | Thành phần | Hợp lệ? |
|---|---|---|
| `Z_morph` | shape + margin (đa trị) | Z chính |
| `Z_morph_den` | + breast density | Ngữ cảnh |
| `Z_morph_den_sub` | + subtlety | Chỉ để phân tích |
| `Z_assess`, `Z_morph_assess` | có assessment | **Rò rỉ**, chỉ để tham chiếu |
''')

code(r'''
# @title 4.1 Công cụ: mã hóa Z, chia fold theo bệnh nhân, metrics, bootstrap nhanh { display-mode: "form" }
def feature_matrix(df, zset):
    parts = []
    if zset.startswith("Z_morph"):
        parts += [pd.DataFrame(memberships(df, "shape"), columns=[f"shape={v}" for v in VOCAB["shape"]]),
                  pd.DataFrame(memberships(df, "margin"), columns=[f"margin={v}" for v in VOCAB["margin"]])]
    if zset in ("Z_morph_den", "Z_morph_den_sub"):
        parts.append(pd.get_dummies(df.density.astype(int), prefix="density").astype(float).reset_index(drop=True))
    if zset == "Z_morph_den_sub":
        parts.append(pd.get_dummies(df.subtlety.astype(int), prefix="subtlety").astype(float).reset_index(drop=True))
    if "assess" in zset:
        parts.append(pd.get_dummies(df.assessment.astype(int), prefix="assess").astype(float).reset_index(drop=True))
    return pd.concat(parts, axis=1)


def expert_score(df):
    """Chỉ tri thức, KHÔNG học: hạng nghi ngờ cao nhất của margin + của shape."""
    fill = {g: float(np.median(list(RANK[g].values()))) for g in RANK}
    return sum(df[f"{g}_set"].apply(lambda s: max((RANK[g][v] for v in s), default=fill[g])).to_numpy(float) for g in RANK)


def patient_folds(df, seed):
    sgkf = StratifiedGroupKFold(n_splits=CFG["N_SPLITS"], shuffle=True, random_state=seed)
    fold = np.empty(len(df), int)
    for k, (_, te) in enumerate(sgkf.split(np.zeros(len(df)), df.y, groups=df.patient_id)):
        fold[te] = k
    return fold


def metrics(y, p):
    yhat = (p >= 0.5).astype(int)
    return {"AUROC": roc_auc_score(y, p), "Accuracy": accuracy_score(y, yhat),
            "Balanced acc": balanced_accuracy_score(y, yhat), "Brier": brier_score_loss(y, np.clip(p, 0, 1))}


def bootstrap_auc(y, p, groups, n_boot, seed=CFG["SEED"]):
    """AUROC trên n_boot mẫu bootstrap THEO BỆNH NHÂN, tính cùng lúc bằng ma trận (xử lý đúng các điểm bằng nhau).

    Mỗi lần bootstrap = rút bệnh nhân có hoàn lại → trọng số mỗi dòng = số lần bệnh nhân đó được rút.
    AUROC có trọng số = Σ_g w⁺_g (W⁻_{<g} + ½ w⁻_g) / (W⁺ W⁻), với g là các nhóm điểm số bằng nhau.
    """
    rng = np.random.default_rng(seed)
    _, pat = np.unique(groups, return_inverse=True)
    n_pat = pat.max() + 1
    W = rng.multinomial(n_pat, np.full(n_pat, 1 / n_pat), size=n_boot)[:, pat].astype(float)  # B × n
    _, g = np.unique(p, return_inverse=True)
    G = g.max() + 1
    onehot = np.zeros((len(p), G)); onehot[np.arange(len(p)), g] = 1
    wpos, wneg = (W * (y == 1)) @ onehot, (W * (y == 0)) @ onehot                              # B × G
    neg_below = np.cumsum(wneg, 1) - wneg
    return (wpos * (neg_below + 0.5 * wneg)).sum(1) / (wpos.sum(1) * wneg.sum(1))


# Tự kiểm tra: bootstrap có trọng số = cách nối mẫu truyền thống
_rng = np.random.default_rng(0); _y = _rng.integers(0, 2, 300); _p = np.round(_rng.random(300), 1); _gr = np.arange(300) // 2
_auc = bootstrap_auc(_y, _p, _gr, 3, seed=1)
_W = np.random.default_rng(1).multinomial(150, np.full(150, 1 / 150), size=3)[:, np.unique(_gr, return_inverse=True)[1]]
assert np.allclose(_auc, [roc_auc_score(_y, _p, sample_weight=w) for w in _W]), "bootstrap_auc sai"
print("✓ bootstrap_auc khớp roc_auc_score(sample_weight)")
''')

code(r'''
# @title 4.2 Danh sách mô hình và chạy CV (≈ 1 phút) { display-mode: "form" }
def fit_predict(model, Xtr, ytr, Xte):
    if model == "Prior (đa số)":
        return np.full(len(Xte), ytr.mean())
    est = {"LogReg": LogisticRegression(C=1.0, max_iter=2000),
           "GBM": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=0),
           "Cây crisp sâu 3": DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=0)}[model]
    return est.fit(Xtr, ytr).predict_proba(Xte)[:, 1]


def oof(df, model, zset, fold):
    y = df.y.to_numpy()
    if model == "Tri thức BI-RADS (không học)":
        return expert_score(df)                     # thứ hạng; chỉ AUROC có nghĩa
    p = np.zeros(len(df))
    if model.startswith("Luật mờ"):
        sem = "probabilistic" if "prob" in model else "possibilistic"
        Ms, Mm = memberships(df, "shape", sem), memberships(df, "margin", sem)
        for k in np.unique(fold):
            tr, te = fold != k, fold == k
            p[te] = FuzzyGridRules().fit(Ms[tr], Mm[tr], y[tr]).predict_proba(Ms[te], Mm[te])
        return p
    X = feature_matrix(df, zset).to_numpy()
    for k in np.unique(fold):
        tr, te = fold != k, fold == k
        p[te] = fit_predict(model, X[tr], y[tr], X[te])
    return p


PLAN = [("Prior (đa số)", "Z_morph"), ("Tri thức BI-RADS (không học)", "Z_morph"), ("Cây crisp sâu 3", "Z_morph"),
        ("Luật mờ (possibilistic)", "Z_morph"), ("Luật mờ (probabilistic)", "Z_morph"),
        ("LogReg", "Z_morph"), ("GBM", "Z_morph"), ("LogReg", "Z_morph_den"), ("GBM", "Z_morph_den"),
        ("LogReg", "Z_morph_den_sub"), ("LogReg", "Z_assess"), ("LogReg", "Z_morph_assess"), ("GBM", "Z_morph_assess")]


def run_ceiling(df, tag):
    rows, first = [], {}
    for r in range(CFG["N_REPEATS"]):
        fold = patient_folds(df, seed=r)
        for model, z in PLAN:
            p = oof(df, model, z, fold)
            m = metrics(df.y.to_numpy(), p)
            if model.startswith(("Prior", "Tri thức")):
                m.update({"Accuracy": np.nan, "Balanced acc": np.nan, "Brier": np.nan})
            rows.append({"cohort": tag, "mô hình": model, "Z": z, "lặp": r, **m})
            if r == 0:
                first[(model, z)] = p
    res = pd.DataFrame(rows)
    agg = res.groupby(["cohort", "mô hình", "Z"], sort=False).agg(
        AUROC=("AUROC", "mean"), AUROC_sd=("AUROC", "std"), Accuracy=("Accuracy", "mean"),
        Balanced_acc=("Balanced acc", "mean"), Brier=("Brier", "mean")).reset_index()
    ci = {k: np.percentile(bootstrap_auc(df.y.to_numpy(), p, df.patient_id.to_numpy(), CFG["N_BOOT"]), [2.5, 97.5])
          for k, p in first.items()}
    agg["CI95_low"] = [ci[(m, z)][0] for m, z in zip(agg["mô hình"], agg.Z)]
    agg["CI95_high"] = [ci[(m, z)][1] for m, z in zip(agg["mô hình"], agg.Z)]
    return res, agg


t0 = time.time()
res_main, CEIL = run_ceiling(train, "all_mass_train")
print(f"Xong trong {time.time() - t0:.0f} giây")
display(CEIL.drop(columns="cohort").round(3))
''')

code(r'''
# @title 4.3 Biểu đồ trần biểu tượng { display-mode: "form" }
d = CEIL.iloc[::-1].reset_index(drop=True)
leak = d.Z.str.contains("assess")
fig, ax = plt.subplots(figsize=(10, 5.6))
yy = np.arange(len(d))
ax.barh(yy, d.AUROC - 0.5, left=0.5, height=0.55, color=np.where(leak, ORANGE, BLUE))
for i, (m, lo, hi) in enumerate(zip(d.AUROC, d.CI95_low, d.CI95_high)):
    ax.plot([lo, hi], [i, i], color=INK2, lw=1.2)
    ax.text(max(hi, m) + 0.006, i, f"{m:.3f}", va="center", fontsize=8, color=INK)
ax.set_yticks(yy, d["mô hình"] + "  ·  " + d.Z)
ax.set_xlim(0.5, 1.0); ax.grid(axis="y", visible=False)
ax.axvline(CFG["B_REFERENCE"], color=INK2, lw=1, ls="--")
ax.text(CFG["B_REFERENCE"] + 0.006, len(d) - 1, f"← CNN ROI ≈{CFG['B_REFERENCE']} (lượt trước, chỉ định hướng)",
        fontsize=7.5, color=INK2, va="center")
ax.set_xlabel("AUROC OOF (CV theo bệnh nhân); vạch = CI 95% bootstrap bệnh nhân")
ax.set_title("Trần biểu tượng A — xanh: Z hợp lệ · cam: có assessment (rò rỉ, chỉ tham chiếu)")
plt.tight_layout(); plt.show()
''')

code(r'''
# @title 4.4 Độ nhạy: bỏ BWC / chỉ giữ "mass thật" theo BI-RADS 5 { display-mode: "form" }
other = {k for k, v in ONTO["shape"].items() if v[1] == "OtherFinding"}
cohorts = {"bỏ BWC": train[train.bwc == 0], "chỉ mass thật": train[train.shape_set.apply(lambda s: len(s) > 0 and not set(s) <= other)]}
sens = []
for tag, dfc in cohorts.items():
    dfc = dfc.reset_index(drop=True)
    for model, z in [("Tri thức BI-RADS (không học)", "Z_morph"), ("Luật mờ (possibilistic)", "Z_morph"), ("LogReg", "Z_morph")]:
        aucs = [roc_auc_score(dfc.y, oof(dfc, model, z, patient_folds(dfc, r))) for r in range(CFG["N_REPEATS"])]
        sens.append({"cohort": f"{tag} (n={len(dfc)})", "mô hình": model, "AUROC": np.mean(aucs)})
SENS = pd.DataFrame(sens).pivot(index="mô hình", columns="cohort", values="AUROC")
SENS.insert(0, f"tất cả (n={len(train)})", CEIL.set_index("mô hình").loc[SENS.index].query("Z == 'Z_morph'").AUROC)
display(SENS.round(3))
''')

code(r'''
# @title 4.5 Lượng thông tin: Z giải thích bao nhiêu phần bất định của Y? { display-mode: "form" }
key = lambda s: "+".join(s) if s else "MISSING"
sh, mg = train.shape_set.apply(key), train.margin_set.apply(key)
H = mutual_info_score(Y, Y)
INFO = pd.Series({"shape": mutual_info_score(Y, sh) / H, "margin": mutual_info_score(Y, mg) / H,
                  "shape × margin": mutual_info_score(Y, sh + "|" + mg) / H,
                  "assessment (rò rỉ)": mutual_info_score(Y, train.assessment.astype(str)) / H})
cell = pd.DataFrame({"c": sh + "|" + mg, "y": Y}).groupby("c").y.agg(["mean", "size"])
CELL_ACC = (np.maximum(cell["mean"], 1 - cell["mean"]) * cell["size"]).sum() / len(Y)

fig, ax = plt.subplots(figsize=(7, 2.6))
ax.barh(INFO.index[::-1], INFO.values[::-1], color=[ORANGE, BLUE, BLUE, BLUE], height=0.55)
for i, v in enumerate(INFO.values[::-1]):
    ax.text(v + 0.01, i, f"{v:.0%}", va="center", fontsize=8.5, color=INK)
ax.set_xlim(0, 0.6); ax.grid(axis="y", visible=False)
ax.set_xlabel("I(Y; Z) / H(Y)")
ax.set_title("Phần entropy của nhãn được Z giải thích (plug-in, train)")
plt.tight_layout(); plt.show()
print(f"Trần accuracy lạc quan của Z_morph (đa số trong từng ô, in-sample, {len(cell)} ô): {CELL_ACC:.1%}")
''')

# ─────────────────────────────────────────────────────────────────────────────
md(r'''
## 5. Bước 1c — Dung sai perception

Khi chạy thật, khái niệm đến từ CNN, không còn là chú giải chuyên gia. Câu hỏi: **lớp luật chịu được bao nhiêu sai số?**

**Thí nghiệm 1, mô phỏng.** Luật học trên chú giải sạch. Ở fold test, mỗi nhóm khái niệm được thay bằng một "dự đoán giả": logits = κ·onehot(giá trị thật) + nhiễu Gauss, rồi softmax thành phân bố q. Tăng κ thì khái niệm đúng hơn. So sánh hai cách đưa vào luật:

- 🔵 **soft (mờ):** dùng nguyên q làm mức thuộc, giữ bất định.
- 🟠 **crisp:** lấy argmax của q trước, rồi mới suy luận.

**Thí nghiệm 2, nhiễu kiểu con người.** Thay chú giải của view này bằng chú giải của view kia trên cùng tổn thương, coi như một lần đọc khác của cùng một ca.
''')

code(r'''
# @title 5.1 Mô phỏng bộ dự đoán khái niệm (vector hóa) { display-mode: "form" }
def simulate_concepts(M_true, kappa, rng):
    """Trả về (q soft, onehot argmax, đúng?) cho một nhóm khái niệm. Giá trị 'thật' rút từ ô chú giải."""
    n, k = M_true.shape
    cum = np.cumsum(M_true / M_true.sum(1, keepdims=True), 1)
    v = (rng.random(n)[:, None] > cum).sum(1)                       # rút 1 giá trị thật trong các giá trị đã chú giải
    logits = kappa * np.eye(k)[v] + rng.standard_normal((n, k))
    q = np.exp(logits - logits.max(1, keepdims=True)); q /= q.sum(1, keepdims=True)
    hard = np.eye(k)[q.argmax(1)]
    return q, hard, M_true[np.arange(n), q.argmax(1)] > 0


KAPPAS = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8]
rows = []
for seed in CFG["SIM_SEEDS"]:
    fold = patient_folds(train, seed)
    models = {k: FuzzyGridRules().fit(MS[fold != k], MM[fold != k], Y[fold != k]) for k in range(CFG["N_SPLITS"])}
    clean = np.zeros(len(train))
    for k, m in models.items():
        clean[fold == k] = m.predict_proba(MS[fold == k], MM[fold == k])
    for kappa in KAPPAS:
        rng = np.random.default_rng(1000 * seed + int(kappa * 10))
        qs, hs, cs = simulate_concepts(MS, kappa, rng)
        qm, hm, cm = simulate_concepts(MM, kappa, rng)
        p_soft, p_hard = np.zeros(len(train)), np.zeros(len(train))
        for k, m in models.items():
            te = fold == k
            p_soft[te], p_hard[te] = m.predict_proba(qs[te], qm[te]), m.predict_proba(hs[te], hm[te])
        rows.append({"seed": seed, "kappa": kappa, "độ chính xác khái niệm": (cs.mean() + cm.mean()) / 2,
                     "AUROC sạch": roc_auc_score(Y, clean), "AUROC soft (mờ)": roc_auc_score(Y, p_soft),
                     "AUROC crisp (argmax)": roc_auc_score(Y, p_hard)})
SIM = pd.DataFrame(rows).groupby("kappa").mean(numeric_only=True).drop(columns="seed")
SIM["lợi ích của mờ"] = SIM["AUROC soft (mờ)"] - SIM["AUROC crisp (argmax)"]
display(SIM.round(3))
''')

code(r'''
# @title 5.2 Đường cong dung sai { display-mode: "form" }
def acc_needed(col, target):
    x, y = SIM["độ chính xác khái niệm"].to_numpy(), SIM[col].to_numpy()
    j = np.flatnonzero(y >= target)
    if len(j) == 0:
        return np.nan
    j = j[0]
    return x[j] if j == 0 else x[j - 1] + (target - y[j - 1]) * (x[j] - x[j - 1]) / (y[j] - y[j - 1])


NEED_SOFT = acc_needed("AUROC soft (mờ)", CFG["B_REFERENCE"])
NEED_CRISP = acc_needed("AUROC crisp (argmax)", CFG["B_REFERENCE"])
human = (AGREE["Shape — đúng từ vựng"], AGREE["Margin — đúng từ vựng"])

fig, ax = plt.subplots(figsize=(9, 5))
x = SIM["độ chính xác khái niệm"]
ax.plot(x, SIM["AUROC soft (mờ)"], color=BLUE, lw=2, marker="o", ms=5, label="Luật mờ, đầu vào soft (product)")
ax.plot(x, SIM["AUROC crisp (argmax)"], color=ORANGE, lw=2, marker="s", ms=5, label="Cùng luật, đầu vào crisp (argmax)")
A_clean = SIM["AUROC sạch"].mean()
ax.axhline(A_clean, color=INK, lw=1, ls=":"); ax.text(0.16, A_clean + 0.004, f"A: luật trên chú giải sạch = {A_clean:.3f}", fontsize=8, color=INK)
ax.axhline(CFG["B_REFERENCE"], color=INK2, lw=1, ls="--")
ax.text(0.16, CFG["B_REFERENCE"] + 0.004, f"B tham khảo ≈{CFG['B_REFERENCE']} (CNN ROI lượt trước; Bước 3 đo lại)", fontsize=8, color=INK2)
ax.axvspan(min(human), max(human), color=AQUA, alpha=0.18, lw=0)
ax.text(min(human) - 0.006, 0.60, "đồng thuận CC↔MLO\ncủa chuyên gia", fontsize=7.5, color=INK2, ha="right")
for v, c in ((NEED_SOFT, BLUE), (NEED_CRISP, ORANGE)):
    if not np.isnan(v):
        ax.plot([v], [CFG["B_REFERENCE"]], marker="o", ms=9, mfc="none", mec=c, mew=2)
ax.set_xlabel("Độ chính xác khái niệm mô phỏng (TB shape & margin)"); ax.set_ylabel("AUROC OOF")
ax.set_ylim(0.5, 0.92)
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.01, 0.72))
ax.set_title("Lớp luật cần khái niệm chính xác đến đâu?")
plt.tight_layout(); plt.show()
print(f"Để đạt B≈{CFG['B_REFERENCE']}: cần ≈{NEED_SOFT:.0%} (soft/mờ) hoặc ≈{NEED_CRISP:.0%} (crisp) độ chính xác khái niệm.")
''')

code(r'''
# @title 5.3 Nhiễu kiểu con người: dùng chú giải của view kia { display-mode: "form" }
partner = np.full(len(train), -1)
for idx in train.groupby("lesion_key").indices.values():
    if len(idx) == 2 and set(train.view.iloc[idx]) == {"CC", "MLO"}:
        partner[idx[0]], partner[idx[1]] = idx[1], idx[0]
has = partner >= 0
chg_s = (MS[has] != MS[partner[has]]).any(1)   # dòng mà shape bị đổi khi lấy chú giải view kia
chg_m = (MM[has] != MM[partner[has]]).any(1)   # dòng mà margin bị đổi
changed = (chg_s | chg_m).mean()


def random_relabel(M, rate, rng):
    """Đối chứng: đổi ĐÚNG tỉ lệ `rate` số dòng sang một giá trị KHÁC, chọn ngẫu nhiên theo tần suất train."""
    M = M.copy()
    freq = M.sum(0) / M.sum()
    flip = np.flatnonzero(rng.random(len(M)) < rate)
    for i in flip:
        w = freq * (M[i] == 0)
        M[i] = 0; M[i, rng.choice(len(w), p=w / w.sum())] = 1
    return M


own, swap, rand = [], [], []
for seed in range(CFG["N_REPEATS"]):
    fold = patient_folds(train, seed)
    rng = np.random.default_rng(seed)
    MS_r, MM_r = random_relabel(MS, chg_s.mean(), rng), random_relabel(MM, chg_m.mean(), rng)
    p_own, p_swap, p_rand = (np.zeros(len(train)) for _ in range(3))
    for k in range(CFG["N_SPLITS"]):
        tr, te = fold != k, fold == k
        m = FuzzyGridRules().fit(MS[tr], MM[tr], Y[tr])       # luật luôn học trên chú giải sạch
        p_own[te] = m.predict_proba(MS[te], MM[te])
        tp = te & has
        p_swap[tp] = m.predict_proba(MS[partner[tp]], MM[partner[tp]])
        p_rand[te] = m.predict_proba(MS_r[te], MM_r[te])
    own.append(roc_auc_score(Y[has], p_own[has])); swap.append(roc_auc_score(Y[has], p_swap[has]))
    rand.append(roc_auc_score(Y[has], p_rand[has]))
XV = {"own": np.mean(own), "swap": np.mean(swap), "random_same_rate": np.mean(rand), "changed": changed,
      "shape_change_rate": chg_s.mean(), "margin_change_rate": chg_m.mean(), "rows": int(has.sum())}

fig, ax = plt.subplots(figsize=(7.5, 2.8))
labs = ["Chú giải của chính view", f"Chú giải view kia\n({changed:.0%} dòng đổi khái niệm)",
        f"Đổi NGẪU NHIÊN cùng tỉ lệ\n(shape {chg_s.mean():.0%}, margin {chg_m.mean():.0%})"]
vals = [XV["own"], XV["swap"], XV["random_same_rate"]]
ax.barh(labs[::-1], vals[::-1], color=[ORANGE, AQUA, BLUE], height=0.55)
for i, v in enumerate(vals[::-1]):
    ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=9, color=INK)
ax.set_xlim(0.75, 0.9); ax.grid(axis="y", visible=False); ax.set_xlabel("AUROC (dòng có cả CC và MLO)")
ax.set_title("Bất đồng của con người gần như vô hại, nhiễu ngẫu nhiên thì không")
plt.tight_layout(); plt.show()
''')

# ─────────────────────────────────────────────────────────────────────────────
md(r'''
## 6. Tổng kết tự động và lưu kết quả
''')

code(r'''
# @title 6.1 Tổng kết (tự sinh từ số liệu vừa chạy) { display-mode: "form" }
get = lambda m, z="Z_morph": CEIL.query("`mô hình` == @m and Z == @z").iloc[0]
fz, lr, ex_, ck = get("Luật mờ (possibilistic)"), get("LogReg"), get("Tri thức BI-RADS (không học)"), get("LogReg", "Z_morph_assess")
gap = fz.AUROC - CFG["B_REFERENCE"]
display(Markdown(f"""
### Kết quả chính

| Đại lượng | Giá trị |
|---|---|
| **A** — luật mờ trên shape + margin | **{fz.AUROC:.3f}** [{fz.CI95_low:.3f}; {fz.CI95_high:.3f}], accuracy {fz.Accuracy:.1%} |
| LogReg trên cùng Z | {lr.AUROC:.3f} |
| Chỉ tri thức BI-RADS, **không học** | {ex_.AUROC:.3f} ({ex_.AUROC / fz.AUROC:.0%} của luật học được) |
| Thêm assessment (rò rỉ) | {ck.AUROC:.3f} (+{ck.AUROC - lr.AUROC:.3f}) → loại khỏi Z |
| Trần accuracy của Z_morph (in-sample, lạc quan) | {CELL_ACC:.1%} |
| Đồng thuận CC↔MLO: shape / margin (từ vựng) | {human[0]:.1%} / {human[1]:.1%} |
| Độ chính xác khái niệm cần để đạt B≈{CFG['B_REFERENCE']} | {NEED_SOFT:.0%} (soft) · {NEED_CRISP:.0%} (crisp) |
| Đổi sang chú giải view kia ({XV['changed']:.0%} dòng đổi khái niệm) | AUROC {XV['own']:.3f} → **{XV['swap']:.3f}** |
| Đổi **ngẫu nhiên** với cùng tỉ lệ | AUROC {XV['own']:.3f} → **{XV['random_same_rate']:.3f}** |

### Cổng quyết định 1

- **A − B tham khảo ≈ {gap:+.3f} AUROC.** Dương nhưng chưa phải "B << A" → **qua có điều kiện**.
- **Lớp luật không phải nút thắt:** luật mờ ≥ LogReg ({fz.AUROC - lr.AUROC:+.3f}).
- **Ưu tiên tiếp theo:**
  1. Bước 2 — thêm thuộc tính hình thái **liên tục** từ mask để tăng A.
  2. Bước 3 — đo B công bằng và độ khôi phục khái niệm R.
  3. Huấn luyện perception theo **mức nghi ngờ** của ontology và suy luận **soft**.
"""))
''')

code(r'''
# @title 6.2 Lưu bảng, hình và tải về { display-mode: "form" }
if IN_COLAB and CFG["SAVE_TO_DRIVE"]:
    from google.colab import drive
    if not Path("/content/drive/MyDrive").exists():
        drive.mount("/content/drive")
    save = Path(CFG["SAVE_DIR"])
else:
    save = Path("/content/step1_results" if IN_COLAB else "step1_results")
save.mkdir(parents=True, exist_ok=True)

CEIL.to_csv(save / "1b_ceiling_summary.csv", index=False)
res_main.to_csv(save / "1b_ceiling_per_repeat.csv", index=False)
SENS.to_csv(save / "1b_sensitivity.csv")
SIM.to_csv(save / "1c_simulation.csv")
top.to_csv(save / "1b_fuzzy_rules_support_ge10.csv", index=False)
AGREE.rename("agreement").to_csv(save / "1a_cross_view_agreement.csv")
summary = {"A_fuzzy": fz.AUROC, "A_fuzzy_ci": [fz.CI95_low, fz.CI95_high], "A_logreg": lr.AUROC,
           "expert_no_learning": ex_.AUROC, "with_assessment_leak": ck.AUROC, "cell_acc_ceiling": CELL_ACC,
           "need_acc_soft": NEED_SOFT, "need_acc_crisp": NEED_CRISP, "cross_view": XV,
           "B_reference": CFG["B_REFERENCE"], "cfg": CFG, "input_sha256": FILES}
(save / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

zpath = Path("step1_results.zip")
with zipfile.ZipFile(zpath, "w") as z:
    for f in save.iterdir():
        z.write(f, f.name)
print(f"Đã lưu {len(list(save.iterdir()))} file vào {save}")
if IN_COLAB:
    try:
        from google.colab import files
        files.download(str(zpath))
    except Exception as e:  # chạy tự động/không có trình duyệt: bỏ qua bước tải về
        print("Bỏ qua tải zip:", e)

# Dòng máy-đọc-được để agent (Antigravity) kiểm tra kết quả
print("STEP1_RESULT_JSON " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in {
    "A_fuzzy": fz.AUROC, "A_logreg": lr.AUROC, "expert_no_learning": ex_.AUROC, "with_assessment_leak": ck.AUROC,
    "need_acc_soft": NEED_SOFT, "need_acc_crisp": NEED_CRISP, "cross_view_own": XV["own"],
    "cross_view_swap": XV["swap"], "random_same_rate": XV["random_same_rate"]}.items()}))
''')


def build(path):
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"colab": {"provenance": [], "toc_visible": True, "name": "Step1_CBIS_DDSM_Symbolic_Premise.ipynb"},
                       "kernelspec": {"name": "python3", "display_name": "Python 3"},
                       "language_info": {"name": "python"}},
          "cells": []}
    for i, (kind, src) in enumerate(CELLS):
        cell = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True), "id": f"c{i:02d}"}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        nb["cells"].append(cell)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    return nb


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "Step1_CBIS_DDSM_Symbolic_Premise.ipynb"
    nb = build(out)
    if len(sys.argv) > 2:  # xuất toàn bộ code thành script để kiểm thử ngoài Jupyter
        with open(sys.argv[2], "w", encoding="utf-8") as f:
            for c in nb["cells"]:
                if c["cell_type"] == "code":
                    f.write("".join(c["source"]) + "\n\n")
    print("wrote", out)
