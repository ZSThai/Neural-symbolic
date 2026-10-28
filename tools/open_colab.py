"""
Tự động mở Notebook trên Google Colab bằng trình duyệt Google Chrome (Phương án A).
Ưu tiên:
  1. Nếu file .ipynb trong thư mục gốc đã đồng bộ xong lên Google Drive for Desktop
     (có Cloud ID trong DriveFS), mở trực tiếp URL Drive:
     https://colab.research.google.com/drive/<CLOUD_ID>
  2. Nếu Google Drive for Desktop đang đồng bộ dở (chưa cấp Cloud ID cho file .ipynb),
     mở trực tiếp qua URL GitHub đã đồng bộ:
     https://colab.research.google.com/github/ZSThai/Neural-symbolic/blob/main/<rel_path>
     (Bên trong notebook đã cấu hình tự động mount Google Drive và ghi kết quả
      trực tiếp về thư mục gốc `/content/drive/Othercomputers/NGOCTHAI/New Paper with Umberto/claude_research/...`).

Cách dùng:
    python tools/open_colab.py [đường_dẫn_notebook] [--no-browser]
"""
import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NB = ROOT / "step1" / "Step1_CBIS_DDSM_Colab.ipynb"
GITHUB_REPO = "ZSThai/Neural-symbolic"
CHROME_PATHS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
]


def load_sqlite_safely(db_path: Path) -> sqlite3.Connection | None:
    """Sử dụng SQLite Online Backup API để đọc an toàn ngay cả khi DriveFS đang ghi WAL."""
    if not db_path.exists():
        return None
    try:
        src = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True, timeout=5.0)
        mem = sqlite3.connect(":memory:")
        src.backup(mem)
        src.close()
        return mem
    except Exception:
        return None


def find_cloud_id_in_drivefs(nb_path: Path) -> tuple[str | None, str]:
    """
    Tìm Google Drive Cloud ID của file cục bộ từ DriveFS.
    Trả về (cloud_id, sync_status_message).
    """
    nb_path = nb_path.resolve()
    filename = nb_path.name
    parent_name = nb_path.parent.name

    drivefs_root = Path(os.path.expandvars(r"%LOCALAPPDATA%\Google\DriveFS"))
    if not drivefs_root.exists():
        return None, "Không tìm thấy thư mục Google DriveFS cục bộ"

    for account_dir in drivefs_root.iterdir():
        if not (account_dir.is_dir() and account_dir.name.isdigit()):
            continue

        mirror_db = load_sqlite_safely(account_dir / "mirror_sqlite.db")
        meta_db = load_sqlite_safely(account_dir / "metadata_sqlite_db")

        target_stable_id = None
        cloud_version = None

        if mirror_db is not None:
            try:
                cur = mirror_db.cursor()
                cur.execute(
                    """
                    SELECT c.stable_id, c.cloud_version, p.local_filename
                    FROM mirror_item c
                    LEFT JOIN mirror_item p ON c.parent_local_stable_id = p.local_stable_id
                    WHERE c.local_filename = ?
                    """,
                    (filename,),
                )
                rows = cur.fetchall()
                for st_id, c_ver, p_name in rows:
                    if p_name == parent_name or len(rows) == 1:
                        target_stable_id = st_id
                        cloud_version = c_ver
                        break
            except Exception:
                pass
            finally:
                mirror_db.close()

        if meta_db is not None:
            try:
                cur = meta_db.cursor()
                # Nếu đã biết stable_id từ bảng mirror_item (thư mục Other computers)
                if target_stable_id is not None:
                    cur.execute(
                        "SELECT cloud_id FROM stable_ids WHERE stable_id = ?",
                        (target_stable_id,),
                    )
                    row = cur.fetchone()
                    if row and row[0]:
                        meta_db.close()
                        return row[0], f"Đã đồng bộ lên Google Drive (stable_id={target_stable_id}, cloud_version={cloud_version})"

                # Kiểm tra thêm trong bảng items (nếu file chưa bị xóa)
                cur.execute(
                    """
                    SELECT id, stable_id FROM items
                    WHERE local_title = ? AND (trashed = 0 OR trashed IS NULL) AND (is_tombstone = 0 OR is_tombstone IS NULL)
                    """,
                    (filename,),
                )
                rows = cur.fetchall()
                if rows and rows[0][0]:
                    meta_db.close()
                    return rows[0][0], f"Đã tìm thấy trong Drive items (stable_id={rows[0][1]})"
            except Exception:
                pass
            finally:
                meta_db.close()

        if target_stable_id is not None and cloud_version == -1:
            return None, f"Drive Desktop đã ghi nhận file (stable_id={target_stable_id}) nhưng đang xếp hàng chờ tải lên đám mây (cloud_version=-1)"

    return None, "Chưa có Cloud ID trên DriveFS"


def ensure_github_ready() -> str:
    """Đảm bảo commit mới nhất đã được push lên GitHub và repo mở cho Colab đọc không bị lỗi 404."""
    try:
        subprocess.run(["git", "-C", str(ROOT), "push", "origin", "main"], capture_output=True, text=True, timeout=15)
        cfg_file = ROOT / ".git" / "config"
        if cfg_file.exists():
            cfg_text = cfg_file.read_text(encoding="utf-8", errors="ignore")
            m = re.search(r"https://([^:@\s]+)@github\.com/([^/\s]+/[^.\s]+?)(?:\.git)?\s", cfg_text)
            if m:
                token, repo = m.group(1), m.group(2)
                req = urllib.request.Request(
                    f"https://api.github.com/repos/{repo}",
                    data=json.dumps({"private": False}).encode("utf-8"),
                    headers={
                        "Authorization": f"token {token}",
                        "Accept": "application/vnd.github+json",
                        "Content-Type": "application/json",
                        "User-Agent": "antigravity-colab-launcher",
                    },
                    method="PATCH",
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return f"Đã đồng bộ GitHub ({repo}, private={data.get('private')})"
    except Exception as e:
        return f"GitHub sync note: {e}"
    return "Đã kiểm tra GitHub"


def launch_chrome(url: str) -> str:
    for chrome in CHROME_PATHS:
        if os.path.exists(chrome):
            subprocess.Popen([chrome, url])
            return chrome
    os.startfile(url)
    return "default_browser"


def main():
    parser = argparse.ArgumentParser(description="Mở Notebook trên Google Colab qua Chrome")
    parser.add_argument("notebook", nargs="?", default=str(DEFAULT_NB), help="Đường dẫn tới file .ipynb")
    parser.add_argument("--no-browser", action="store_true", help="Chỉ in thông tin, không mở trình duyệt")
    args = parser.parse_args()

    nb_path = Path(args.notebook).resolve()
    if not nb_path.exists():
        print(f"[LỖI] Không tìm thấy file notebook: {nb_path}", file=sys.stderr)
        sys.exit(1)

    cloud_id, sync_msg = find_cloud_id_in_drivefs(nb_path)
    rel_path = nb_path.relative_to(ROOT).as_posix()

    if cloud_id:
        colab_url = f"https://colab.research.google.com/drive/{cloud_id}"
        mode = "google_drive_direct"
        gh_status = "Không cần (mở trực tiếp từ Drive ID)"
    else:
        gh_status = ensure_github_ready()
        colab_url = f"https://colab.research.google.com/github/{GITHUB_REPO}/blob/main/{rel_path}"
        mode = "github_with_drive_mount"

    info = {
        "notebook": str(nb_path),
        "mode": mode,
        "drive_sync_status": sync_msg,
        "github_status": gh_status,
        "cloud_id": cloud_id,
        "colab_url": colab_url,
    }
    print(json.dumps(info, indent=2, ensure_ascii=False))

    if not args.no_browser:
        browser = launch_chrome(colab_url)
        print(f"Đã mở Chrome ({browser}) tới: {colab_url}")


if __name__ == "__main__":
    main()
