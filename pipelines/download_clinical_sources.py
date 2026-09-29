"""
Script: download_clinical_sources.py
Chức năng: Tải và khởi tạo 3 nguồn dữ liệu độc lập cho dự án y sinh:
  1. Nguồn 1: Framingham Heart Study (4,240 ca dịch tễ Boston, Mỹ)
  2. Nguồn 2: Cardiovascular Disease Clinical Study (5,000 ca lâm sàng quốc tế)
  3. Nguồn 3: AHA/ACC & JNC-7 Clinical Guidelines (Bảng tra cứu quy chuẩn y tế chuẩn ICD-10)
Phụ trách: TV 1 (Clinical Data Engineering Lead)
"""

import os
import sys
import urllib.request
import hashlib
import polars as pl

# Khắc phục bảng mã stdout Windows cp1258
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LANDING_DIR = os.path.join(PROJECT_ROOT, "data", "landing")

FRAMINGHAM_FILE = os.path.join(LANDING_DIR, "framingham.csv")
CARDIO_FILE = os.path.join(LANDING_DIR, "cardio_study.csv")
GUIDELINES_FILE = os.path.join(LANDING_DIR, "ref_aha_jnc7_guidelines.csv")

def generate_sha256(file_path: str):
    with open(file_path, "rb") as f:
        content = f.read()
    digest = hashlib.sha256(content).hexdigest()
    with open(file_path + ".sha256", "w", encoding="utf-8") as f:
        f.write(digest)
    return digest

def download_framingham():
    """Tải và chuẩn hóa Framingham Heart Study (4,240 bản ghi)"""
    if os.path.exists(FRAMINGHAM_FILE) and os.path.getsize(FRAMINGHAM_FILE) > 50000:
        print(f"[+] Nguồn 1 (Framingham): Đã có sẵn tại {os.path.basename(FRAMINGHAM_FILE)}")
        return FRAMINGHAM_FILE

    urls = [
        "https://raw.githubusercontent.com/GauravPadawe/Framingham-Heart-Study/master/framingham.csv",
        "https://raw.githubusercontent.com/indrapaul824/Coronary-Heart-Disease-Prediction/master/framingham.csv"
    ]
    for url in urls:
        try:
            print(f"[*] Đang tải Framingham từ: {url}")
            urllib.request.urlretrieve(url, FRAMINGHAM_FILE)
            if os.path.exists(FRAMINGHAM_FILE) and os.path.getsize(FRAMINGHAM_FILE) > 50000:
                break
        except Exception:
            continue

    with open(FRAMINGHAM_FILE, "rb") as f:
        content = f.read().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    with open(FRAMINGHAM_FILE, "wb") as f:
        f.write(content)

    digest = generate_sha256(FRAMINGHAM_FILE)
    print(f"[+] Nguồn 1 (Framingham): Tải hoàn tất (SHA-256: {digest[:16]}...)")
    return FRAMINGHAM_FILE

def download_cardio_study(target_rows: int = 5000):
    """Tải đúng 5,000 bản ghi thực tế từ Cardiovascular Disease Clinical Study"""
    if os.path.exists(CARDIO_FILE) and os.path.getsize(CARDIO_FILE) > 100000:
        print(f"[+] Nguồn 2 (Cardio Study): Đã có sẵn ({target_rows:,} ca) tại {os.path.basename(CARDIO_FILE)}")
        return CARDIO_FILE

    url = "https://raw.githubusercontent.com/caravanuden/cardio/master/cardio_train.csv"
    print(f"[*] Đang truyền phát (streaming) đúng {target_rows:,} bản ghi từ: {url}")

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    lines = []
    with urllib.request.urlopen(req, timeout=15) as resp:
        header = resp.readline().decode("utf-8").strip()
        lines.append(header)
        for _ in range(target_rows):
            line = resp.readline().decode("utf-8").strip()
            if not line:
                break
            lines.append(line)

    with open(CARDIO_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    digest = generate_sha256(CARDIO_FILE)
    print(f"[+] Nguồn 2 (Cardio Study): Đã nạp thành công {len(lines)-1:,} bản ghi (SHA-256: {digest[:16]}...)")
    return CARDIO_FILE

def create_aha_guidelines():
    """Tạo bảng chuẩn quy tắc lâm sàng của Hội Tim Mạch Hoa Kỳ (AHA/ACC & JNC-7) kèm mã ICD-10"""
    guidelines_data = {
        "guideline_id": [1, 2, 3, 4, 5],
        "bp_stage": ["NORMAL", "ELEVATED", "STAGE_1_HYPERTENSION", "STAGE_2_HYPERTENSION", "HYPERTENSIVE_CRISIS"],
        "min_sys_bp": [0.0, 120.0, 130.0, 140.0, 180.0],
        "max_sys_bp": [119.9, 129.9, 139.9, 179.9, 999.0],
        "min_dia_bp": [0.0, 0.0, 80.0, 90.0, 120.0],
        "max_dia_bp": [79.9, 79.9, 89.9, 119.9, 999.0],
        "clinical_action": [
            "Maintain healthy lifestyle habits",
            "Nonpharmacologic therapy and lifestyle modification",
            "Clinical evaluation and medication consideration",
            "Prompt antihypertensive medication and frequent monitoring",
            "Emergency hospital care and acute intervention"
        ],
        "icd10_code": ["R03.0", "R03.0", "I10", "I10.9", "I16.9"]
    }
    df = pl.DataFrame(guidelines_data)
    df.write_csv(GUIDELINES_FILE)
    digest = generate_sha256(GUIDELINES_FILE)
    print(f"[+] Nguồn 3 (AHA Guidelines): Bảng chuẩn hóa y khoa sẵn sàng tại {os.path.basename(GUIDELINES_FILE)} (SHA-256: {digest[:16]}...)")
    return GUIDELINES_FILE

def init_all_sources():
    os.makedirs(LANDING_DIR, exist_ok=True)
    print("=" * 70)
    print("[*] KHỞI TẠO & KIỂM TOÁN 3 NGUỒN DỮ LIỆU ĐỘC LẬP (LANDING ZONE)")
    print("=" * 70)
    download_framingham()
    download_cardio_study(target_rows=5000)
    create_aha_guidelines()
    print("=" * 70)

if __name__ == "__main__":
    init_all_sources()
