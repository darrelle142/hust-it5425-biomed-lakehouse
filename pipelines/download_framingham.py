"""
Script tải bộ dữ liệu Framingham Heart Study từ nguồn mở uy tín về thư mục data/landing/
Kèm chuẩn hóa ký tự xuống dòng (LF) và chữ ký kiểm toán SHA-256 theo chuẩn FDA 21 CFR Part 11
"""

import os
import sys
import urllib.request
import hashlib

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "landing")
TARGET_FILE = os.path.join(DATA_DIR, "framingham.csv")
HASH_FILE = os.path.join(DATA_DIR, "framingham.csv.sha256")

URLS = [
    "https://raw.githubusercontent.com/GauravPadawe/Framingham-Heart-Study/master/framingham.csv",
    "https://raw.githubusercontent.com/indrapaul824/Coronary-Heart-Disease-Prediction/master/framingham.csv",
    "https://raw.githubusercontent.com/kabdi101/CHD_prediction/master/framingham.csv"
]

def download_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"[*] Bắt đầu kiểm tra và tải bộ dữ liệu Framingham Heart Study...")
    
    downloaded = False
    if not os.path.exists(TARGET_FILE) or os.path.getsize(TARGET_FILE) < 50000:
        for url in URLS:
            try:
                print(f"[*] Đang tải từ mirror: {url}")
                urllib.request.urlretrieve(url, TARGET_FILE)
                if os.path.exists(TARGET_FILE) and os.path.getsize(TARGET_FILE) > 50000:
                    print(f"[+] Tải thành công! Kích thước thô: {os.path.getsize(TARGET_FILE):,} bytes")
                    downloaded = True
                    break
            except Exception as e:
                print(f"[-] Thất bại từ mirror này ({e}), đang thử mirror tiếp theo...")
    else:
        downloaded = True

    if not downloaded:
        print("[-] Không thể tải tự động từ các mirror online. Vui lòng kiểm tra kết nối mạng.")
        sys.exit(1)

    # Chuẩn hóa line endings \r thành \n
    with open(TARGET_FILE, "rb") as f:
        content = f.read()

    normalized_content = content.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    with open(TARGET_FILE, "wb") as f:
        f.write(normalized_content)

    # Tính toán SHA-256
    digest = hashlib.sha256(normalized_content).hexdigest()
    with open(HASH_FILE, "w", encoding="utf-8") as f:
        f.write(digest)

    print(f"[+] Đã chuẩn hóa ký tự xuống dòng và tạo chữ ký kiểm toán SHA-256: {digest}")
    print(f"[+] Dữ liệu đã sẵn sàng tại: {TARGET_FILE}")

if __name__ == "__main__":
    download_dataset()
