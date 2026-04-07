import os
import requests

from config import UPLOAD_DIR, SERVER_PORT

base_url = os.getenv("TEST_BASE_URL", f"http://127.0.0.1:{SERVER_PORT}")
pdf_path = f"{UPLOAD_DIR}/grade-9-science-and-technology.pdf"

with open(pdf_path, "rb") as f:
    res1 = requests.post(f"{base_url}/api/upload", files={"file": f})

res2 = requests.post(f"{base_url}/api/analyze_global")

with open("test_out.txt", "w", encoding="utf-8") as f:
    f.write(f"Status: {res2.status_code}\n")
    f.write(res2.text)
