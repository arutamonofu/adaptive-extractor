#!/usr/bin/env python3
import time, requests
from pathlib import Path

HOST = "https://uniparser.dp.tech"
DELAY = 2  # секунды между запросами (защита от rate-limit)

def parse_file(pdf_path: str, out_path: str, api_key: str) -> None:
    token = Path(pdf_path).stem[:15]
    
    # 1. Отправка (sync=False → асинхронно)
    with open(pdf_path, "rb") as f:
        r = requests.post(f"{HOST}/trigger-file-async",
            files={"file": f},
            data={"token": token, "sync": False, "textual": 2, "table": 1, 
                  "equation": 1, "expression": 1, "chart": 0, "molecule": 0, "figure": 0},
            headers={"X-API-Key": api_key}, timeout=1200)
    r.raise_for_status()

    # 2. Опрос результата
    fmt = {"token": token, "textual": "markdown", "table": "markdown", 
           "equation": "markdown", "expression": "markdown"}
    
    for _ in range(120):  # макс. 10 мин
        r = requests.post(f"{HOST}/get-formatted", json=fmt, 
                          headers={"X-API-Key": api_key}, timeout=1200)
        r.raise_for_status()
        res = r.json()
        if res.get("status") == "success":
            Path(out_path).write_text(res.get("content", ""), encoding="utf-8")
            return
        time.sleep(5)
    raise TimeoutError(f"Timeout: {pdf_path}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("input_dir", help="Папка с PDF")
    p.add_argument("output_dir", help="Папка для MD")
    p.add_argument("-k", "--api-key", required=True)
    args = p.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    pdfs = list(Path(args.input_dir).glob("*.pdf"))
    
    print(f"Найдено файлов: {len(pdfs)}")
    for i, pdf in enumerate(pdfs, 1):
        out = Path(args.output_dir) / f"{pdf.stem}.md"
        print(f"[{i}/{len(pdfs)}] {pdf.name}...", end=" ", flush=True)
        try:
            parse_file(str(pdf), str(out), args.api_key)
            print("✓")
        except Exception as e:
            print(f"✗ ({e})")
        if i < len(pdfs): time.sleep(DELAY)  # пауза между файлами