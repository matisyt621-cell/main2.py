"""
OMEGA - proces renderujący w tle.
URUCHAMIANY jako osobny, w pełni niezależny proces przez omega_app.py
(subprocess.Popen). Działa całkowicie oddzielnie od sesji Streamlit -
zamknięcie karty przeglądarki NIE przerywa renderowania.

Użycie: python omega_worker.py --config <ścieżka_do_job_config.json>
"""

import argparse
import json
import os
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed

from omega_engine import _render_single_video


def _write_status(path, status):
    # Zapis atomowy (tmp + replace), żeby Streamlit nigdy nie odczytało
    # "połowicznie zapisanego" pliku statusu podczas pollingu.
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(status, f)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        job_config = json.load(f)

    jobs = job_config["jobs"]
    pack_size = job_config["pack_size"]
    parallel_workers = job_config["parallel_workers"]
    status_path = job_config["status_path"]
    output_dir = job_config["output_dir"]

    status = {
        "total": len(jobs), "done": 0, "results": [], "errors": [],
        "state": "running", "zip_files": []
    }
    _write_status(status_path, status)

    with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
        future_to_idx = {executor.submit(_render_single_video, job): job["idx"] for job in jobs}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            result = future.result()
            if isinstance(result, str) and result.startswith("__ERROR__"):
                status["errors"].append(result)
            else:
                status["results"].append(result)
            status["done"] += 1
            _write_status(status_path, status)

    # --- Pakowanie do ZIP: ZAWSZE po 20 filmów (stała wartość, na życzenie) ---
    zip_files = []
    video_paths = status["results"]
    for i in range(0, len(video_paths), pack_size):
        chunk = video_paths[i:i + pack_size]
        part_num = i // pack_size + 1
        zip_name = os.path.join(output_dir, f"OMEGA_PART_{part_num}.zip")
        with zipfile.ZipFile(zip_name, "w", compression=zipfile.ZIP_STORED) as z:
            for v in chunk:
                if os.path.exists(v):
                    z.write(v, arcname=os.path.basename(v))
        zip_files.append(zip_name)

    status["zip_files"] = zip_files
    status["state"] = "done"
    _write_status(status_path, status)


# WAŻNE: guard __main__ jest tu konieczny z tego samego powodu co wcześniej -
# ProcessPoolExecutor na Windows/macOS ("spawn") re-importuje ten plik w
# każdym procesie roboczym. Bez guarda main() (z parsowaniem argparse)
# wykonałoby się ponownie w każdym z nich.
if __name__ == "__main__":
    main()
