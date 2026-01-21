import os
import subprocess
import sys


def write_excel(rows, out_path):
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: pandas. Install with: python -m pip install pandas openpyxl"
        ) from exc
    try:
        import openpyxl  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: openpyxl. Install with: python -m pip install openpyxl"
        ) from exc
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_excel(out_path, index=False)


def open_excel_file(path):
    if not path:
        return
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)
            return
        if sys.platform == "darwin":
            subprocess.run(["open", path], check=False)
            return
        subprocess.run(["xdg-open", path], check=False)
    except Exception:
        return
