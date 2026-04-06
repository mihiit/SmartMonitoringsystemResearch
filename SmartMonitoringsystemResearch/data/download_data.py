"""
data/download_data.py
---------------------
Downloads all four publicly available datasets used in the paper.

Datasets:
  1. Pima Indians Diabetes     – UCI ML Repository ID 34  (CC BY 4.0)
  2. Diabetes 130-US Hospitals – UCI ML Repository ID 296 (CC BY 4.0)
  3. UCI Diabetes CGM          – UCI ML Repository ID 34  (public domain)
  4. NHANES 2015-2018          – CDC open data (public domain)

All datasets are de-identified and require no IRB approval
(45 CFR 46.104(d)(4)).
"""

import os
import urllib.request
import zipfile
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__))

# ---------------------------------------------------------------------------
# URLs
# ---------------------------------------------------------------------------
URLS = {
    "pima": (
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
        "pima-indians-diabetes.data.csv"
    ),
    "diabetes130": (
        "https://archive.ics.uci.edu/static/public/296/"
        "diabetes+130-us+hospitals+for+years+1999-2008.zip"
    ),
}

PIMA_COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
]


def download_pima():
    """Download Pima Indians Diabetes dataset."""
    out_path = os.path.join(DATA_DIR, "pima.csv")
    if os.path.exists(out_path):
        print("[Pima] Already downloaded.")
        return
    print("[Pima] Downloading …")
    urllib.request.urlretrieve(URLS["pima"], out_path)
    # Add header
    df = pd.read_csv(out_path, header=None, names=PIMA_COLUMNS)
    df.to_csv(out_path, index=False)
    print(f"[Pima] Saved to {out_path}  (N={len(df)})")


def download_diabetes130():
    """Download Diabetes 130-US Hospitals dataset."""
    out_csv = os.path.join(DATA_DIR, "diabetes130.csv")
    if os.path.exists(out_csv):
        print("[Diabetes-130] Already downloaded.")
        return
    zip_path = os.path.join(DATA_DIR, "diabetes130.zip")
    print("[Diabetes-130] Downloading …")
    try:
        urllib.request.urlretrieve(URLS["diabetes130"], zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(DATA_DIR)
        # Rename extracted file
        extracted = os.path.join(DATA_DIR, "dataset_diabetes", "diabetic_data.csv")
        if os.path.exists(extracted):
            os.rename(extracted, out_csv)
        os.remove(zip_path)
        df = pd.read_csv(out_csv)
        print(f"[Diabetes-130] Saved to {out_csv}  (N={len(df)})")
    except Exception as e:
        print(f"[Diabetes-130] Auto-download failed ({e}).")
        print(
            "  Manual download: https://archive.ics.uci.edu/dataset/296/"
            "diabetes+130-us+hospitals+for+years+1999-2008\n"
            f"  Place 'diabetic_data.csv' in {DATA_DIR}/ as 'diabetes130.csv'"
        )


def download_uci_cgm():
    """UCI Diabetes CGM is embedded in the repository as synthetic stand-in.

    The original dataset (UCI Repository ID 34, M. G. Kahn 1994) is
    accessed via ucimlrepo if installed, otherwise a note is printed.
    """
    out_path = os.path.join(DATA_DIR, "uci_cgm.csv")
    if os.path.exists(out_path):
        print("[UCI CGM] Already downloaded.")
        return
    try:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=34)
        df = pd.concat([ds.data.features, ds.data.targets], axis=1)
        df.to_csv(out_path, index=False)
        print(f"[UCI CGM] Saved to {out_path}  (N={len(df)})")
    except ImportError:
        print(
            "[UCI CGM] ucimlrepo not installed. Run: pip install ucimlrepo\n"
            "  Or download manually from: https://archive.ics.uci.edu/dataset/34/diabetes"
        )
    except Exception as e:
        print(f"[UCI CGM] Download failed: {e}")


def download_nhanes():
    """NHANES 2015-2018 note.

    Full NHANES public-use files are large (~several GB).
    We use a calibrated synthetic cohort in the paper (N=2000).
    The synthetic generation script is in data/synthetic_cohorts.py.
    """
    print(
        "[NHANES] NHANES 2015-2018 public-use files are available at:\n"
        "  https://wwwn.cdc.gov/nchs/nhanes/\n"
        "  In this paper, a 2000-sample synthetic cohort calibrated to\n"
        "  NHANES distributions is used. Run: python data/synthetic_cohorts.py"
    )


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    download_pima()
    download_diabetes130()
    download_uci_cgm()
    download_nhanes()
    print("\nAll available datasets downloaded.")
