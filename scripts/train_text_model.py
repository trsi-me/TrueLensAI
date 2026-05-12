# -*- coding: utf-8 -*-
# Train fake-news classifier: TF-IDF + PassiveAggressiveClassifier (WELFake CSV).
import argparse
import glob
import io
import os
import re
import shutil
import sys
import tempfile
import unicodedata

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Paths from this file (avoid import config: cwd / duplicate modules).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(_SCRIPT_DIR)
DATASET_ROOT = os.path.join(BASE_DIR, "datasate")
WELFAKE_CSV_PATH = os.path.join(DATASET_ROOT, "WELFake_Dataset.csv")
DEFAULT_CSV_GLOB = os.path.join(DATASET_ROOT, "welfake", "**", "*.csv")
FALLBACK_CSV_GLOB = os.path.join(DATASET_ROOT, "**", "*.csv")
OUT_MODEL = os.path.join(BASE_DIR, "ml_models", "saved_models", "text_model.pkl")
OUT_VEC = os.path.join(BASE_DIR, "ml_models", "saved_models", "tfidf_vectorizer.pkl")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = unicodedata.normalize("NFC", text)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"http\S+", " ", t)
    t = re.sub(r"[^\w\s\u0600-\u06FF]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def find_csv(path_glob: str) -> str:
    matches = sorted(glob.glob(path_glob, recursive=True))
    if not matches:
        raise FileNotFoundError("No CSV found. Download the dataset or pass --csv.")
    return matches[0]


def resolve_csv_path(csv_arg: str, glob_arg: str) -> str:
    if csv_arg:
        return csv_arg
    if os.path.isfile(WELFAKE_CSV_PATH):
        return WELFAKE_CSV_PATH
    # Kaggle unzip often creates a folder named WELFake_Dataset.csv with the file inside
    if os.path.isdir(WELFAKE_CSV_PATH):
        nested = os.path.join(WELFAKE_CSV_PATH, "WELFake_Dataset.csv")
        if os.path.isfile(nested):
            return nested
    try:
        return find_csv(glob_arg)
    except FileNotFoundError:
        if glob_arg == DEFAULT_CSV_GLOB:
            try:
                return find_csv(FALLBACK_CSV_GLOB)
            except FileNotFoundError as err:
                raise FileNotFoundError(
                    "No CSV found. Place WELFake_Dataset.csv in datasate/ or use --csv."
                ) from err
        raise


def _read_csv_via_temp_copy(csv_path: str):
    # Copy to temp when another process locks the original path.
    fd, tmp = tempfile.mkstemp(suffix=".csv", prefix="welfake_")
    os.close(fd)
    try:
        shutil.copy2(csv_path, tmp)
        return pd.read_csv(tmp, encoding="utf-8-sig", on_bad_lines="skip")
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _read_csv_win32_shared(csv_path: str):
    # Win32: FILE_SHARE_READ|WRITE; CreateFileW restype must be c_void_p (64-bit handle).
    import ctypes
    import msvcrt

    path = os.path.abspath(csv_path)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.restype = ctypes.c_void_p
    kernel32.CreateFileW.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]

    GENERIC_READ = 0x80000000
    FILE_SHARE_READ = 0x00000001
    FILE_SHARE_WRITE = 0x00000002
    FILE_SHARE_DELETE = 0x00000004
    OPEN_EXISTING = 3
    FILE_ATTRIBUTE_NORMAL = 0x80

    handle = kernel32.CreateFileW(
        path,
        GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        None,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        None,
    )
    inv = ctypes.c_void_p(-1).value
    if isinstance(handle, ctypes.c_void_p):
        h = handle.value
    else:
        h = int(handle)
    if h == inv or h == 0xFFFFFFFFFFFFFFFF:
        # ACCESS_DENIED or locked; temp copy often still works
        return _read_csv_via_temp_copy(csv_path)

    try:
        fd = msvcrt.open_osfhandle(h, os.O_RDONLY)
    except OSError as e:
        if getattr(e, "errno", None) == 9:
            return _read_csv_via_temp_copy(csv_path)
        raise
    binf = os.fdopen(fd, "rb", closefd=True)
    text = io.TextIOWrapper(binf, encoding="utf-8-sig", newline="")
    try:
        return pd.read_csv(text, on_bad_lines="skip")
    finally:
        text.close()


def _read_csv_pandas(csv_path: str):
    try:
        return pd.read_csv(csv_path, encoding="utf-8-sig", on_bad_lines="skip")
    except PermissionError:
        pass
    except OSError as e:
        # Windows often raises OSError(13) instead of PermissionError for locked files
        if getattr(e, "errno", None) not in (13, 5):
            raise
    if sys.platform != "win32":
        raise PermissionError(
            "Permission denied reading CSV. Close other apps using the file or use --csv with a copy."
        )
    try:
        return _read_csv_win32_shared(csv_path)
    except (PermissionError, OSError) as err:
        try:
            return _read_csv_via_temp_copy(csv_path)
        except Exception:
            raise PermissionError(
                "Permission denied reading CSV. Close Excel, "
                "ensure OneDrive file is fully downloaded, or use --csv path\\to\\copy.csv"
            ) from err


def load_dataframe(csv_path: str) -> tuple:
    df = _read_csv_pandas(csv_path)
    cols = {c.lower(): c for c in df.columns}
    label_col = None
    for key in ("label", "labels", "class"):
        if key in cols:
            label_col = cols[key]
            break
    if label_col is None:
        for c in df.columns:
            if str(c).lower() not in ("title", "text", "article"):
                label_col = c
                break
    if label_col is None:
        raise ValueError("Could not detect label column.")
    if "title" in cols and "text" in cols:
        texts = (
            df[cols["title"]].fillna("").astype(str)
            + " "
            + df[cols["text"]].fillna("").astype(str)
        )
    else:
        text_col = None
        for key in ("text", "title", "article"):
            if key in cols:
                text_col = cols[key]
                break
        if text_col is None:
            text_col = df.columns[0]
        texts = df[text_col].fillna("").astype(str)
    y = df[label_col]
    if y.dtype == object:
        y = y.map(lambda x: 1 if str(x).strip() in ("1", "true", "real", "REAL") else 0)
    y = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)
    y = y.clip(0, 1)
    X = texts.map(clean_text)
    mask = X.str.len() >= 10
    X = X[mask]
    y = y[mask]
    return X.tolist(), y.values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="Path to CSV file")
    ap.add_argument("--glob", default=DEFAULT_CSV_GLOB, help="Glob pattern to find CSV")
    ap.add_argument(
        "--swap-binary-labels",
        action="store_true",
        help="After loading labels, set y := 1 - y (use if CSV uses 0=real, 1=fake instead of WELFake 0=fake, 1=real).",
    )
    args = ap.parse_args()
    csv_path = resolve_csv_path(args.csv, args.glob)
    print("Using CSV:", csv_path)
    X, y = load_dataframe(csv_path)
    if args.swap_binary_labels:
        y = 1 - y.astype(int)
    print("Samples after filtering:", len(y))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=50000,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                PassiveAggressiveClassifier(
                    loss="hinge",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print("Accuracy:", round(acc, 4))
    print(classification_report(y_test, pred, digits=4))
    os.makedirs(os.path.dirname(OUT_MODEL), exist_ok=True)
    joblib.dump(pipeline.named_steps["clf"], OUT_MODEL)
    joblib.dump(pipeline.named_steps["tfidf"], OUT_VEC)
    print("Saved classifier:", OUT_MODEL)
    print("Saved vectorizer:", OUT_VEC)
    if acc < 0.93:
        print(
            "Warning: accuracy below 93%. Try more data or tune TF-IDF.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
