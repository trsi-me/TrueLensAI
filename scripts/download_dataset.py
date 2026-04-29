# -*- coding: utf-8 -*-
# Download Kaggle datasets into datasate/ (requires Kaggle API credentials).
import argparse
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "datasate")


def _kaggle_json_path():
    return os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")


def _has_kaggle_credentials():
    if os.path.isfile(_kaggle_json_path()):
        return True
    user = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    return bool(user and key)


def _print_kaggle_setup_error():
    cfg = _kaggle_json_path()
    print("Kaggle API credentials not found.", file=sys.stderr)
    print("", file=sys.stderr)
    print("Option A (config file):", file=sys.stderr)
    print("  1. On kaggle.com: Account -> Settings -> API -> Create New Token", file=sys.stderr)
    print("  2. Save the downloaded file as:", file=sys.stderr)
    print("     " + cfg, file=sys.stderr)
    print("  3. Create folder if needed: " + os.path.dirname(cfg), file=sys.stderr)
    print("", file=sys.stderr)
    print("Option B (environment variables, same session):", file=sys.stderr)
    print("  set KAGGLE_USERNAME=your_kaggle_username", file=sys.stderr)
    print("  set KAGGLE_KEY=your_api_key", file=sys.stderr)
    print("", file=sys.stderr)
    print("Then run this script again.", file=sys.stderr)


def _import_kaggle_api():
    if not _has_kaggle_credentials():
        _print_kaggle_setup_error()
        sys.exit(1)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("Install package: pip install kaggle", file=sys.stderr)
        sys.exit(1)
    return KaggleApi


def download_welfake():
    KaggleApi = _import_kaggle_api()
    os.makedirs(RAW_DIR, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    slug = "saurabhshahane/fake-news-classification"
    out = os.path.join(RAW_DIR, "welfake")
    os.makedirs(out, exist_ok=True)
    api.dataset_download_files(slug, path=out, unzip=True)
    print("Downloaded WELFake to:", out)


def download_cifake():
    KaggleApi = _import_kaggle_api()
    os.makedirs(RAW_DIR, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    slug = "birdy654/cifake-real-and-ai-generated-synthetic-images"
    out = os.path.join(RAW_DIR, "cifake")
    os.makedirs(out, exist_ok=True)
    api.dataset_download_files(slug, path=out, unzip=True)
    print("Downloaded CIFAKE to:", out)


def main():
    p = argparse.ArgumentParser(description="Download dataset from Kaggle")
    p.add_argument(
        "dataset",
        choices=["welfake", "cifake", "all"],
        help="Dataset name",
    )
    args = p.parse_args()
    if args.dataset in ("welfake", "all"):
        download_welfake()
    if args.dataset in ("cifake", "all"):
        download_cifake()


if __name__ == "__main__":
    main()
