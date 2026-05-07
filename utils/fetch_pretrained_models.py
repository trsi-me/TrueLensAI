# -*- coding: utf-8 -*-
# Download pretrained model files when URLs / base URL are set (deployment without git-lfs blobs).
from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request
from urllib.parse import urlparse

from config import Config

CHUNK_BYTES = 4 * 1024 * 1024
USER_AGENT = "TrueLensAI-fetch-pretrained/1.0"


def _truthy(env_value: str) -> bool:
    return env_value.strip().lower() in ("1", "true", "yes", "on")


def pretrained_sources_configured() -> bool:
    return bool(_configured_urls())


def _configured_urls() -> list[tuple[str, str]]:
    base = os.environ.get("PRETRAINED_MODELS_BASE_URL", "").strip().rstrip("/")
    out: list[tuple[str, str]] = []
    # Optional: only explicit URL (not derived from PRETRAINED_MODELS_BASE_URL) to avoid 404s.
    tflite_explicit = os.environ.get("IMAGE_MODEL_TFLITE_DOWNLOAD_URL", "").strip()
    if tflite_explicit:
        out.append(
            (tflite_explicit, _dest_for(os.path.basename(Config.IMAGE_MODEL_TFLITE_PATH)))
        )
    items: list[tuple[str, str, str]] = [
        (
            "IMAGE_MODEL_DOWNLOAD_URL",
            os.environ.get("IMAGE_MODEL_DOWNLOAD_URL", "").strip(),
            os.path.basename(Config.IMAGE_MODEL_PATH),
        ),
        (
            "TEXT_MODEL_DOWNLOAD_URL",
            os.environ.get("TEXT_MODEL_DOWNLOAD_URL", "").strip(),
            os.path.basename(Config.TEXT_MODEL_PATH),
        ),
        (
            "TFIDF_VECTORIZER_DOWNLOAD_URL",
            os.environ.get("TFIDF_VECTORIZER_DOWNLOAD_URL", "").strip(),
            os.path.basename(Config.TFIDF_PATH),
        ),
    ]
    for _env_key, explicit, filename in items:
        if explicit:
            out.append((explicit, _dest_for(filename)))
            continue
        if base:
            out.append((f"{base}/{filename}", _dest_for(filename)))
    return out


def _dest_for(filename: str) -> str:
    return os.path.join(Config.BASE_DIR, "ml_models", "saved_models", filename)


def _should_skip(dest: str, min_bytes: int) -> bool:
    try:
        if os.path.isfile(dest) and os.path.getsize(dest) >= min_bytes:
            return True
    except OSError:
        pass
    return False


def _download(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            # Some CDNs return 403 without Accept
            "Accept": "*/*",
        },
        method="GET",
    )
    # Large .h5: long timeout prevents slow CDN / cold storage from failing mid-transfer.
    read_timeout_sec = int(os.environ.get("PRETRAINED_FETCH_TIMEOUT_SECONDS", "7200"))
    try:
        with urllib.request.urlopen(req, timeout=read_timeout_sec) as resp:
            with open(tmp, "wb") as f:
                while True:
                    chunk = resp.read(CHUNK_BYTES)
                    if not chunk:
                        break
                    f.write(chunk)
        os.replace(tmp, dest)
    except BaseException:
        try:
            if os.path.isfile(tmp):
                os.remove(tmp)
        except OSError:
            pass
        raise


def fetch_if_configured(*, force: bool = False, skip_existing: bool = True) -> dict:
    """
    If PRETRAINED_MODELS_BASE_URL and/or per-file *_DOWNLOAD_URL env vars are set,
    download artifacts into ml_models/saved_models/.

    Returns dict: ok (bool), downloaded (list), skipped (list), errors (list of str).
    """
    pairs = _configured_urls()
    result: dict = {
        "ok": True,
        "downloaded": [],
        "skipped": [],
        "errors": [],
    }
    if not pairs:
        print(
            "fetch_pretrained_models: no URLs configured "
            "(set PRETRAINED_MODELS_BASE_URL or *_DOWNLOAD_URL env vars).",
            file=sys.stderr,
        )
        return result

    min_bytes = 0 if force else 512

    for url, dest in pairs:
        netloc = urlparse(url).netloc
        if not netloc:
            result["ok"] = False
            result["errors"].append(f"Invalid URL (missing host): {url!r}")
            continue
        if skip_existing and not force and _should_skip(dest, min_bytes):
            result["skipped"].append(dest)
            print("fetch_pretrained_models: skip existing", dest, file=sys.stderr)
            continue
        try:
            print("fetch_pretrained_models: downloading", url, "->", dest, file=sys.stderr)
            _download(url, dest)
            result["downloaded"].append(dest)
        except urllib.error.HTTPError as e:
            result["ok"] = False
            result["errors"].append(f"HTTP {e.code} for {url}: {e.reason}")
        except urllib.error.URLError as e:
            result["ok"] = False
            result["errors"].append(f"URL error for {url}: {e.reason!r}")
        except OSError as e:
            result["ok"] = False
            result["errors"].append(f"IO error for {dest}: {e}")
    return result


def main() -> int:
    if _truthy(os.environ.get("PRETRAINED_FETCH_REQUIRED", "")) and not pretrained_sources_configured():
        print(
            "fetch_pretrained_models: PRETRAINED_FETCH_REQUIRED is set but no download URLs configured.",
            file=sys.stderr,
        )
        return 1
    force = _truthy(os.environ.get("PRETRAINED_FETCH_FORCE", ""))
    res = fetch_if_configured(force=force, skip_existing=not force)
    for err in res["errors"]:
        print(err, file=sys.stderr)
    if not res["errors"] and (res["downloaded"] or res["skipped"]):
        print(
            "fetch_pretrained_models: done (downloaded=%d skipped=%d)"
            % (len(res["downloaded"]), len(res["skipped"])),
            file=sys.stderr,
        )
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
