#!/usr/bin/env python3
"""Prefetch MedGemma and MultiCaRe assets for group_proj_vanda___fixed.py.

This script performs downloads only. It does not train or allocate a GPU.
Set HF_TOKEN in the shell or use an existing Hugging Face login.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the model and datasets into an isolated Hugging Face cache."
    )
    parser.add_argument(
        "--cache-dir",
        default="/scratch/<NUS-ID>/group_proj_cache_fast/hf_cache",
        help="Target cache directory.",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_HUB_CACHE"] = str(cache_dir / "hub")
    os.environ.pop("HF_HUB_OFFLINE", None)

    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)
        print("Hugging Face authentication succeeded.")
    else:
        print("No token environment variable found; using the cached Hugging Face login if available.")

    print("Cache directory:", cache_dir)

    model_path = snapshot_download(
        "google/medgemma-4b-it",
        ignore_patterns=["*.gguf", "*.msgpack", "*.h5"],
    )
    print("MedGemma downloaded:", model_path)

    image_path = snapshot_download(
        "OpenMed/multicare-images",
        repo_type="dataset",
        allow_patterns=["data/*.parquet"],
    )
    print("MultiCaRe image dataset downloaded:", image_path)

    cases_path = snapshot_download(
        "OpenMed/multicare-cases",
        repo_type="dataset",
        allow_patterns=["cases.parquet"],
    )
    print("MultiCaRe cases dataset downloaded:", cases_path)

    print("Prefetch complete.")


if __name__ == "__main__":
    main()
