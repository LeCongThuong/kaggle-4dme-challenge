#!/usr/bin/env bash
set -e

ZIP=inference_artifacts.zip
DEST=data

python - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="Thuong/kaggle-4DMR-features",
    repo_type="dataset",
    filename="inference_artifacts.zip",
    local_dir=".")
PY

unzip -q "$ZIP" -d "$DEST"