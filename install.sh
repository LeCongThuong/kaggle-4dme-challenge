#!/usr/bin/env bash
set -e

# ---------- paths ----------
REPO_URL="https://github.com/LeCongThuong/kaggle-4dme-challenge.git"  # change if forked
DATA_DIR="data"               # where the zip will be unpacked
ZIP_NAME="inference_artifacts.zip"

# ---------- clone & env ----------
conda create -n 4dmr python=3.9 -y
conda activate 4dmr

# git clone "$REPO_URL"
cd kaggle-4dme-challenge

# ---------- deps ----------
pip install -r requirements.txt

# ---------- download feature bundle (public, no token needed) ----------
pip install -q huggingface_hub
python - <<'PY'
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id="Thuong/kaggle-4DMR-features",
    repo_type="dataset",
    filename="inference_artifacts.zip",
    local_dir=".")
print("Downloaded to", path)
PY

# ---------- unzip ----------
unzip -q "$ZIP_NAME" -d "$DATA_DIR"

# directory check
echo "\nUnpacked structure:" && tree -L 2 "$DATA_DIR/inference_artifacts" || true