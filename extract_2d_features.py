#!/usr/bin/env python3
"""
RADIO summary-feature extractor, letting you choose exactly where the model
weights are cached.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image
from torch.nn import functional as F
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm

# ─────────────────────────── CLI ──────────────────────────────────────────────
def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RADIO summary extractor")
    add_yaml_defaults(p, "extract_2d")      
    p.add_argument("--dataset_root")
    p.add_argument("--features_root")
    p.add_argument("--model_version")
    p.add_argument("--model_cache")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    return args


def add_yaml_defaults(parser: argparse.ArgumentParser, section: str):
    """Inject --config flag and push YAML values into parser defaults."""
    parser.add_argument("--config", help="path to exp.yaml", nargs="?")
    cfg_path, _ = parser.parse_known_args()  
    if cfg_path.config:
        with open(cfg_path.config, "r") as fh:
            blob = yaml.safe_load(fh)
        if section not in blob:
            print(f"[ERR] Section '{section}' not found in {cfg_path.config}", file=sys.stderr)
            sys.exit(1)
        parser.set_defaults(**blob[section])
# ────────────────────────────────

def load_model(version: str, device: str, model_cache_dir: str):
    """
    Download (if needed) and load RADIO under `model_cache_dir`.
    """
    cache_path = Path(model_cache_dir).expanduser()
    cache_path.mkdir(parents=True, exist_ok=True)

    os.environ["TORCH_HOME"] = str(cache_path)          
    torch.hub.set_dir(str(cache_path))                 

    model = torch.hub.load("NVlabs/RADIO", "radio_model",
                           version=version,
                           progress=True, skip_validation=True)
    return model.to(device).eval()


def save_feature(tensor: torch.Tensor, img_path: Path, features_root: Path):
    rel = Path(img_path).stem
    seq_id = str(img_path).split("/")[-2]
    out_dir_path = os.path.join(features_root, seq_id)
    Path(out_dir_path).mkdir(exist_ok=True, parents=True)
    out_path = os.path.join(out_dir_path, f"{rel}.pt")
    torch.save(tensor.cpu(), out_path)


# ─────────────────────────── MAIN ─────────────────────────────────────────────
@torch.no_grad()
def main() -> None:
    args = get_args()
    dataset_root  = Path(args.dataset_root).expanduser()
    features_root = Path(args.features_root).expanduser()

    model = load_model(args.model_version, args.device, args.model_cache)

    all_imgs = list(Path(dataset_root).rglob("*.png"))
    if not all_imgs:
        print("No images found", file=sys.stderr)
        sys.exit(1)

    for img_path in tqdm(all_imgs, desc="Extracting", unit="img"):
        x = pil_to_tensor(Image.open(img_path).convert("RGB")).to(
                dtype=torch.float32, device=args.device) / 255.0
        x = x.unsqueeze(0)
        nearest_res = model.get_nearest_supported_resolution(*x.shape[-2:])
        x = F.interpolate(x, nearest_res, mode='bilinear', align_corners=False)
        summary, _ = model(x, feature_fmt="NCHW")
        save_feature(summary, img_path, features_root)

    print("All features written to", features_root)

if __name__ == "__main__":
    main()
