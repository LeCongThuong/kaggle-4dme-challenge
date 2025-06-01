#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_full.py  ·  inference + post-processing in one go
"""

from __future__ import annotations
import argparse, re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import os, random, re 
import yaml, argparse, sys
# ───────────────────────── constants ─────────────────────────
D_MODEL     = 1664
K_FRAMES    = 16
CLASS_NAMES = ["Negative", "Positive", "Repression", "Surprise", "Others"]

def set_deterministic(seed: int = 0) -> None:
    """
    Force torch / numpy / python-random (and CUDA if present) to behave
    deterministically so that repeated runs give identical results.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # cuDNN specific
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    

# ─────────────────────── helper functions ────────────────────

def add_yaml_defaults(parser: argparse.ArgumentParser, section: str):
    """Inject --config flag and push YAML values into parser defaults."""
    parser.add_argument("--config", help="path to exp.yaml", nargs="?")
    # we only need to know --config for now
    cfg_path, _ = parser.parse_known_args()  
    if cfg_path.config:
        with open(cfg_path.config, "r") as fh:
            blob = yaml.safe_load(fh)
        if section not in blob:
            print(f"[ERR] Section '{section}' not found in {cfg_path.config}", file=sys.stderr)
            sys.exit(1)
        parser.set_defaults(**blob[section])
# ────────────────────────────────

def _norm(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / (v.norm(p=2) + eps)

def load_expr_coeff(npy_path: Path) -> torch.Tensor:
    coeffs = np.load(npy_path, allow_pickle=True).item()["coeffs"]
    return torch.from_numpy(coeffs[:, 156:327].ravel()).float()

def load_tensor(p: Path) -> torch.Tensor:
    if p.suffix == ".pt":
        return torch.load(p, map_location="cpu", weights_only=True)
    return torch.as_tensor(np.load(p, mmap_mode="r"), dtype=torch.float32, device="cpu")

def uniform_sample(n: int, k: int) -> List[int]:
    """
    Return k frame indices spaced as evenly as possible across [0, n-1].

    • When k ≥ n  →   just return [0, 1, …, n-1] and pad the tail with (n-1)
                      
    • When k == 1  →  pick the midpoint frame (same as n//2).
    • Otherwise    →  round the float positions produced by linspace.
    """
    if k >= n:
        return list(range(n)) + [n - 1] * (k - n)

    if k == 1:
        return [n // 2]

    return [round(i * (n - 1) / (k - 1)) for i in range(k)]

def build_clip(seq_name: str, feat2d_root: Path, feat3d_root: Path) -> torch.Tensor:
    dir2d, dir3d = feat2d_root / seq_name, feat3d_root / seq_name
    files2d = sorted(dir2d.glob("*.pt")) or sorted(dir2d.glob("*.npy"))
    files3d = sorted(dir3d.glob("*.npy"))
    if not files3d or not files2d:
        raise FileNotFoundError(f"Missing features for {seq_name}")

    n = min(len(files2d), len(files3d))
    idxs = uniform_sample(n, K_FRAMES)

    frames = []
    for i in idxs:
        f2d = _norm(load_tensor(files2d[i]).flatten())
        f3d = _norm(load_expr_coeff(files3d[i]))
        frames.append(torch.cat([f2d, f3d], dim=0))

    return torch.stack(frames)[:, :D_MODEL]          # (K, D_MODEL)

# ───────────────────────── inference ──────────────────────────
def run_inference(model_path: Path,
                  feat2d_root: Path,
                  feat3d_root: Path) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = torch.jit.load(model_path, map_location=device).eval()

    seqs = sorted([p.name for p in feat2d_root.iterdir() if p.is_dir()])
    rows = []

    with torch.no_grad():
        for seq in seqs:
            clip  = build_clip(seq, feat2d_root, feat3d_root).unsqueeze(0)
            probs = torch.sigmoid(model(clip.to(device)))[0].cpu().numpy()
            rows.append({"path_new": seq, **dict(zip(CLASS_NAMES, probs))})

    return pd.DataFrame(rows, columns=["path_new", *CLASS_NAMES])

# ─────────────────────── post-processing ─────────────────────
def post_process(df: pd.DataFrame,
                 thresholds: np.ndarray,
                 apply_fix: bool) -> pd.DataFrame:
    df[CLASS_NAMES] = (df[CLASS_NAMES].values >= thresholds).astype(int)

    df["Id"] = df["path_new"].str.replace(r"_[lr]$", "", regex=True)
    df = df.groupby("Id", as_index=False)[CLASS_NAMES].max()

    if apply_fix:
        df.loc[df["Negative"] == 1, "Positive"] = 0 # based on the 4DME: A Spontaneous 4D Micro-Expression Dataset With Multimodalities that: negative and positive are both 1.

    df["Id"] = df["Id"].str.extract(r'(\d+)').astype(int)
    return df.sort_values("Id")[["Id", *CLASS_NAMES]]

# ─────────────────────────── CLI ─────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="inference + post-process")
    add_yaml_defaults(p, "run_full")             # ← NEW
    p.add_argument("--feat2d", type=Path)
    p.add_argument("--feat3d", type=Path)
    p.add_argument("--model",  type=Path)
    p.add_argument("--out",    type=Path)
    p.add_argument("--thresholds", nargs=5, type=float)
    p.add_argument("--no_fix", action="store_true")
    args = p.parse_args()
    return args

# ─────────────────────────── main ────────────────────────────
def main():
    args     = parse_args()
    set_deterministic()
    apply_fix = not args.no_fix                 
    thr_arr   = np.array(args.thresholds, dtype=float)

    raw_df   = run_inference(args.model, args.feat2d, args.feat3d)
    final_df = post_process(raw_df, thr_arr, apply_fix)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(args.out, index=False)
    print(f"Saved → {args.out}")

if __name__ == "__main__":
    main()
