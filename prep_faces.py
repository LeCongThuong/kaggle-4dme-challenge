#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prep_faces.py  ·  split L/R, crop to consistent face box, resize
"""

from __future__ import annotations
import sys, re, argparse, yaml
from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

# ───────────────────────── constants ──────────────────────────
ALLOWED_EXT = (".jpg", ".jpeg", ".png")
mp_face = mp.solutions.face_detection 

# ───────────── YAML override helper (tiny & self-contained) ─────────────
def add_yaml_defaults(parser: argparse.ArgumentParser, section: str):
    """
    Inject --config flag and, if present, push YAML[section] values
    into parser defaults *before* full parse.
    """
    parser.add_argument("--config", help="path to exp.yaml", nargs="?")
    cfg_stub, _ = parser.parse_known_args()
    if cfg_stub.config:
        with open(cfg_stub.config, "r") as fh:
            blob = yaml.safe_load(fh)
        if section not in blob:
            sys.exit(f"[ERR] Section '{section}' not found in {cfg_stub.config}")
        parser.set_defaults(**blob[section])

# ─────────────────────────── helpers ───────────────────────────
def detect_face_bbox(img: Image.Image, expand: float = 1.5) -> Tuple[int, int, int, int] | None:
    """Return (x_min, y_min, x_max, y_max) in pixel coords or None."""
    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.005) as fd:
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        res = fd.process(img_bgr)
        if not res.detections:
            return None

        bb = res.detections[0].location_data.relative_bounding_box
        h, w = img.size[1], img.size[0]
        x_min, y_min = int(bb.xmin * w), int(bb.ymin * h)
        box_w, box_h = int(bb.width * w), int(bb.height * h)

        cx, cy = x_min + box_w // 2, y_min + box_h // 2
        new_w, new_h = int(box_w * expand), int(box_h * expand)

        x0 = max(cx - new_w // 2, 0)
        y0 = max(cy - new_h // 2, 0)
        x1 = min(cx + new_w // 2, w)
        y1 = min(cy + new_h // 2, h)
        return x0, y0, x1, y1

def split_lr(img: Image.Image) -> Tuple[Image.Image, Image.Image]:
    w, h = img.size
    mid = w // 2
    return img.crop((0, 0, mid, h)), img.crop((mid, 0, w, h))

def numeric_sort_key(name: str) -> Tuple[int, str]:
    m = re.search(r'(\d+)', name)
    return (int(m.group(1)) if m else -1, name)

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def process_sequence(seq_dir: Path, out_root: Path, target_size: Tuple[int, int]):
    files = sorted([f for f in seq_dir.iterdir() if f.suffix.lower() in ALLOWED_EXT],
                   key=lambda p: numeric_sort_key(p.name))
    if not files:
        return

    # compute face boxes from halves of first frame
    first_img = Image.open(files[0]).convert("RGB")
    left0, right0 = split_lr(first_img)
    left_box  = detect_face_bbox(left0)  or (0, 400, 1200, 1600)
    right_box = detect_face_bbox(right0) or (0, 400, 1200, 1600)

    seq_name  = seq_dir.name
    out_left  = out_root / f"{seq_name}_l"
    out_right = out_root / f"{seq_name}_r"
    ensure_dir(out_left); ensure_dir(out_right)

    for idx, img_path in enumerate(files):
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Skipped {img_path}: {e}")
            continue

        left, right = split_lr(img)
        left_c  = left.crop(left_box).resize(target_size, Image.LANCZOS)
        right_c = right.crop(right_box).resize(target_size, Image.LANCZOS)

        fname = f"f_{idx:06d}.png"
        left_c.save(out_left  / fname)
        right_c.save(out_right / fname)

    print(f"{seq_name}: {len(files)} frames: {out_left.name}, {out_right.name}")

# ───────────────────────────── main ────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="split L/R, crop, resize")
    add_yaml_defaults(ap, "prep_faces")    
    ap.add_argument("--input",  type=Path, required=False)
    ap.add_argument("--output", type=Path, required=False)
    args = ap.parse_args()

    target_size = tuple(args.target_size)
    ensure_dir(args.output)

    seq_dirs = [d for d in args.input.iterdir() if d.is_dir()]
    if not seq_dirs:
        sys.exit("No sequences found in input directory.")

    for seq in sorted(seq_dirs, key=lambda p: numeric_sort_key(p.name)):
        process_sequence(seq, args.output, target_size)

if __name__ == "__main__":
    main()