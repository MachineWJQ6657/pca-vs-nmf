"""Rebuild data/olivetti_preprocessed.npz from the public Olivetti faces dataset.

Usage:  python make_data.py            # writes data/olivetti_preprocessed.npz.rebuilt and compares
        python make_data.py --check    # only compares with the existing data/olivetti_preprocessed.npz

What it does
  1. Loads the 400 Olivetti faces (AT&T Laboratories Cambridge; 40 people x 10 photos, 64 x 64 pixels)
     through scikit-learn's fetch_olivetti_faces (needs internet once; the file is cached afterwards).
     shuffle=False keeps the original order: image i belongs to person i // 10.
  2. Draws one fixed split, stratified by person (8 training and 2 test photos per person):
         train_test_split(np.arange(400), test_size=0.2, stratify=person, random_state=42)
     This recipe reproduces the stored train_idx / test_idx exactly (checked by the author of this file).
  3. Saves X_train, X_test (float32, flattened to 4096), y_train, y_test (int32), train_idx, test_idx.
  4. Compares the result with the team's existing file and prints whether every array matches.

If step 4 prints a mismatch for X_train / X_test, the team's file was made from a differently
scaled copy of the images; then document how Person 1 loaded and scaled them in README.md.
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

TARGET = Path("data") / "olivetti_preprocessed.npz"


def load_faces():
    from sklearn.datasets import fetch_olivetti_faces
    faces = fetch_olivetti_faces(shuffle=False)
    return faces.data.astype(np.float32), faces.target.astype(np.int32)


def build(faces_x, faces_y):
    assert faces_x.shape == (400, 4096) and faces_y.shape == (400,)
    assert np.array_equal(faces_y, np.repeat(np.arange(40), 10)), "images are not in the original person order"
    train_idx, test_idx = train_test_split(np.arange(400), test_size=0.2, stratify=faces_y, random_state=42)
    return dict(X_train=faces_x[train_idx], X_test=faces_x[test_idx],
                y_train=faces_y[train_idx], y_test=faces_y[test_idx],
                train_idx=train_idx.astype(np.int64), test_idx=test_idx.astype(np.int64))


def compare(new, path):
    with np.load(path) as old:
        ok = True
        for key in ("train_idx", "test_idx", "y_train", "y_test", "X_train", "X_test"):
            same = np.array_equal(new[key], old[key])
            extra = "" if same or not key.startswith("X") else f" (max abs difference {np.abs(new[key].astype(float) - old[key]).max():.3g})"
            print(f"  {key:10s} identical: {same}{extra}")
            ok &= same
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="compare only; do not write a file")
    args = parser.parse_args()

    arrays = build(*load_faces())
    if TARGET.is_file():
        print(f"Comparing with {TARGET} (sha256 {hashlib.sha256(TARGET.read_bytes()).hexdigest()[:16]}...):")
        print("MATCH" if compare(arrays, TARGET) else "DIFFERENT")
    if not args.check:
        out = TARGET.with_suffix(".npz.rebuilt")
        out.parent.mkdir(exist_ok=True)
        with open(out, "wb") as handle:
            np.savez_compressed(handle, **arrays)
        print("wrote", out)
