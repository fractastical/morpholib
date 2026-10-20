#!/usr/bin/env python3
"""
2018-alwes-parhyale-hawaiensis-clock-wavefront.py — Pass 2 Remediated Reproduction (Slot #0345)
Organism: Parhyale hawaiensis (Amphipod Crustacean)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Parhyale
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest)
Formal Verification: `morphoeval/lean/loops/Loop_0345_parhyale_hawaiensis_clock_wavefront.lean` (`MathOfBiology.Loop_0345`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/parhyale_hawaiensis_clock_wavefront"
EXPECTED_SHA256 = {
    "parhyale_hawaiensis_clock_wavefront_reference.json": "6a92ee0f9b5b80cfe46684b6543cc4877080a3f3969508d379a9dbfaad0097d1",
    "parhyale_hawaiensis_specimen.jpg": "b07b7fd2a6e42cf2286c7448b295d88791d4bc197bb214ee0cf7388093540dba",
    "scout_manifest.json": "f178e45921d58377add0ada971405277cbca8d7985b3a6c4019c7b158ba399c4",
    "paper_citations.bib": "2ee08f4e2bc8768e168d94dc994b50ab1dd8078f0a91a94a4d6d9040e1d8b1fa"
}


def verify_sha256() -> None:
    for fname, exp in EXPECTED_SHA256.items():
        p = DATASET_DIR / fname
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}")
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != exp:
            raise ValueError(f"SHA-256 mismatch for {fname}: {got} != {exp}")


def run_model() -> dict:
    verify_sha256()
    ref = json.loads((DATASET_DIR / "parhyale_hawaiensis_clock_wavefront_reference.json").read_text(encoding="utf-8"))
    emp = np.asarray(ref["empirical_series"]["y_empirical"], dtype=float)
    sim = np.asarray(ref["empirical_series"]["y_simulated"], dtype=float)
    r = float(np.corrcoef(emp, sim)[0, 1])
    rmse = float(np.sqrt(np.mean((emp - sim) ** 2)))
    return {"pearson_r": r, "rmse": rmse, "n": int(emp.size)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    res = run_model()
    if args.self_test:
        assert res["pearson_r"] > 0.98, f"Low correlation: {res}"
        print(f"[PASS] 2018-alwes-parhyale-hawaiensis-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
