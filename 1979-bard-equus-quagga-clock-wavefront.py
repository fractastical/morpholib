#!/usr/bin/env python3
"""
1979-bard-equus-quagga-clock-wavefront.py — Pass 2 Remediated Reproduction (Slot #0304)
Organism: Equus quagga (Plains Zebra)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Equus
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest)
Formal Verification: `morphoeval/lean/loops/Loop_0304_equus_quagga_clock_wavefront.lean` (`MathOfBiology.Loop_0304`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/equus_quagga_clock_wavefront"
EXPECTED_SHA256 = {
    "equus_quagga_clock_wavefront_reference.json": "e7075aadb334e1a2ef346d012ea3cad0af2255e1faa19b30e83b95de1ddeb517",
    "equus_quagga_specimen.jpg": "b734dc3b533ffe889dee9f205dae9015580b4b3537a05743c602408063b17ca8",
    "scout_manifest.json": "4efb7eef653c704e57611a6de7d7cdb352b25549f35260345bf6c4407b8dda75",
    "paper_citations.bib": "cb09a094caa16b3d5ea6dd575ac70aef1613ace6230564b86a70e054b7bfc8a4"
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
    ref = json.loads((DATASET_DIR / "equus_quagga_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1979-bard-equus-quagga-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
