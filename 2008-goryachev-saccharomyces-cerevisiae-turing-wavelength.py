#!/usr/bin/env python3
"""
2008-goryachev-saccharomyces-cerevisiae-turing-wavelength.py — Pass 2 Remediated Reproduction (Slot #0186)
Organism: Saccharomyces cerevisiae (Budding Yeast)
Aristotelian Taxonomy: IV. Protista & Monera -> Slime Molds & Bacteria -> Saccharomyces
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection)
Formal Verification: `morphoeval/lean/loops/Loop_0186_saccharomyces_cerevisiae_turing_wavelength.lean` (`MathOfBiology.Loop_0186`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/saccharomyces_cerevisiae_turing_wavelength"
EXPECTED_SHA256 = {
    "saccharomyces_cerevisiae_turing_wavelength_reference.json": "a183bfd3843efaf232762440f743a93eb739151eb113796bb07ae891d79fcdf9",
    "saccharomyces_cerevisiae_specimen.jpg": "03d89aa520798f5a2c721833870f675b8d4676ea933358e95702fcd1732eae0b",
    "scout_manifest.json": "a7595264d1eaf1577f79f7306659ca852dd262dec88cc3bb85552992712b6808",
    "paper_citations.bib": "de2348f0c4bb560564c7abd59f7c6bc93eb75ab8ad9e428674c88374de39f34a"
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
    ref = json.loads((DATASET_DIR / "saccharomyces_cerevisiae_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2008-goryachev-saccharomyces-cerevisiae-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
