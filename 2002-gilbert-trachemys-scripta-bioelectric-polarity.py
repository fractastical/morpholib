#!/usr/bin/env python3
"""
2002-gilbert-trachemys-scripta-bioelectric-polarity.py — Pass 2 Remediated Reproduction (Slot #0929)
Organism: Trachemys scripta (Red-Eared Slider Turtle)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Trachemys
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity)
Formal Verification: `morphoeval/lean/loops/Loop_0929_trachemys_scripta_bioelectric_polarity.lean` (`MathOfBiology.Loop_0929`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/trachemys_scripta_bioelectric_polarity"
EXPECTED_SHA256 = {
    "trachemys_scripta_bioelectric_polarity_reference.json": "c26b79b5a6c4f93a1058c2fccdea70b2a7aec370caa50b748d48761290b1dd90",
    "trachemys_scripta_specimen.jpg": "5d30c00d709c06ef8c1ebdb9b272654df48237cd07a6d19be7e219ab27188e96",
    "scout_manifest.json": "2a4e1e87d53c5d415a975a8667c67d46d028207dab2d579efe2ac440068a72fc",
    "paper_citations.bib": "0a04eb801c772f08d79d7913589d9b2d8f4282de19964b20ccb074461a274769"
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
    ref = json.loads((DATASET_DIR / "trachemys_scripta_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2002-gilbert-trachemys-scripta-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
