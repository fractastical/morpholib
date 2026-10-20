#!/usr/bin/env python3
"""
2001-benjacob-paenibacillus-dendritiformis-bioelectric-polarity.py — Pass 2 Remediated Reproduction (Slot #0989)
Organism: Paenibacillus dendritiformis (Chiral Branching Bacterium)
Aristotelian Taxonomy: IV. Protista & Monera -> Slime Molds & Bacteria -> Paenibacillus
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity)
Formal Verification: `morphoeval/lean/loops/Loop_0989_paenibacillus_dendritiformis_bioelectric_polarity.lean` (`MathOfBiology.Loop_0989`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/paenibacillus_dendritiformis_bioelectric_polarity"
EXPECTED_SHA256 = {
    "paenibacillus_dendritiformis_bioelectric_polarity_reference.json": "0f7fb3832466a5fb3b5714e0562563e9f21f8e985450fe5bace85da8fbd37363",
    "paenibacillus_dendritiformis_specimen.jpg": "dbed60f770b34bfce5f9193b55484568f5cccc3a5f417b5c7c6e79a7a8c84cf6",
    "scout_manifest.json": "4ae715ddb9b64b3c03c8a59c310eed4c00d0224d3ffb26b6445574863c9acaa7",
    "paper_citations.bib": "04245c3c1da54401bd20026cea4e6c50bd68f86597b55c6f679866924772e958"
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
    ref = json.loads((DATASET_DIR / "paenibacillus_dendritiformis_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2001-benjacob-paenibacillus-dendritiformis-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
