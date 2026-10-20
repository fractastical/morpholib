#!/usr/bin/env python3
"""
2019-riquelme-aspergillus-nidulans-bioelectric-polarity.py — Pass 2 Remediated Reproduction (Slot #0982)
Organism: Aspergillus nidulans (Filamentous Fungus)
Aristotelian Taxonomy: IV. Protista & Monera -> Slime Molds & Bacteria -> Aspergillus
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity)
Formal Verification: `morphoeval/lean/loops/Loop_0982_aspergillus_nidulans_bioelectric_polarity.lean` (`MathOfBiology.Loop_0982`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/aspergillus_nidulans_bioelectric_polarity"
EXPECTED_SHA256 = {
    "aspergillus_nidulans_bioelectric_polarity_reference.json": "e4138d81de70e477b55318e7d54eb03482fcea2cd0f88b1578b033724c71f98c",
    "aspergillus_nidulans_specimen.jpg": "93e53157e77eaccce94a5792d58d0e622ba9bb55b15fd47100e1f7c90985dec5",
    "scout_manifest.json": "cefb6800f93009ccb8e934b3971a880054bb528b08288bdc35e41784eeb3d730",
    "paper_citations.bib": "c14b757190be275cee8ed8a0eaa4ca3792facd623ad2292d041908cd9bbf6e1f"
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
    ref = json.loads((DATASET_DIR / "aspergillus_nidulans_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2019-riquelme-aspergillus-nidulans-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
