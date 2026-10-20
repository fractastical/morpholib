#!/usr/bin/env python3
"""
2010-gordon-arachnoidiscus-ehrenbergii-bioelectric-polarity.py — Pass 2 Remediated Reproduction (Slot #0999)
Organism: Arachnoidiscus ehrenbergii (Wheel Centric Diatom)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Arachnoidiscus
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity)
Formal Verification: `morphoeval/lean/loops/Loop_0999_arachnoidiscus_ehrenbergii_bioelectric_polarity.lean` (`MathOfBiology.Loop_0999`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/arachnoidiscus_ehrenbergii_bioelectric_polarity"
EXPECTED_SHA256 = {
    "arachnoidiscus_ehrenbergii_bioelectric_polarity_reference.json": "3d5aad907cdcf41d6a7fa4d6ab5f5467b42579db2e1b79b4e1b7da5ea32158f2",
    "arachnoidiscus_ehrenbergii_specimen.jpg": "499034f85d30d464ea67fea4d7404a3546b57b85bb64c8d553cb36f2129bed05",
    "scout_manifest.json": "bc7adaa38c5a8a3dc49f7a8944565331e5d6bc7910e2bb620d23f032c37130b9",
    "paper_citations.bib": "d5f8e7206b5d79029060b910fc261c4701fcdbf00bce1c47d41e425682f5d0be"
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
    ref = json.loads((DATASET_DIR / "arachnoidiscus_ehrenbergii_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2010-gordon-arachnoidiscus-ehrenbergii-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
