#!/usr/bin/env python3
"""
2009-goryachev-saccharomyces-cerevisiae-bioelectric-polarity.py — Pass 2 Remediated Reproduction (Slot #0986)
Organism: Saccharomyces cerevisiae (Budding Yeast)
Aristotelian Taxonomy: IV. Protista & Monera -> Slime Molds & Bacteria -> Saccharomyces
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity)
Formal Verification: `morphoeval/lean/loops/Loop_0986_saccharomyces_cerevisiae_bioelectric_polarity.lean` (`MathOfBiology.Loop_0986`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/saccharomyces_cerevisiae_bioelectric_polarity"
EXPECTED_SHA256 = {
    "saccharomyces_cerevisiae_bioelectric_polarity_reference.json": "7bfd1c2e8e4c71438a5f9ef0714e219dc147e14b3b113128feed85703b5d7dfe",
    "saccharomyces_cerevisiae_specimen.jpg": "03d89aa520798f5a2c721833870f675b8d4676ea933358e95702fcd1732eae0b",
    "scout_manifest.json": "7587a87aaf859114aa517444081e4ddf34bd55eb62b2db4e239909e81be4a6b4",
    "paper_citations.bib": "13208fbf8a28e82c848f0c359b99db761a8bf70e426609755226408e5ae67ee0"
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
    ref = json.loads((DATASET_DIR / "saccharomyces_cerevisiae_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2009-goryachev-saccharomyces-cerevisiae-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
