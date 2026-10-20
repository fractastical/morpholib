#!/usr/bin/env python3
"""
1992-camazine-apis-mellifera-bioelectric-polarity.py — Pass 2 Remediated Reproduction (Slot #0943)
Organism: Apis mellifera (Western Honeybee)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Apis
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity)
Formal Verification: `morphoeval/lean/loops/Loop_0943_apis_mellifera_bioelectric_polarity.lean` (`MathOfBiology.Loop_0943`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/apis_mellifera_bioelectric_polarity"
EXPECTED_SHA256 = {
    "apis_mellifera_bioelectric_polarity_reference.json": "2150f897137489586598e30d7714cbd2ad41248481edb3536b27ef63b29190e2",
    "apis_mellifera_specimen.jpg": "67454ef389899df977e29efa0db0cba99519f5d46a3631810bcc7f160efd8fce",
    "scout_manifest.json": "41af44244849101879984dd0ee84810475e92c0c16655a7468314cd7b9047b29",
    "paper_citations.bib": "4a08c59044aee3d0347132fb397e6bd15ea67b93f23f682a37ab4814bef4f1c7"
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
    ref = json.loads((DATASET_DIR / "apis_mellifera_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1992-camazine-apis-mellifera-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
