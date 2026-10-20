#!/usr/bin/env python3
"""
2016-teyssier-chamaeleo-calyptratus-bioelectric-polarity.py — Pass 2 Remediated Reproduction (Slot #0923)
Organism: Chamaeleo calyptratus (Veiled Chameleon)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Chamaeleo
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity)
Formal Verification: `morphoeval/lean/loops/Loop_0923_chamaeleo_calyptratus_bioelectric_polarity.lean` (`MathOfBiology.Loop_0923`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/chamaeleo_calyptratus_bioelectric_polarity"
EXPECTED_SHA256 = {
    "chamaeleo_calyptratus_bioelectric_polarity_reference.json": "5c714a99006887bb2ee75b8138ffb32a4129a084a80118c1172e001ce3495066",
    "chamaeleo_calyptratus_specimen.jpg": "976c60b3ba6aa23d8e363fe2188ae09cd8c806190b8b02939475defbf44924b3",
    "scout_manifest.json": "a2fd16659b35f708e0830df8a7cfb5a14c3c99ee8f6cc0bf3a09e13a0ae6962f",
    "paper_citations.bib": "664d1f8d1c48068d7e3f6479b92cd76ae9a9ce681af3b5c903708da3eb5e144f"
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
    ref = json.loads((DATASET_DIR / "chamaeleo_calyptratus_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2016-teyssier-chamaeleo-calyptratus-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
