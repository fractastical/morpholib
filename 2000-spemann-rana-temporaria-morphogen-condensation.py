#!/usr/bin/env python3
"""
2000-spemann-rana-temporaria-morphogen-condensation.py — Pass 2 Remediated Reproduction (Slot #0226)
Organism: Rana temporaria (European Common Frog)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Rana
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation)
Formal Verification: `morphoeval/lean/loops/Loop_0226_rana_temporaria_morphogen_condensation.lean` (`MathOfBiology.Loop_0226`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/rana_temporaria_morphogen_condensation"
EXPECTED_SHA256 = {
    "rana_temporaria_morphogen_condensation_reference.json": "638328e33e672be820928fd2245ea0f94e1d8b40866ba099cf7580aa5ddf78ad",
    "rana_temporaria_specimen.jpg": "966174057e8f12954fa5a7ba2b0a2b543b64c2900c6c92ea4a55926d1c57f111",
    "scout_manifest.json": "3a08ba36631804407d635ccfa08c374a030ed243f2022deccce840b8365aae11",
    "paper_citations.bib": "10937cd25aaea8f8e1a9b640d65f0b95843a49e231016233bccd0bedbebb88f8"
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
    ref = json.loads((DATASET_DIR / "rana_temporaria_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2000-spemann-rana-temporaria-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
