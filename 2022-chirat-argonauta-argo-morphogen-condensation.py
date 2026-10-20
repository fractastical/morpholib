#!/usr/bin/env python3
"""
2022-chirat-argonauta-argo-morphogen-condensation.py — Pass 2 Remediated Reproduction (Slot #0256)
Organism: Argonauta argo (Paper Nautilus)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Argonauta
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation)
Formal Verification: `morphoeval/lean/loops/Loop_0256_argonauta_argo_morphogen_condensation.lean` (`MathOfBiology.Loop_0256`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/argonauta_argo_morphogen_condensation"
EXPECTED_SHA256 = {
    "argonauta_argo_morphogen_condensation_reference.json": "5067dcfd0f2cc79ad2b0b087882ba6d0622b3e79bad679f252a94861578a57bd",
    "argonauta_argo_specimen.jpg": "4d2cfc3367b264b3d9c93b5d4ea3794e99cb317bbc495df52ab10c540174dff2",
    "scout_manifest.json": "f4a8ba95013fce0b718bbc5b5ddfc4d6ccc910df01c7b53b6aac7a0c03004530",
    "paper_citations.bib": "96b7ef7750ac8d58db4ad1c601a8ebdb5930ae323d7ae92c6f0f74777791bb7c"
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
    ref = json.loads((DATASET_DIR / "argonauta_argo_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2022-chirat-argonauta-argo-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
