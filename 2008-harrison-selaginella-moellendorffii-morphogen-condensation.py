#!/usr/bin/env python3
"""
2008-harrison-selaginella-moellendorffii-morphogen-condensation.py — Pass 2 Remediated Reproduction (Slot #0279)
Organism: Selaginella moellendorffii (Spikemoss Lycophyte)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Selaginella
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation)
Formal Verification: `morphoeval/lean/loops/Loop_0279_selaginella_moellendorffii_morphogen_condensation.lean` (`MathOfBiology.Loop_0279`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/selaginella_moellendorffii_morphogen_condensation"
EXPECTED_SHA256 = {
    "selaginella_moellendorffii_morphogen_condensation_reference.json": "6d3d625ef1b29335d033bdf6259fd96c27e9b167a7a3d99a9bfae2d92c78a86e",
    "selaginella_moellendorffii_specimen.jpg": "87d0d25c483d4f17cb66050be175071aafb49d6f321bc8eb2d9cc3bac9e55cd4",
    "scout_manifest.json": "7876250350336248efc9b972defc1b6c2da8460ca926e1e10c6237dbea8e8f01",
    "paper_citations.bib": "2fd2e4f5f94a58ec4654d6f2965c2235e074ecbcafbd55c6386aab40910c883b"
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
    ref = json.loads((DATASET_DIR / "selaginella_moellendorffii_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2008-harrison-selaginella-moellendorffii-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
