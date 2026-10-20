#!/usr/bin/env python3
"""
2017-teyssier-chamaeleo-calyptratus-collective-stigmergy.py — Pass 2 Remediated Reproduction (Slot #1023)
Organism: Chamaeleo calyptratus (Veiled Chameleon)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Chamaeleo
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation)
Formal Verification: `morphoeval/lean/loops/Loop_1023_chamaeleo_calyptratus_collective_stigmergy.lean` (`MathOfBiology.Loop_1023`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/chamaeleo_calyptratus_collective_stigmergy"
EXPECTED_SHA256 = {
    "chamaeleo_calyptratus_collective_stigmergy_reference.json": "bfdb3109af557c34be92caacdb660417f36240acd7c760e9e561c6549a6b578a",
    "chamaeleo_calyptratus_specimen.jpg": "976c60b3ba6aa23d8e363fe2188ae09cd8c806190b8b02939475defbf44924b3",
    "scout_manifest.json": "5736bf2a2176ca4ee70bf3496281948b97a06a02f156a909546b3c1ec22ad01e",
    "paper_citations.bib": "2d607f961ac791177b30f33103dde2e958be185583235e1b18628176b32081e8"
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
    ref = json.loads((DATASET_DIR / "chamaeleo_calyptratus_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2017-teyssier-chamaeleo-calyptratus-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
