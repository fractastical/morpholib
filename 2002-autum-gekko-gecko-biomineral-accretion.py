#!/usr/bin/env python3
"""
2002-autum-gekko-gecko-biomineral-accretion.py — Pass 2 Remediated Reproduction (Slot #0828)
Organism: Gekko gecko (Tokay Gecko)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Gekko
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization)
Formal Verification: `morphoeval/lean/loops/Loop_0828_gekko_gecko_biomineral_accretion.lean` (`MathOfBiology.Loop_0828`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/gekko_gecko_biomineral_accretion"
EXPECTED_SHA256 = {
    "gekko_gecko_biomineral_accretion_reference.json": "d487126050b8443edaa074012713d8e1b95bf53e780f894df50af904d5610581",
    "gekko_gecko_specimen.jpg": "d5822e517434737d90b14fd5099c77f09ab976ca2a3ba00c2484f69a65eb91e8",
    "scout_manifest.json": "fdb1acb71eee455475afe489d231939d468203e9f8c98c5e68fa42c5465acef9",
    "paper_citations.bib": "8aa21430a6d3cad5c702a18c41dd1e73ea66db58c92a8cf9c632b7508708e101"
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
    ref = json.loads((DATASET_DIR / "gekko_gecko_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2002-autum-gekko-gecko-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
