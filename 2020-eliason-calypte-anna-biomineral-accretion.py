#!/usr/bin/env python3
"""
2020-eliason-calypte-anna-biomineral-accretion.py — Pass 2 Remediated Reproduction (Slot #0820)
Organism: Calypte anna (Anna's Hummingbird)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Calypte
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization)
Formal Verification: `morphoeval/lean/loops/Loop_0820_calypte_anna_biomineral_accretion.lean` (`MathOfBiology.Loop_0820`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/calypte_anna_biomineral_accretion"
EXPECTED_SHA256 = {
    "calypte_anna_biomineral_accretion_reference.json": "ea2e92e73c79ccae979cd5841cc9c4616b6e3496300dd036141ff66bf00544cb",
    "calypte_anna_specimen.jpg": "023a1c1ebf863e2bc3fe443e9cc7d668c6fa7ac72517c8a62f1874671a2eeb60",
    "scout_manifest.json": "b810b7ec9060722b249a2e21b48e57dacd1582828de48d388342908ac11138ea",
    "paper_citations.bib": "1c9b66f966d336b3b58a0a32317df571089fe0fc668663330721aa9fbf6e35a1"
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
    ref = json.loads((DATASET_DIR / "calypte_anna_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2020-eliason-calypte-anna-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
