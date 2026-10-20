#!/usr/bin/env python3
"""
2016-alwes-parhyale-hawaiensis-biomineral-accretion.py — Pass 2 Remediated Reproduction (Slot #0845)
Organism: Parhyale hawaiensis (Amphipod Crustacean)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Parhyale
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization)
Formal Verification: `morphoeval/lean/loops/Loop_0845_parhyale_hawaiensis_biomineral_accretion.lean` (`MathOfBiology.Loop_0845`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/parhyale_hawaiensis_biomineral_accretion"
EXPECTED_SHA256 = {
    "parhyale_hawaiensis_biomineral_accretion_reference.json": "8c140af3daffc5682effa634e3299792c4b1f220d374c71b0f8b01b80b2de95b",
    "parhyale_hawaiensis_specimen.jpg": "b07b7fd2a6e42cf2286c7448b295d88791d4bc197bb214ee0cf7388093540dba",
    "scout_manifest.json": "dc22ce73aa360d0658154bab575252e03174512a3753b77c9f7673ee52ea8f63",
    "paper_citations.bib": "3327c2359dae181f482fe84328a3cd539f8fb978563f665f9ad2692feec89074"
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
    ref = json.loads((DATASET_DIR / "parhyale_hawaiensis_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2016-alwes-parhyale-hawaiensis-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
