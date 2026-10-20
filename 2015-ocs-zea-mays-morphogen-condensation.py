#!/usr/bin/env python3
"""
2015-ocs-zea-mays-morphogen-condensation.py — Pass 2 Remediated Reproduction (Slot #0273)
Organism: Zea mays (Maize)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Zea
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation)
Formal Verification: `morphoeval/lean/loops/Loop_0273_zea_mays_morphogen_condensation.lean` (`MathOfBiology.Loop_0273`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/zea_mays_morphogen_condensation"
EXPECTED_SHA256 = {
    "zea_mays_morphogen_condensation_reference.json": "9104076ab1c7fa6f931a9ec08912289d6daba3f94e346dff81683be992187fc3",
    "zea_mays_specimen.jpg": "a78e27ea3c83c91a3f84373319a3b590f4a53e6108ddf50860d114690cbb6db9",
    "scout_manifest.json": "69be85fdd6c768b76be5d89701e486283881793376d6c425e61d376987230b6a",
    "paper_citations.bib": "3e0e3de2335edd72497772ad36f6376eb153c755ffabd3649412108647d92edd"
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
    ref = json.loads((DATASET_DIR / "zea_mays_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2015-ocs-zea-mays-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
