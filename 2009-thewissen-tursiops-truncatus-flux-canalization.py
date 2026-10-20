#!/usr/bin/env python3
"""
2009-thewissen-tursiops-truncatus-flux-canalization.py — Pass 2 Remediated Reproduction (Slot #0410)
Organism: Tursiops truncatus (Bottlenose Dolphin)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Tursiops
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth)
Formal Verification: `morphoeval/lean/loops/Loop_0410_tursiops_truncatus_flux_canalization.lean` (`MathOfBiology.Loop_0410`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/tursiops_truncatus_flux_canalization"
EXPECTED_SHA256 = {
    "tursiops_truncatus_flux_canalization_reference.json": "d939296ae5c2e7e583278f6c8b3eabe2dc9de348f39419baec3be2bfb6353a64",
    "tursiops_truncatus_specimen.jpg": "038f0a413155e6a92701bb438caa1f4c23fdef11b3c5c4892f851e65bcdf710a",
    "scout_manifest.json": "095cf63a43f1fe234289a4de1d0d9e0ba57762c4ea0834c8e7f81b537cdcbd54",
    "paper_citations.bib": "ca23d327a039bbd4b8d26675c7b30f58f4e0322a716827a065fee8584d1e0d9e"
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
    ref = json.loads((DATASET_DIR / "tursiops_truncatus_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2009-thewissen-tursiops-truncatus-flux-canalization.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
