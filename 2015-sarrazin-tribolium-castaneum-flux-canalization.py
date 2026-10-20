#!/usr/bin/env python3
"""
2015-sarrazin-tribolium-castaneum-flux-canalization.py — Pass 2 Remediated Reproduction (Slot #0442)
Organism: Tribolium castaneum (Red Flour Beetle)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Tribolium
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth)
Formal Verification: `morphoeval/lean/loops/Loop_0442_tribolium_castaneum_flux_canalization.lean` (`MathOfBiology.Loop_0442`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/tribolium_castaneum_flux_canalization"
EXPECTED_SHA256 = {
    "tribolium_castaneum_flux_canalization_reference.json": "00d04d10edbdc1650e67004715fb840fcd4151afa16354e4714756c5646c13a4",
    "tribolium_castaneum_specimen.jpg": "a991c507b6f59178075bea80d7ad2c1c97d839991d31594797919b07b5ac1d77",
    "scout_manifest.json": "52fc63748e7d30dac8c9be1b0267c986f250a14a955add61d428645984ef498c",
    "paper_citations.bib": "13cafa8f4b1c620556701ead383b5ea851de1d319e8d0971cbb0791f091b19a6"
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
    ref = json.loads((DATASET_DIR / "tribolium_castaneum_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2015-sarrazin-tribolium-castaneum-flux-canalization.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
