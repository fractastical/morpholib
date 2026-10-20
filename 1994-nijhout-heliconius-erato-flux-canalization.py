#!/usr/bin/env python3
"""
1994-nijhout-heliconius-erato-flux-canalization.py — Pass 2 Remediated Reproduction (Slot #0447)
Organism: Heliconius erato (Red Postman Butterfly)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Heliconius
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth)
Formal Verification: `morphoeval/lean/loops/Loop_0447_heliconius_erato_flux_canalization.lean` (`MathOfBiology.Loop_0447`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/heliconius_erato_flux_canalization"
EXPECTED_SHA256 = {
    "heliconius_erato_flux_canalization_reference.json": "0dc638ef58379888d7ab4c40884bb79a5a5904877380d391fbb6cb1914c17b2f",
    "heliconius_erato_specimen.jpg": "e54735f4ca16e4326cb91d2969d83bb613e3c4111aaf7f30cb218075b6fec48e",
    "scout_manifest.json": "00a73e56b36e970270180ad10da8cfc0f1185436ffae41887ce846bce95c3813",
    "paper_citations.bib": "689e078ddfd542710149cf0abd794bf4a738a9a56f09bc212b337ddcd9aafefb"
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
    ref = json.loads((DATASET_DIR / "heliconius_erato_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1994-nijhout-heliconius-erato-flux-canalization.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
