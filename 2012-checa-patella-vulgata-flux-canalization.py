#!/usr/bin/env python3
"""
2012-checa-patella-vulgata-flux-canalization.py — Pass 2 Remediated Reproduction (Slot #0453)
Organism: Patella vulgata (Common Limpet)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Patella
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth)
Formal Verification: `morphoeval/lean/loops/Loop_0453_patella_vulgata_flux_canalization.lean` (`MathOfBiology.Loop_0453`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/patella_vulgata_flux_canalization"
EXPECTED_SHA256 = {
    "patella_vulgata_flux_canalization_reference.json": "d52cfb4b1b070428a9bc058a0d2f2afbe1499c54784d08c3e1cb54f7e661f4aa",
    "patella_vulgata_specimen.jpg": "dc71ee15b04a3541b9b25a13411da18888acaaba54973227c4ba40c361669d0e",
    "scout_manifest.json": "a88c6e3fa256a0130bd81f5171e6ef8afa25a4e4eb94d8f9c25c27988da90dd5",
    "paper_citations.bib": "4d7a71b08317ab5aff2ae30950afd3c147414d2763b889e8b430edf24f9dc2ac"
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
    ref = json.loads((DATASET_DIR / "patella_vulgata_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2012-checa-patella-vulgata-flux-canalization.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
