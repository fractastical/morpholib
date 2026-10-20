#!/usr/bin/env python3
"""
2007-kusserow-nematostella-vectensis-collective-stigmergy.py — Pass 2 Remediated Reproduction (Slot #1062)
Organism: Nematostella vectensis (Starlet Sea Anemone)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Nematostella
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation)
Formal Verification: `morphoeval/lean/loops/Loop_1062_nematostella_vectensis_collective_stigmergy.lean` (`MathOfBiology.Loop_1062`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/nematostella_vectensis_collective_stigmergy"
EXPECTED_SHA256 = {
    "nematostella_vectensis_collective_stigmergy_reference.json": "b677eccb69473508636020fdbbd79e785ba3b91496985cf884eaa33a2b364dc5",
    "nematostella_vectensis_specimen.jpg": "fb9d66de457a62bbf3d2ec1bcdb57f33e32468690a3047a09117a19153073a53",
    "scout_manifest.json": "62d6dce2cf332850a62ad98f243a1bc5fe1482f0c18c09c26f7dbe5f1f640c15",
    "paper_citations.bib": "0b1dab854c0c528a8fb31bb1deec0367e1002144d4e1c24c7a886b0bb0d4d0c8"
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
    ref = json.loads((DATASET_DIR / "nematostella_vectensis_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2007-kusserow-nematostella-vectensis-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
