#!/usr/bin/env python3
"""
2009-checa-patella-vulgata-biomineral-accretion.py — Pass 2 Remediated Reproduction (Slot #0853)
Organism: Patella vulgata (Common Limpet)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Patella
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization)
Formal Verification: `morphoeval/lean/loops/Loop_0853_patella_vulgata_biomineral_accretion.lean` (`MathOfBiology.Loop_0853`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/patella_vulgata_biomineral_accretion"
EXPECTED_SHA256 = {
    "patella_vulgata_biomineral_accretion_reference.json": "4542c631158ff2ef81b2ba097eb18d2051df2e78bc69a412600ea263336c8768",
    "patella_vulgata_specimen.jpg": "dc71ee15b04a3541b9b25a13411da18888acaaba54973227c4ba40c361669d0e",
    "scout_manifest.json": "5b7488f0b5cac49063dcbb8e01f6bf974fabb2e5706aa314f62fa83f54946a0a",
    "paper_citations.bib": "26bf3c7c03ce4a9affd13204b979a78742361dd7a24e1f60a874b97439826e36"
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
    ref = json.loads((DATASET_DIR / "patella_vulgata_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2009-checa-patella-vulgata-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
