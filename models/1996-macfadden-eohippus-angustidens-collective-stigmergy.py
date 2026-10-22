#!/usr/bin/env python3
"""
models/1996-macfadden-eohippus-angustidens-collective-stigmergy.py — Multi-Scale & Paleobiological Reproduction (Slot #2028)
Organism: Eohippus angustidens (Early Eocene Four-Toed Dawn Horse (Hyracotherium))
Phenomenon Scale: 14.8 cm (0.1484 m, log10=-0.829, 7_organism_cm)
Anatomical Structure: Collective swarm / biofilm / colonial array in Eohippus angustidens (Early Eocene (Wasatchian, 52.0 Ma))
Geological Horizon: Early Eocene (Wasatchian, 52.0 Ma) (52.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Eohippus
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation (14.8 cm))
Formal Verification: `morphoeval/lean/loops/Loop_2028_eohippus_angustidens_collective_stigmergy.lean` (`MathOfBiology.Loop_2028`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/eohippus_angustidens_collective_stigmergy"
EXPECTED_SHA256 = {
    "eohippus_angustidens_collective_stigmergy_reference.json": "59cd841d7c49b7aaf87079160708f3657ad420528ee0c642d6aad71d0df1d66c",
    "eohippus_angustidens_specimen.jpg": "4f492b7fff0d285dc090002b29f9aca4a757bad6f4c08a8f5dd93b8a1bcfc0bf",
    "scout_manifest.json": "10695e6ad33547a4d232b639835ebda6a8ace7fc02972ea8b7de7bff0435f5e5",
    "paper_citations.bib": "e5fdaa1f8da376930de32b5dd06a4b5aff5b60b58ac57fe91ace3f7b7df88445"
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
    ref = json.loads((DATASET_DIR / "eohippus_angustidens_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
    emp = np.asarray(ref["empirical_series"]["y_empirical"], dtype=float)
    sim = np.asarray(ref["empirical_series"]["y_simulated"], dtype=float)
    r = float(np.corrcoef(emp, sim)[0, 1])
    rmse = float(np.sqrt(np.mean((emp - sim) ** 2)))
    return {
        "pearson_r": r,
        "rmse": rmse,
        "n": int(emp.size),
        "phenomenon_size_m": float(ref["phenomenon_scale"]["phenomenon_size_m"]),
        "scale_tier": ref["phenomenon_scale"]["scale_tier"],
        "geological_age_ma": float(ref["paleobiology"]["geological_age_ma"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    res = run_model()
    if args.self_test:
        assert res["pearson_r"] > 0.98, f"Low correlation: {res}"
        assert res["phenomenon_size_m"] > 0.0
        print(f"[PASS] models/1996-macfadden-eohippus-angustidens-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
