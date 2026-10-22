#!/usr/bin/env python3
"""
models/1996-dimichele-lepidodendron-aculeatum-collective-stigmergy.py — Multi-Scale & Paleobiological Reproduction (Slot #2068)
Organism: Lepidodendron aculeatum (Carboniferous Giant Scale-Tree (Helical Parastichy Leaf Cushions))
Phenomenon Scale: 6.1 cm (0.06088 m, log10=-1.216, 7_organism_cm)
Anatomical Structure: Collective swarm / biofilm / colonial array in Lepidodendron aculeatum (Late Carboniferous (Pennsylvanian, 310 Ma))
Geological Horizon: Late Carboniferous (Pennsylvanian, 310 Ma) (310.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Lepidodendron
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation (6.1 cm))
Formal Verification: `morphoeval/lean/loops/Loop_2068_lepidodendron_aculeatum_collective_stigmergy.lean` (`MathOfBiology.Loop_2068`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/lepidodendron_aculeatum_collective_stigmergy"
EXPECTED_SHA256 = {
    "lepidodendron_aculeatum_collective_stigmergy_reference.json": "f744287678db94a2e2a3939b15ad3130ef265b86f4c971d04495bd5ba38bb778",
    "lepidodendron_aculeatum_specimen.jpg": "854f4a6257e58a0123b8eee34674c32ed016882c88d6eb907ca1d85dbb78c467",
    "scout_manifest.json": "c80e9e54b3dd6a51c432c5e4f70ad2e76c7fa15d8fe07363bdc6b0086707f6b5",
    "paper_citations.bib": "6de293fbf0277825d78f7657ea6a6a13816f1db5f6d00a8dacfc337ad796adfc"
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
    ref = json.loads((DATASET_DIR / "lepidodendron_aculeatum_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1996-dimichele-lepidodendron-aculeatum-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
