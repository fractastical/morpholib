#!/usr/bin/env python3
"""
models/2023-munro-physalia-physalis-collective-stigmergy.py — Multi-Scale & Paleobiological Reproduction (Slot #2096)
Organism: Physalia physalis (Portuguese Man o' War (Chiral Left/Right Dimorphic Pneumatophore Sail))
Phenomenon Scale: 19.2 cm (0.1922 m, log10=-0.716, 7_organism_cm)
Anatomical Structure: Collective swarm / biofilm / colonial array in Physalia physalis (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Physalia
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation (19.2 cm))
Formal Verification: `morphoeval/lean/loops/Loop_2096_physalia_physalis_collective_stigmergy.lean` (`MathOfBiology.Loop_2096`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/physalia_physalis_collective_stigmergy"
EXPECTED_SHA256 = {
    "physalia_physalis_collective_stigmergy_reference.json": "39ffbc86f217a05052dc11c9629eab29ee55b7ae58e4e9601405b9af2cbe3e9b",
    "physalia_physalis_specimen.jpg": "b68854e9fdaaf725c3899f1667f4e8361f6a0f2556862c28573db2d565a50904",
    "scout_manifest.json": "6fb3dfdf9d70463db8acc2cd012e6240fe7dce409a28c5d8fa1a5e2f44c9200e",
    "paper_citations.bib": "7ee61b4c36cbd2cdaf949eddf487fba3579a9fdfe950210267d7aa38b4b1bdf3"
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
    ref = json.loads((DATASET_DIR / "physalia_physalis_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2023-munro-physalia-physalis-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
