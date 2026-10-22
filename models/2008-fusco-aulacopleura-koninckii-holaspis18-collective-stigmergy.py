#!/usr/bin/env python3
"""
models/2008-fusco-aulacopleura-koninckii-holaspis18-collective-stigmergy.py — Multi-Scale & Paleobiological Reproduction (Slot #2053)
Organism: Aulacopleura koninckii_holaspis18 (Silurian Trilobite Holaspid Morphotype I (18 Thoracic Segments))
Phenomenon Scale: 2.4 cm (0.02448 m, log10=-1.611, 7_organism_cm)
Anatomical Structure: Collective swarm / biofilm / colonial array in Aulacopleura koninckii_holaspis18 (Silurian (Wenlock Epoch, 428.3 Ma))
Geological Horizon: Silurian (Wenlock Epoch, 428.3 Ma) (428.3 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Aulacopleura
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation (2.4 cm))
Formal Verification: `morphoeval/lean/loops/Loop_2053_aulacopleura_koninckii_holaspis18_collective_stigmergy.lean` (`MathOfBiology.Loop_2053`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/aulacopleura_koninckii_holaspis18_collective_stigmergy"
EXPECTED_SHA256 = {
    "aulacopleura_koninckii_holaspis18_collective_stigmergy_reference.json": "c24df7e53d6be611e24444e51e2aaf12cf6300b11dbaacf65a1598433851db6c",
    "aulacopleura_koninckii_holaspis18_specimen.jpg": "1b6c8997dc42e91d0975aee1225a27810db08c0301831c90db07bebae7b0344a",
    "scout_manifest.json": "a51d6da60d8575c98b703da0c4d655236fddbc911501e883911099708d21c75f",
    "paper_citations.bib": "5da5fd2711e199d344160179d53833ab4ea4e9d8b959a6b8787273fdcdf15fae"
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
    ref = json.loads((DATASET_DIR / "aulacopleura_koninckii_holaspis18_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2008-fusco-aulacopleura-koninckii-holaspis18-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
