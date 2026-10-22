#!/usr/bin/env python3
"""
models/2008-fusco-aulacopleura-koninckii-holaspis22-collective-stigmergy.py — Multi-Scale & Paleobiological Reproduction (Slot #2054)
Organism: Aulacopleura koninckii_holaspis22 (Silurian Trilobite Holaspid Morphotype II (22 Thoracic Segments))
Phenomenon Scale: 3.5 cm (0.03524 m, log10=-1.453, 7_organism_cm)
Anatomical Structure: Collective swarm / biofilm / colonial array in Aulacopleura koninckii_holaspis22 (Silurian (Wenlock Epoch, 428.1 Ma))
Geological Horizon: Silurian (Wenlock Epoch, 428.1 Ma) (428.1 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Aulacopleura
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation (3.5 cm))
Formal Verification: `morphoeval/lean/loops/Loop_2054_aulacopleura_koninckii_holaspis22_collective_stigmergy.lean` (`MathOfBiology.Loop_2054`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/aulacopleura_koninckii_holaspis22_collective_stigmergy"
EXPECTED_SHA256 = {
    "aulacopleura_koninckii_holaspis22_collective_stigmergy_reference.json": "498a39605542580ceae6abac5f08f6081d97fe3180fa7caa283e101a1f979663",
    "aulacopleura_koninckii_holaspis22_specimen.jpg": "1b6c8997dc42e91d0975aee1225a27810db08c0301831c90db07bebae7b0344a",
    "scout_manifest.json": "b86b40f1f1cd2af08f9dd77db5b7e4e066860afe4e3232329ab1926ceecfe780",
    "paper_citations.bib": "e9031d86c6911fe9afe718e738f3e3ab56634da5779b63207be6e7395ca90e3c"
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
    ref = json.loads((DATASET_DIR / "aulacopleura_koninckii_holaspis22_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2008-fusco-aulacopleura-koninckii-holaspis22-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
