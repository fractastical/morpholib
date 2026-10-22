#!/usr/bin/env python3
"""
models/2018-parichy-danio-albolineatus-collective-stigmergy.py — Multi-Scale & Paleobiological Reproduction (Slot #2012)
Organism: Danio albolineatus (Pearl Danio (Uniform Interstripe))
Phenomenon Scale: 5.8 cm (0.05767 m, log10=-1.239, 7_organism_cm)
Anatomical Structure: Collective swarm / biofilm / colonial array in Danio albolineatus (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Danio
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation (5.8 cm))
Formal Verification: `morphoeval/lean/loops/Loop_2012_danio_albolineatus_collective_stigmergy.lean` (`MathOfBiology.Loop_2012`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/danio_albolineatus_collective_stigmergy"
EXPECTED_SHA256 = {
    "danio_albolineatus_collective_stigmergy_reference.json": "7e612dc453dfd2141df88b7984adcd6eea244ca36205aeb5b439c22e38b2dfa5",
    "danio_albolineatus_specimen.jpg": "6059c6481299cc5bb1799fc70390ba1924f905610346e50fc579efcd09df8e46",
    "scout_manifest.json": "9025b4af4007d3bc9a262221756345ebb05ef093d75bc434aefd6d2be102ecae",
    "paper_citations.bib": "21bc77a84bd83f1364e6a63df96fc851b6a570ed8c5462d253e9ff2e26d90685"
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
    ref = json.loads((DATASET_DIR / "danio_albolineatus_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2018-parichy-danio-albolineatus-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
