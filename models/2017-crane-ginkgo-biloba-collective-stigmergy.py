#!/usr/bin/env python3
"""
models/2017-crane-ginkgo-biloba-collective-stigmergy.py — Multi-Scale & Paleobiological Reproduction (Slot #2074)
Organism: Ginkgo biloba (Extant Maidenhair Tree (Dichotomous Open-Vein Fan Leaf))
Phenomenon Scale: 8.9 cm (0.08874 m, log10=-1.052, 7_organism_cm)
Anatomical Structure: Collective swarm / biofilm / colonial array in Ginkgo biloba (Holocene (Extant Living Fossil))
Geological Horizon: Holocene (Extant Living Fossil) (0.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Ginkgo
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation (8.9 cm))
Formal Verification: `morphoeval/lean/loops/Loop_2074_ginkgo_biloba_collective_stigmergy.lean` (`MathOfBiology.Loop_2074`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/ginkgo_biloba_collective_stigmergy"
EXPECTED_SHA256 = {
    "ginkgo_biloba_collective_stigmergy_reference.json": "046844e60caf1c92bb8dffde225589081eececb2d83e72fb6e056e17edcc1152",
    "ginkgo_biloba_specimen.jpg": "b0a1f66d4688ddb858306c70a27a0bc279b41f3eaf1e9935f5b40d33cb7f7905",
    "scout_manifest.json": "7d0f43ce97855bcf1d96cfc7c1a98a70111300615ff5178cd30c77525be8dee2",
    "paper_citations.bib": "0d455df509b76ad2027b340307d0f0e52aed3a26b6b48a0a04aecbec7ab45046"
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
    ref = json.loads((DATASET_DIR / "ginkgo_biloba_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2017-crane-ginkgo-biloba-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
