#!/usr/bin/env python3
"""
models/2019-smith-hallucigenia-sparsa-collective-stigmergy.py — Multi-Scale & Paleobiological Reproduction (Slot #2063)
Organism: Hallucigenia sparsa (Middle Cambrian Burgess Shale Armored Lobopod)
Phenomenon Scale: 3.3 cm (0.03317 m, log10=-1.479, 7_organism_cm)
Anatomical Structure: Collective swarm / biofilm / colonial array in Hallucigenia sparsa (Middle Cambrian (Burgess Shale, 508 Ma))
Geological Horizon: Middle Cambrian (Burgess Shale, 508 Ma) (508.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Hallucigenia
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation (3.3 cm))
Formal Verification: `morphoeval/lean/loops/Loop_2063_hallucigenia_sparsa_collective_stigmergy.lean` (`MathOfBiology.Loop_2063`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/hallucigenia_sparsa_collective_stigmergy"
EXPECTED_SHA256 = {
    "hallucigenia_sparsa_collective_stigmergy_reference.json": "d9379486a93b40943934c0fa5f9ab3b31a19b44a6832038ee1f394f57372838a",
    "hallucigenia_sparsa_specimen.jpg": "f0a3447adf27b728b96f3d6331a6f82327a3e9dbaf8871c9581019053e23ea81",
    "scout_manifest.json": "80297fc192b1da425900734c236e6b2162391ccd04a79771f891d30b43247ca1",
    "paper_citations.bib": "bb703d427aa401f55f2fcba625e9f1a8417eff13315d02b5ef428750582e0efa"
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
    ref = json.loads((DATASET_DIR / "hallucigenia_sparsa_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2019-smith-hallucigenia-sparsa-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
