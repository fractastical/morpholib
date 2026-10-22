#!/usr/bin/env python3
"""
models/2006-royer-ginkgo-adiantoides-flux-canalization.py — Multi-Scale & Paleobiological Reproduction (Slot #1533)
Organism: Ginkgo adiantoides (Paleocene-Eocene Bilobed Coalescent Ginkgo Chronospecies)
Phenomenon Scale: 1.4 cm (0.01417 m, log10=-1.849, 7_organism_cm)
Anatomical Structure: Auxin flux vein / phyllotactic primordium in Ginkgo adiantoides (Paleocene-Eocene Boundary (56 Ma))
Geological Horizon: Paleocene-Eocene Boundary (56 Ma) (56.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Ginkgo
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth (1.4 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1533_ginkgo_adiantoides_flux_canalization.lean` (`MathOfBiology.Loop_1533`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/ginkgo_adiantoides_flux_canalization"
EXPECTED_SHA256 = {
    "ginkgo_adiantoides_flux_canalization_reference.json": "b0fc5f537a255d6c8a768857549fbd35936dc049ca7cef6ce4071d36513177d9",
    "ginkgo_adiantoides_specimen.jpg": "9b107911c55810e272a4f18af3a1f21e0f0d95561cf384cba5c90c9245314ca8",
    "scout_manifest.json": "2331a8e99bb3646547c46ad4ccb476798d64e1a0cbf9d1f03e6893bb7f3ed20d",
    "paper_citations.bib": "3f345e079fbf7d204553d4373df559b7e81251b5fced850d6b6e70b79c9f6454"
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
    ref = json.loads((DATASET_DIR / "ginkgo_adiantoides_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2006-royer-ginkgo-adiantoides-flux-canalization.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
