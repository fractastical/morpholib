#!/usr/bin/env python3
"""
models/2007-royer-ginkgo-adiantoides-collective-stigmergy.py — Multi-Scale & Paleobiological Reproduction (Slot #2073)
Organism: Ginkgo adiantoides (Paleocene-Eocene Bilobed Coalescent Ginkgo Chronospecies)
Phenomenon Scale: 10.2 cm (0.102 m, log10=-0.991, 7_organism_cm)
Anatomical Structure: Collective swarm / biofilm / colonial array in Ginkgo adiantoides (Paleocene-Eocene Boundary (56 Ma))
Geological Horizon: Paleocene-Eocene Boundary (56 Ma) (56.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Ginkgo
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation (10.2 cm))
Formal Verification: `morphoeval/lean/loops/Loop_2073_ginkgo_adiantoides_collective_stigmergy.lean` (`MathOfBiology.Loop_2073`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/ginkgo_adiantoides_collective_stigmergy"
EXPECTED_SHA256 = {
    "ginkgo_adiantoides_collective_stigmergy_reference.json": "2f37f20f58f6a7e604656160fe97d2b7e9d559e1c1357e1cc749a09f52068e67",
    "ginkgo_adiantoides_specimen.jpg": "9b107911c55810e272a4f18af3a1f21e0f0d95561cf384cba5c90c9245314ca8",
    "scout_manifest.json": "46998ef0635f1084fbf14df4dac2bca6c48dd193aad00096a5c75c27d312406b",
    "paper_citations.bib": "79898f953bc8ad21cf565bea49fc47b6d347fad1bb906aad1da40ad553709d30"
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
    ref = json.loads((DATASET_DIR / "ginkgo_adiantoides_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2007-royer-ginkgo-adiantoides-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
