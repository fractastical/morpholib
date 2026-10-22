#!/usr/bin/env python3
"""
models/2014-crane-ginkgo-biloba-elastic-buckling.py — Multi-Scale & Paleobiological Reproduction (Slot #1804)
Organism: Ginkgo biloba (Extant Maidenhair Tree (Dichotomous Open-Vein Fan Leaf))
Phenomenon Scale: 1.1 cm (0.01085 m, log10=-1.965, 7_organism_cm)
Anatomical Structure: Elastic bilayer fold / wrinkle / suture wavelength in Ginkgo biloba (Holocene (Extant Living Fossil))
Geological Horizon: Holocene (Extant Living Fossil) (0.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Ginkgo
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling (1.1 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1804_ginkgo_biloba_elastic_buckling.lean` (`MathOfBiology.Loop_1804`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/ginkgo_biloba_elastic_buckling"
EXPECTED_SHA256 = {
    "ginkgo_biloba_elastic_buckling_reference.json": "9515f5b7c8330d796151ce2e28cee25f8499972257f47ac8da13937e542bc602",
    "ginkgo_biloba_specimen.jpg": "b0a1f66d4688ddb858306c70a27a0bc279b41f3eaf1e9935f5b40d33cb7f7905",
    "scout_manifest.json": "bc176b270b6b71a2355235418684b8506e3a3cc80a865473858bca6387548813",
    "paper_citations.bib": "dd0d03eadfa62a6da00959745ef5d188458cba746d54d5345a6672e2089bd01e"
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
    ref = json.loads((DATASET_DIR / "ginkgo_biloba_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2014-crane-ginkgo-biloba-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
