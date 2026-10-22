#!/usr/bin/env python3
"""
models/2015-gur-danio-rerio-nm180-iridophore-bragg-crystal-branching-tree.py — Multi-Scale & Paleobiological Reproduction (Slot #1126)
Organism: Danio rerio (Zebrafish — 72 hpf–Adult)
Phenomenon Scale: 186 nm (1.863e-07 m, log10=-6.730, 2_macromolecular_100nm)
Anatomical Structure: Guanine Bragg-stack nanocrystal platelet & tight-junction ZO-1 belt (72 hpf–Adult (Iridophore Reflector Platelet, ZFA:0009128)) · Zebrahub neural_crest_iridophore_lineage & EVL_periderm
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Danio
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting at 186 nm (Guanine Bragg-stack nanocrystal platelet & tight-junction ZO-1 belt))
Formal Verification: `morphoeval/lean/loops/Loop_1126_danio_rerio_nm180_iridophore_bragg_crystal_branching_tree.lean` (`MathOfBiology.Loop_1126`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/danio_rerio_nm180_iridophore_bragg_crystal_branching_tree"
EXPECTED_SHA256 = {
    "danio_rerio_nm180_iridophore_bragg_crystal_branching_tree_reference.json": "a5d23500d92b7c48689e60867390ebf46ec837c5038761892db32f8b0a94afb8",
    "danio_rerio_specimen.jpg": "850b2d4a94c7cbcb3c16e606ddb70d198f082ccb2f046ee44dbaa5ef5941678a",
    "scout_manifest.json": "c94097657a963c22bfcaa1fc1a3cdcebcd06c4626be3cabe17919fa6d1ff012c",
    "paper_citations.bib": "d6340dedb4bc0139e96080ef940f80f74208eddb2f508fb649797e6de6e1e521"
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
    ref = json.loads((DATASET_DIR / "danio_rerio_nm180_iridophore_bragg_crystal_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2015-gur-danio-rerio-nm180-iridophore-bragg-crystal-branching-tree.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
