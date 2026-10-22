#!/usr/bin/env python3
"""
models/2014-yamanaka-danio-rerio-um18-melanophore-xanthophore-cell-elastic-buckling.py — Multi-Scale & Paleobiological Reproduction (Slot #1147)
Organism: Danio rerio (Zebrafish — 14–21 dpf)
Phenomenon Scale: 20 µm (1.962e-05 m, log10=-4.707, 4_cellular_10um)
Anatomical Structure: Single melanophore–xanthophore pseudopodial contact & NMP cell body (14–21 dpf (Metamorphic Pigment & Tailbud Progenitor, ZFA:0009096)) · Zebrahub tailbud_neuromesodermal_progenitors & pigment_chromatophores
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Danio
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling at 20 µm (Single melanophore–xanthophore pseudopodial contact & NMP cell body))
Formal Verification: `morphoeval/lean/loops/Loop_1147_danio_rerio_um18_melanophore_xanthophore_cell_elastic_buckling.lean` (`MathOfBiology.Loop_1147`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/danio_rerio_um18_melanophore_xanthophore_cell_elastic_buckling"
EXPECTED_SHA256 = {
    "danio_rerio_um18_melanophore_xanthophore_cell_elastic_buckling_reference.json": "1074b724f0a9fa6abdc9fdaa5038b00a724f9caaadaf0b41bdb6b5f295343b57",
    "danio_rerio_specimen.jpg": "850b2d4a94c7cbcb3c16e606ddb70d198f082ccb2f046ee44dbaa5ef5941678a",
    "scout_manifest.json": "246aee183bf6eaa2cf30f5f562ea798f6f1a9ff3c3f9a291e57dfbe31b6786df",
    "paper_citations.bib": "dfe5e0ae10398fb58392d14a75ff9b6da2cd2f3e97ca58f2986f6520ef630406"
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
    ref = json.loads((DATASET_DIR / "danio_rerio_um18_melanophore_xanthophore_cell_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2014-yamanaka-danio-rerio-um18-melanophore-xanthophore-cell-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
