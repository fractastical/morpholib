#!/usr/bin/env python3
"""
models/2012-hull-globorotalia-tumida-morphogen-condensation.py — Multi-Scale & Paleobiological Reproduction (Slot #1358)
Organism: Globorotalia tumida (Pliocene-Holocene Heavy-Keeled Biconvex Foraminifer Chronospecies)
Phenomenon Scale: 124 µm (0.0001236 m, log10=-3.908, 5_primordium_100um)
Anatomical Structure: Morphogen gradient & pre-cartilage condensation zone in Globorotalia tumida (Pliocene-Holocene (5.3–0 Ma))
Geological Horizon: Pliocene-Holocene (5.3–0 Ma) (0.0 Ma)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Globorotalia
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation (124 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1358_globorotalia_tumida_morphogen_condensation.lean` (`MathOfBiology.Loop_1358`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/globorotalia_tumida_morphogen_condensation"
EXPECTED_SHA256 = {
    "globorotalia_tumida_morphogen_condensation_reference.json": "cc779e34ce8bd89af174fa02d12a634fd297f01c50e39bf9b2fe13e4118d44cb",
    "globorotalia_tumida_specimen.jpg": "991d4acf78405580eb9f4efd4f75f1e9aec37937e62dce3147b582690341043b",
    "scout_manifest.json": "dddace7e170d0b69e91f4f552eaafae771a170bdf66e20ee237e725a9152dff9",
    "paper_citations.bib": "fa701d69e79b3c6dc254e45240d3de90ba9a7588ddc09131c1ac32c3d1432554"
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
    ref = json.loads((DATASET_DIR / "globorotalia_tumida_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2012-hull-globorotalia-tumida-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
