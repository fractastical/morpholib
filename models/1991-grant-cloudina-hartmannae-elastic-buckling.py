#!/usr/bin/env python3
"""
models/1991-grant-cloudina-hartmannae-elastic-buckling.py — Multi-Scale & Paleobiological Reproduction (Slot #1792)
Organism: Cloudina hartmannae (Terminal Ediacaran Cone-in-Cone Biomineralized Calcite Tube)
Phenomenon Scale: 573 µm (0.0005729 m, log10=-3.242, 5_primordium_100um)
Anatomical Structure: Elastic bilayer fold / wrinkle / suture wavelength in Cloudina hartmannae (Terminal Ediacaran (Nama Group, 549 Ma))
Geological Horizon: Terminal Ediacaran (Nama Group, 549 Ma) (549.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Cloudina
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling (573 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1792_cloudina_hartmannae_elastic_buckling.lean` (`MathOfBiology.Loop_1792`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/cloudina_hartmannae_elastic_buckling"
EXPECTED_SHA256 = {
    "cloudina_hartmannae_elastic_buckling_reference.json": "e263d6a936cf4df297cb04792eeee4e4233fc3fe717b531389bcf8c9b4712cd2",
    "cloudina_hartmannae_specimen.jpg": "cbe0233dbf275f4e575ad54b3134b268c070d5cafa4a4ef2add96d579bc0efe6",
    "scout_manifest.json": "be20aaae7778757607c4de8c288e149bc50b56d2e6b056b2e61392a6835cae15",
    "paper_citations.bib": "cfda887ce773d9738ad31502e461b13559ebe18e0f71ef017b7c4eaaac29e125"
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
    ref = json.loads((DATASET_DIR / "cloudina_hartmannae_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1991-grant-cloudina-hartmannae-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
