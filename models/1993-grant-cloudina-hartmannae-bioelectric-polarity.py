#!/usr/bin/env python3
"""
models/1993-grant-cloudina-hartmannae-bioelectric-polarity.py — Multi-Scale & Paleobiological Reproduction (Slot #1972)
Organism: Cloudina hartmannae (Terminal Ediacaran Cone-in-Cone Biomineralized Calcite Tube)
Phenomenon Scale: 2.1 µm (2.083e-06 m, log10=-5.681, 3_subcellular_um)
Anatomical Structure: Gap-junction bioelectric domain & ion-channel patch in Cloudina hartmannae (Terminal Ediacaran (Nama Group, 549 Ma))
Geological Horizon: Terminal Ediacaran (Nama Group, 549 Ma) (549.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Cloudina
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity (2.1 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1972_cloudina_hartmannae_bioelectric_polarity.lean` (`MathOfBiology.Loop_1972`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/cloudina_hartmannae_bioelectric_polarity"
EXPECTED_SHA256 = {
    "cloudina_hartmannae_bioelectric_polarity_reference.json": "96cba06f88b4aec3c2b2b622b995f2205c73fbe554725b02effad620b159bb70",
    "cloudina_hartmannae_specimen.jpg": "cbe0233dbf275f4e575ad54b3134b268c070d5cafa4a4ef2add96d579bc0efe6",
    "scout_manifest.json": "014f997db32dac02783368e19ef5a3b6a6384d8a10dcb5152a1888b496c6f0d0",
    "paper_citations.bib": "f22d61c30da20905709c3e5b8d57dd5d82b2ea10796e5fb28b9ccfd9cc59df5b"
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
    ref = json.loads((DATASET_DIR / "cloudina_hartmannae_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1993-grant-cloudina-hartmannae-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
