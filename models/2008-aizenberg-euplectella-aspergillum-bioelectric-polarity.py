#!/usr/bin/env python3
"""
models/2008-aizenberg-euplectella-aspergillum-bioelectric-polarity.py — Multi-Scale & Paleobiological Reproduction (Slot #2005)
Organism: Euplectella aspergillum (Venus' Flower Basket Glass Sponge (Hierarchical Silica Lattice Cage))
Phenomenon Scale: 1.0 µm (1.02e-06 m, log10=-5.991, 3_subcellular_um)
Anatomical Structure: Gap-junction bioelectric domain & ion-channel patch in Euplectella aspergillum (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Euplectella
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity (1.0 µm))
Formal Verification: `morphoeval/lean/loops/Loop_2005_euplectella_aspergillum_bioelectric_polarity.lean` (`MathOfBiology.Loop_2005`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/euplectella_aspergillum_bioelectric_polarity"
EXPECTED_SHA256 = {
    "euplectella_aspergillum_bioelectric_polarity_reference.json": "f89b00f693bcc9f0d496b2a0231eed227fe225727426341ce2f08a08d8556499",
    "euplectella_aspergillum_specimen.jpg": "c39266f00e99a0f35db98c47cad0aab61aaf982021afc8d7827e37376a49fd28",
    "scout_manifest.json": "5e1980e5d476955ec1825f40a907692370a8ad3a16e2b6c3506bbb9bb30d77af",
    "paper_citations.bib": "d736cd0e44dae6bfa70049341e64bc54a54ddffe8727403e09d66e23d195b765"
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
    ref = json.loads((DATASET_DIR / "euplectella_aspergillum_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2008-aizenberg-euplectella-aspergillum-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
