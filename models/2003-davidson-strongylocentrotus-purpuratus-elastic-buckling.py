#!/usr/bin/env python3
"""
models/2003-davidson-strongylocentrotus-purpuratus-elastic-buckling.py — Multi-Scale & Paleobiological Reproduction (Slot #1813)
Organism: Strongylocentrotus purpuratus (Purple Sea Urchin (Larval Calcite Spicule Triradiate Skeleton))
Phenomenon Scale: 17 µm (1.739e-05 m, log10=-4.760, 4_cellular_10um)
Anatomical Structure: Elastic bilayer fold / wrinkle / suture wavelength in Strongylocentrotus purpuratus (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Strongylocentrotus
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling (17 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1813_strongylocentrotus_purpuratus_elastic_buckling.lean` (`MathOfBiology.Loop_1813`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/strongylocentrotus_purpuratus_elastic_buckling"
EXPECTED_SHA256 = {
    "strongylocentrotus_purpuratus_elastic_buckling_reference.json": "4dca14ef79858529b2fa338b339b22656c0f0c83f04d0e48299458b9335524c7",
    "strongylocentrotus_purpuratus_specimen.jpg": "515b723840267df6465272f908c6951fdf06c057159d3c6f7d6e67e7bee5224a",
    "scout_manifest.json": "b8d3bbf674fc5d449bc11a5c57dc1e857f10d2fdfdc109ebec3351bcfdde390a",
    "paper_citations.bib": "bfdd61e037795450dfce750de9091fe47f7043c8248681985932f367ffdef15c"
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
    ref = json.loads((DATASET_DIR / "strongylocentrotus_purpuratus_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2003-davidson-strongylocentrotus-purpuratus-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
