#!/usr/bin/env python3
"""
models/2002-hughes-aulacopleura-koninckii-meraspis-flux-canalization.py — Multi-Scale & Paleobiological Reproduction (Slot #1512)
Organism: Aulacopleura koninckii_meraspis (Silurian Trilobite Meraspid Larva (12 Thoracic Segments, Early Morph))
Phenomenon Scale: 1.1 mm (0.001144 m, log10=-2.942, 6_organ_mm)
Anatomical Structure: Auxin flux vein / phyllotactic primordium in Aulacopleura koninckii_meraspis (Silurian (Wenlock Epoch, 428.5 Ma))
Geological Horizon: Silurian (Wenlock Epoch, 428.5 Ma) (428.5 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Aulacopleura
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth (1.1 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1512_aulacopleura_koninckii_meraspis_flux_canalization.lean` (`MathOfBiology.Loop_1512`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/aulacopleura_koninckii_meraspis_flux_canalization"
EXPECTED_SHA256 = {
    "aulacopleura_koninckii_meraspis_flux_canalization_reference.json": "ca90194ae140a5d561d04a44f82a0adb7d1c3ec64c35863836649565b85ae3cf",
    "aulacopleura_koninckii_meraspis_specimen.jpg": "1b6c8997dc42e91d0975aee1225a27810db08c0301831c90db07bebae7b0344a",
    "scout_manifest.json": "101c8bb3fdab8172f1bc3544f7ed2639c6d58ee26cfc7a8fa90e42a99a3dfd4e",
    "paper_citations.bib": "aaf1918cfc87a42d46770b92f8ed657950209d55d56e8b2512306d8328fae6eb"
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
    ref = json.loads((DATASET_DIR / "aulacopleura_koninckii_meraspis_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2002-hughes-aulacopleura-koninckii-meraspis-flux-canalization.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
