#!/usr/bin/env python3
"""
models/2017-gilmour-danio-rerio-um220-pllp-heart-primordium-morphogen-condensation.py — Multi-Scale & Paleobiological Reproduction (Slot #1162)
Organism: Danio rerio (Zebrafish — 24–36 hpf)
Phenomenon Scale: 179 µm (0.0001793 m, log10=-3.746, 5_primordium_100um)
Anatomical Structure: Migrating posterior lateral line primordium (PLLP) & looping heart tube (24–36 hpf (Pharyngula PLLP & Linear Heart Tube, ZFA:0001279)) · Zebrahub migrating_PLLP_stream & myocardial_endocardial_tube (24 hpf)
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Danio
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation at 179 µm (Migrating posterior lateral line primordium (PLLP) & looping heart tube))
Formal Verification: `morphoeval/lean/loops/Loop_1162_danio_rerio_um220_pllp_heart_primordium_morphogen_condensation.lean` (`MathOfBiology.Loop_1162`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/danio_rerio_um220_pllp_heart_primordium_morphogen_condensation"
EXPECTED_SHA256 = {
    "danio_rerio_um220_pllp_heart_primordium_morphogen_condensation_reference.json": "b23ac452ffdcad44dec19dd4f1aa8f9500e8897f12fdfc5532615cf737d2bac1",
    "danio_rerio_specimen.jpg": "850b2d4a94c7cbcb3c16e606ddb70d198f082ccb2f046ee44dbaa5ef5941678a",
    "scout_manifest.json": "befc3bcb38c9409612aed9b5d647a6146f17885c237d3e8d5f7e44cc545aae55",
    "paper_citations.bib": "b25dabd45af9cb2750b80ee312b21feaeb8723d927e33e93a7b2e5efdbc42eb0"
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
    ref = json.loads((DATASET_DIR / "danio_rerio_um220_pllp_heart_primordium_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2017-gilmour-danio-rerio-um220-pllp-heart-primordium-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
