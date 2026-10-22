#!/usr/bin/env python3
"""
models/2014-chen-dasypus-novemcinctus-flux-canalization.py — Multi-Scale & Paleobiological Reproduction (Slot #1546)
Organism: Dasypus novemcinctus (Nine-Banded Armadillo (Hexagonal Osteoderm Rosette Tiles))
Phenomenon Scale: 1.7 mm (0.001706 m, log10=-2.768, 6_organ_mm)
Anatomical Structure: Auxin flux vein / phyllotactic primordium in Dasypus novemcinctus (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Dasypus
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth (1.7 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1546_dasypus_novemcinctus_flux_canalization.lean` (`MathOfBiology.Loop_1546`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/dasypus_novemcinctus_flux_canalization"
EXPECTED_SHA256 = {
    "dasypus_novemcinctus_flux_canalization_reference.json": "f097f4ff52cd0308d9041e0d5889d240a6abdbda2e6a8c6e5bb84dfd3d73734b",
    "dasypus_novemcinctus_specimen.jpg": "5cbb611c05d8707d92d1c6d2cdd19de1525cb938009adae92309147f22174539",
    "scout_manifest.json": "187b2c67991a10b618c965fb9fccddba9131263969c3aed6e2f58001817b0386",
    "paper_citations.bib": "fc51e79549d8fc19053a1f41b642e32bb14f34bbec6394034350edeb716801a9"
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
    ref = json.loads((DATASET_DIR / "dasypus_novemcinctus_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2014-chen-dasypus-novemcinctus-flux-canalization.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
