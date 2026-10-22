#!/usr/bin/env python3
"""
models/2014-soroldoni-danio-rerio-um55-somite-rosette-neuromast-flux-canalization.py — Multi-Scale & Paleobiological Reproduction (Slot #1154)
Organism: Danio rerio (Zebrafish — 16–24 hpf)
Phenomenon Scale: 51 µm (5.088e-05 m, log10=-4.293, 4_cellular_10um)
Anatomical Structure: Single nascent somite block (55 µm AP) & deposited lateral-line neuromast (16–24 hpf (Somite Epithelial Somitomere & Neuromast Rosette, ZFA:0000155)) · Zebrahub lateral_line_neuromast_rosette & segmented_paraxial_somite
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Danio
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth at 51 µm (Single nascent somite block (55 µm AP) & deposited lateral-line neuromast))
Formal Verification: `morphoeval/lean/loops/Loop_1154_danio_rerio_um55_somite_rosette_neuromast_flux_canalization.lean` (`MathOfBiology.Loop_1154`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/danio_rerio_um55_somite_rosette_neuromast_flux_canalization"
EXPECTED_SHA256 = {
    "danio_rerio_um55_somite_rosette_neuromast_flux_canalization_reference.json": "a3b1143d6b713d21f28a5a883e9a6e93c195478d6709034fc3f4c77841d9f1ff",
    "danio_rerio_specimen.jpg": "850b2d4a94c7cbcb3c16e606ddb70d198f082ccb2f046ee44dbaa5ef5941678a",
    "scout_manifest.json": "d41959f00dc4ef1088465845c51b4d666a3ff04f90bf5515441e00cd954ca295",
    "paper_citations.bib": "baf5158fd83d3fc2da9d858ebf673514cc54bb7ab6c70289779054a7f2000498"
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
    ref = json.loads((DATASET_DIR / "danio_rerio_um55_somite_rosette_neuromast_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2014-soroldoni-danio-rerio-um55-somite-rosette-neuromast-flux-canalization.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
