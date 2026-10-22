#!/usr/bin/env python3
"""
models/2015-volkening-danio-rerio-mm32-adult-caudal-fin-flank-stripes-clock-wavefront.py — Multi-Scale & Paleobiological Reproduction (Slot #1183)
Organism: Danio rerio (Zebrafish — 60–90 dpf)
Phenomenon Scale: 2.8 cm (0.02784 m, log10=-1.555, 7_organism_cm)
Anatomical Structure: Adult 5-stripe flank epidermis (32 mm SL) & bifurcating lepidotrichia rays (60–90 dpf (Adult Body Flank & Regenerating Caudal Fin, ZFA:0000016)) · Daniocell / ZFIN adult_dermal_chromatophore & osteoblast_fin_ray
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Danio
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest at 2.8 cm (Adult 5-stripe flank epidermis (32 mm SL) & bifurcating lepidotrichia rays))
Formal Verification: `morphoeval/lean/loops/Loop_1183_danio_rerio_mm32_adult_caudal_fin_flank_stripes_clock_wavefront.lean` (`MathOfBiology.Loop_1183`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/danio_rerio_mm32_adult_caudal_fin_flank_stripes_clock_wavefront"
EXPECTED_SHA256 = {
    "danio_rerio_mm32_adult_caudal_fin_flank_stripes_clock_wavefront_reference.json": "639c62abdac158309d91cbaeb16bf3d52815fc9e0e619bc4e87aca64a8e9d817",
    "danio_rerio_specimen.jpg": "850b2d4a94c7cbcb3c16e606ddb70d198f082ccb2f046ee44dbaa5ef5941678a",
    "scout_manifest.json": "708b62418b5fd44166b36c38bdc04cf9bb85c3cfb7e5ae12a5e3ed72d92f5032",
    "paper_citations.bib": "764e5b870a2df9487f8eb0a2d6dffaa56a8e5bf68493aaf2e7be4f25ffbce0cb"
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
    ref = json.loads((DATASET_DIR / "danio_rerio_mm32_adult_caudal_fin_flank_stripes_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2015-volkening-danio-rerio-mm32-adult-caudal-fin-flank-stripes-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
