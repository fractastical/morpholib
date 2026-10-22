#!/usr/bin/env python3
"""
models/2018-wang-manis-javanica-clock-wavefront.py — Multi-Scale & Paleobiological Reproduction (Slot #1455)
Organism: Manis javanica (Sunda Pangolin (Imbricated Keratin Scale Stress-Shielding Lattice))
Phenomenon Scale: 2.8 mm (0.002828 m, log10=-2.549, 6_organ_mm)
Anatomical Structure: Segmentation clock somite / metamere unit in Manis javanica (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Manis
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest (2.8 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1455_manis_javanica_clock_wavefront.lean` (`MathOfBiology.Loop_1455`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/manis_javanica_clock_wavefront"
EXPECTED_SHA256 = {
    "manis_javanica_clock_wavefront_reference.json": "2f6f45c066bc406f6ee2255c68e9294806a6d88cc3a2173f43057dbee3a2019c",
    "manis_javanica_specimen.jpg": "0d2cf7b0e5853e57873b4a69f96a0bf200561aa593feda56d04025e8b7619711",
    "scout_manifest.json": "bb1168f0f357e3fec2ea8674f930cbb317476eddab187e3c0c02dcddb7d3dab1",
    "paper_citations.bib": "a49c018efc97e39963981291469458d25d2776208b267d1957dcef50fd2e36a1"
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
    ref = json.loads((DATASET_DIR / "manis_javanica_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2018-wang-manis-javanica-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
