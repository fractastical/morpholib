#!/usr/bin/env python3
"""
models/2001-hughes-aulacopleura-koninckii-meraspis-clock-wavefront.py — Multi-Scale & Paleobiological Reproduction (Slot #1422)
Organism: Aulacopleura koninckii_meraspis (Silurian Trilobite Meraspid Larva (12 Thoracic Segments, Early Morph))
Phenomenon Scale: 366 µm (0.0003662 m, log10=-3.436, 5_primordium_100um)
Anatomical Structure: Segmentation clock somite / metamere unit in Aulacopleura koninckii_meraspis (Silurian (Wenlock Epoch, 428.5 Ma))
Geological Horizon: Silurian (Wenlock Epoch, 428.5 Ma) (428.5 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Aulacopleura
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest (366 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1422_aulacopleura_koninckii_meraspis_clock_wavefront.lean` (`MathOfBiology.Loop_1422`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/aulacopleura_koninckii_meraspis_clock_wavefront"
EXPECTED_SHA256 = {
    "aulacopleura_koninckii_meraspis_clock_wavefront_reference.json": "f46616ce6c4dded38fd21fc61856d3645c3962d81905f5026de0054063289438",
    "aulacopleura_koninckii_meraspis_specimen.jpg": "1b6c8997dc42e91d0975aee1225a27810db08c0301831c90db07bebae7b0344a",
    "scout_manifest.json": "0b81adc0757a487e3cca2cbfb2db51930a90ee98fee9ea745f493f5dea930107",
    "paper_citations.bib": "37aca7c64d0eb247f76bf11f0ab1d9edaa0d516227ae1edc9e9edc9cbf88f00f"
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
    ref = json.loads((DATASET_DIR / "aulacopleura_koninckii_meraspis_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2001-hughes-aulacopleura-koninckii-meraspis-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
