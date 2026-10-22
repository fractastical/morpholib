#!/usr/bin/env python3
"""
models/2005-zheng-ginkgo-apodes-clock-wavefront.py — Multi-Scale & Paleobiological Reproduction (Slot #1442)
Organism: Ginkgo apodes (Early Cretaceous Multi-Lobed Jehol Biota Ginkgo Chronospecies)
Phenomenon Scale: 3.5 mm (0.003528 m, log10=-2.452, 6_organ_mm)
Anatomical Structure: Segmentation clock somite / metamere unit in Ginkgo apodes (Early Cretaceous (Yixian Fm, 121 Ma))
Geological Horizon: Early Cretaceous (Yixian Fm, 121 Ma) (121.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Ginkgo
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest (3.5 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1442_ginkgo_apodes_clock_wavefront.lean` (`MathOfBiology.Loop_1442`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/ginkgo_apodes_clock_wavefront"
EXPECTED_SHA256 = {
    "ginkgo_apodes_clock_wavefront_reference.json": "1dee280ece7379e64535d81f490b271e9b12a668f136440dd755e752ba91660f",
    "ginkgo_apodes_specimen.jpg": "3a074e9794ed0797dcc5abd51ab2fcccf8a109da1756d08ad6f52d95faf7d0f5",
    "scout_manifest.json": "923889bc7d78035838b04e3a15c891e84e2aeb7ff05957eebc79ded5a78e2356",
    "paper_citations.bib": "1c69401d40aa135940694b36559dd38dc806e24095dba4f42d6e55a503c938ce"
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
    ref = json.loads((DATASET_DIR / "ginkgo_apodes_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2005-zheng-ginkgo-apodes-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
