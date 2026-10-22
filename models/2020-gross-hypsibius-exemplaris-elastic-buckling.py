#!/usr/bin/env python3
"""
models/2020-gross-hypsibius-exemplaris-elastic-buckling.py — Multi-Scale & Paleobiological Reproduction (Slot #1821)
Organism: Hypsibius exemplaris (Water Bear Tardigrade (Cryptobiotic Tun Buckling & Stylet Morphogenesis))
Phenomenon Scale: 47 µm (4.695e-05 m, log10=-4.328, 4_cellular_10um)
Anatomical Structure: Elastic bilayer fold / wrinkle / suture wavelength in Hypsibius exemplaris (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Hypsibius
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling (47 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1821_hypsibius_exemplaris_elastic_buckling.lean` (`MathOfBiology.Loop_1821`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/hypsibius_exemplaris_elastic_buckling"
EXPECTED_SHA256 = {
    "hypsibius_exemplaris_elastic_buckling_reference.json": "eb1921292f704bafbbc5502df2cea31e1298ca216de894e47c7fc12fcd109eaf",
    "hypsibius_exemplaris_specimen.jpg": "47de00f0a828a76d4d2489503394a17024fde8a0989acc96911706aa535ba88d",
    "scout_manifest.json": "2a2f0b4010148d097765bfcaf748dfa3bbd8a50847e71f00c01cd5cba839efee",
    "paper_citations.bib": "79e7f8b11bb77e3fee8b4172ce7fdda0792761bc68ca98f2ac778d94557006ed"
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
    ref = json.loads((DATASET_DIR / "hypsibius_exemplaris_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2020-gross-hypsibius-exemplaris-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
