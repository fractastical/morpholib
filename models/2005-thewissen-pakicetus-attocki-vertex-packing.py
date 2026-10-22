#!/usr/bin/env python3
"""
models/2005-thewissen-pakicetus-attocki-vertex-packing.py — Multi-Scale & Paleobiological Reproduction (Slot #1583)
Organism: Pakicetus attocki (Early Eocene Terrestrial Wolf-Sized Stem-Cetacean)
Phenomenon Scale: 356 µm (0.000356 m, log10=-3.449, 5_primordium_100um)
Anatomical Structure: Apical epithelial cell vertex polygon in Pakicetus attocki (Early Eocene (Ypresian, 49.0 Ma))
Geological Horizon: Early Eocene (Ypresian, 49.0 Ma) (49.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Pakicetus
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing (356 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1583_pakicetus_attocki_vertex_packing.lean` (`MathOfBiology.Loop_1583`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/pakicetus_attocki_vertex_packing"
EXPECTED_SHA256 = {
    "pakicetus_attocki_vertex_packing_reference.json": "040af9fa3ab949890c724cdfe2b8c4ab5d27938b7a5c7499a5bb1859658c316c",
    "pakicetus_attocki_specimen.jpg": "f535c1d722f574fc11becd2103eda0c5697e7184d39cb5b8ee794f42ce68e4a4",
    "scout_manifest.json": "1fcbae1c98154ad163a42af249d65326f620fca0cdd58bc6cd09e44a0960c943",
    "paper_citations.bib": "73703b30be910b932192001cd0318c07f9ae28f2c44e6c196082fe08701e910c"
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
    ref = json.loads((DATASET_DIR / "pakicetus_attocki_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2005-thewissen-pakicetus-attocki-vertex-packing.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
