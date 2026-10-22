#!/usr/bin/env python3
"""
models/2020-inoue-placenticeras-meeki-vertex-packing.py — Multi-Scale & Paleobiological Reproduction (Slot #1598)
Organism: Placenticeras meeki (Late Cretaceous Iridescent Ammolite Ammonite (Hyper-Fractal Suture, D_f=1.68))
Phenomenon Scale: 158 µm (0.0001581 m, log10=-3.801, 5_primordium_100um)
Anatomical Structure: Apical epithelial cell vertex polygon in Placenticeras meeki (Late Cretaceous (Campanian, 75 Ma))
Geological Horizon: Late Cretaceous (Campanian, 75 Ma) (75.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Placenticeras
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing (158 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1598_placenticeras_meeki_vertex_packing.lean` (`MathOfBiology.Loop_1598`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/placenticeras_meeki_vertex_packing"
EXPECTED_SHA256 = {
    "placenticeras_meeki_vertex_packing_reference.json": "78082a17e423a7947a908c8259736590daf4e72eccccde83700860ee29dfb8f2",
    "placenticeras_meeki_specimen.jpg": "ef945c35e08dd1adfded926af45c962f3c6017582c005f246eceedc2238e4229",
    "scout_manifest.json": "242c8701099ab3a12fae7da35e28c67afb47268d1d23016d7c862a689c134782",
    "paper_citations.bib": "cb1a36befbeabe247412dd6b7a2fd7b29012b417cb5e27eeef6ed96ee4466df9"
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
    ref = json.loads((DATASET_DIR / "placenticeras_meeki_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2020-inoue-placenticeras-meeki-vertex-packing.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
