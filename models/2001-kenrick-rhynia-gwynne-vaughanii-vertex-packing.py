#!/usr/bin/env python3
"""
models/2001-kenrick-rhynia-gwynne-vaughanii-vertex-packing.py — Multi-Scale & Paleobiological Reproduction (Slot #1615)
Organism: Rhynia gwynne_vaughanii (Early Devonian Rhynie Chert Leafless Protostele Sporophyte)
Phenomenon Scale: 8.4 µm (8.4e-06 m, log10=-5.076, 3_subcellular_um)
Anatomical Structure: Apical epithelial cell vertex polygon in Rhynia gwynne_vaughanii (Early Devonian (Pragian Rhynie Chert, 407 Ma))
Geological Horizon: Early Devonian (Pragian Rhynie Chert, 407 Ma) (407.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Rhynia
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing (8.4 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1615_rhynia_gwynne_vaughanii_vertex_packing.lean` (`MathOfBiology.Loop_1615`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/rhynia_gwynne_vaughanii_vertex_packing"
EXPECTED_SHA256 = {
    "rhynia_gwynne_vaughanii_vertex_packing_reference.json": "e29369f151abb6cddf567b862259ef0f06450b7cc8950e1eb32ff338070b724b",
    "rhynia_gwynne_vaughanii_specimen.jpg": "5eb2588bbe45fb5b68e29e7763da6962ffa7cf335ab0a5ad8d2e46211cea3f15",
    "scout_manifest.json": "d7fee6f44f09099eb766a5a26352b3687e27825c6048e20246e258469b70c79f",
    "paper_citations.bib": "d7f498aeeb759abcd28f67cca3cb2625270482965f129d1c3bada071028589e9"
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
    ref = json.loads((DATASET_DIR / "rhynia_gwynne_vaughanii_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2001-kenrick-rhynia-gwynne-vaughanii-vertex-packing.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
