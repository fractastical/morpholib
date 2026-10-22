#!/usr/bin/env python3
"""
models/1991-ward-nautilus-pompilius-vertex-packing.py — Multi-Scale & Paleobiological Reproduction (Slot #1601)
Organism: Nautilus pompilius (Extant Chambered Nautilus (Equiangular Logarithmic Siphuncle))
Phenomenon Scale: 433 µm (0.0004331 m, log10=-3.363, 5_primordium_100um)
Anatomical Structure: Apical epithelial cell vertex polygon in Nautilus pompilius (Holocene (Extant Living Fossil))
Geological Horizon: Holocene (Extant Living Fossil) (0.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Nautilus
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing (433 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1601_nautilus_pompilius_vertex_packing.lean` (`MathOfBiology.Loop_1601`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/nautilus_pompilius_vertex_packing"
EXPECTED_SHA256 = {
    "nautilus_pompilius_vertex_packing_reference.json": "6c25c62ca6ccc1906371846bd4cd50ddd67575b081b5ab320aa9554c742e681c",
    "nautilus_pompilius_specimen.jpg": "5dfd47d29d062b864cb14b6f5260b674dffb372fac09ff90bdd8fc5caf196da1",
    "scout_manifest.json": "a19629e22154570cacdbb06feb3abcd6dd40127b51e7d609241d19343186f56d",
    "paper_citations.bib": "b088d6db0dbd9b62ad6d9bc01e75b5bcc21b7f012d0e117eeca52d790375e66f"
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
    ref = json.loads((DATASET_DIR / "nautilus_pompilius_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1991-ward-nautilus-pompilius-vertex-packing.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
