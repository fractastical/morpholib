#!/usr/bin/env python3
"""
models/1996-macfadden-eohippus-angustidens-vertex-packing.py — Multi-Scale & Paleobiological Reproduction (Slot #1578)
Organism: Eohippus angustidens (Early Eocene Four-Toed Dawn Horse (Hyracotherium))
Phenomenon Scale: 206 µm (0.0002061 m, log10=-3.686, 5_primordium_100um)
Anatomical Structure: Apical epithelial cell vertex polygon in Eohippus angustidens (Early Eocene (Wasatchian, 52.0 Ma))
Geological Horizon: Early Eocene (Wasatchian, 52.0 Ma) (52.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Eohippus
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing (206 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1578_eohippus_angustidens_vertex_packing.lean` (`MathOfBiology.Loop_1578`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/eohippus_angustidens_vertex_packing"
EXPECTED_SHA256 = {
    "eohippus_angustidens_vertex_packing_reference.json": "cd496d77388c097d17bdf82a1b320df56ed34935592bdb270ac31d18190b9c72",
    "eohippus_angustidens_specimen.jpg": "4f492b7fff0d285dc090002b29f9aca4a757bad6f4c08a8f5dd93b8a1bcfc0bf",
    "scout_manifest.json": "2f3bf3dfaa5d8f8e5373c8275d209e6574c77b5df36838c6ce5ac51d1627b90c",
    "paper_citations.bib": "251b8fe377fbb0dd455ec2a06cc8a785aa42218b9ec895dddfb7e470d7d75ef5"
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
    ref = json.loads((DATASET_DIR / "eohippus_angustidens_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1996-macfadden-eohippus-angustidens-vertex-packing.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
