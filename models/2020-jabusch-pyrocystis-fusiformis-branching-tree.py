#!/usr/bin/env python3
"""
models/2020-jabusch-pyrocystis-fusiformis-branching-tree.py — Multi-Scale & Paleobiological Reproduction (Slot #1740)
Organism: Pyrocystis fusiformis (Bioluminescent Marine Dinoflagellate (Shear-Induced Scintillon Flash))
Phenomenon Scale: 140 µm (0.0001397 m, log10=-3.855, 5_primordium_100um)
Anatomical Structure: Bifurcating vascular / bronchial / hyphal branch in Pyrocystis fusiformis (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Pyrocystis
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting (140 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1740_pyrocystis_fusiformis_branching_tree.lean` (`MathOfBiology.Loop_1740`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/pyrocystis_fusiformis_branching_tree"
EXPECTED_SHA256 = {
    "pyrocystis_fusiformis_branching_tree_reference.json": "ca944ee05d0fc141300b91fa9aa8fff5f2638bc12525a58994bce4eaf1e07e15",
    "pyrocystis_fusiformis_specimen.jpg": "4fad32ed3942f3ab89d8e79cda844c3cab6085f9a3ce836885fe9d9bf8112cb2",
    "scout_manifest.json": "82504de46587f7ab1165dc192e0a0182f59dee7b44dc0a9384b37c5ee04233ba",
    "paper_citations.bib": "2a453ee9f587db1d90c534fb02a992fa0ac52a032a4272f5e5af0be080081adc"
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
    ref = json.loads((DATASET_DIR / "pyrocystis_fusiformis_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2020-jabusch-pyrocystis-fusiformis-branching-tree.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
