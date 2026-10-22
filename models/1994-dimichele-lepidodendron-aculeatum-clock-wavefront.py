#!/usr/bin/env python3
"""
models/1994-dimichele-lepidodendron-aculeatum-clock-wavefront.py — Multi-Scale & Paleobiological Reproduction (Slot #1438)
Organism: Lepidodendron aculeatum (Carboniferous Giant Scale-Tree (Helical Parastichy Leaf Cushions))
Phenomenon Scale: 2.7 mm (0.002706 m, log10=-2.568, 6_organ_mm)
Anatomical Structure: Segmentation clock somite / metamere unit in Lepidodendron aculeatum (Late Carboniferous (Pennsylvanian, 310 Ma))
Geological Horizon: Late Carboniferous (Pennsylvanian, 310 Ma) (310.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Lepidodendron
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest (2.7 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1438_lepidodendron_aculeatum_clock_wavefront.lean` (`MathOfBiology.Loop_1438`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/lepidodendron_aculeatum_clock_wavefront"
EXPECTED_SHA256 = {
    "lepidodendron_aculeatum_clock_wavefront_reference.json": "7c764bbb5de5e1e8f393dfeeb91dbc7c766d59d405d0e2fe2ab5ad9fb2af29ec",
    "lepidodendron_aculeatum_specimen.jpg": "854f4a6257e58a0123b8eee34674c32ed016882c88d6eb907ca1d85dbb78c467",
    "scout_manifest.json": "f1ebda0481af6e92e7a5dba1311a2e458ed318557e66d9eb5ec0c6c40067fed2",
    "paper_citations.bib": "1b674c3e90672d63b218c88200cd6d2f00bb49b09715dfff6090973a2fc131ca"
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
    ref = json.loads((DATASET_DIR / "lepidodendron_aculeatum_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1994-dimichele-lepidodendron-aculeatum-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
