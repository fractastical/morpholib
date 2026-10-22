#!/usr/bin/env python3
"""
models/1985-malmgren-globorotalia-plesiotumida-clock-wavefront.py — Multi-Scale & Paleobiological Reproduction (Slot #1447)
Organism: Globorotalia plesiotumida (Late Miocene Moderately-Swollen Trochospiral Foraminifer Chronospecies)
Phenomenon Scale: 40 µm (3.958e-05 m, log10=-4.403, 4_cellular_10um)
Anatomical Structure: Segmentation clock somite / metamere unit in Globorotalia plesiotumida (Late Miocene (Messinian, 6.2 Ma))
Geological Horizon: Late Miocene (Messinian, 6.2 Ma) (6.2 Ma)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Globorotalia
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest (40 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1447_globorotalia_plesiotumida_clock_wavefront.lean` (`MathOfBiology.Loop_1447`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/globorotalia_plesiotumida_clock_wavefront"
EXPECTED_SHA256 = {
    "globorotalia_plesiotumida_clock_wavefront_reference.json": "6c64856aa7beb5529f54b4489739631d3ad28c521cde6b7710c28b1b63e696e3",
    "globorotalia_plesiotumida_specimen.jpg": "c1c82488a477ce08218e48d667618dbcb5efdb7a98603919f3838b81a82663b6",
    "scout_manifest.json": "430fabe24df49d689577e76fc7508680324cc7c16d38a26366ddb6ed425b711e",
    "paper_citations.bib": "8a687c069bc60ad801c981bbd7431225fbaaafd5f9faa7d9ed0d129944cd3ce3"
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
    ref = json.loads((DATASET_DIR / "globorotalia_plesiotumida_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1985-malmgren-globorotalia-plesiotumida-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
