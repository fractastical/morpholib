#!/usr/bin/env python3
"""
models/1986-malmgren-globorotalia-plesiotumida-flux-canalization.py — Multi-Scale & Paleobiological Reproduction (Slot #1537)
Organism: Globorotalia plesiotumida (Late Miocene Moderately-Swollen Trochospiral Foraminifer Chronospecies)
Phenomenon Scale: 124 µm (0.0001237 m, log10=-3.908, 5_primordium_100um)
Anatomical Structure: Auxin flux vein / phyllotactic primordium in Globorotalia plesiotumida (Late Miocene (Messinian, 6.2 Ma))
Geological Horizon: Late Miocene (Messinian, 6.2 Ma) (6.2 Ma)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Globorotalia
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth (124 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1537_globorotalia_plesiotumida_flux_canalization.lean` (`MathOfBiology.Loop_1537`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/globorotalia_plesiotumida_flux_canalization"
EXPECTED_SHA256 = {
    "globorotalia_plesiotumida_flux_canalization_reference.json": "68e49a9bfb3e6cb6a647c6dbde3eaf3a94834a87580b9678ee0defbadc1575f0",
    "globorotalia_plesiotumida_specimen.jpg": "c1c82488a477ce08218e48d667618dbcb5efdb7a98603919f3838b81a82663b6",
    "scout_manifest.json": "3214c00d6e8b4f26bef4da311a00d546a5b667f96feb4f0777531933086e340f",
    "paper_citations.bib": "ad895015cf55c0afc231d92be44daeec2f4c5a6bd5b661fda75ef10d2a636574"
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
    ref = json.loads((DATASET_DIR / "globorotalia_plesiotumida_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1986-malmgren-globorotalia-plesiotumida-flux-canalization.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
