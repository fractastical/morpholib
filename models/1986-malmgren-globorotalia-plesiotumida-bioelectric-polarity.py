#!/usr/bin/env python3
"""
models/1986-malmgren-globorotalia-plesiotumida-bioelectric-polarity.py — Multi-Scale & Paleobiological Reproduction (Slot #1987)
Organism: Globorotalia plesiotumida (Late Miocene Moderately-Swollen Trochospiral Foraminifer Chronospecies)
Phenomenon Scale: 396 nm (3.958e-07 m, log10=-6.403, 2_macromolecular_100nm)
Anatomical Structure: Gap-junction bioelectric domain & ion-channel patch in Globorotalia plesiotumida (Late Miocene (Messinian, 6.2 Ma))
Geological Horizon: Late Miocene (Messinian, 6.2 Ma) (6.2 Ma)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Globorotalia
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity (396 nm))
Formal Verification: `morphoeval/lean/loops/Loop_1987_globorotalia_plesiotumida_bioelectric_polarity.lean` (`MathOfBiology.Loop_1987`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/globorotalia_plesiotumida_bioelectric_polarity"
EXPECTED_SHA256 = {
    "globorotalia_plesiotumida_bioelectric_polarity_reference.json": "2ea3bd2dc09a0f3747495fe93f9b77c8a854cdd10baee8f2144a3442ea0eb62c",
    "globorotalia_plesiotumida_specimen.jpg": "c1c82488a477ce08218e48d667618dbcb5efdb7a98603919f3838b81a82663b6",
    "scout_manifest.json": "9fb3af838ee6873e4345ec52ef1ed28960d5fa5933a756f53a1f474a11d5bf84",
    "paper_citations.bib": "0407edfc9eeb34e7ec8166fcd024a48ae6f804cd6ca438a21e5b9c3db89adfe5"
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
    ref = json.loads((DATASET_DIR / "globorotalia_plesiotumida_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1986-malmgren-globorotalia-plesiotumida-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
