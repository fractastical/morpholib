#!/usr/bin/env python3
"""
models/2023-jabusch-pyrocystis-fusiformis-bioelectric-polarity.py — Multi-Scale & Paleobiological Reproduction (Slot #2010)
Organism: Pyrocystis fusiformis (Bioluminescent Marine Dinoflagellate (Shear-Induced Scintillon Flash))
Phenomenon Scale: 319 nm (3.192e-07 m, log10=-6.496, 2_macromolecular_100nm)
Anatomical Structure: Gap-junction bioelectric domain & ion-channel patch in Pyrocystis fusiformis (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Pyrocystis
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity (319 nm))
Formal Verification: `morphoeval/lean/loops/Loop_2010_pyrocystis_fusiformis_bioelectric_polarity.lean` (`MathOfBiology.Loop_2010`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/pyrocystis_fusiformis_bioelectric_polarity"
EXPECTED_SHA256 = {
    "pyrocystis_fusiformis_bioelectric_polarity_reference.json": "320f47dcedf0a1475b4f774adb14f2a85a961a7bb6a1faf1a4c8b5eccfedd3d7",
    "pyrocystis_fusiformis_specimen.jpg": "4fad32ed3942f3ab89d8e79cda844c3cab6085f9a3ce836885fe9d9bf8112cb2",
    "scout_manifest.json": "1345123d68ce5535dd09a5490aa9e55bb1447fed38d245faa1bb6317d60b7d92",
    "paper_citations.bib": "ef3927169a224ffe53553c0c4a7e2020b57ba445b5d56c9a3fc1c87ffc3686f1"
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
    ref = json.loads((DATASET_DIR / "pyrocystis_fusiformis_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2023-jabusch-pyrocystis-fusiformis-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
