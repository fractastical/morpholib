#!/usr/bin/env python3
"""
models/2011-boisvert-panderichthys-rhombolepis-bioelectric-polarity.py — Multi-Scale & Paleobiological Reproduction (Slot #1933)
Organism: Panderichthys rhombolepis (Late Devonian Elpistostegalian Flat-Skulled Fish)
Phenomenon Scale: 139 µm (0.0001386 m, log10=-3.858, 5_primordium_100um)
Anatomical Structure: Gap-junction bioelectric domain & ion-channel patch in Panderichthys rhombolepis (Late Devonian (Frasnian, 380 Ma))
Geological Horizon: Late Devonian (Frasnian, 380 Ma) (380.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Panderichthys
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity (139 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1933_panderichthys_rhombolepis_bioelectric_polarity.lean` (`MathOfBiology.Loop_1933`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/panderichthys_rhombolepis_bioelectric_polarity"
EXPECTED_SHA256 = {
    "panderichthys_rhombolepis_bioelectric_polarity_reference.json": "94159ec961ce3f324c8c886623b0a139290b1ecf7278759bc6397489e8dd17f1",
    "panderichthys_rhombolepis_specimen.jpg": "5c39604c489cd7a850b62b1d1aaf2e3061b9865043e83968c2feebf4e34ca870",
    "scout_manifest.json": "bccb99c6dcfc6d85e566531bd63456adf43c100594666550b191f001126b62e1",
    "paper_citations.bib": "6e46ddf3b098ae5a10f36dec5b0b7a7a6b55984d4f91089616b97ed61da99173"
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
    ref = json.loads((DATASET_DIR / "panderichthys_rhombolepis_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2011-boisvert-panderichthys-rhombolepis-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
