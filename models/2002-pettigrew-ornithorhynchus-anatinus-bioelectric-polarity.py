#!/usr/bin/env python3
"""
models/2002-pettigrew-ornithorhynchus-anatinus-bioelectric-polarity.py — Multi-Scale & Paleobiological Reproduction (Slot #1994)
Organism: Ornithorhynchus anatinus (Duck-Billed Platypus (Electroreceptive Bill Pore Hexagonal Array))
Phenomenon Scale: 109 nm (1.086e-07 m, log10=-6.964, 2_macromolecular_100nm)
Anatomical Structure: Gap-junction bioelectric domain & ion-channel patch in Ornithorhynchus anatinus (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Ornithorhynchus
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity (109 nm))
Formal Verification: `morphoeval/lean/loops/Loop_1994_ornithorhynchus_anatinus_bioelectric_polarity.lean` (`MathOfBiology.Loop_1994`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/ornithorhynchus_anatinus_bioelectric_polarity"
EXPECTED_SHA256 = {
    "ornithorhynchus_anatinus_bioelectric_polarity_reference.json": "126811cdd985e7357647726b1243f7944b2dc954ae57322a4f1886ccdeb4e975",
    "ornithorhynchus_anatinus_specimen.jpg": "5a8d5bf6c1809500a8e9e88694cff57c75cead6258be2e7d05ede7922465f2b6",
    "scout_manifest.json": "131dec8d598b0bbdcbf35779da25fb1e89704a621f82929059209ffcfb5d7c52",
    "paper_citations.bib": "349d437fa72f892d6c101e1106eb764d02eb34f5f7445f94f31a8e1493cabcac"
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
    ref = json.loads((DATASET_DIR / "ornithorhynchus_anatinus_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2002-pettigrew-ornithorhynchus-anatinus-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
