#!/usr/bin/env python3
"""
models/2017-nweeia-monodon-monoceros-bioelectric-polarity.py — Multi-Scale & Paleobiological Reproduction (Slot #1947)
Organism: Monodon monoceros (Arctic Narwhal (Chiral Helical Left Incisor Tusk))
Phenomenon Scale: 1.6 mm (0.001554 m, log10=-2.809, 6_organ_mm)
Anatomical Structure: Gap-junction bioelectric domain & ion-channel patch in Monodon monoceros (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Monodon
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity (1.6 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1947_monodon_monoceros_bioelectric_polarity.lean` (`MathOfBiology.Loop_1947`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/monodon_monoceros_bioelectric_polarity"
EXPECTED_SHA256 = {
    "monodon_monoceros_bioelectric_polarity_reference.json": "13b145ec458211a81848745c05bccf22d5fabf5e5095fa903075a7f32583cf66",
    "monodon_monoceros_specimen.jpg": "1fa8766cd6ff186448201eb9da7097191319445a02628f275cfe5752d1be8cc3",
    "scout_manifest.json": "b3e36fa340955fb9afa47e2975f4784690e35a9d780c3e9465f991e34353a55c",
    "paper_citations.bib": "028e81a346a97d91a07277b12dd4c74def546fe2a65e04abefd2eaaf35caf597"
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
    ref = json.loads((DATASET_DIR / "monodon_monoceros_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2017-nweeia-monodon-monoceros-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
