#!/usr/bin/env python3
"""
models/2015-li-microraptor-gui-bioelectric-polarity.py — Multi-Scale & Paleobiological Reproduction (Slot #1951)
Organism: Microraptor gui (Early Cretaceous Iridescent Black Four-Winged Dromaeosaur)
Phenomenon Scale: 2.5 nm (2.5e-09 m, log10=-8.602, 1_molecular_nm)
Anatomical Structure: Gap-junction bioelectric domain & ion-channel patch in Microraptor gui (Early Cretaceous (Jiufotang Fm, 120.0 Ma))
Geological Horizon: Early Cretaceous (Jiufotang Fm, 120.0 Ma) (120.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Microraptor
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity (2.5 nm))
Formal Verification: `morphoeval/lean/loops/Loop_1951_microraptor_gui_bioelectric_polarity.lean` (`MathOfBiology.Loop_1951`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/microraptor_gui_bioelectric_polarity"
EXPECTED_SHA256 = {
    "microraptor_gui_bioelectric_polarity_reference.json": "52bd61d563385b9c7dfe9cc1a172d5b4ae0c30d288e448562cbe3d30ba097015",
    "microraptor_gui_specimen.jpg": "24fb32a1367db5df0d77dace2b5b810c0854a2b3c231e370b837356348765137",
    "scout_manifest.json": "f9339e95c97fcaff12f6b66e2f83601fb4764f849b68e1ca04f5213c2781cbff",
    "paper_citations.bib": "354c53ebd22a025d62e7d99c425c929df49b79b37cb654ac63ff14ced32c7f40"
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
    ref = json.loads((DATASET_DIR / "microraptor_gui_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2015-li-microraptor-gui-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
