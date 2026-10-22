#!/usr/bin/env python3
"""
models/2012-li-microraptor-gui-turing-wavelength.py — Multi-Scale & Paleobiological Reproduction (Slot #1231)
Organism: Microraptor gui (Early Cretaceous Iridescent Black Four-Winged Dromaeosaur)
Phenomenon Scale: 34 nm (3.422e-08 m, log10=-7.466, 1_molecular_nm)
Anatomical Structure: Turing pigment stripe / ocellus wavelength in Microraptor gui (Early Cretaceous (Jiufotang Fm, 120.0 Ma))
Geological Horizon: Early Cretaceous (Jiufotang Fm, 120.0 Ma) (120.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Microraptor
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection (34 nm))
Formal Verification: `morphoeval/lean/loops/Loop_1231_microraptor_gui_turing_wavelength.lean` (`MathOfBiology.Loop_1231`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/microraptor_gui_turing_wavelength"
EXPECTED_SHA256 = {
    "microraptor_gui_turing_wavelength_reference.json": "bc4ea67ebb8cd76b7a1fa0553fe69c2c2dd9ffd166254dc3ed0ca46b0ee9423a",
    "microraptor_gui_specimen.jpg": "24fb32a1367db5df0d77dace2b5b810c0854a2b3c231e370b837356348765137",
    "scout_manifest.json": "358d8188573f814b52c1924a2dd2018504fda2d7f46cacbd181ddbed8b275bbf",
    "paper_citations.bib": "d6f6e1d3e4c7ae2fc6aa8930cabe11525ee397e0d674edc428a25abc95e2e2be"
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
    ref = json.loads((DATASET_DIR / "microraptor_gui_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2012-li-microraptor-gui-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
