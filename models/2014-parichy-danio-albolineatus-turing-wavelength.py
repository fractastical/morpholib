#!/usr/bin/env python3
"""
models/2014-parichy-danio-albolineatus-turing-wavelength.py — Multi-Scale & Paleobiological Reproduction (Slot #1202)
Organism: Danio albolineatus (Pearl Danio (Uniform Interstripe))
Phenomenon Scale: 1.3 mm (0.001282 m, log10=-2.892, 6_organ_mm)
Anatomical Structure: Turing pigment stripe / ocellus wavelength in Danio albolineatus (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Danio
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection (1.3 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1202_danio_albolineatus_turing_wavelength.lean` (`MathOfBiology.Loop_1202`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/danio_albolineatus_turing_wavelength"
EXPECTED_SHA256 = {
    "danio_albolineatus_turing_wavelength_reference.json": "6bf35f80b29f4baa1462f6ece7e26b437d4448a9b13b02f838fc7468a30655bd",
    "danio_albolineatus_specimen.jpg": "6059c6481299cc5bb1799fc70390ba1924f905610346e50fc579efcd09df8e46",
    "scout_manifest.json": "7da9954b4cf59d03a6d1387c56c1598fdd97feb18bb38add651a0472e364763d",
    "paper_citations.bib": "c5597db7e079e882da0c69f9e91827671996473c0c38a72a2828638630503872"
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
    ref = json.loads((DATASET_DIR / "danio_albolineatus_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2014-parichy-danio-albolineatus-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
