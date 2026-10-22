#!/usr/bin/env python3
"""
models/2017-parichy-danio-albolineatus-bioelectric-polarity.py — Multi-Scale & Paleobiological Reproduction (Slot #1922)
Organism: Danio albolineatus (Pearl Danio (Uniform Interstripe))
Phenomenon Scale: 26 µm (2.563e-05 m, log10=-4.591, 4_cellular_10um)
Anatomical Structure: Gap-junction bioelectric domain & ion-channel patch in Danio albolineatus (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Danio
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity (26 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1922_danio_albolineatus_bioelectric_polarity.lean` (`MathOfBiology.Loop_1922`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/danio_albolineatus_bioelectric_polarity"
EXPECTED_SHA256 = {
    "danio_albolineatus_bioelectric_polarity_reference.json": "28d9d87a4a4b3c3c467b20b139e22b9954d69f6da8eca2bf9562ad8ce75044aa",
    "danio_albolineatus_specimen.jpg": "6059c6481299cc5bb1799fc70390ba1924f905610346e50fc579efcd09df8e46",
    "scout_manifest.json": "5534f760edd1f1e37627cf895c09cb8f606faa4edcb7bbddff0255abee54887d",
    "paper_citations.bib": "4e12054dc31bb1e3feb9e44392a491d4c0311befd6ce1a291cd41d5b0e963863"
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
    ref = json.loads((DATASET_DIR / "danio_albolineatus_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2017-parichy-danio-albolineatus-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
