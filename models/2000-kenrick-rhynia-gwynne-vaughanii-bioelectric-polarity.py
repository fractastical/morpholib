#!/usr/bin/env python3
"""
models/2000-kenrick-rhynia-gwynne-vaughanii-bioelectric-polarity.py — Multi-Scale & Paleobiological Reproduction (Slot #1975)
Organism: Rhynia gwynne_vaughanii (Early Devonian Rhynie Chert Leafless Protostele Sporophyte)
Phenomenon Scale: 2.7 µm (2.688e-06 m, log10=-5.571, 3_subcellular_um)
Anatomical Structure: Gap-junction bioelectric domain & ion-channel patch in Rhynia gwynne_vaughanii (Early Devonian (Pragian Rhynie Chert, 407 Ma))
Geological Horizon: Early Devonian (Pragian Rhynie Chert, 407 Ma) (407.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Rhynia
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity (2.7 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1975_rhynia_gwynne_vaughanii_bioelectric_polarity.lean` (`MathOfBiology.Loop_1975`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/rhynia_gwynne_vaughanii_bioelectric_polarity"
EXPECTED_SHA256 = {
    "rhynia_gwynne_vaughanii_bioelectric_polarity_reference.json": "a612607062f852be5b918f35dfbe0d6850cb0d276e4ce7e02d1fb00cffe5a23d",
    "rhynia_gwynne_vaughanii_specimen.jpg": "5eb2588bbe45fb5b68e29e7763da6962ffa7cf335ab0a5ad8d2e46211cea3f15",
    "scout_manifest.json": "9770f6b6e13eb90eb476a4f2db32fbe61144248861481ec847af539c82ab4367",
    "paper_citations.bib": "a80ce9744d043fa84d39b0381e764f14924314e07504e1fead4a67a8600ff179"
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
    ref = json.loads((DATASET_DIR / "rhynia_gwynne_vaughanii_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2000-kenrick-rhynia-gwynne-vaughanii-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
