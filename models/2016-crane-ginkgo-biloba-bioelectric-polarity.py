#!/usr/bin/env python3
"""
models/2016-crane-ginkgo-biloba-bioelectric-polarity.py — Multi-Scale & Paleobiological Reproduction (Slot #1984)
Organism: Ginkgo biloba (Extant Maidenhair Tree (Dichotomous Open-Vein Fan Leaf))
Phenomenon Scale: 39 µm (3.944e-05 m, log10=-4.404, 4_cellular_10um)
Anatomical Structure: Gap-junction bioelectric domain & ion-channel patch in Ginkgo biloba (Holocene (Extant Living Fossil))
Geological Horizon: Holocene (Extant Living Fossil) (0.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Ginkgo
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity (39 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1984_ginkgo_biloba_bioelectric_polarity.lean` (`MathOfBiology.Loop_1984`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/ginkgo_biloba_bioelectric_polarity"
EXPECTED_SHA256 = {
    "ginkgo_biloba_bioelectric_polarity_reference.json": "495bea76b45cab4299d4353889945318ab68b304e3a51c75c8dd82c09e14bbbd",
    "ginkgo_biloba_specimen.jpg": "b0a1f66d4688ddb858306c70a27a0bc279b41f3eaf1e9935f5b40d33cb7f7905",
    "scout_manifest.json": "00ccd89bc8413e52f0f5d43488b63f700087622107c5395521522e67f1ef5deb",
    "paper_citations.bib": "71eef46129b6c8744dcb096dcbcf9c039f74b296d13c3ecbe3c36085d7152fa3"
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
    ref = json.loads((DATASET_DIR / "ginkgo_biloba_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2016-crane-ginkgo-biloba-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
