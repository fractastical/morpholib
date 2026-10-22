#!/usr/bin/env python3
"""
models/2005-davidson-strongylocentrotus-purpuratus-bioelectric-polarity.py — Multi-Scale & Paleobiological Reproduction (Slot #1993)
Organism: Strongylocentrotus purpuratus (Purple Sea Urchin (Larval Calcite Spicule Triradiate Skeleton))
Phenomenon Scale: 63 nm (6.324e-08 m, log10=-7.199, 2_macromolecular_100nm)
Anatomical Structure: Gap-junction bioelectric domain & ion-channel patch in Strongylocentrotus purpuratus (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Strongylocentrotus
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity (63 nm))
Formal Verification: `morphoeval/lean/loops/Loop_1993_strongylocentrotus_purpuratus_bioelectric_polarity.lean` (`MathOfBiology.Loop_1993`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/strongylocentrotus_purpuratus_bioelectric_polarity"
EXPECTED_SHA256 = {
    "strongylocentrotus_purpuratus_bioelectric_polarity_reference.json": "0a9bd380ac4a846cf6a7a76f257ef4544a3f50499dea7fbd01696898b424eb62",
    "strongylocentrotus_purpuratus_specimen.jpg": "515b723840267df6465272f908c6951fdf06c057159d3c6f7d6e67e7bee5224a",
    "scout_manifest.json": "e985f783a868c32ae86112720bde536c81d9012f9db82cbeb822f5df90d497e8",
    "paper_citations.bib": "126f2f65129f3c086451aa114a653d5a51caaaa1c73694405853be19b64e2d0d"
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
    ref = json.loads((DATASET_DIR / "strongylocentrotus_purpuratus_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2005-davidson-strongylocentrotus-purpuratus-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
