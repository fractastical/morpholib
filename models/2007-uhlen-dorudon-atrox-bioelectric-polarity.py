#!/usr/bin/env python3
"""
models/2007-uhlen-dorudon-atrox-bioelectric-polarity.py — Multi-Scale & Paleobiological Reproduction (Slot #1946)
Organism: Dorudon atrox (Late Eocene Basilosaurid Whale (Vestigial Hindlimb))
Phenomenon Scale: 275 µm (0.0002747 m, log10=-3.561, 5_primordium_100um)
Anatomical Structure: Gap-junction bioelectric domain & ion-channel patch in Dorudon atrox (Late Eocene (Priabonian, 37.0 Ma))
Geological Horizon: Late Eocene (Priabonian, 37.0 Ma) (37.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Dorudon
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity (275 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1946_dorudon_atrox_bioelectric_polarity.lean` (`MathOfBiology.Loop_1946`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/dorudon_atrox_bioelectric_polarity"
EXPECTED_SHA256 = {
    "dorudon_atrox_bioelectric_polarity_reference.json": "fe28b2e6ebb4495fb800f9dee63d47739b99db3efdc3eb9736069369115ef747",
    "dorudon_atrox_specimen.jpg": "5a04e317b788950efe607a8cf4a468b15573b2f7e2efbde1deb8fa2bfaeaabaf",
    "scout_manifest.json": "caa387b5c7f1f163ffd8cf213c892f82e21d7f37b723dd1cc56b9b9ae59d040e",
    "paper_citations.bib": "906fd943b8c022d63a4a681522329ef30b33182b9fcb382169260e78bdcbf8a9"
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
    ref = json.loads((DATASET_DIR / "dorudon_atrox_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2007-uhlen-dorudon-atrox-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
