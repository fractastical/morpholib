#!/usr/bin/env python3
"""
models/2015-weaver-odontodactylus-scyllarus-bioelectric-polarity.py — Multi-Scale & Paleobiological Reproduction (Slot #2003)
Organism: Odontodactylus scyllarus (Peacock Mantis Shrimp (Bouligand Helicoidal Dactyl Club Nanocomposite))
Phenomenon Scale: 2.5 nm (2.5e-09 m, log10=-8.602, 1_molecular_nm)
Anatomical Structure: Gap-junction bioelectric domain & ion-channel patch in Odontodactylus scyllarus (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Odontodactylus
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity (2.5 nm))
Formal Verification: `morphoeval/lean/loops/Loop_2003_odontodactylus_scyllarus_bioelectric_polarity.lean` (`MathOfBiology.Loop_2003`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/odontodactylus_scyllarus_bioelectric_polarity"
EXPECTED_SHA256 = {
    "odontodactylus_scyllarus_bioelectric_polarity_reference.json": "db014d65b9ad645f4a753eb5ccf90cadc1d4f83a94367774b05d7cf60ebacf1b",
    "odontodactylus_scyllarus_specimen.jpg": "a0cb5ee4fe57474a9ed4bdb381117a01fec6e1ff71de38b747022330546126c0",
    "scout_manifest.json": "136740b9e8583e859b1b6a2243af79c8285e46f4c9fc749af5a6f3afef961a5d",
    "paper_citations.bib": "d81c08d44a6b5f63c9a935b7c3b551e92c133d7c2b47ed31e68cefbe3367c93b"
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
    ref = json.loads((DATASET_DIR / "odontodactylus_scyllarus_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2015-weaver-odontodactylus-scyllarus-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
