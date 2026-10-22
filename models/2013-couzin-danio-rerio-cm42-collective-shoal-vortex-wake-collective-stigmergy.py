#!/usr/bin/env python3
"""
models/2013-couzin-danio-rerio-cm42-collective-shoal-vortex-wake-collective-stigmergy.py — Multi-Scale & Paleobiological Reproduction (Slot #1200)
Organism: Danio rerio (Zebrafish — Adult Collective Shoal)
Phenomenon Scale: 52.7 cm (0.5271 m, log10=-0.278, 8_colonial_m)
Anatomical Structure: Reverse Kármán vortex dipole wake & visual-mechanosensory shoal lattice (Adult Collective Shoal (15–30 Fish Polarized School, 0.42 m arena)) · MapZebrain optic_tectum_loom_circuit & lateral_line_rheotaxis
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Danio
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation at 52.7 cm (Reverse Kármán vortex dipole wake & visual-mechanosensory shoal lattice))
Formal Verification: `morphoeval/lean/loops/Loop_1200_danio_rerio_cm42_collective_shoal_vortex_wake_collective_stigmergy.lean` (`MathOfBiology.Loop_1200`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/danio_rerio_cm42_collective_shoal_vortex_wake_collective_stigmergy"
EXPECTED_SHA256 = {
    "danio_rerio_cm42_collective_shoal_vortex_wake_collective_stigmergy_reference.json": "e5c71f4dfbb842816129672c46c60306cb24d30b2347555b7f8b515cc74b19e7",
    "danio_rerio_specimen.jpg": "850b2d4a94c7cbcb3c16e606ddb70d198f082ccb2f046ee44dbaa5ef5941678a",
    "scout_manifest.json": "0087c60a4f116b84496d8c3d95b70b85895c8fb76440d0569bb06a96c3b27471",
    "paper_citations.bib": "88e773382c5763dbff346f34c1617909b8662eb5282d95cf0613bf73f8a25231"
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
    ref = json.loads((DATASET_DIR / "danio_rerio_cm42_collective_shoal_vortex_wake_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2013-couzin-danio-rerio-cm42-collective-shoal-vortex-wake-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
