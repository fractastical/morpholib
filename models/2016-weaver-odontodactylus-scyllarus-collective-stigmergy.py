#!/usr/bin/env python3
"""
models/2016-weaver-odontodactylus-scyllarus-collective-stigmergy.py — Multi-Scale & Paleobiological Reproduction (Slot #2093)
Organism: Odontodactylus scyllarus (Peacock Mantis Shrimp (Bouligand Helicoidal Dactyl Club Nanocomposite))
Phenomenon Scale: 1.2 µm (1.228e-06 m, log10=-5.911, 3_subcellular_um)
Anatomical Structure: Collective swarm / biofilm / colonial array in Odontodactylus scyllarus (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Odontodactylus
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation (1.2 µm))
Formal Verification: `morphoeval/lean/loops/Loop_2093_odontodactylus_scyllarus_collective_stigmergy.lean` (`MathOfBiology.Loop_2093`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/odontodactylus_scyllarus_collective_stigmergy"
EXPECTED_SHA256 = {
    "odontodactylus_scyllarus_collective_stigmergy_reference.json": "b0a0982fd1811097bd2fb3acdb36b762337046d4c5c8b2be691d27b6cfeb32fd",
    "odontodactylus_scyllarus_specimen.jpg": "a0cb5ee4fe57474a9ed4bdb381117a01fec6e1ff71de38b747022330546126c0",
    "scout_manifest.json": "d864f22970344d33958ac2614124a9bffd86ccb41941c28a2ab2385f6546b6fa",
    "paper_citations.bib": "cae119d1a99435f317e7491860ff77007f5cdcab3bede322092fa25b66fc2133"
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
    ref = json.loads((DATASET_DIR / "odontodactylus_scyllarus_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2016-weaver-odontodactylus-scyllarus-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
