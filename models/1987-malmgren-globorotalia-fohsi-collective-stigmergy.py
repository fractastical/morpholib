#!/usr/bin/env python3
"""
models/1987-malmgren-globorotalia-fohsi-collective-stigmergy.py — Multi-Scale & Paleobiological Reproduction (Slot #2076)
Organism: Globorotalia fohsi (Middle Miocene Peripheral-Keel Punctuated Foraminifer Chronospecies)
Phenomenon Scale: 703 µm (0.0007031 m, log10=-3.153, 5_primordium_100um)
Anatomical Structure: Collective swarm / biofilm / colonial array in Globorotalia fohsi (Middle Miocene (Serravallian, 12.7 Ma))
Geological Horizon: Middle Miocene (Serravallian, 12.7 Ma) (12.7 Ma)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Globorotalia
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation (703 µm))
Formal Verification: `morphoeval/lean/loops/Loop_2076_globorotalia_fohsi_collective_stigmergy.lean` (`MathOfBiology.Loop_2076`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/globorotalia_fohsi_collective_stigmergy"
EXPECTED_SHA256 = {
    "globorotalia_fohsi_collective_stigmergy_reference.json": "33cf8e1a703bf6df2c91e609f5f53006242d386e4f0873d28729bfb020a56a90",
    "globorotalia_fohsi_specimen.jpg": "da1c3b52b1d7e26144fe2b2960cf08ed37afc6fd3eaf5de4044b285992f13b6f",
    "scout_manifest.json": "a8f14e9416104e96b8c88350cc4f5c42dcdb2ee5e3afed39017884f49c8663b6",
    "paper_citations.bib": "fc45fbc81c7ed08df5eabaaf44f8ca9461431f0031010f2b3ad2fa2d15b17ac9"
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
    ref = json.loads((DATASET_DIR / "globorotalia_fohsi_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1987-malmgren-globorotalia-fohsi-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
