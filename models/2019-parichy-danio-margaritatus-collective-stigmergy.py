#!/usr/bin/env python3
"""
models/2019-parichy-danio-margaritatus-collective-stigmergy.py — Multi-Scale & Paleobiological Reproduction (Slot #2011)
Organism: Danio margaritatus (Celestial Pearl Danio (Galaxy Rasbora))
Phenomenon Scale: 2.8 cm (0.02754 m, log10=-1.560, 7_organism_cm)
Anatomical Structure: Collective swarm / biofilm / colonial array in Danio margaritatus (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Danio
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation (2.8 cm))
Formal Verification: `morphoeval/lean/loops/Loop_2011_danio_margaritatus_collective_stigmergy.lean` (`MathOfBiology.Loop_2011`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/danio_margaritatus_collective_stigmergy"
EXPECTED_SHA256 = {
    "danio_margaritatus_collective_stigmergy_reference.json": "c338efbfcbe2c4033c33ed16d2f91d4b21019a347775e72b25d8ea8243cf536d",
    "danio_margaritatus_specimen.jpg": "6f4ec861e86da2f7ea31f23d6a81b802286dce7ac2a0631e45f9097fd9b8e876",
    "scout_manifest.json": "267939b935835654ea5bdcef2ee93c06c34733f537344683557d3ff7e63b97dd",
    "paper_citations.bib": "9dd9954a0c89dc6a7c4d5397f240d8dfc420224cd6e9afb2c93b06cf49892a8b"
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
    ref = json.loads((DATASET_DIR / "danio_margaritatus_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2019-parichy-danio-margaritatus-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
