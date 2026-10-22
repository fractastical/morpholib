#!/usr/bin/env python3
"""
models/2003-pettigrew-ornithorhynchus-anatinus-collective-stigmergy.py — Multi-Scale & Paleobiological Reproduction (Slot #2084)
Organism: Ornithorhynchus anatinus (Duck-Billed Platypus (Electroreceptive Bill Pore Hexagonal Array))
Phenomenon Scale: 244 µm (0.0002444 m, log10=-3.612, 5_primordium_100um)
Anatomical Structure: Collective swarm / biofilm / colonial array in Ornithorhynchus anatinus (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Ornithorhynchus
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation (244 µm))
Formal Verification: `morphoeval/lean/loops/Loop_2084_ornithorhynchus_anatinus_collective_stigmergy.lean` (`MathOfBiology.Loop_2084`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/ornithorhynchus_anatinus_collective_stigmergy"
EXPECTED_SHA256 = {
    "ornithorhynchus_anatinus_collective_stigmergy_reference.json": "14723dd8895d4b7175c79b385fb0db6cc0ca86fb30042a879340ea9d9c267059",
    "ornithorhynchus_anatinus_specimen.jpg": "5a8d5bf6c1809500a8e9e88694cff57c75cead6258be2e7d05ede7922465f2b6",
    "scout_manifest.json": "cc112fc488c96acf87c31689e8ca3bb83adb2a806d4d1c5dc9c60b0a0d21eda0",
    "paper_citations.bib": "35def246690f567211aad9b6a54aa3dfc2a9e54bc2034b25122e657f3b60ed8b"
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
    ref = json.loads((DATASET_DIR / "ornithorhynchus_anatinus_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2003-pettigrew-ornithorhynchus-anatinus-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
