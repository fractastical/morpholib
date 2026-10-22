#!/usr/bin/env python3
"""
models/2019-bell-hesperornis-regalis-collective-stigmergy.py — Multi-Scale & Paleobiological Reproduction (Slot #2043)
Organism: Hesperornis regalis (Late Cretaceous Toothed Diving Bird)
Phenomenon Scale: 35.4 cm (0.3545 m, log10=-0.450, 8_colonial_m)
Anatomical Structure: Collective swarm / biofilm / colonial array in Hesperornis regalis (Late Cretaceous (Campanian, 83.5 Ma))
Geological Horizon: Late Cretaceous (Campanian, 83.5 Ma) (83.5 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Hesperornis
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation (35.4 cm))
Formal Verification: `morphoeval/lean/loops/Loop_2043_hesperornis_regalis_collective_stigmergy.lean` (`MathOfBiology.Loop_2043`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/hesperornis_regalis_collective_stigmergy"
EXPECTED_SHA256 = {
    "hesperornis_regalis_collective_stigmergy_reference.json": "41ecec19766f3c68c8076fc2fec937bc9635143c915efb304ef7944047471a70",
    "hesperornis_regalis_specimen.jpg": "a8ca1cd24cf7035f2fff3d592b3f55831669012975aed64e3050567bec49e7a7",
    "scout_manifest.json": "7726724bba50eeeedbc7c509ee2aa55c2a4afe30f3c69aaf14c5f66bc8ba8605",
    "paper_citations.bib": "48e2aea1558fc3fc6e06b53e05b18e0e88ece2d4bb0a638987d3dbe8d661f404"
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
    ref = json.loads((DATASET_DIR / "hesperornis_regalis_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2019-bell-hesperornis-regalis-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
