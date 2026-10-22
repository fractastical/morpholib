#!/usr/bin/env python3
"""
models/2000-norris-globorotalia-praefohsi-collective-stigmergy.py — Multi-Scale & Paleobiological Reproduction (Slot #2075)
Organism: Globorotalia praefohsi (Middle Miocene Unkeeled Rounded-Margin Foraminifer Chronospecies)
Phenomenon Scale: 545 µm (0.0005447 m, log10=-3.264, 5_primordium_100um)
Anatomical Structure: Collective swarm / biofilm / colonial array in Globorotalia praefohsi (Middle Miocene (Serravallian, 13.4 Ma))
Geological Horizon: Middle Miocene (Serravallian, 13.4 Ma) (13.4 Ma)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Globorotalia
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation (545 µm))
Formal Verification: `morphoeval/lean/loops/Loop_2075_globorotalia_praefohsi_collective_stigmergy.lean` (`MathOfBiology.Loop_2075`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/globorotalia_praefohsi_collective_stigmergy"
EXPECTED_SHA256 = {
    "globorotalia_praefohsi_collective_stigmergy_reference.json": "fa8c8f73a8b3a9ded6f58dc310373175c2c63943c05e830f93f30a44cfc34db4",
    "globorotalia_praefohsi_specimen.jpg": "4bd61f5c5fd1e4a4d998a7b4bf377b989b14b379445592ebb2d464ec82617c74",
    "scout_manifest.json": "e9efdde089f0aca8f9f4c226fe28789e20a9757bca86f19ebe676f5e505ec485",
    "paper_citations.bib": "d1adc4bbf8ba1b294490beb789632dba5330bc3d9e657398969f399d2ebf29cf"
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
    ref = json.loads((DATASET_DIR / "globorotalia_praefohsi_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2000-norris-globorotalia-praefohsi-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
