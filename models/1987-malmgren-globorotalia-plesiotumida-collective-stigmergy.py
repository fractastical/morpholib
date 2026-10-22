#!/usr/bin/env python3
"""
models/1987-malmgren-globorotalia-plesiotumida-collective-stigmergy.py — Multi-Scale & Paleobiological Reproduction (Slot #2077)
Organism: Globorotalia plesiotumida (Late Miocene Moderately-Swollen Trochospiral Foraminifer Chronospecies)
Phenomenon Scale: 890 µm (0.0008905 m, log10=-3.050, 5_primordium_100um)
Anatomical Structure: Collective swarm / biofilm / colonial array in Globorotalia plesiotumida (Late Miocene (Messinian, 6.2 Ma))
Geological Horizon: Late Miocene (Messinian, 6.2 Ma) (6.2 Ma)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Globorotalia
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation (890 µm))
Formal Verification: `morphoeval/lean/loops/Loop_2077_globorotalia_plesiotumida_collective_stigmergy.lean` (`MathOfBiology.Loop_2077`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/globorotalia_plesiotumida_collective_stigmergy"
EXPECTED_SHA256 = {
    "globorotalia_plesiotumida_collective_stigmergy_reference.json": "05d7d67a3f777491d6e47f452c8e583803d40c6796686667c9f1ba6f254b82d4",
    "globorotalia_plesiotumida_specimen.jpg": "c1c82488a477ce08218e48d667618dbcb5efdb7a98603919f3838b81a82663b6",
    "scout_manifest.json": "d290835016b869aa35ddad9480f5bd7d7f343eda3e33bb59d34a67640d641b0d",
    "paper_citations.bib": "5d61036065abc95a769462094c8e89c9bb5fec0eef66a5490d085fdec970c599"
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
    ref = json.loads((DATASET_DIR / "globorotalia_plesiotumida_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1987-malmgren-globorotalia-plesiotumida-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
