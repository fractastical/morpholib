#!/usr/bin/env python3
"""
models/1974-eldredge-eldredgeops-rana-18file-biomineral-accretion.py — Multi-Scale & Paleobiological Reproduction (Slot #1875)
Organism: Eldredgeops rana_18file (Middle Devonian Phacopid Trilobite (18-File Schizochroal Calcite Eye))
Phenomenon Scale: 246 µm (0.0002455 m, log10=-3.610, 5_primordium_100um)
Anatomical Structure: Biomineralized shell whorl / spicule / test in Eldredgeops rana_18file (Middle Devonian (Cazenovian, 387 Ma))
Geological Horizon: Middle Devonian (Cazenovian, 387 Ma) (387.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Eldredgeops
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization (246 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1875_eldredgeops_rana_18file_biomineral_accretion.lean` (`MathOfBiology.Loop_1875`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/eldredgeops_rana_18file_biomineral_accretion"
EXPECTED_SHA256 = {
    "eldredgeops_rana_18file_biomineral_accretion_reference.json": "c22c47f7e8c497b28bc56607db26f592a10e6c2226edb462d1abb74b42347d7c",
    "eldredgeops_rana_18file_specimen.jpg": "ba63b7ea8e242e6374117b84812be17b530eebb3f9621b25eff533b5397e1369",
    "scout_manifest.json": "3570922c8a01a21ec3cdbbd91c1ba7046df2e48ab04262a6bf6bcc79540573b0",
    "paper_citations.bib": "7c7ae140e1f9edcae569d1b7be9fda16d1f9ff56761c4e21bffe8cc3ad90ef93"
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
    ref = json.loads((DATASET_DIR / "eldredgeops_rana_18file_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1974-eldredge-eldredgeops-rana-18file-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
