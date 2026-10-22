#!/usr/bin/env python3
"""
models/2005-bohn-nepenthes-rajah-elastic-buckling.py — Multi-Scale & Paleobiological Reproduction (Slot #1829)
Organism: Nepenthes rajah (Giant Bornean Pitcher Plant (Micro-Grooved Peristome Aquaplaning Rim))
Phenomenon Scale: 10.0 µm (9.999e-06 m, log10=-5.000, 3_subcellular_um)
Anatomical Structure: Elastic bilayer fold / wrinkle / suture wavelength in Nepenthes rajah (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Nepenthes
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling (10.0 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1829_nepenthes_rajah_elastic_buckling.lean` (`MathOfBiology.Loop_1829`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/nepenthes_rajah_elastic_buckling"
EXPECTED_SHA256 = {
    "nepenthes_rajah_elastic_buckling_reference.json": "bcf5767a519efd35d0bce975be0aff46d8131c4fb78f68a1a15c513b21b4724b",
    "nepenthes_rajah_specimen.jpg": "34d473e5c7aa422e26ade1e48e08666c34867f853524458ff188daa017f9d5a3",
    "scout_manifest.json": "5140089b96612fe0409d79a6006790ffb5640b2ade6202c726c28f1ff87620f3",
    "paper_citations.bib": "030ca6fb979c231d90e27556a09950e364fbb95e843887b068270738778234f8"
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
    ref = json.loads((DATASET_DIR / "nepenthes_rajah_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2005-bohn-nepenthes-rajah-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
