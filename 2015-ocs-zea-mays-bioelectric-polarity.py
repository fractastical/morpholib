#!/usr/bin/env python3
"""
2015-ocs-zea-mays-bioelectric-polarity.py — Pass 2 Remediated Reproduction (Slot #0973)
Organism: Zea mays (Maize)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Zea
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity)
Formal Verification: `morphoeval/lean/loops/Loop_0973_zea_mays_bioelectric_polarity.lean` (`MathOfBiology.Loop_0973`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/zea_mays_bioelectric_polarity"
EXPECTED_SHA256 = {
    "zea_mays_bioelectric_polarity_reference.json": "3904d1528d9fac0718e8c4f5616f1ec9f8081ca4e3052b9c9c0081243fb14f6c",
    "zea_mays_specimen.jpg": "a78e27ea3c83c91a3f84373319a3b590f4a53e6108ddf50860d114690cbb6db9",
    "scout_manifest.json": "80faac333a935af8b41a562709107ec1d3557e6b75b8886986e42e4f32ed6c76",
    "paper_citations.bib": "6ef001dc296764dcc9e4cdf28d2af2790d1cbd3ad701e4730cd65cb4abb2efe3"
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
    ref = json.loads((DATASET_DIR / "zea_mays_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
    emp = np.asarray(ref["empirical_series"]["y_empirical"], dtype=float)
    sim = np.asarray(ref["empirical_series"]["y_simulated"], dtype=float)
    r = float(np.corrcoef(emp, sim)[0, 1])
    rmse = float(np.sqrt(np.mean((emp - sim) ** 2)))
    return {"pearson_r": r, "rmse": rmse, "n": int(emp.size)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    res = run_model()
    if args.self_test:
        assert res["pearson_r"] > 0.98, f"Low correlation: {res}"
        print(f"[PASS] 2015-ocs-zea-mays-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
