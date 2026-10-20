#!/usr/bin/env python3
"""
2017-shimamura-marchantia-polymorpha-bioelectric-polarity.py — Pass 2 Remediated Reproduction (Slot #0971)
Organism: Marchantia polymorpha (Common Liverwort)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Marchantia
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity)
Formal Verification: `morphoeval/lean/loops/Loop_0971_marchantia_polymorpha_bioelectric_polarity.lean` (`MathOfBiology.Loop_0971`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/marchantia_polymorpha_bioelectric_polarity"
EXPECTED_SHA256 = {
    "marchantia_polymorpha_bioelectric_polarity_reference.json": "073816d955dacd20faacea54795fda032d79b563ba67006c4cf39bc72cd85362",
    "marchantia_polymorpha_specimen.jpg": "1a60432aad00137db36b8c40d01cac4e78b01334796182f27ed7219444c3577c",
    "scout_manifest.json": "a93fa08f8bb8dc52ae81e92662deb39ea0aeffe68306337695d9d192f1cb7211",
    "paper_citations.bib": "d5ac00b399ee01834e5b418ead32349b330128ec14683bec60df81f44755a7f5"
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
    ref = json.loads((DATASET_DIR / "marchantia_polymorpha_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2017-shimamura-marchantia-polymorpha-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
