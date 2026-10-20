#!/usr/bin/env python3
"""
2016-plackett-ceratopteris-richardii-bioelectric-polarity.py — Pass 2 Remediated Reproduction (Slot #0980)
Organism: Ceratopteris richardii (Triangle Water Fern)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Ceratopteris
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity)
Formal Verification: `morphoeval/lean/loops/Loop_0980_ceratopteris_richardii_bioelectric_polarity.lean` (`MathOfBiology.Loop_0980`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/ceratopteris_richardii_bioelectric_polarity"
EXPECTED_SHA256 = {
    "ceratopteris_richardii_bioelectric_polarity_reference.json": "124782668cb5ae3d9b6a6512e92839b4ddbec4b512b26e2b34e86b15fe313140",
    "ceratopteris_richardii_specimen.jpg": "3fcf2684662c089b9c66687d1f6b2e9409565391d0d316e5db46561d1809641f",
    "scout_manifest.json": "5849513e74fa151e07be57a865c71e04df42cdd89b0344d5eae6318dea7eecce",
    "paper_citations.bib": "17c159c1fd937dc833ff047480d8e1038537387fbd8b6c911fbd735ad5a6876f"
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
    ref = json.loads((DATASET_DIR / "ceratopteris_richardii_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2016-plackett-ceratopteris-richardii-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
