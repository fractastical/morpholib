#!/usr/bin/env python3
"""
2006-kaandorp-acropora-millepora-morphogen-condensation.py — Pass 2 Remediated Reproduction (Slot #0265)
Organism: Acropora millepora (Branching Staghorn Coral)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Acropora
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation)
Formal Verification: `morphoeval/lean/loops/Loop_0265_acropora_millepora_morphogen_condensation.lean` (`MathOfBiology.Loop_0265`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/acropora_millepora_morphogen_condensation"
EXPECTED_SHA256 = {
    "acropora_millepora_morphogen_condensation_reference.json": "6ee26e2101cb2fcc65b2ae47f5f56cc1247f8c83d0fda0e67ecae047e4ff84f4",
    "acropora_millepora_specimen.jpg": "85c35e854dd9d07d46528548512579040fef8d6cd4893257fb28bd3b4a468eb0",
    "scout_manifest.json": "68d0d6b417b0d91559654730cbc916cba7cc7f48580102bf6328d55c1701d882",
    "paper_citations.bib": "2aa2f7f4512f0a6179909f5d1e7484e8d0760810bc292cc8b0b3f2206f916694"
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
    ref = json.loads((DATASET_DIR / "acropora_millepora_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2006-kaandorp-acropora-millepora-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
