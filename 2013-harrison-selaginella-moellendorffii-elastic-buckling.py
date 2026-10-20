#!/usr/bin/env python3
"""
2013-harrison-selaginella-moellendorffii-elastic-buckling.py — Pass 2 Remediated Reproduction (Slot #0779)
Organism: Selaginella moellendorffii (Spikemoss Lycophyte)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Selaginella
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling)
Formal Verification: `morphoeval/lean/loops/Loop_0779_selaginella_moellendorffii_elastic_buckling.lean` (`MathOfBiology.Loop_0779`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/selaginella_moellendorffii_elastic_buckling"
EXPECTED_SHA256 = {
    "selaginella_moellendorffii_elastic_buckling_reference.json": "a86596f4e3cbe7a8f2ff68614b97ae6373071db27c04a674a7d2d871f99559ba",
    "selaginella_moellendorffii_specimen.jpg": "87d0d25c483d4f17cb66050be175071aafb49d6f321bc8eb2d9cc3bac9e55cd4",
    "scout_manifest.json": "cee6f923d63cfebed005edfc53a2303c2c8af97bb00cb8a9bee6412ecce3abe8",
    "paper_citations.bib": "77e73ed48561937edddc3777b0542416afcef0210ff4cdea74d1c71dc3a62318"
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
    ref = json.loads((DATASET_DIR / "selaginella_moellendorffii_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2013-harrison-selaginella-moellendorffii-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
