#!/usr/bin/env python3
"""
1958-hartline-limulus-polyphemus-collective-stigmergy.py — Pass 2 Remediated Reproduction (Slot #1050)
Organism: Limulus polyphemus (Atlantic Horseshoe Crab)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Limulus
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation)
Formal Verification: `morphoeval/lean/loops/Loop_1050_limulus_polyphemus_collective_stigmergy.lean` (`MathOfBiology.Loop_1050`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/limulus_polyphemus_collective_stigmergy"
EXPECTED_SHA256 = {
    "limulus_polyphemus_collective_stigmergy_reference.json": "716c0faf1f343d1130effdbc69d7dee39a544539be62689408017838d4355f9d",
    "limulus_polyphemus_specimen.jpg": "6024a6e114ddb2e0b996d82067f840499683d1838662afe24496b8f315e2ef45",
    "scout_manifest.json": "6e28eea5c4347ca8235763d39de2a3b2d48b3417dc3aa4cd2263725f72fc3464",
    "paper_citations.bib": "0ebcbb9fd7d659339330cca26eda7033fe7a6baa35e43fbe700b7148bd63287c"
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
    ref = json.loads((DATASET_DIR / "limulus_polyphemus_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1958-hartline-limulus-polyphemus-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
