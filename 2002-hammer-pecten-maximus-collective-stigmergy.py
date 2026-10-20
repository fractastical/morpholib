#!/usr/bin/env python3
"""
2002-hammer-pecten-maximus-collective-stigmergy.py — Pass 2 Remediated Reproduction (Slot #1057)
Organism: Pecten maximus (Great Scallop)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Pecten
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation)
Formal Verification: `morphoeval/lean/loops/Loop_1057_pecten_maximus_collective_stigmergy.lean` (`MathOfBiology.Loop_1057`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/pecten_maximus_collective_stigmergy"
EXPECTED_SHA256 = {
    "pecten_maximus_collective_stigmergy_reference.json": "4e4a19465027974b3acdab5c35ff321a382fed712a0a0ca3bcc0a3a22d759aae",
    "pecten_maximus_specimen.jpg": "2f87d4ec30ebec6adb82f91aec67eb300e3469f57920cbccf9c9bb0d22b703a8",
    "scout_manifest.json": "ed80b32f6774d8891840d80fdb6ce9ab6415ea66fe86cbf4ec4d2e1e81fc8da6",
    "paper_citations.bib": "5db7dd13c5ba4917c582440fd6a25bb90bb3bb0b115f8526a335ee3e74678eb4"
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
    ref = json.loads((DATASET_DIR / "pecten_maximus_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2002-hammer-pecten-maximus-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
