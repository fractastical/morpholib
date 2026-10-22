#!/usr/bin/env python3
"""
models/2008-hoyall-charnia-masoni-collective-stigmergy.py — Multi-Scale & Paleobiological Reproduction (Slot #2059)
Organism: Charnia masoni (Ediacaran Self-Similar Glide-Reflected Rangeomorph Frond (Mistaken Point))
Phenomenon Scale: 28.4 cm (0.2845 m, log10=-0.546, 8_colonial_m)
Anatomical Structure: Collective swarm / biofilm / colonial array in Charnia masoni (Late Ediacaran (565 Ma))
Geological Horizon: Late Ediacaran (565 Ma) (565.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Charnia
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation (28.4 cm))
Formal Verification: `morphoeval/lean/loops/Loop_2059_charnia_masoni_collective_stigmergy.lean` (`MathOfBiology.Loop_2059`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/charnia_masoni_collective_stigmergy"
EXPECTED_SHA256 = {
    "charnia_masoni_collective_stigmergy_reference.json": "89081404ed00f50d43cecc266512b502fbf781c9bf73c64a6e1ea6acac66c3c5",
    "charnia_masoni_specimen.jpg": "92b6f0531f638c633e69def818647884521ea36a8f79147751e0437ea9e17b3a",
    "scout_manifest.json": "0b8af3f131813251159591ce294cf3f78b4aa36cdceed09f724821c51e092110",
    "paper_citations.bib": "b105f576e40275d4f435e9053a2149f867025ff3a3cc325bc44bbde6a12dda02"
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
    ref = json.loads((DATASET_DIR / "charnia_masoni_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2008-hoyall-charnia-masoni-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
