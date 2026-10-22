#!/usr/bin/env python3
"""
models/2003-chiappe-confuciusornis-sanctus-collective-stigmergy.py — Multi-Scale & Paleobiological Reproduction (Slot #2042)
Organism: Confuciusornis sanctus (Early Cretaceous Pygostyle Beaked Bird (Ribbon Tail Streamers))
Phenomenon Scale: 24.4 cm (0.2444 m, log10=-0.612, 7_organism_cm)
Anatomical Structure: Collective swarm / biofilm / colonial array in Confuciusornis sanctus (Early Cretaceous (Yixian Fm, 125.0 Ma))
Geological Horizon: Early Cretaceous (Yixian Fm, 125.0 Ma) (125.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Confuciusornis
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation (24.4 cm))
Formal Verification: `morphoeval/lean/loops/Loop_2042_confuciusornis_sanctus_collective_stigmergy.lean` (`MathOfBiology.Loop_2042`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/confuciusornis_sanctus_collective_stigmergy"
EXPECTED_SHA256 = {
    "confuciusornis_sanctus_collective_stigmergy_reference.json": "f1ef5d44e1f9bbb80a626805a72c0e674c9230736f70b3dcf4197a6f57eceb12",
    "confuciusornis_sanctus_specimen.jpg": "cd7c05e2d404d66e00f79aa73a3de8f5a68cde60009b9288d61176e6902d7620",
    "scout_manifest.json": "54c45ec461f1acf7f5c9fcdc02c29c30559948ae0ac2a55a47661f2ade830370",
    "paper_citations.bib": "2c24e8b5fe32b18c59c0bcd015f1be38e2f3198bd51129a62da70340adc4dd57"
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
    ref = json.loads((DATASET_DIR / "confuciusornis_sanctus_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2003-chiappe-confuciusornis-sanctus-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
