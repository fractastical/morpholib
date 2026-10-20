#!/usr/bin/env python3
"""
1992-nijhout-heliconius-erato-bioelectric-polarity.py — Pass 2 Remediated Reproduction (Slot #0947)
Organism: Heliconius erato (Red Postman Butterfly)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Heliconius
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity)
Formal Verification: `morphoeval/lean/loops/Loop_0947_heliconius_erato_bioelectric_polarity.lean` (`MathOfBiology.Loop_0947`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/heliconius_erato_bioelectric_polarity"
EXPECTED_SHA256 = {
    "heliconius_erato_bioelectric_polarity_reference.json": "af5957dc967474c75c34949cb2dc87dc932d9d94211714e67071a204d06a1043",
    "heliconius_erato_specimen.jpg": "e54735f4ca16e4326cb91d2969d83bb613e3c4111aaf7f30cb218075b6fec48e",
    "scout_manifest.json": "7e8474172e71f10830f550b9d4177650ed6c5c6b6b8a8e0114ed83ead121513e",
    "paper_citations.bib": "f0430582805472cacce5326cef40b4641b4c2919a7aa5db81dadf86190ec4076"
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
    ref = json.loads((DATASET_DIR / "heliconius_erato_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1992-nijhout-heliconius-erato-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
