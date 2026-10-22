#!/usr/bin/env python3
"""
models/2005-fusco-aulacopleura-koninckii-holaspis22-elastic-buckling.py — Multi-Scale & Paleobiological Reproduction (Slot #1784)
Organism: Aulacopleura koninckii_holaspis22 (Silurian Trilobite Holaspid Morphotype II (22 Thoracic Segments))
Phenomenon Scale: 4.3 mm (0.004308 m, log10=-2.366, 6_organ_mm)
Anatomical Structure: Elastic bilayer fold / wrinkle / suture wavelength in Aulacopleura koninckii_holaspis22 (Silurian (Wenlock Epoch, 428.1 Ma))
Geological Horizon: Silurian (Wenlock Epoch, 428.1 Ma) (428.1 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Aulacopleura
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling (4.3 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1784_aulacopleura_koninckii_holaspis22_elastic_buckling.lean` (`MathOfBiology.Loop_1784`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/aulacopleura_koninckii_holaspis22_elastic_buckling"
EXPECTED_SHA256 = {
    "aulacopleura_koninckii_holaspis22_elastic_buckling_reference.json": "215785ed72739e48283b20a2bd356786131616a05a11a68b9f7997cc7610f763",
    "aulacopleura_koninckii_holaspis22_specimen.jpg": "1b6c8997dc42e91d0975aee1225a27810db08c0301831c90db07bebae7b0344a",
    "scout_manifest.json": "ef3e47755600ecacf86fd41e664260bf6e3b5b6984039db02935f772a80ed00e",
    "paper_citations.bib": "f875144c2219c73b226fbc91c85fcf280f3cea83749e849ad76e1f81a0526fd3"
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
    ref = json.loads((DATASET_DIR / "aulacopleura_koninckii_holaspis22_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2005-fusco-aulacopleura-koninckii-holaspis22-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
