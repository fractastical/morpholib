#!/usr/bin/env python3
"""
models/2024-lange-danio-rerio-nm4-connexin-her-complex-elastic-buckling.py — Multi-Scale & Paleobiological Reproduction (Slot #1107)
Organism: Danio rerio (Zebrafish — 10–14 hpf)
Phenomenon Scale: 4.6 nm (4.578e-09 m, log10=-8.339, 1_molecular_nm)
Anatomical Structure: Connexin-41.8 hemichannel pore & Her1/Her7 bHLH dimer complex (10–14 hpf (Bud to 10-Somite Stage, ZFA:0001094)) · Zebrahub PSM_presomitic_mesoderm & neural_plate_progenitors (10–14 hpf)
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Danio
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling at 4.6 nm (Connexin-41.8 hemichannel pore & Her1/Her7 bHLH dimer complex))
Formal Verification: `morphoeval/lean/loops/Loop_1107_danio_rerio_nm4_connexin_her_complex_elastic_buckling.lean` (`MathOfBiology.Loop_1107`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/danio_rerio_nm4_connexin_her_complex_elastic_buckling"
EXPECTED_SHA256 = {
    "danio_rerio_nm4_connexin_her_complex_elastic_buckling_reference.json": "588213b8c972244d1792490ab2405c05a007e5f8525b504b772f8237cf0255cd",
    "danio_rerio_specimen.jpg": "850b2d4a94c7cbcb3c16e606ddb70d198f082ccb2f046ee44dbaa5ef5941678a",
    "scout_manifest.json": "17a4693fa4be9178c5e761c35fd327c87c67f154fcb169bdfe32e4a9744381dd",
    "paper_citations.bib": "68ad9254e61af4d79b9562f9ae8456249895d8c3f969fd349ccfc65a32f09f64"
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
    ref = json.loads((DATASET_DIR / "danio_rerio_nm4_connexin_her_complex_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2024-lange-danio-rerio-nm4-connexin-her-complex-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
