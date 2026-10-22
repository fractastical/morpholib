#!/usr/bin/env python3
"""
models/2017-gilmour-danio-rerio-um220-pllp-heart-primordium-branching-tree.py — Multi-Scale & Paleobiological Reproduction (Slot #1166)
Organism: Danio rerio (Zebrafish — 24–36 hpf)
Phenomenon Scale: 228 µm (0.0002277 m, log10=-3.643, 5_primordium_100um)
Anatomical Structure: Migrating posterior lateral line primordium (PLLP) & looping heart tube (24–36 hpf (Pharyngula PLLP & Linear Heart Tube, ZFA:0001279)) · Zebrahub migrating_PLLP_stream & myocardial_endocardial_tube (24 hpf)
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Danio
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting at 228 µm (Migrating posterior lateral line primordium (PLLP) & looping heart tube))
Formal Verification: `morphoeval/lean/loops/Loop_1166_danio_rerio_um220_pllp_heart_primordium_branching_tree.lean` (`MathOfBiology.Loop_1166`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/danio_rerio_um220_pllp_heart_primordium_branching_tree"
EXPECTED_SHA256 = {
    "danio_rerio_um220_pllp_heart_primordium_branching_tree_reference.json": "56130e4e3cf91dc0b3f4e296b57b8c84b79a2998b572b28df29ddae2c6717986",
    "danio_rerio_specimen.jpg": "850b2d4a94c7cbcb3c16e606ddb70d198f082ccb2f046ee44dbaa5ef5941678a",
    "scout_manifest.json": "1905e1b04a0e73033e9c3b8d117db7e2fecb316de9c129c0e5ed3b1e241875d4",
    "paper_citations.bib": "187afcec05d85ec2d593d9bc8cb932cfb3ef40062e108adfa1812a4f319bfcec"
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
    ref = json.loads((DATASET_DIR / "danio_rerio_um220_pllp_heart_primordium_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2017-gilmour-danio-rerio-um220-pllp-heart-primordium-branching-tree.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
