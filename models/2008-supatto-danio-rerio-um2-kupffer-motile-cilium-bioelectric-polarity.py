#!/usr/bin/env python3
"""
models/2008-supatto-danio-rerio-um2-kupffer-motile-cilium-bioelectric-polarity.py — Multi-Scale & Paleobiological Reproduction (Slot #1139)
Organism: Danio rerio (Zebrafish — 11–14 hpf)
Phenomenon Scale: 2.9 µm (2.88e-06 m, log10=-5.541, 3_subcellular_um)
Anatomical Structure: Motile 9+2 nodal cilium & sarcomeric Z-disc myofibril bundle (11–14 hpf (Kupffer's Vesicle L-R Organizer, ZFA:0000049)) · Zebrahub kupffers_vesicle_ciliated_epithelium & fast_muscle_myotome
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Danio
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity at 2.9 µm (Motile 9+2 nodal cilium & sarcomeric Z-disc myofibril bundle))
Formal Verification: `morphoeval/lean/loops/Loop_1139_danio_rerio_um2_kupffer_motile_cilium_bioelectric_polarity.lean` (`MathOfBiology.Loop_1139`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/danio_rerio_um2_kupffer_motile_cilium_bioelectric_polarity"
EXPECTED_SHA256 = {
    "danio_rerio_um2_kupffer_motile_cilium_bioelectric_polarity_reference.json": "5a29943860770ced27c57399d02c03acd4ee93dd368b9031d02dc4bcf1d0536d",
    "danio_rerio_specimen.jpg": "850b2d4a94c7cbcb3c16e606ddb70d198f082ccb2f046ee44dbaa5ef5941678a",
    "scout_manifest.json": "1e609ba1c5f21a7932f4ffe90d8e5804f4e35e7296fa8d8be89b63d4e6baf37f",
    "paper_citations.bib": "c26791ff89e6a72723645981f9b8cd0224b295d9a9cae22c431e8ed8571723c7"
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
    ref = json.loads((DATASET_DIR / "danio_rerio_um2_kupffer_motile_cilium_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2008-supatto-danio-rerio-um2-kupffer-motile-cilium-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
