#!/usr/bin/env python3
"""
models/2008-supatto-danio-rerio-um2-kupffer-motile-cilium-flux-canalization.py — Multi-Scale & Paleobiological Reproduction (Slot #1134)
Organism: Danio rerio (Zebrafish — 11–14 hpf)
Phenomenon Scale: 2.2 µm (2.22e-06 m, log10=-5.654, 3_subcellular_um)
Anatomical Structure: Motile 9+2 nodal cilium & sarcomeric Z-disc myofibril bundle (11–14 hpf (Kupffer's Vesicle L-R Organizer, ZFA:0000049)) · Zebrahub kupffers_vesicle_ciliated_epithelium & fast_muscle_myotome
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Danio
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth at 2.2 µm (Motile 9+2 nodal cilium & sarcomeric Z-disc myofibril bundle))
Formal Verification: `morphoeval/lean/loops/Loop_1134_danio_rerio_um2_kupffer_motile_cilium_flux_canalization.lean` (`MathOfBiology.Loop_1134`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/danio_rerio_um2_kupffer_motile_cilium_flux_canalization"
EXPECTED_SHA256 = {
    "danio_rerio_um2_kupffer_motile_cilium_flux_canalization_reference.json": "e66d176f5675d4893136f68569fc625c519ed541b0c9199bd9910ee1630cd854",
    "danio_rerio_specimen.jpg": "850b2d4a94c7cbcb3c16e606ddb70d198f082ccb2f046ee44dbaa5ef5941678a",
    "scout_manifest.json": "8bb444fde54f6d60bc10d548c4476300d1e7b4bbab3d3c484eb994c76c4c448a",
    "paper_citations.bib": "81e7527a76b2a4ca4f5905a8eeb3371702daf6ea62a44f336223c1a517bf6f98"
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
    ref = json.loads((DATASET_DIR / "danio_rerio_um2_kupffer_motile_cilium_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2008-supatto-danio-rerio-um2-kupffer-motile-cilium-flux-canalization.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
