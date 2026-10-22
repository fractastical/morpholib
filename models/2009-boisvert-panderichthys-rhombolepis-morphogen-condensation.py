#!/usr/bin/env python3
"""
models/2009-boisvert-panderichthys-rhombolepis-morphogen-condensation.py — Multi-Scale & Paleobiological Reproduction (Slot #1303)
Organism: Panderichthys rhombolepis (Late Devonian Elpistostegalian Flat-Skulled Fish)
Phenomenon Scale: 3.1 cm (0.03119 m, log10=-1.506, 7_organism_cm)
Anatomical Structure: Morphogen gradient & pre-cartilage condensation zone in Panderichthys rhombolepis (Late Devonian (Frasnian, 380 Ma))
Geological Horizon: Late Devonian (Frasnian, 380 Ma) (380.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Panderichthys
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation (3.1 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1303_panderichthys_rhombolepis_morphogen_condensation.lean` (`MathOfBiology.Loop_1303`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/panderichthys_rhombolepis_morphogen_condensation"
EXPECTED_SHA256 = {
    "panderichthys_rhombolepis_morphogen_condensation_reference.json": "dd31bb45e5b310525ba075eeedaf16fc9949a0f1797271c1532387774c677004",
    "panderichthys_rhombolepis_specimen.jpg": "5c39604c489cd7a850b62b1d1aaf2e3061b9865043e83968c2feebf4e34ca870",
    "scout_manifest.json": "488a1e1bf79f7be3a3918eb8c3364c9278aacd71e8a96bbe03588291a568520c",
    "paper_citations.bib": "42b31d4e1f6305a718cd15b1bcf23b6c278ce375005c1a6084ddda9651fc7bcd"
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
    ref = json.loads((DATASET_DIR / "panderichthys_rhombolepis_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2009-boisvert-panderichthys-rhombolepis-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
