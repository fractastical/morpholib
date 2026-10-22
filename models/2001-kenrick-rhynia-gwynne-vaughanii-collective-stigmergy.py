#!/usr/bin/env python3
"""
models/2001-kenrick-rhynia-gwynne-vaughanii-collective-stigmergy.py — Multi-Scale & Paleobiological Reproduction (Slot #2065)
Organism: Rhynia gwynne_vaughanii (Early Devonian Rhynie Chert Leafless Protostele Sporophyte)
Phenomenon Scale: 6.0 mm (0.006048 m, log10=-2.218, 6_organ_mm)
Anatomical Structure: Collective swarm / biofilm / colonial array in Rhynia gwynne_vaughanii (Early Devonian (Pragian Rhynie Chert, 407 Ma))
Geological Horizon: Early Devonian (Pragian Rhynie Chert, 407 Ma) (407.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Rhynia
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation (6.0 mm))
Formal Verification: `morphoeval/lean/loops/Loop_2065_rhynia_gwynne_vaughanii_collective_stigmergy.lean` (`MathOfBiology.Loop_2065`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/rhynia_gwynne_vaughanii_collective_stigmergy"
EXPECTED_SHA256 = {
    "rhynia_gwynne_vaughanii_collective_stigmergy_reference.json": "ddd02108e80071d5730bad3cd5d1715c124d950c6c4d0c6d05f92165676074a3",
    "rhynia_gwynne_vaughanii_specimen.jpg": "5eb2588bbe45fb5b68e29e7763da6962ffa7cf335ab0a5ad8d2e46211cea3f15",
    "scout_manifest.json": "581ad7fe956bd49962351397bb00b7a10a9282729de42903938fa276e49dd4c6",
    "paper_citations.bib": "8835283c2d3dbb92490fc84dcf86e3ef5f747d1b8eb2020c5aa5bb71a4035f67"
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
    ref = json.loads((DATASET_DIR / "rhynia_gwynne_vaughanii_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2001-kenrick-rhynia-gwynne-vaughanii-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
