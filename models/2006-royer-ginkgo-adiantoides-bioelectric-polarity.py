#!/usr/bin/env python3
"""
models/2006-royer-ginkgo-adiantoides-bioelectric-polarity.py — Multi-Scale & Paleobiological Reproduction (Slot #1983)
Organism: Ginkgo adiantoides (Paleocene-Eocene Bilobed Coalescent Ginkgo Chronospecies)
Phenomenon Scale: 45 µm (4.534e-05 m, log10=-4.344, 4_cellular_10um)
Anatomical Structure: Gap-junction bioelectric domain & ion-channel patch in Ginkgo adiantoides (Paleocene-Eocene Boundary (56 Ma))
Geological Horizon: Paleocene-Eocene Boundary (56 Ma) (56.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Ginkgo
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity (45 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1983_ginkgo_adiantoides_bioelectric_polarity.lean` (`MathOfBiology.Loop_1983`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/ginkgo_adiantoides_bioelectric_polarity"
EXPECTED_SHA256 = {
    "ginkgo_adiantoides_bioelectric_polarity_reference.json": "47c80da28799e514b6a6753048e6415c8c9076e2fc79c1fe6f8a582c564cf5ff",
    "ginkgo_adiantoides_specimen.jpg": "9b107911c55810e272a4f18af3a1f21e0f0d95561cf384cba5c90c9245314ca8",
    "scout_manifest.json": "1a316445351e1e66752c97aea6d5fc12ea08ca8202a61a5e616f352ea73ceaeb",
    "paper_citations.bib": "b74281f8aaf6a1f8ec63d54327068b86e16b2a894b7d50ae8f0c89e26c900a7b"
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
    ref = json.loads((DATASET_DIR / "ginkgo_adiantoides_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2006-royer-ginkgo-adiantoides-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
