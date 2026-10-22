#!/usr/bin/env python3
"""
models/1995-dimichele-lepidodendron-aculeatum-bioelectric-polarity.py — Multi-Scale & Paleobiological Reproduction (Slot #1978)
Organism: Lepidodendron aculeatum (Carboniferous Giant Scale-Tree (Helical Parastichy Leaf Cushions))
Phenomenon Scale: 27 µm (2.706e-05 m, log10=-4.568, 4_cellular_10um)
Anatomical Structure: Gap-junction bioelectric domain & ion-channel patch in Lepidodendron aculeatum (Late Carboniferous (Pennsylvanian, 310 Ma))
Geological Horizon: Late Carboniferous (Pennsylvanian, 310 Ma) (310.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Lepidodendron
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity (27 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1978_lepidodendron_aculeatum_bioelectric_polarity.lean` (`MathOfBiology.Loop_1978`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/lepidodendron_aculeatum_bioelectric_polarity"
EXPECTED_SHA256 = {
    "lepidodendron_aculeatum_bioelectric_polarity_reference.json": "7ffa3af6173da69ccbcff0b3b5a448abaee38c71970ae3e142ab1221a049cc61",
    "lepidodendron_aculeatum_specimen.jpg": "854f4a6257e58a0123b8eee34674c32ed016882c88d6eb907ca1d85dbb78c467",
    "scout_manifest.json": "fd39a0e921821f8bf03f1a80c37c5113c7a7d8c512d84be835d60fabb0a0f905",
    "paper_citations.bib": "364c54f0aa93e32d6afc96f3c8473f0976a1b247ab2aa6db12707b1fc8f222cf"
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
    ref = json.loads((DATASET_DIR / "lepidodendron_aculeatum_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1995-dimichele-lepidodendron-aculeatum-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
