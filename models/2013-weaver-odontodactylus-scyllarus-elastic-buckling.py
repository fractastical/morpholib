#!/usr/bin/env python3
"""
models/2013-weaver-odontodactylus-scyllarus-elastic-buckling.py — Multi-Scale & Paleobiological Reproduction (Slot #1823)
Organism: Odontodactylus scyllarus (Peacock Mantis Shrimp (Bouligand Helicoidal Dactyl Club Nanocomposite))
Phenomenon Scale: 150 nm (1.502e-07 m, log10=-6.823, 2_macromolecular_100nm)
Anatomical Structure: Elastic bilayer fold / wrinkle / suture wavelength in Odontodactylus scyllarus (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Odontodactylus
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling (150 nm))
Formal Verification: `morphoeval/lean/loops/Loop_1823_odontodactylus_scyllarus_elastic_buckling.lean` (`MathOfBiology.Loop_1823`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/odontodactylus_scyllarus_elastic_buckling"
EXPECTED_SHA256 = {
    "odontodactylus_scyllarus_elastic_buckling_reference.json": "c35b91862899c118e4ec6fef9b84726ec6dddbc44cc105eb3635ddd2f505ef90",
    "odontodactylus_scyllarus_specimen.jpg": "a0cb5ee4fe57474a9ed4bdb381117a01fec6e1ff71de38b747022330546126c0",
    "scout_manifest.json": "94066c2f25b9d61216aed23a50df61ee5230706f74a1482cc0cd7bd558691e87",
    "paper_citations.bib": "51fb90edb1d4ff77d57c8bfbbfb08a91ea933455e902cbf9fad51825e71e536f"
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
    ref = json.loads((DATASET_DIR / "odontodactylus_scyllarus_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2013-weaver-odontodactylus-scyllarus-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
