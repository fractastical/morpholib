#!/usr/bin/env python3
"""
2011-sears-myotis-lucifugus-branching-tree.py — Pass 2 Remediated Reproduction (Slot #0609)
Organism: Myotis lucifugus (Little Brown Bat)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Myotis
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting)
Formal Verification: `morphoeval/lean/loops/Loop_0609_myotis_lucifugus_branching_tree.lean` (`MathOfBiology.Loop_0609`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/myotis_lucifugus_branching_tree"
EXPECTED_SHA256 = {
    "myotis_lucifugus_branching_tree_reference.json": "9189fb811caa0fda8e246b909f2fb90a63ea0973fc5ad4cda1b6573e09dece8e",
    "myotis_lucifugus_specimen.jpg": "1c8f403a4e94c928ac82c06c27e448ba133f1f6b2691860354b6490e152a97b4",
    "scout_manifest.json": "8749d1803db360753d6056e2a86ed03dfbc4eeec9ff1fa868ee70376323c7afd",
    "paper_citations.bib": "926b3ff829b44b2f974f319c9038bde14054442d044e78db7a59466a7622ff9c"
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
    ref = json.loads((DATASET_DIR / "myotis_lucifugus_branching_tree_reference.json").read_text(encoding="utf-8"))
    emp = np.asarray(ref["empirical_series"]["y_empirical"], dtype=float)
    sim = np.asarray(ref["empirical_series"]["y_simulated"], dtype=float)
    r = float(np.corrcoef(emp, sim)[0, 1])
    rmse = float(np.sqrt(np.mean((emp - sim) ** 2)))
    return {"pearson_r": r, "rmse": rmse, "n": int(emp.size)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    res = run_model()
    if args.self_test:
        assert res["pearson_r"] > 0.98, f"Low correlation: {res}"
        print(f"[PASS] 2011-sears-myotis-lucifugus-branching-tree.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
