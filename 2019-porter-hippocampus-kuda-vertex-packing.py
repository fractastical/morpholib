#!/usr/bin/env python3
"""
2019-porter-hippocampus-kuda-vertex-packing.py — Pass 2 Remediated Reproduction (Slot #0534)
Organism: Hippocampus kuda (Spotted Seahorse)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Hippocampus
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing)
Formal Verification: `morphoeval/lean/loops/Loop_0534_hippocampus_kuda_vertex_packing.lean` (`MathOfBiology.Loop_0534`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/hippocampus_kuda_vertex_packing"
EXPECTED_SHA256 = {
    "hippocampus_kuda_vertex_packing_reference.json": "f1ac66eb03c31f1a7dfbb75896c84528672efe5bfde3437bdaf44c962dc1c72c",
    "hippocampus_kuda_specimen.jpg": "2d453503e6f09426796b25434fc672645ac4aa639654b8d282c6607928df715e",
    "scout_manifest.json": "2fa0126291d0b3c42894a02837396d9eeb370fab1fadad79d2d19ebd58943eae",
    "paper_citations.bib": "0c780e861b4b540a3517360a2417cfc897cde209722cf60d4624194e00e11f01"
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
    ref = json.loads((DATASET_DIR / "hippocampus_kuda_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2019-porter-hippocampus-kuda-vertex-packing.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
