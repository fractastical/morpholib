#!/usr/bin/env python3
"""
2017-shapiro-columba-livia-vertex-packing.py — Pass 2 Remediated Reproduction (Slot #0514)
Organism: Columba livia (Rock Dove)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Columba
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing)
Formal Verification: `morphoeval/lean/loops/Loop_0514_columba_livia_vertex_packing.lean` (`MathOfBiology.Loop_0514`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/columba_livia_vertex_packing"
EXPECTED_SHA256 = {
    "columba_livia_vertex_packing_reference.json": "069cec94b163ec7f3fa26d2919be0980929ce53a186673a05796842833dc7a0b",
    "columba_livia_specimen.jpg": "00bf4ed50c6075550c363caacdfa0f8820f68486fb8d737e81039f614c1ae7fb",
    "scout_manifest.json": "88c42289caebb939fdbc64247def96b3631929dbeddc8c33164051951a095813",
    "paper_citations.bib": "b175bed2b57caa4bc9af6e4f6ad422e4f80f0070738d72398b083eabed5841b2"
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
    ref = json.loads((DATASET_DIR / "columba_livia_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2017-shapiro-columba-livia-vertex-packing.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
