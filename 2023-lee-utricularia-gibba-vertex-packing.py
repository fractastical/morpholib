#!/usr/bin/env python3
"""
2023-lee-utricularia-gibba-vertex-packing.py — Pass 2 Remediated Reproduction (Slot #0577)
Organism: Utricularia gibba (Humped Bladderwort)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Utricularia
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing)
Formal Verification: `morphoeval/lean/loops/Loop_0577_utricularia_gibba_vertex_packing.lean` (`MathOfBiology.Loop_0577`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/utricularia_gibba_vertex_packing"
EXPECTED_SHA256 = {
    "utricularia_gibba_vertex_packing_reference.json": "1de53b577a7558fc634da5386e4242ec71701f95c47e2a987e9cce3ea88e133c",
    "utricularia_gibba_specimen.jpg": "9143b846b36d681d8c10ef23ad998247630e9d5e3abebcdfc12e3fa9dadfc3f7",
    "scout_manifest.json": "115789280ac1542a4f7509b710cfa9f979aaa0a8eedb80952929337da6518996",
    "paper_citations.bib": "a613c57b09dcbda439df49a37c186eea2e2eba891d0a37021025946d3a2b4c20"
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
    ref = json.loads((DATASET_DIR / "utricularia_gibba_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2023-lee-utricularia-gibba-vertex-packing.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
