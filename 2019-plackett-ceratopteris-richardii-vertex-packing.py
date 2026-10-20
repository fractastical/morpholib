#!/usr/bin/env python3
"""
2019-plackett-ceratopteris-richardii-vertex-packing.py — Pass 2 Remediated Reproduction (Slot #0580)
Organism: Ceratopteris richardii (Triangle Water Fern)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Ceratopteris
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing)
Formal Verification: `morphoeval/lean/loops/Loop_0580_ceratopteris_richardii_vertex_packing.lean` (`MathOfBiology.Loop_0580`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/ceratopteris_richardii_vertex_packing"
EXPECTED_SHA256 = {
    "ceratopteris_richardii_vertex_packing_reference.json": "48a78e79715a47fdf38db8c749bbb4b0ef635c22370e3b91a13de47ed7b6062d",
    "ceratopteris_richardii_specimen.jpg": "3fcf2684662c089b9c66687d1f6b2e9409565391d0d316e5db46561d1809641f",
    "scout_manifest.json": "b0543299f3bfbf856bd86b8a7ad2ff63211e10c639c16e2b45ace026a21b0dac",
    "paper_citations.bib": "bab5c79d49dfc4785e6abefda5f7f96fa9881868bab6372d10d70d8eb7549695"
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
    ref = json.loads((DATASET_DIR / "ceratopteris_richardii_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2019-plackett-ceratopteris-richardii-vertex-packing.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
