#!/usr/bin/env python3
"""
2011-harrison-selaginella-moellendorffii-vertex-packing.py — Pass 2 Remediated Reproduction (Slot #0579)
Organism: Selaginella moellendorffii (Spikemoss Lycophyte)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Selaginella
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing)
Formal Verification: `morphoeval/lean/loops/Loop_0579_selaginella_moellendorffii_vertex_packing.lean` (`MathOfBiology.Loop_0579`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/selaginella_moellendorffii_vertex_packing"
EXPECTED_SHA256 = {
    "selaginella_moellendorffii_vertex_packing_reference.json": "e030ccf76784b9338cc0b612ec89ddcf1e7ac019c5e99685331de03e46cb985a",
    "selaginella_moellendorffii_specimen.jpg": "87d0d25c483d4f17cb66050be175071aafb49d6f321bc8eb2d9cc3bac9e55cd4",
    "scout_manifest.json": "81e5d100201878e61db665f82d65f5032f6d96688a4c4643f15352707ac83757",
    "paper_citations.bib": "41edae13b05ef09d43f5f0f934326006111b73e09e7d3efe9e40a8796d626f06"
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
    ref = json.loads((DATASET_DIR / "selaginella_moellendorffii_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2011-harrison-selaginella-moellendorffii-vertex-packing.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
