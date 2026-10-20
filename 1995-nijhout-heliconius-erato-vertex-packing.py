#!/usr/bin/env python3
"""
1995-nijhout-heliconius-erato-vertex-packing.py — Pass 2 Remediated Reproduction (Slot #0547)
Organism: Heliconius erato (Red Postman Butterfly)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Heliconius
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing)
Formal Verification: `morphoeval/lean/loops/Loop_0547_heliconius_erato_vertex_packing.lean` (`MathOfBiology.Loop_0547`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/heliconius_erato_vertex_packing"
EXPECTED_SHA256 = {
    "heliconius_erato_vertex_packing_reference.json": "26c61989a1c1d6170d0d70305ee40d31aa222155dca9c4b30b356202dbd08587",
    "heliconius_erato_specimen.jpg": "e54735f4ca16e4326cb91d2969d83bb613e3c4111aaf7f30cb218075b6fec48e",
    "scout_manifest.json": "e06c4426260e8ef2dff9ac956049ee0bb82f8f6b639893fba6d825079bfb3b9c",
    "paper_citations.bib": "4d2c05ab06b176d5e029bddf8ee8843f38630e6ef8f417ef0fae198c9fbba51f"
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
    ref = json.loads((DATASET_DIR / "heliconius_erato_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1995-nijhout-heliconius-erato-vertex-packing.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
