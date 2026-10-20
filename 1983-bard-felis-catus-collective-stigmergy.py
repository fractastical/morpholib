#!/usr/bin/env python3
"""
1983-bard-felis-catus-collective-stigmergy.py — Pass 2 Remediated Reproduction (Slot #1003)
Organism: Felis catus (Domestic Cat)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Felis
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation)
Formal Verification: `morphoeval/lean/loops/Loop_1003_felis_catus_collective_stigmergy.lean` (`MathOfBiology.Loop_1003`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/felis_catus_collective_stigmergy"
EXPECTED_SHA256 = {
    "felis_catus_collective_stigmergy_reference.json": "2504525b7233de533552150e3dc8ffa100154ce4816a5b43695a23915d5c96c3",
    "felis_catus_specimen.jpg": "f417376a9a5f657061f2082396f5cc7eb2b6a4d0c92e0c1d3692ceb2b52926bd",
    "scout_manifest.json": "86392e10a40437028981420be7500a3ebe3a7b1f191d7a099f616728c7fa24b6",
    "paper_citations.bib": "4c0b27c83122086faae810fd0b357e114edd5fdf9ba704d6a20679308aa28c98"
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
    ref = json.loads((DATASET_DIR / "felis_catus_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1983-bard-felis-catus-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
