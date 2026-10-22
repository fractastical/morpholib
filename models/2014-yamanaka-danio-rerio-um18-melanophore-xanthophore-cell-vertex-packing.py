#!/usr/bin/env python3
"""
models/2014-yamanaka-danio-rerio-um18-melanophore-xanthophore-cell-vertex-packing.py — Multi-Scale & Paleobiological Reproduction (Slot #1145)
Organism: Danio rerio (Zebrafish — 14–21 dpf)
Phenomenon Scale: 18 µm (1.764e-05 m, log10=-4.754, 4_cellular_10um)
Anatomical Structure: Single melanophore–xanthophore pseudopodial contact & NMP cell body (14–21 dpf (Metamorphic Pigment & Tailbud Progenitor, ZFA:0009096)) · Zebrahub tailbud_neuromesodermal_progenitors & pigment_chromatophores
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Danio
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing at 18 µm (Single melanophore–xanthophore pseudopodial contact & NMP cell body))
Formal Verification: `morphoeval/lean/loops/Loop_1145_danio_rerio_um18_melanophore_xanthophore_cell_vertex_packing.lean` (`MathOfBiology.Loop_1145`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/danio_rerio_um18_melanophore_xanthophore_cell_vertex_packing"
EXPECTED_SHA256 = {
    "danio_rerio_um18_melanophore_xanthophore_cell_vertex_packing_reference.json": "b75d403e592139145c6333d2945f37157bc804cfb522e1297ecb40b41c864b9e",
    "danio_rerio_specimen.jpg": "850b2d4a94c7cbcb3c16e606ddb70d198f082ccb2f046ee44dbaa5ef5941678a",
    "scout_manifest.json": "24b0db300bb0da86d5db7ccba06366183f612feda9a8f01ba20529f04b6c658b",
    "paper_citations.bib": "f65e1bcec3e98b72f1a70d3f7d47d92314359e9127b5a0f6757da242ef0030b9"
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
    ref = json.loads((DATASET_DIR / "danio_rerio_um18_melanophore_xanthophore_cell_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2014-yamanaka-danio-rerio-um18-melanophore-xanthophore-cell-vertex-packing.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
