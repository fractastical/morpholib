#!/usr/bin/env python3
"""
Re-test Thompson's Fig. 370 amphipod series (Harpinia -> Stegocephalus,
Harpinia -> Hyperia) on G.O. Sars's 1895 lithographs (Crustacea of Norway I,
Pls. 52, 69, 2), the habitus figures Thompson's outlines match, alongside his
own coarser 1917 engraving.

For each source (lithograph, engraving) and each profile set (full dorsal +
ventral profile, dorsal-only) the script compares three maps fitted to the
same semilandmarks, in-sample and by leave-one-out (hat-matrix) CV:
  - 4p rectangular box stretch
  - 8p Thompson coaxial-arch / radial-fan law (as in run_all_1917_arthropods_audit.py)
  - 8p generic quadratic with the same parameter count (control)

Writes thompson_1917_amphipod_lithograph_check.json.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = REPO_ROOT / "thompson_1917_amphipod_lithograph_check.json"

SARS_CROPS = {
    "harpinia": ("sars1895_pl52_harpinia_plumosa.jpg", (199, 169, 1054, 596)),
    "stegocephalus": ("sars1895_pl69_stegocephalus_inflatus.jpg", (348, 189, 1064, 765)),
    "hyperia": ("sars1895_pl2_hyperia_galba.jpg", (388, 189, 934, 656)),
}
ENGRAVING_ROWS = {"harpinia": (5, 175), "stegocephalus": (195, 415), "hyperia": (445, 655)}
TARGET_WIDTH_PX = 430
N_DORSAL = 12


def load_module(name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fit_models(S0: np.ndarray, T0: np.ndarray, loo_hat_r2) -> dict:
    u, v = S0[:, 0], S0[:, 1]
    one = np.ones(len(u))
    designs = {
        "box_4p": (np.column_stack([u, one]), np.column_stack([v, one])),
        "thompson_8p": (np.column_stack([u, u * v, v ** 2, one]), np.column_stack([v, u * v, u ** 2, one])),
        "generic_quadratic_8p": (np.column_stack([u, u ** 2, v, one]), np.column_stack([v, u, u ** 2, one])),
    }
    ss0 = float(np.sum((T0 - S0) ** 2))
    out = {}
    for name, (Ax, Ay) in designs.items():
        mx = np.linalg.lstsq(Ax, T0[:, 0], rcond=None)[0]
        my = np.linalg.lstsq(Ay, T0[:, 1], rcond=None)[0]
        P = np.column_stack([Ax @ mx, Ay @ my])
        out[name] = {
            "insample_r2_pct": round((1.0 - float(np.sum((T0 - P) ** 2)) / ss0) * 100.0, 1),
            "loo_cv_r2_pct": round(loo_hat_r2(Ax, Ay, S0, T0), 1),
        }
    return out


def main() -> None:
    arth = load_module("arth", "thompson/run_all_1917_arthropods_audit.py")
    plates = load_module("plates", "thompson/generate_arthropod_museum_plates_figure.py")

    profiles = {"sars_1895_lithograph": {}, "thompson_1917_engraving": {}}
    for key, (fname, box) in SARS_CROPS.items():
        im = Image.open(plates.fetch_monograph_plate(fname)).convert("L").crop(box)
        im = im.resize((TARGET_WIDTH_PX, int(im.height * TARGET_WIDTH_PX / im.width)), Image.LANCZOS)
        profiles["sars_1895_lithograph"][key] = arth.extract_amphipod_profile_from_png(np.asarray(im))[0]

    eng = np.asarray(Image.open(arth.CRAB_DIR / "thompson_1917_fig370_amphipods.png").convert("L"))
    for key, (y0, y1) in ENGRAVING_ROWS.items():
        profiles["thompson_1917_engraving"][key] = arth.extract_amphipod_profile_from_png(eng[y0:y1, :])[0]

    results = {}
    for source, prof in profiles.items():
        S = prof["harpinia"]
        results[source] = {}
        for target in ("stegocephalus", "hyperia"):
            T = prof[target]
            results[source][f"harpinia_to_{target}"] = {
                "full_profile_24": fit_models(S, T, arth.loo_hat_r2),
                "dorsal_only_12": fit_models(S[:N_DORSAL], T[:N_DORSAL], arth.loo_hat_r2),
            }

    out = {
        "sources": {
            "sars_1895_lithograph": "G.O. Sars (1895) An Account of the Crustacea of Norway I, Pls. 52, 69, 2 (Wikimedia Commons, public domain, SHA-256 checked)",
            "thompson_1917_engraving": "Thompson (1917) Fig. 370, Project Gutenberg i746.png",
        },
        "method": "24 profile semilandmarks (12 dorsal + 12 ventral) via extract_amphipod_profile_from_png; LOO-CV by hat matrix",
        "results": results,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    for source, pairs in results.items():
        for pair, sets in pairs.items():
            for pset, models in sets.items():
                row = "  ".join(f"{m} {v['insample_r2_pct']:5.1f}/{v['loo_cv_r2_pct']:6.1f}" for m, v in models.items())
                print(f"{source:24s} {pair:26s} {pset:16s} {row}")
    print(f"Saved {OUT_JSON}")


if __name__ == "__main__":
    main()
