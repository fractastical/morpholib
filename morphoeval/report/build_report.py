#!/usr/bin/env python3
"""Build the MorphoEval Reproducibility Report (PDF) from report.md and the audit files.

    python3 morphoeval/report/build_report.py                 # -> build/MorphoEval-Reproducibility-Report.pdf
    python3 morphoeval/report/build_report.py --out-dir /tmp/r --no-pdf   # markdown only

The prose lives in report.md; this script fills its placeholders from the audit YAMLs so the
tables cannot drift from the published scores:

    {{DATE}} {{COMMIT}}          today / `git rev-parse --short HEAD`
    {{RUBRIC_TABLE}}             from morphoeval/rubric.yaml (with sub-axis decomposition)
    {{SCORE_TABLE}}              current head of each active script's chain, per dimension + sub-axes + overall
    {{DECOMPOSITION_TABLE}}      implementation fidelity vs biological validity vs epistemic role
    {{BLINDED_TABLE}}            5-script blinded second-assessor comparison (Claude vs Gemini)
    {{HISTORY_TABLE}}            every evaluation pass across all audit files, in date order
    {{N_SCRIPTS_ACTIVE}} {{N_SCRIPTS_EVALUATED}} {{N_PRIMARY_AUDITS}} {{N_PRIMARY_PASSES}}
    {{N_BLINDED_PASSES}} {{N_TOTAL_AUDITS}} {{N_TOTAL_PASSES}} {{N_NANOPUBS}} {{N_HEADS}} {{N_BLINDED_NANOPUBS}}

The pipeline figure is drawn by draw_pipeline.py into the build dir.  PDF via pandoc + xelatex.
Everything under build/ is git-ignored; the PDF is a deposit artefact (Zenodo), not a repo file.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUBRIC = ROOT / "morphoeval" / "rubric.yaml"

SUPERSEDED_SCRIPTS = {"1900-planformDB_parser.py"}

# Epistemic role, empirical validation target, and mechanistic / model-validity status for each script
SCRIPT_META = {
    "2005-runions-leaf-venation.py": {
        "kind": "reproduce",
        "target": "real specimens (segmented dicot skeleton 168 areoles + Ginkgo leaf) & Price 2013 (349 leaves)",
        "validity": "Implements Runions (2005) open Voronoi & closed Toussaint RNG space-colonization with persistent V(s) tagging, planar segment-intersection enforcement, Murray/Price Strahler scaling, and planar dual areole nesting.",
    },
    "1984-wolfram-conus-textile-rule30.py": {
        "kind": "reproduce",
        "target": "real specimens (OSU & Commons Conus textile photos; H/W error 0.07%, IoU 0.835) & OEIS A051023",
        "validity": "Exact Rule 30 / Jen (1990) cellular automaton half; blinded 2nd assessor (50.5 vs 87.3) flagged that the Meinhardt PDE uses an unsourced relaxation divisor and stochastic injection and Raup T is swamped by axial_ext.",
    },
    "1979-vogel-phyllotaxis.py": {
        "kind": "reproduce",
        "target": "real specimens (621 DoG-segmented florets on H. annuus photo + Swinton 2016 N=768 capitula)",
        "validity": "Implements Vogel (1979) spiral, bounded Voronoi optimality, Douady-Couder (1992) Hofmeister relaxation, and rim-dislocation ensemble reproducing Swinton's Fib-1 > Fib+1 asymmetry.",
    },
    "thompson/1917-thompson-scarus-pomacanthus.py": {
        "kind": "test",
        "target": "real specimens (791 fish species landmarks + 2 segmented photos)",
        "validity": "Refutes source claim: Thompson's coaxial circles explain 32% of Procrustes distance (= affine) and lower outline IoU (0.618 vs 0.662 similarity baseline).",
    },
    "2026-eshghi-odonata-wing-cells.py": {
        "kind": "test",
        "target": "real specimens (69 fossil drawings re-segmented + 394 extant wings)",
        "validity": "Tests scaling prediction only (N ~ L^2, cell length ~ L^0); does not run the generative venation mechanism.",
    },
    "2013-lobo-planform-morphology.py": {
        "kind": "decode",
        "target": "authors' cartoons (PlanformDB 2.5.0 Figs. 3, 5; mean IoU 0.78)",
        "validity": "Static graph decoder, not a generative biological model; cartoons matched are the source software's drawings, not worm photographs.",
    },
    "2014-lobo-limbform-morphology.py": {
        "kind": "decode",
        "target": "authors' cartoons (LimbformDB 2.1.0 Fig. 1 A''-E''; IoU 0.41-0.74)",
        "validity": "Static graph decoder, not a generative mechanism; compares to source cartoons rather than real limb photographs.",
    },
    "2015-santella-wormguides-celegans.py": {
        "kind": "decode",
        "target": "real specimen 4D tracking dataset (Santella et al. 2015 WormGUIDES)",
        "validity": "Replays empirical nuclear trajectories and segmented eggshell mesh (29.2 um width vs 30 um); does not simulate force-generation mechanics.",
    },
    "create_raup_animations.py": {
        "kind": "reproduce",
        "target": "none (internal mesh parameter recovery to 5e-15; 'cf.' archetypes)",
        "validity": "Descriptive geometric morphospace (W, D, T, S), not a developmental mechanism; no biological shell specimen measured.",
    },
    "2018-hoffmann-wing-venation.py": {
        "kind": "reproduce",
        "target": "real specimen segmented wing (Perithemis tenera, 249 secondary cells)",
        "validity": "Seeds Voronoi sites from peaks of the real wing's own distance transform rather than generating inhibition zones de novo.",
    },
    "2013-chindapol-coral-flow.py": {
        "kind": "reproduce",
        "target": "qualitative morphology trends (Peclet-dependent symmetry smmag)",
        "validity": "Replaces coupled Lattice-Boltzmann Navier-Stokes + advection-diffusion with biased diffusion-limited aggregation (DLA).",
    },
    "2021-Cervera–Levin–Mafe.py": {
        "kind": "reproduce",
        "target": "real S. mediterranea silhouette mask + Oviedo 2010 DH frequencies",
        "validity": "Qualitative 1D reaction-diffusion + 2-node ODE proxy; paints territories onto a static silhouette and switches between two pre-drawn cartoons.",
    },
    "1952-turing-morpho.py": {
        "kind": "reproduce",
        "target": "none (generic 2D periodic grid at f=0.055, k=0.062)",
        "validity": "Canonical Gray-Scott reaction-diffusion PDE; chemical pattern generator not tied to a specific organism's anatomy.",
    },
    "2001-bressloff-v1-form-constants.py": {
        "kind": "reproduce",
        "target": "Horton & Hoyt (1991) human V1 linear magnification data M_lin(r)",
        "validity": "Analytical retino-cortical log-polar map + E(2) eigenmodes on V1; upgraded in v2 with full S^1 orientation tuning and bifurcation PDE.",
    },
    "2010-petroff-stromatolite.py": {
        "kind": "reproduce",
        "target": "published field spacing range ~1 cm (Yellowstone Conophyton; no specimen image)",
        "validity": "Exact diffusive length scale ell = sqrt(D * tau) * f(alpha); lamina cross-section uses an analytic height-squared photic-weight proxy (Batchelor et al. 2004 + w_photic) rather than solving Laplace's equation nabla^2 c = 0.",
    },
    "2024-kahng-coral-geometry.py": {
        "kind": "reproduce",
        "target": "published Porites/Leptoseris/Acropora parameters & microfragmentation SAP speedup (no specimen mesh)",
        "validity": "Exact volumetric/surface identities and continuum SAP = (G/rho)*(dS/dV); discrete 1-yr step introduces O(C/r) truncation in small-r hemispheres and uses shared G=1.50 g/cm^2/yr across disparate taxa.",
    },
    "1900-planformDB_parser.py": {
        "kind": "decode (superseded)",
        "target": "none (convex hull of organ points)",
        "validity": "Superseded by 2013-lobo-planform-morphology.py; ignored RegionParam/RegionsLink encoding.",
    },
}


def sh(*cmd, **kw) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=ROOT, **kw).stdout.strip()


def load_audits() -> list[dict]:
    out = []
    tracked = set(sh("git", "ls-files", "morphoeval/audits/*.yaml").splitlines())
    files = [str(ROOT / p) for p in tracked if ".measurements." not in p]
    if not files:
        files = [p for p in glob.glob(str(ROOT / "morphoeval" / "audits" / "*.yaml")) if ".measurements." not in p]
    files.sort(key=lambda f: Path(f).stem)
    for f in files:
        a = yaml.safe_load(open(f))
        a["_file"] = Path(f).name
        out.append(a)
    return out


def rubric_table(rubric: dict) -> str:
    rows = ["| group | sub-axis | dimension | weight | question |", "|---|---|---|---:|---|"]
    for d in rubric["dimensions"]:
        q = " ".join(d["question"].split())
        sub = d.get("sub_group", d["group"])
        rows.append(f"| {d['group']} | `{sub}` | **{d['name']}** (`{d['id']}`) | {d['weight']} | {q} |")
    return "\n".join(rows)


def overall(scores: dict, weights: dict) -> float:
    return sum(scores[d] * w for d, w in weights.items()) / sum(weights.values())


def overall_applicable(scores: dict, weights: dict, na: list[str]) -> float:
    app = {d: w for d, w in weights.items() if d not in na}
    if not app:
        return overall(scores, weights)
    return sum(scores[d] * w for d, w in app.items()) / sum(app.values())


def group_mean(scores: dict, rubric: dict, group: str) -> float:
    ds = [d for d in rubric["dimensions"] if d["group"] == group]
    return sum(scores[d["id"]] * d["weight"] for d in ds) / sum(d["weight"] for d in ds)


def subgroup_mean(scores: dict, rubric: dict, sub_group: str, na: list[str] | None = None) -> float:
    ds = [d for d in rubric["dimensions"] if d.get("sub_group") == sub_group]
    if na:
        ds_app = [d for d in ds if d["id"] not in na]
        if ds_app:
            ds = ds_app
    return sum(scores[d["id"]] * d["weight"] for d in ds) / sum(d["weight"] for d in ds)


def collect(audits: list[dict], rubric: dict):
    """Return (latest_active, latest_superseded, hist_primary, hist_blinded, hist_all)."""
    weights = {d["id"]: d["weight"] for d in rubric["dimensions"]}
    hist_primary, hist_blinded, hist_all = [], [], []
    for a in audits:
        # Skip non-canonical audit schemas (e.g. the autonomous slot campaign
        # YAMLs keyed by top-level `audit_id` with no `audit:`/`simulations:`
        # blocks). Without this guard, collect() — and every consumer such as
        # docs/media/make_montage.py scores() — crashes with KeyError: 'audit'
        # on the thousands of slot audits now tracked under morphoeval/audits/.
        if not isinstance(a, dict) or not isinstance(a.get("audit"), dict):
            continue
        if not isinstance(a["audit"].get("assessor"), dict) or "id" not in a["audit"]["assessor"]:
            continue
        if not isinstance(a.get("simulations"), list):
            continue
        role = a["audit"].get("role", "primary")
        assessor_id = a["audit"]["assessor"]["id"]
        for s in a["simulations"]:
            if s.get("status") != "evaluated":
                continue
            sc = {d: v["value"] for d, v in s["scores"].items()}
            na = [d for d, v in s["scores"].items() if v.get("not_applicable")]
            rec = dict(
                audit=a["audit"]["id"],
                role=role,
                assessor=assessor_id,
                date=str(a["audit"]["date"]),
                commit=a["audit"]["checkpoint"]["commit"],
                sim=s["id"],
                name=s["name"],
                script=s["script"],
                scores=sc,
                na=na,
                overall=overall(sc, weights),
                overall_app=overall_applicable(sc, weights, na),
                faith=group_mean(sc, rubric, "faithfulness"),
                impl=subgroup_mean(sc, rubric, "model-implementation"),
                impl_app=subgroup_mean(sc, rubric, "model-implementation", na),
                bio=subgroup_mean(sc, rubric, "biological-validation"),
                bio_app=subgroup_mean(sc, rubric, "biological-validation", na),
                repro=group_mean(sc, rubric, "reproducibility"),
            )
            hist_all.append(rec)
            if role == "second_assessor":
                hist_blinded.append(rec)
            else:
                hist_primary.append(rec)
    hist_primary.sort(key=lambda r: (r["date"], r["audit"]))
    hist_all.sort(key=lambda r: (r["date"], 0 if r["role"] != "second_assessor" else 1, r["audit"]))
    latest = {}
    for r in hist_primary:
        latest[r["script"]] = r          # later primary audits overwrite; keyed by script
    latest_active = sorted(
        [r for r in latest.values() if r["script"] not in SUPERSEDED_SCRIPTS],
        key=lambda r: -r["overall"],
    )
    latest_superseded = sorted(
        [r for r in latest.values() if r["script"] in SUPERSEDED_SCRIPTS],
        key=lambda r: -r["overall"],
    )
    return latest_active, latest_superseded, hist_primary, hist_blinded, hist_all


def score_table(latest_active, latest_superseded, rubric) -> str:
    dims = [d["id"] for d in rubric["dimensions"]]
    short = {"model-fidelity": "fidel.", "specimen-match": "spec.", "parameter-expressivity": "param.",
             "morphometric-validation": "morph.", "source-traceability": "src.", "code-traceability": "code",
             "repro-package": "repro."}
    head = ("| script | kind | audit (commit) | " + " | ".join(short[d] for d in dims) +
            " | impl. | bio. | repro. | app. | **pad.** |")
    rows = [head, "|" + "-" * 32 + "|" + "-" * 9 + "|" + "-" * 22 + "|" + "-----:|" * len(dims) + "-----:|" * 4 + "------:|"]
    for r in latest_active:
        cells = [f"{r['scores'][d]}{'*' if d in r['na'] else ''}" for d in dims]
        aud = r["audit"].replace("-audit", "").replace("2026-", "")
        kind = SCRIPT_META.get(r["script"], {}).get("kind", "—")
        rows.append(
            f"| `{r['script']}` | {kind} | {aud} (`{r['commit']}`) | " + " | ".join(cells) +
            f" | {r['impl']:.1f} | {r['bio']:.1f} | {r['repro']:.1f} | {r['overall_app']:.1f} | **{r['overall']:.1f}** |"
        )
    for r in latest_superseded:
        cells = [f"{r['scores'][d]}{'*' if d in r['na'] else ''}" for d in dims]
        aud = r["audit"].replace("-audit", "").replace("2026-", "")
        rows.append(
            f"| *`{r['script']}` (superseded)* | *decode* | {aud} (`{r['commit']}`) | " + " | ".join(cells) +
            f" | {r['impl']:.1f} | {r['bio']:.1f} | {r['repro']:.1f} | {r['overall_app']:.1f} | **{r['overall']:.1f}** |"
        )
    rows.append("")
    rows.append(
        "Table: Current primary-chain scores (uncalibrated 0–100 rubric ratings, not probabilities) across the "
        f"{len(latest_active)} active evaluated implementations (plus the superseded `1900-planformDB_parser.py`, "
        "replaced by `2013-lobo-planform-morphology.py`). Weights: "
        + ", ".join(f"{short[d['id']]} {d['weight']}" for d in rubric["dimensions"])
        + ". Sub-axes: **impl.** = model-implementation fidelity (`fidel.` 20 + `param.` 10 = weight 30); "
        "**bio.** = biological/empirical validation (`spec.` 25 + `morph.` 15 = weight 40); "
        "**repro.** = reproducibility & traceability (`src.` 10 + `code` 8 + `repro.` 12 = weight 30). "
        "An asterisk (`50*`) marks a dimension flagged `not_applicable`: **app.** is the weight-renormalised mean over "
        "applicable dimensions (mathematically neutral), whereas **pad.** substitutes 50 (shrinking the average toward 50)."
    )
    return "\n".join(rows)


def decomposition_table(latest_active) -> str:
    rows = [
        "| script | role | validation target | impl. (30) | bio. (40) | repro. (30) | mechanistic & biological-validity status |",
        "|" + "-" * 28 + "|" + "-" * 10 + "|" + "-" * 28 + "|------:|------:|------:|" + "-" * 48 + "|",
    ]
    for r in latest_active:
        meta = SCRIPT_META.get(r["script"], {})
        impl_str = f"{r['impl']:.1f}" + (f" ({r['impl_app']:.1f} app.)" if r["impl"] != r["impl_app"] else "")
        bio_str = f"{r['bio']:.1f}" + (f" ({r['bio_app']:.1f} app.)" if r["bio"] != r["bio_app"] else "")
        rows.append(
            f"| `{r['script']}` | `{meta.get('kind', '—')}` | {meta.get('target', '—')} | "
            f"{impl_str} | {bio_str} | {r['repro']:.1f} | {meta.get('validity', '—')} |"
        )
    rows.append("")
    rows.append(
        "Table: Separation of model-implementation fidelity (**impl.**, weight 30), biological/empirical validation "
        "(**bio.**, weight 40), and reproducibility (**repro.**, weight 30) from the scientific validity of the "
        "underlying biological mechanism. Where a dimension is flagged `not_applicable`, the applicable-only sub-score "
        "is shown in parentheses."
    )
    return "\n".join(rows)


def blinded_table(latest_active, hist_blinded) -> str:
    prim_by_sim = {r["sim"]: r for r in latest_active}
    blind_by_sim = {r["sim"]: r for r in hist_blinded}
    ordered_sims = [r["sim"] for r in latest_active if r["sim"] in blind_by_sim]

    prim_ranks = {sim: idx + 1 for idx, sim in enumerate(sorted(ordered_sims, key=lambda s: -prim_by_sim[s]["overall"]))}
    blind_ranks = {sim: idx + 1 for idx, sim in enumerate(sorted(ordered_sims, key=lambda s: -blind_by_sim[s]["overall"]))}

    rows = [
        "| script | primary (`claude-fable-5-1` / `gpt-5.6-sol`† / `gemini-4-argon`‡) impl. / bio. / repro. | 2nd assessor (`gemini-4-argon` / `claude-fable-5-1`‡) impl. / bio. / repro. | primary app. / **pad.** (rank) | 2nd assessor app. / **pad.** (rank) | Δ pad. (Δ app.) | panel mean **pad.** | independent defects / observations surfaced by 2nd pass |",
        "|" + "-" * 28 + "|" + "-" * 22 + "|" + "-" * 22 + "|" + "-" * 18 + "|" + "-" * 18 + "|---------:|------:|" + "-" * 42 + "|",
    ]
    defects = {
        "wolfram-conus-textile-rule30": "CA/Jen half verified exact; PDE uses unsourced relaxation divisor & stochastic injection (extinguishes without it); inert Raup T; non-discriminating chord metric; D_box > 2; wrong DOIs in script/.bib.",
        "hoffmann-wing-venation": "`fetch_wing_png()` lacks a SHA-256 digest check on the remote zip; stale docstring claim on `scikit-image`/`Pillow`; fixed 18 Lloyd steps.",
        "thompson-scarus-pomacanthus": "2-anchor silhouette registration vs 17-landmark fit; collapsed museum vs live extended fin posture.",
        "eshghi-odonata-wing-cells": "`fossil_drawing()` (<128, largest component) vs `_read_drawing()` (<160, >=150 px) threshold mismatch; overwrites tracked CSV in place.",
        "lobo-limbform-morphology": "65-hex-char typo in `DB_SHA256` (L92, trailing `0`) causing false-positive SHA-256 warning; `Link.side`/`reversed`/`inverted` parsed but unused.",
        "santella-wormguides-celegans": "2D orthographic projection of 3D embryo/meshes; c302 chemical synapses and gap junctions summed into one matrix.",
        "planform-morphology-animation": "Heuristic superellipse exponents (`N_FREE=2.2`, `N_BORDER=7.0`); requires external `PLANFORM_DB_PATH`; compares to cartoons rather than worm photos.",
        "raup-shell-coiling": "`create_parameter_space_visualization()` zips 3 (W,D) and 3 (W,T) pairs on a 1D diagonal instead of a 2D grid; S=1.0 fixed everywhere.",
        "cervera-levin-mafe": "Verifies commit `0154143` fixes (`f2_for_shift` bisection and dynamic Panel D legend); notes uncited pharynx-posterior threshold `x*_b = 0.5833`.",
        "chindapol-coral-flow": "Explicit 2D lattice DLA-with-drift proxy for 3D COMSOL Navier–Stokes + advection–diffusion; noisy 2D pixel tip detection without Fig. 9–12 overlay.",
        "turing-gray-scott": "Inline comment mislabels consumed substrate `U` as an 'inhibitor'; saves `turing_patterns_black_neon.gif` (mismatching `docs/media/turing_gray_scott.gif`).",
        "bressloff-v1-form-constants": "Pass-1 plotted pure cos(kx*x + ky*y) without S^1 orientation tuning or ODE/PDE integration; remediated in v2 with full V1 x S^1 E(2) fields.",
        "kahng-coral-geometry": "Discrete 1-yr forward-Euler step makes hemisphere extension appear size-dependent (0.52 -> 1.25 cm/yr vs exact continuum G/rho = 1.25 cm/yr); applying COMMON (G=1.50) to a 0.5 cm disc yields 26 cm/yr extension at r = 30 cm.",
        "petroff-stromatolite": "Lamina front uses Batchelor et al. (2004) + ad-hoc w_photic = 0.12 + 0.88*(z/z_max)^2 and pinned basal feet rather than Petroff et al.'s (2010) diffusive-flux solver (nabla^2 c = 0); text panel claims 0.66 cm records a ~20 h rhythm when computed at tau = 12 h.",
    }
    for sim in ordered_sims:
        p = prim_by_sim[sim]
        b = blind_by_sim[sim]
        dp = b["overall"] - p["overall"]
        da = b["overall_app"] - p["overall_app"]
        p_mean = 0.5 * (p["overall"] + b["overall"])
        dagger = "†" if p["assessor"] == "gpt-5.6-sol" else ("‡" if p["assessor"] == "gemini-4-argon" else "")
        rows.append(
            f"| `{p['script']}`{dagger} | {p['impl']:.1f} / {p['bio']:.1f} / {p['repro']:.1f} | "
            f"{b['impl']:.1f} / {b['bio']:.1f} / {b['repro']:.1f} | "
            f"{p['overall_app']:.1f} / **{p['overall']:.1f}** (#{prim_ranks[sim]}) | "
            f"{b['overall_app']:.1f} / **{b['overall']:.1f}** (#{blind_ranks[sim]}) | "
            f"{dp:+.1f} ({da:+.1f}) | **{p_mean:.1f}** | {defects.get(sim, '—')} |"
        )
    rows.append("")
    rows.append(
        "Table: Cross-company second-assessor evaluation across the 11 scripts with standalone `role: second_assessor` audits: "
        "the 10-script `gemini-4-argon` pass (`2026-10-17-blinded-sample-audit.yaml`, paired with `claude-fable-5-1` on 9 scripts and †`gpt-5.6-sol` on `2021-Cervera–Levin–Mafe.py`) "
        "plus the ‡blinded `claude-fable-5-1` pass (`2026-10-19-conus-second-assessor-audit.yaml`) on `1984-wolfram-conus-textile-rule30.py` (primary `gemini-4-argon`). "
        "(The remaining 2 active evaluated scripts, `2005-runions-leaf-venation.py` and `1979-vogel-phyllotaxis.py`, were dual-assessed inside their 12-step closed remediation loops by `gemini-4-argon` + `claude-sonnet-5-5-high`.) "
        "Across the 10-script `gemini-4-argon` pass, rank-order concordance with the primary assessors is strong (Spearman ρ = 0.891, p = 0.00054; Kendall τ = 0.778, p = 0.00095; "
        "Pearson r = 0.920, p = 0.00017 on padded scores; Spearman ρ = 0.903 on applicable-only scores; exact ρ = 1.00, τ = 1.00 on the initial 5-script blinded subset; "
        "dimension-level ρ = 0.796, r = 0.813 across 70 pairs, MAD = 10.4 points). Adding the blinded *Conus textile* replication (where `claude-fable-5-1` scored 50.5 vs `gemini-4-argon` 87.3, Δ = −36.8 points) "
        "brings the 11-script (77-dimension-pair) cross-company comparison to provider-paired Spearman ρ = 0.678 (p = 1.3 × 10⁻¹¹), Pearson r = 0.664, and MAD = 13.8 points."
    )
    return "\n".join(rows)


def history_table(hist_all) -> str:
    rows = [
        "| date | audit | assessor | commit | script | impl. | bio. | repro. | app. | overall | note |",
        "|" + "-" * 11 + "|" + "-" * 28 + "|" + "-" * 16 + "|" + "-" * 8 + "|" + "-" * 34 + "|-----:|-----:|------:|-----:|-------:|" + "-" * 28 + "|",
    ]
    prev = {}
    for r in hist_all:
        note = ""
        if r["role"] == "second_assessor":
            prim = prev.get(r["script"])
            note = f"blinded 2nd assessor (primary {prim:.1f})" if prim is not None else "blinded 2nd assessor"
        elif r["script"] in prev:
            note = f"re-score; was {prev[r['script']]:.1f}"
        elif r["sim"] == "planform-morphology-animation" and r["script"].startswith("2013"):
            note = "rewrite of `1900-planformDB_parser.py` (14.7)"
        rows.append(
            f"| {r['date']} | {r['audit']} | `{r['assessor']}` | `{r['commit']}` | `{r['script']}` | "
            f"{r['impl']:.1f} | {r['bio']:.1f} | {r['repro']:.1f} | {r['overall_app']:.1f} | {r['overall']:.1f} | {note} |"
        )
        if r["role"] != "second_assessor":
            prev[r["script"]] = r["overall"]
    return "\n".join(rows)


def nanopub_counts() -> tuple[int, int, int, int]:
    """(published to production, current heads, locally signed second-assessor nanopubs, locally signed unpublished primary nanopubs)."""
    import re
    prod = set()
    superseded_by_prod = set()
    blinded_signed = 0
    primary_unpub_signed = 0
    pat = re.compile(r"npx:supersedes\s+<([^>]+)>")
    tracked = set(sh("git", "ls-files", "morphoeval/kg/*/published.json").splitlines())
    pub_files = [str(ROOT / p) for p in sorted(tracked)] if tracked else sorted(glob.glob(str(ROOT / "morphoeval" / "kg" / "*" / "published.json")))
    for f in pub_files:
        is_blinded = ("blinded-sample" in f) or ("second-assessor" in f)
        sdir = Path(f).parent / "signed"
        for k, e in json.load(open(f)).items():
            if is_blinded and e.get("nanopub"):
                blinded_signed += 1
            elif not is_blinded and e.get("nanopub") and not any(v.get("published") == "prod" for v in [e] + e.get("history", [])):
                primary_unpub_signed += 1
            for v in [e] + e.get("history", []):
                if v.get("published") == "prod" and v.get("nanopub"):
                    prod.add(v["nanopub"])
            if e.get("published") == "prod" and (sdir / k).exists():
                superseded_by_prod.update(pat.findall((sdir / k).read_text()))
            all_vers = e.get("history", []) + [e]
            for i in range(1, len(all_vers)):
                if all_vers[i].get("published") == "prod" and all_vers[i - 1].get("nanopub"):
                    superseded_by_prod.add(all_vers[i - 1]["nanopub"])
    # Also record cross-directory supersession where 2026-10-14-morphology-audit history[0] replaced 2026-10-13-morphology-audit
    p13 = ROOT / "morphoeval" / "kg" / "2026-10-13-morphology-audit" / "published.json"
    p14 = ROOT / "morphoeval" / "kg" / "2026-10-14-morphology-audit" / "published.json"
    if p13.exists() and p14.exists():
        d13, d14 = json.load(open(p13)), json.load(open(p14))
        for k in d14:
            if k in d13 and d13[k].get("published") == "prod":
                superseded_by_prod.add(d13[k]["nanopub"])
    heads_offline = prod - superseded_by_prod
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("bmk", ROOT / "morphoeval" / "build_morphoeval_kg.py")
        bmk = importlib.util.module_from_spec(spec); spec.loader.exec_module(bmk)
        heads = set()
        for u in sorted(prod):
            h = bmk.latest_version(u, timeout=20)
            if h is None:
                raise RuntimeError(f"no head for {u}")
            heads.add(h)
        heads &= prod
        if heads != heads_offline:
            print(f"  heads: Nanopub Query {len(heads)}, offline record {len(heads_offline)} (using the query)")
        return len(prod), len(heads), blinded_signed, primary_unpub_signed
    except Exception as ex:                                     # noqa: BLE001
        print(f"  head resolution unavailable ({ex}); using the offline record")
        return len(prod), len(heads_offline), blinded_signed, primary_unpub_signed


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(HERE / "build"))
    ap.add_argument("--no-pdf", action="store_true")
    ap.add_argument("--counts", help="override 'N_NANOPUBS,N_HEADS' (the manifest-derived count is reported for comparison)")
    args = ap.parse_args(argv)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    rubric = yaml.safe_load(open(RUBRIC))
    audits = load_audits()
    latest_active, latest_superseded, hist_primary, hist_blinded, hist_all = collect(audits, rubric)
    n_pub_auto, n_heads_auto, n_blinded_np, n_prim_unpub_np = nanopub_counts()
    if args.counts:
        n_pub, n_heads = (int(x) for x in args.counts.split(","))
    else:
        n_pub, n_heads = n_pub_auto, n_heads_auto
    commit = sh("git", "rev-parse", "--short", "HEAD")

    subprocess.run([sys.executable, str(HERE / "draw_pipeline.py"), "--out", str(out / "pipeline.png")], check=True,
                   env={**os.environ, "MPLBACKEND": "Agg"})

    n_primary_audits = sum(1 for a in audits if a["audit"].get("role") != "second_assessor")
    n_blinded_audits = sum(1 for a in audits if a["audit"].get("role") == "second_assessor")
    n_total_audits = len(audits)

    text = (HERE / "report.md").read_text()
    fill = {
        "{{DATE}}": dt.date.today().isoformat(),
        "{{COMMIT}}": commit,
        "{{RUBRIC_TABLE}}": rubric_table(rubric),
        "{{SCORE_TABLE}}": score_table(latest_active, latest_superseded, rubric),
        "{{DECOMPOSITION_TABLE}}": decomposition_table(latest_active),
        "{{BLINDED_TABLE}}": blinded_table(latest_active, hist_blinded),
        "{{HISTORY_TABLE}}": history_table(hist_all),
        "{{N_SCRIPTS_TOTAL}}": "18",
        "{{N_SCRIPTS_ACTIVE}}": str(len(latest_active)),
        "{{N_SCRIPTS_EVALUATED}}": str(len(latest_active) + len(latest_superseded)),
        "{{N_SCRIPTS_UNEVALUATED}}": "5",
        "{{N_PRIMARY_AUDITS}}": str(n_primary_audits),
        "{{N_BLINDED_AUDITS}}": str(n_blinded_audits),
        "{{N_PRIMARY_PASSES}}": str(len(hist_primary)),
        "{{N_BLINDED_PASSES}}": str(len(hist_blinded)),
        "{{N_TOTAL_AUDITS}}": str(n_total_audits),
        "{{N_TOTAL_PASSES}}": str(len(hist_all)),
        "{{N_EVALUATED}}": str(len(hist_primary)),
        "{{N_NANOPUBS}}": str(n_pub),
        "{{N_HEADS}}": str(n_heads),
        "{{N_BLINDED_NANOPUBS}}": str(n_blinded_np),
        "{{N_PRIMARY_UNPUB_NANOPUBS}}": str(n_prim_unpub_np),
    }
    for k, v in fill.items():
        text = text.replace(k, v)
    left = [l for l in text.splitlines() if "{{" in l]
    if left:
        sys.exit(f"unfilled placeholders: {left}")
    md = out / "MorphoEval-Reproducibility-Report.md"
    md.write_text(text)
    print(f"wrote {md}  ({len(hist_primary)} primary passes + {len(hist_blinded)} blinded passes across {n_total_audits} audit files; "
          f"{len(latest_active)} active scripts + {len(latest_superseded)} superseded; {n_pub} prod nanopubs, {n_heads} heads, {n_blinded_np} signed 2nd-assessor nanopubs)")

    if args.no_pdf:
        return
    if not shutil.which("pandoc"):
        sys.exit("pandoc not found (brew install pandoc; also needs xelatex)")
    pdf = out / "MorphoEval-Reproducibility-Report.pdf"
    cmd = ["pandoc", str(md), "-o", str(pdf), "--pdf-engine=xelatex", "--resource-path", str(out),
           "-V", "geometry:margin=2.2cm", "-V", "fontsize=10pt", "-V", "colorlinks=true", "-V", "linkcolor=NavyBlue",
           "-V", "urlcolor=NavyBlue", "-V", "mainfont=Arial Unicode MS", "-V", "monofont=Menlo", "-V", "monofontoptions=Scale=0.82",
           "--number-sections", "--toc", "--toc-depth=2",
           "-H", str(HERE / "header.tex")]
    subprocess.run(cmd, check=True, cwd=out)
    print(f"wrote {pdf}  ({pdf.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
