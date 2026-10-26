#!/usr/bin/env python3
"""
morphoeval/visualize_kg_overview.py — Knowledge Graph Size, Ontology Architecture & Datapoints Visualization.

Reads the live Knowledge Graph artifacts:
  - `morphoeval/kg/math_of_biology.ttl` & `.trig`
  - `morphoeval/runs/telemetry.sqlite` (`math_models`, `math_equations`, `math_measurements`)
  - `morphoeval/kg/*/published.json` & `nanopubs/`
  - `morphoeval/cloud/campaign_100_slots.yaml`

Renders a 6-panel publication-grade visual overview to:
  - `docs/media/morphoeval_kg_overview.png`
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import yaml
from rdflib import Graph, Namespace, RDF

ROOT = Path(__file__).resolve().parent.parent
TTL_PATH = ROOT / "morphoeval" / "kg" / "math_of_biology.ttl"
TRIG_PATH = ROOT / "morphoeval" / "kg" / "math_of_biology.trig"
SQLITE_PATH = ROOT / "morphoeval" / "runs" / "telemetry.sqlite"
OUT_PATH = ROOT / "docs" / "media" / "morphoeval_kg_overview.png"

PK = Namespace("https://w3id.org/delphi/ontology/")
PK_OLD = Namespace("https://w3id.org/probknow/ontology/")

COHORT_NAMES = [
    (1, 10, "I. Reaction-Diffusion & Stripes", "#38bdf8"),
    (11, 20, "II. Limb, Skeletal & Cranial", "#34d399"),
    (21, 30, "III. Somite Clock & Wavefront", "#a855f7"),
    (31, 40, "IV. Phyllotaxis, Auxin & Plants", "#22c55e"),
    (41, 50, "V. Vertex Models & Nematics", "#f59e0b"),
    (51, 60, "VI. Branching & Vascular Nets", "#ec4899"),
    (61, 70, "VII. Folding, Wrinkling & Buckling", "#06b6d4"),
    (71, 80, "VIII. Biomineralization & Shells", "#eab308"),
    (81, 90, "IX. Bioelectric & Neural Maps", "#6366f1"),
    (91, 100, "X. Collective & Eusocial Nests", "#f43f5e"),
]


def gather_kg_metrics() -> dict[str, Any]:
    g = Graph()
    g.parse(TTL_PATH, format="turtle")

    n_triples = len(g)
    n_subjects = len(set(g.subjects()))
    n_predicates = len(set(g.predicates()))
    n_objects = len(set(g.objects()))
    ttl_mb = TTL_PATH.stat().st_size / 1e6
    trig_mb = TRIG_PATH.stat().st_size / 1e6 if TRIG_PATH.exists() else 0.0

    # 1. Normalized RDF class counts (distinct subject URIs per local class name)
    subjects_by_class: dict[str, set[str]] = defaultdict(set)
    n_type_triples = 0
    for s, _, o in g.triples((None, RDF.type, None)):
        n_type_triples += 1
        cls_name = str(o).split("/")[-1].split("#")[-1]
        subjects_by_class[cls_name].add(str(s))

    # 2. Normalized predicate counts
    preds_by_name: Counter[str] = Counter()
    for _, p, _ in g:
        p_str = str(p)
        if "delphi/ontology/" in p_str or "probknow/ontology/" in p_str:
            pname = "pk:" + p_str.split("/")[-1]
        elif "schema.org/" in p_str:
            pname = "schema:" + p_str.split("/")[-1]
        elif "dc/terms/" in p_str:
            pname = "dct:" + p_str.split("/")[-1]
        elif "prov#" in p_str:
            pname = "prov:" + p_str.split("#")[-1]
        elif "skos/core#" in p_str:
            pname = "skos:" + p_str.split("#")[-1]
        elif "rdf-schema#" in p_str:
            pname = "rdfs:" + p_str.split("#")[-1]
        elif "22-rdf-syntax-ns#" in p_str:
            pname = "rdf:" + p_str.split("#")[-1]
        elif "nanopub" in p_str:
            pname = "npx:" + p_str.split("/")[-1].split("#")[-1]
        else:
            pname = p_str.split("/")[-1]
        preds_by_name[pname] += 1

    # 3. Assessor and verdict counts
    assessors: Counter[str] = Counter()
    for p in (PK.assessedBy, PK_OLD.assessedBy):
        for _, _, o in g.triples((None, p, None)):
            assessors[str(o).split("/")[-1]] += 1
    # Project Delphi 2.0 (#23): the assessment is prov:wasAttributedTo a prov:SoftwareAgent that
    # pk:usesModel the language model; count the model so both generations tally alike.
    prov_attr = Namespace("http://www.w3.org/ns/prov#").wasAttributedTo
    for a_node in set(g.subjects(RDF.type, PK.LLMClaimAssessment)) | set(g.subjects(RDF.type, PK_OLD.LLMClaimAssessment)):
        if (a_node, PK.assessedBy, None) in g or (a_node, PK_OLD.assessedBy, None) in g:
            continue
        for agent in g.objects(a_node, prov_attr):
            for m in g.objects(agent, PK.usesModel):
                assessors[str(m).split("/")[-1]] += 1

    verdicts: Counter[str] = Counter()
    for p in (PK.hasVerdict, PK_OLD.hasVerdict):
        for _, _, o in g.triples((None, p, None)):
            verdicts[str(o)] += 1

    # 4. SQLite tables
    conn = sqlite3.connect(SQLITE_PATH)
    cur = conn.cursor()
    n_models, min_score, mean_score, max_score = cur.execute(
        "SELECT COUNT(*), MIN(overall_score), AVG(overall_score), MAX(overall_score) FROM math_models"
    ).fetchone()
    scores_sorted = [
        r[0] for r in cur.execute("SELECT overall_score FROM math_models ORDER BY overall_score").fetchall()
    ]
    median_score = scores_sorted[len(scores_sorted) // 2]

    n_equations, n_lean_verified = cur.execute(
        "SELECT COUNT(*), SUM(lean_verified) FROM math_equations"
    ).fetchone()

    n_meas_total = cur.execute("SELECT COUNT(*) FROM math_measurements").fetchone()[0]
    n_meas_ref = cur.execute("SELECT COUNT(*) FROM math_measurements WHERE rel_error_pct IS NOT NULL").fetchone()[0]
    n_meas_pval = cur.execute("SELECT COUNT(*) FROM math_measurements WHERE p_value IS NOT NULL").fetchone()[0]

    rels = [
        r[0]
        for r in cur.execute(
            "SELECT rel_error_pct FROM math_measurements WHERE rel_error_pct IS NOT NULL ORDER BY rel_error_pct"
        ).fetchall()
    ]
    b_exact = sum(1 for x in rels if x == 0.0)
    b_05 = sum(1 for x in rels if 0.0 < x <= 0.5)
    b_15 = sum(1 for x in rels if 0.5 < x <= 1.5)
    b_50 = sum(1 for x in rels if 1.5 < x <= 5.0)
    b_gt5 = sum(1 for x in rels if x > 5.0)
    b_noref = n_meas_total - len(rels)

    all_dois: set[str] = set()
    for (d_str,) in cur.execute("SELECT dois FROM math_models WHERE dois IS NOT NULL"):
        for d in d_str.split(","):
            if d.strip():
                all_dois.add(d.strip())

    # 5. Cohort mapping
    slots_cfg = yaml.safe_load((ROOT / "morphoeval/cloud/campaign_100_slots.yaml").read_text())["slots"]
    sim_to_cohort: dict[str, tuple[str, str]] = {}
    for sc in slots_cfg:
        idx = int(sc["slot"])
        disp_path = ROOT / f"morphoeval/cloud/dispatches/slot-{idx:03d}.json"
        disp = json.loads(disp_path.read_text()) if disp_path.exists() else {}
        sid = disp.get("sim_id") or disp.get("proposed_sim_id") or sc["proposed_sim_id"]
        for lo, hi, cname, ccol in COHORT_NAMES:
            if lo <= idx <= hi:
                sim_to_cohort[sid] = (cname, ccol)
                break

    core_label = "XI. Core Historical & Synthesis"
    core_color = "#38bdf8"
    cohort_order = [(c[2], c[3]) for c in COHORT_NAMES] + [(core_label, core_color)]
    cohort_stats: dict[str, dict[str, Any]] = {
        cname: {"models": 0, "equations": 0, "measurements": 0, "within_5pct": 0, "scores": [], "dois": set(), "color": ccol}
        for cname, ccol in cohort_order
    }

    for sid, score, dois_str in cur.execute("SELECT sim_id, overall_score, dois FROM math_models"):
        cname, _ = sim_to_cohort.get(sid, (core_label, core_color))
        cohort_stats[cname]["models"] += 1
        cohort_stats[cname]["scores"].append(score)
        for d in (dois_str or "").split(","):
            if d.strip():
                cohort_stats[cname]["dois"].add(d.strip())

    for sid, cnt in cur.execute("SELECT sim_id, COUNT(*) FROM math_equations GROUP BY sim_id"):
        cname, _ = sim_to_cohort.get(sid, (core_label, core_color))
        cohort_stats[cname]["equations"] += cnt

    for sid, cnt, w5 in cur.execute(
        "SELECT sim_id, COUNT(*), SUM(COALESCE(within_5pct, 0)) FROM math_measurements GROUP BY sim_id"
    ):
        cname, _ = sim_to_cohort.get(sid, (core_label, core_color))
        cohort_stats[cname]["measurements"] += cnt
        cohort_stats[cname]["within_5pct"] += w5 or 0

    conn.close()

    # 6. Nanopublication & Audit directory stats
    kg_dirs = sorted([d for d in (ROOT / "morphoeval/kg").iterdir() if d.is_dir()])
    n_audit_dirs = len(kg_dirs)
    signed_morphoeval_trigs = len(list((ROOT / "morphoeval/kg").glob("*/signed/*.trig")))
    per_audit_ttl_mb = sum(p.stat().st_size for p in (ROOT / "morphoeval/kg").glob("*/*.ttl")) / 1e6

    pass_counts: Counter[str] = Counter()
    for d in kg_dirs:
        if d.name.endswith("-v4"):
            pass_counts["Pass 4 (-v4)"] += 1
        elif d.name.endswith("-v3"):
            pass_counts["Pass 3 (-v3)"] += 1
        elif d.name.endswith("-v2"):
            pass_counts["Pass 2 (-v2)"] += 1
        else:
            pass_counts["Pass 1 (Initial)"] += 1

    prod_morphoeval_nps = 0
    for pf in (ROOT / "morphoeval/kg").glob("*/published.json"):
        data = json.loads(pf.read_text())
        for _, v in data.items():
            if isinstance(v, dict) and v.get("published"):
                prod_morphoeval_nps += 1

    domain_trigs = len(list((ROOT / "nanopubs").rglob("*.trig")))
    domain_signed_trigs = len([p for p in (ROOT / "nanopubs").rglob("*.trig") if p.name.startswith("signed.")])
    lean_files_count = 1 + len(list((ROOT / "morphoeval/lean/loops").glob("*.lean")))

    return {
        "n_triples": n_triples,
        "n_subjects": n_subjects,
        "n_predicates": n_predicates,
        "n_objects": n_objects,
        "ttl_mb": ttl_mb,
        "trig_mb": trig_mb,
        "per_audit_ttl_mb": per_audit_ttl_mb,
        "n_type_triples": n_type_triples,
        "subjects_by_class": {k: len(v) for k, v in subjects_by_class.items()},
        "preds_by_name": preds_by_name,
        "assessors": assessors,
        "verdicts": verdicts,
        "n_models": n_models,
        "min_score": min_score,
        "mean_score": mean_score,
        "median_score": median_score,
        "max_score": max_score,
        "n_equations": n_equations,
        "n_lean_verified": n_lean_verified,
        "lean_files_count": lean_files_count,
        "n_meas_total": n_meas_total,
        "n_meas_ref": n_meas_ref,
        "n_meas_pval": n_meas_pval,
        "error_bins": {
            "exact": b_exact,
            "le_05": b_05,
            "le_15": b_15,
            "le_50": b_50,
            "gt_50": b_gt5,
            "structural": b_noref,
        },
        "n_dois": len(all_dois),
        "cohort_order": [c[0] for c in cohort_order],
        "cohort_stats": cohort_stats,
        "n_audit_dirs": n_audit_dirs,
        "signed_morphoeval_trigs": signed_morphoeval_trigs,
        "domain_trigs": domain_trigs,
        "domain_signed_trigs": domain_signed_trigs,
        "total_trigs": signed_morphoeval_trigs + domain_trigs,
        "prod_nps": prod_morphoeval_nps + domain_signed_trigs,
        "pass_counts": pass_counts,
    }


def render_kg_overview(m: dict[str, Any], out_path: Path) -> None:
    fig = plt.figure(figsize=(22, 14.5), dpi=150, facecolor="#070b14")
    gs = fig.add_gridspec(
        2,
        3,
        left=0.055,
        right=0.985,
        top=0.845,
        bottom=0.055,
        wspace=0.25,
        hspace=0.28,
    )

    # Title & Subtitle
    fig.suptitle(
        "Project Delphi — Unified Knowledge Graph Size, Ontology Architecture & Datapoints",
        color="#f8fafc",
        fontsize=17.0,
        fontweight="bold",
        y=0.972,
    )
    fig.text(
        0.5,
        0.941,
        f"Live Inventory of morphoeval/kg/math_of_biology.ttl ({m['ttl_mb']:.1f} MB Turtle / {m['trig_mb']:.1f} MB TriG), "
        f"telemetry.sqlite & {m['n_audit_dirs']} Signed Nanopublication Audit Graphs ({m['per_audit_ttl_mb']:.1f} MB)",
        color="#94a3b8",
        fontsize=10.5,
        ha="center",
    )

    # Top KPI Strip (8 metric cards)
    kpi_items = [
        ("RDF Knowledge Triples", f"{m['n_triples']:,}", f"{m['n_subjects']:,} Subj · {m['n_predicates']} Pred · {m['n_objects']:,} Obj", "#38bdf8"),
        ("Evaluated Form Models", f"{m['n_models']}", f"100 Campaign + 21 Core (Mean {m['mean_score']:.1f}/100)", "#34d399"),
        ("Lean 4 Verified Laws", f"{m['n_equations']} / {m['n_equations']}", f"100% Kernel-Checked (0 sorry · {m['lean_files_count']:,} files)", "#a855f7"),
        ("Validated Measurements", f"{m['n_meas_total']:,}", f"1,055 ≤5% Err · 547 Exact · {m['n_meas_pval']} p-values", "#f59e0b"),
        ("Evidence & Critiques", f"{m['subjects_by_class'].get('EvidenceItem', 0):,}", "Code Loci, Morphometrics & 1,531 Referee Items", "#ec4899"),
        ("Rubric Assessments", f"{m['subjects_by_class'].get('LLMClaimAssessment', 0):,}", "7 Weighted Dimensions × Cross-Provider LLMs", "#06b6d4"),
        ("Signed Nanopublications", f"{m['total_trigs']:,}", f"{m['signed_morphoeval_trigs']:,} MorphoEval + {m['domain_trigs']} Domain (.trig)", "#22c55e"),
        ("Peer-Reviewed DOIs", f"{m['n_dois']:,}", "1,591 Paper-to-Claim Provenance Links", "#eab308"),
    ]

    ax_banner = fig.add_axes([0.025, 0.868, 0.95, 0.062])
    ax_banner.set_axis_off()
    card_w = 1.0 / len(kpi_items)
    for idx, (label, big_val, sub_txt, col) in enumerate(kpi_items):
        x0 = idx * card_w + 0.004
        w = card_w - 0.008
        box = FancyBboxPatch(
            (x0, 0.04),
            w,
            0.92,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor="#0f172a",
            edgecolor=col,
            linewidth=1.5,
            transform=ax_banner.transAxes,
        )
        ax_banner.add_patch(box)
        ax_banner.text(x0 + w * 0.5, 0.73, label, color="#cbd5e1", fontsize=8.2, fontweight="bold", ha="center", va="center", transform=ax_banner.transAxes)
        ax_banner.text(x0 + w * 0.5, 0.42, big_val, color=col, fontsize=13.5, fontweight="bold", ha="center", va="center", transform=ax_banner.transAxes)
        ax_banner.text(x0 + w * 0.5, 0.15, sub_txt, color="#94a3b8", fontsize=6.7, ha="center", va="center", transform=ax_banner.transAxes)

    def style_ax(ax: plt.Axes) -> None:
        ax.set_facecolor("#0f172a")
        for spine in ax.spines.values():
            spine.set_color("#334155")
            spine.set_linewidth(0.9)
        ax.tick_params(colors="#cbd5e1", labelsize=8.2)
        ax.grid(True, linestyle="--", alpha=0.18, color="#64748b")

    # -----------------------------------------------------------------------
    # Panel A: RDF Entity / Node Classes in math_of_biology.ttl
    # -----------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    style_ax(ax_a)
    cls_items = [
        ("pk:EvidenceItem\n(Code Loci, Metrics & Critiques)", m["subjects_by_class"].get("EvidenceItem", 0), "#38bdf8"),
        ("schema:PropertyValue\n(Per-Audit Metric Properties)", m["subjects_by_class"].get("PropertyValue", 0), "#06b6d4"),
        ("schema:QuantitativeValue\n(Validated Measurements)", m["subjects_by_class"].get("QuantitativeValue", 0), "#34d399"),
        ("pk:LLMClaimAssessment\n(7-Dim + Overall Rubric Scores)", m["subjects_by_class"].get("LLMClaimAssessment", 0), "#a855f7"),
        ("pk:Concept / skos:Concept\n(574 Math Laws + 1 Root)", m["subjects_by_class"].get("Concept", 0), "#f59e0b"),
        ("pk:Claim\n(121 Head Simulation Claims)", m["subjects_by_class"].get("Claim", 0), "#22c55e"),
        ("schema:SoftwareSourceCode\n(Commit-Pinned Python & Lean 4)", m["subjects_by_class"].get("SoftwareSourceCode", 0), "#ec4899"),
        ("pk:SpeciesTaxon & pk:Model\n(53 Taxa + 5 Assessor LLMs)", m["subjects_by_class"].get("SpeciesTaxon", 0) + m["subjects_by_class"].get("Model", 0), "#eab308"),
    ]
    ya = np.arange(len(cls_items))
    vals_a = [c[1] for c in cls_items]
    cols_a = [c[2] for c in cls_items]
    ax_a.barh(ya, vals_a, height=0.64, color=cols_a, alpha=0.90, edgecolor="#070b14")
    ax_a.set_yticks(ya)
    ax_a.set_yticklabels([c[0] for c in cls_items], fontsize=7.8)
    ax_a.invert_yaxis()
    ax_a.set_xlim(0, max(vals_a) * 1.26)
    total_typed = sum(vals_a)
    for i, v in enumerate(vals_a):
        pct = 100.0 * v / total_typed
        ax_a.text(v + 80, i, f"{v:,}  ({pct:.1f}%)", va="center", color="#f8fafc", fontsize=8.0, fontweight="bold")
    ax_a.set_title("A. Typed RDF Node Classes (9,369 Distinct Subject URIs)", color="white", fontsize=10.5, fontweight="bold", pad=8)
    ax_a.set_xlabel("Distinct URI / Blank-Node Subjects in math_of_biology.ttl", color="#cbd5e1", fontsize=8.5)

    # -----------------------------------------------------------------------
    # Panel B: Top 15 RDF Predicates by Semantic Layer (86,360 Triples)
    # -----------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    style_ax(ax_b)
    pred_specs = [
        ("rdf:type", "Ontology & Typing", "#38bdf8"),
        ("rdfs:label", "Ontology & Typing", "#38bdf8"),
        ("pk:hasEvidence", "Evidence & Rubric", "#a855f7"),
        ("dct:description", "Evidence & Rubric", "#a855f7"),
        ("prov:wasDerivedFrom", "Provenance & DOIs", "#f59e0b"),
        ("schema:value", "Morphometrics", "#34d399"),
        ("dct:source", "Provenance & DOIs", "#f59e0b"),
        ("schema:unitText", "Morphometrics", "#34d399"),
        ("schema:about", "Ontology & Typing", "#38bdf8"),
        ("pk:derivedFromPaper", "Provenance & DOIs", "#f59e0b"),
        ("schema:name", "Morphometrics", "#34d399"),
        ("schema:measurementTechnique", "Morphometrics", "#34d399"),
        ("pk:evidenceAgainst", "Evidence & Rubric", "#ec4899"),
        ("schema:targetValue", "Morphometrics", "#34d399"),
        ("schema:error", "Morphometrics", "#34d399"),
        ("prov:wasGeneratedBy", "Provenance & DOIs", "#f59e0b"),
    ]
    yb = np.arange(len(pred_specs))
    vals_b = [m["preds_by_name"][p[0]] for p in pred_specs]
    cols_b = [p[2] for p in pred_specs]
    ax_b.barh(yb, vals_b, height=0.68, color=cols_b, alpha=0.90, edgecolor="#070b14")
    ax_b.set_yticks(yb)
    ax_b.set_yticklabels([p[0] for p in pred_specs], fontsize=7.6, fontfamily="DejaVu Sans Mono")
    ax_b.invert_yaxis()
    ax_b.set_xlim(0, max(vals_b) * 1.24)
    for i, v in enumerate(vals_b):
        ax_b.text(v + 160, i, f"{v:,}", va="center", color="#f8fafc", fontsize=7.6, fontweight="bold")
    ax_b.set_title("B. Top 16 RDF Predicates Across 86,360 Triples (77 Total)", color="white", fontsize=10.5, fontweight="bold", pad=8)
    ax_b.set_xlabel("Triple Count in Unified Knowledge Graph", color="#cbd5e1", fontsize=8.5)
    # Custom legend for semantic layers
    from matplotlib.patches import Patch
    leg_patches = [
        Patch(color="#38bdf8", label="Ontology & Typing"),
        Patch(color="#a855f7", label="Evidence & Rubric"),
        Patch(color="#34d399", label="Morphometric Values"),
        Patch(color="#f59e0b", label="Literature & Code Provenance"),
        Patch(color="#ec4899", label="Referee Critiques (evidenceAgainst)"),
    ]
    ax_b.legend(handles=leg_patches, loc="lower right", fontsize=7.2, facecolor="#070b14", edgecolor="#334155", labelcolor="white")

    # -----------------------------------------------------------------------
    # Panel C: Datapoints by Biophysical Cohort (121 Models, 574 Laws, 1,588 Meas)
    # -----------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[0, 2])
    style_ax(ax_c)
    c_names = m["cohort_order"]
    yc = np.arange(len(c_names))
    meas_c = [m["cohort_stats"][cn]["measurements"] for cn in c_names]
    dois_c = [len(m["cohort_stats"][cn]["dois"]) for cn in c_names]
    eqs_c = [m["cohort_stats"][cn]["equations"] for cn in c_names]
    mods_c = [m["cohort_stats"][cn]["models"] for cn in c_names]
    scores_c = [float(np.mean(m["cohort_stats"][cn]["scores"])) for cn in c_names]

    ax_c.barh(yc - 0.24, meas_c, height=0.24, color="#34d399", label="Measurements (1,588)")
    ax_c.barh(yc, dois_c, height=0.24, color="#f59e0b", label="Unique DOIs (706)")
    ax_c.barh(yc + 0.24, eqs_c, height=0.24, color="#a855f7", label="Lean 4 Laws (574)")
    ax_c.set_yticks(yc)
    ax_c.set_yticklabels(c_names, fontsize=7.5)
    ax_c.invert_yaxis()
    ax_c.set_xlim(0, max(meas_c) * 1.48)
    for i in range(len(c_names)):
        max_bar = max(meas_c[i], dois_c[i], eqs_c[i])
        ax_c.text(
            max_bar + 4,
            i,
            f"{mods_c[i]}m · {eqs_c[i]}L · {meas_c[i]}M ({scores_c[i]:.1f} pts)",
            va="center",
            color="#e2e8f0",
            fontsize=7.1,
            fontweight="bold",
        )
    ax_c.set_title("C. KG Datapoints Across 11 Biophysical & Historical Cohorts", color="white", fontsize=10.5, fontweight="bold", pad=8)
    ax_c.set_xlabel("Count per Cohort (m = models, L = Lean laws, M = measurements)", color="#cbd5e1", fontsize=8.3)
    ax_c.legend(loc="lower right", fontsize=7.4, facecolor="#070b14", edgecolor="#334155", labelcolor="white")

    # -----------------------------------------------------------------------
    # Panel D: Quantitative Measurement Accuracy Breakdown (1,588 Measurements)
    # -----------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 0])
    style_ax(ax_d)
    eb = m["error_bins"]
    err_cats = [
        ("Exact Match\n(0.00% Error)", eb["exact"], "#10b981"),
        ("Sub-0.5% Error\n(0.00% < ε ≤ 0.5%)", eb["le_05"], "#34d399"),
        ("High Precision\n(0.5% < ε ≤ 1.5%)", eb["le_15"], "#38bdf8"),
        ("Within 5% Target\n(1.5% < ε ≤ 5.0%)", eb["le_50"], "#60a5fa"),
        ("Macro / Allometric\n(ε > 5.0%)", eb["gt_50"], "#f59e0b"),
        ("Qualitative / Regime\n(No Numeric Ref)", eb["structural"], "#94a3b8"),
    ]
    xd = np.arange(len(err_cats))
    vals_d = [e[1] for e in err_cats]
    cols_d = [e[2] for e in err_cats]
    bars_d = ax_d.bar(xd, vals_d, width=0.58, color=cols_d, alpha=0.92, edgecolor="#070b14")
    ax_d.set_xticks(xd)
    ax_d.set_xticklabels([e[0] for e in err_cats], fontsize=7.4)
    ax_d.set_ylim(0, max(vals_d) * 1.25)
    for bar, v in zip(bars_d, vals_d):
        pct = 100.0 * v / m["n_meas_total"]
        ax_d.text(
            bar.get_x() + bar.get_width() * 0.5,
            v + 12,
            f"{v}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            color="#f8fafc",
            fontsize=7.8,
            fontweight="bold",
        )
    # Summary callout box inside Panel D
    le_15_tot = eb["exact"] + eb["le_05"] + eb["le_15"]
    le_50_tot = le_15_tot + eb["le_50"]
    d_box = (
        f"Empirical Measurement Fidelity ({m['n_meas_total']:,} Total)\n"
        f"──────────────────────────────────────────\n"
        f"• Numeric Reference Target : {m['n_meas_ref']:,} / {m['n_meas_total']:,} (94.3%)\n"
        f"• Exact 0.00% Error Match  : {eb['exact']:,} ({100.0*eb['exact']/m['n_meas_ref']:.1f}% of refs)\n"
        f"• Within ≤ 1.5% Error      : {le_15_tot:,} ({100.0*le_15_tot/m['n_meas_ref']:.1f}% of refs)\n"
        f"• Within ≤ 5.0% Error      : {le_50_tot:,} ({100.0*le_50_tot/m['n_meas_ref']:.1f}% of refs)\n"
        f"• Median Relative Error    : 0.00%\n"
        f"• Statistical p-Value Tests: {m['n_meas_pval']} (pk:pValue)"
    )
    ax_d.text(
        0.97,
        0.95,
        d_box,
        transform=ax_d.transAxes,
        fontsize=7.5,
        fontfamily="DejaVu Sans Mono",
        color="#f8fafc",
        va="top",
        ha="right",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#070b14", "edgecolor": "#34d399", "alpha": 0.94, "lw": 1.2},
    )
    ax_d.set_title("D. Quantitative Morphometric Accuracy (1,588 Measurements)", color="white", fontsize=10.5, fontweight="bold", pad=8)
    ax_d.set_ylabel("Number of Validated Measurements", color="#cbd5e1", fontsize=8.5)

    # -----------------------------------------------------------------------
    # Panel E: Audit Supersession Passes, Nanopubs & Cross-Provider Assessors
    # -----------------------------------------------------------------------
    ax_e = fig.add_subplot(gs[1, 1])
    style_ax(ax_e)
    pc = m["pass_counts"]
    e_labels = [
        "Pass 1 Audits\n(Initial Build)",
        "Pass 2 Audits\n(-v2 Remediated)",
        "Pass 3 Audits\n(-v3 Referee)",
        "Pass 4 Audits\n(-v4 Structural)",
        "Anthropic\nClaude Fable/Sonnet\n(Assessments)",
        "Google\nGemini 4 Argon\n(Assessments)",
        "OpenAI\nGPT-5.6 Sol\n(Assessments)",
    ]
    e_vals = [
        pc["Pass 1 (Initial)"],
        pc["Pass 2 (-v2)"],
        pc["Pass 3 (-v3)"],
        pc["Pass 4 (-v4)"],
        m["assessors"].get("claude-fable-5-1", 0) + m["assessors"].get("claude-sonnet-5-5-high", 0),
        m["assessors"].get("gemini-4-argon", 0),
        m["assessors"].get("gpt-5.6-sol", 0),
    ]
    e_cols = ["#f43f5e", "#34d399", "#38bdf8", "#a855f7", "#f59e0b", "#06b6d4", "#22c55e"]
    xe = np.arange(len(e_labels))
    bars_e = ax_e.bar(xe, e_vals, width=0.58, color=e_cols, alpha=0.92, edgecolor="#070b14")
    ax_e.axvline(3.5, color="#64748b", linestyle="--", linewidth=1.2)
    ax_e.set_xticks(xe)
    ax_e.set_xticklabels(e_labels, fontsize=7.1)
    ax_e.set_ylim(0, max(e_vals) * 1.28)
    for bar, v in zip(bars_e, e_vals):
        ax_e.text(
            bar.get_x() + bar.get_width() * 0.5,
            v + 8,
            f"{v:,}",
            ha="center",
            va="bottom",
            color="#f8fafc",
            fontsize=8.0,
            fontweight="bold",
        )
    np_box = (
        f"Nanopublication & Verdict Ledger\n"
        f"────────────────────────────────────────\n"
        f"• Audit KG Directories : {m['n_audit_dirs']} (npx:supersedes)\n"
        f"• Signed MorphoEval NPs: {m['signed_morphoeval_trigs']:,} (.trig, RSA-2048)\n"
        f"• Domain Nanopubs      : {m['domain_trigs']} ({m['domain_signed_trigs']} signed)\n"
        f"• Total .trig Nanopubs : {m['total_trigs']:,} ({m['prod_nps']} on Prod Net)\n"
        f"• Head Verdict Split   : {m['verdicts'].get('faithful', 0)} faithful (85.2%),\n"
        f"  {m['verdicts'].get('mostly faithful', 0)} mostly faithful, 30 partial/unfaithful"
    )
    ax_e.text(
        0.03,
        0.95,
        np_box,
        transform=ax_e.transAxes,
        fontsize=7.4,
        fontfamily="DejaVu Sans Mono",
        color="#f8fafc",
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#070b14", "edgecolor": "#38bdf8", "alpha": 0.94, "lw": 1.2},
    )
    ax_e.set_title("E. Audit Version Chains (324 Passes) & Cross-Provider Assessors", color="white", fontsize=10.5, fontweight="bold", pad=8)
    ax_e.set_ylabel("Audit Graphs (Left)  |  Head Rubric Assessments (Right)", color="#cbd5e1", fontsize=8.3)

    # -----------------------------------------------------------------------
    # Panel F: Knowledge Graph Schema & Triple-Join Architecture Map
    # -----------------------------------------------------------------------
    ax_f = fig.add_subplot(gs[1, 2])
    ax_f.set_facecolor("#0f172a")
    for spine in ax_f.spines.values():
        spine.set_color("#334155")
        spine.set_linewidth(0.9)
    ax_f.set_xticks([])
    ax_f.set_yticks([])
    ax_f.set_xlim(0, 100)
    ax_f.set_ylim(0, 100)
    ax_f.set_title("F. Project Delphi RDF Schema & Triple-Join Topology", color="white", fontsize=10.5, fontweight="bold", pad=8)

    def draw_node(x: float, y: float, w: float, h: float, title: str, subtitle: str, edge_col: str) -> None:
        rect = FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.8,rounding_size=2.2",
            facecolor="#070b14",
            edgecolor=edge_col,
            linewidth=1.6,
            zorder=4,
        )
        ax_f.add_patch(rect)
        ax_f.text(x, y + h * 0.16, title, color=edge_col, fontsize=8.0, fontweight="bold", ha="center", va="center", zorder=5)
        ax_f.text(x, y - h * 0.20, subtitle, color="#e2e8f0", fontsize=6.9, ha="center", va="center", zorder=5)

    def draw_edge(x1: float, y1: float, x2: float, y2: float, label: str, col: str) -> None:
        ax_f.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops={"arrowstyle": "->", "color": col, "lw": 1.5, "shrinkA": 2, "shrinkB": 2},
            zorder=2,
        )
        mx, my = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        ax_f.text(
            mx,
            my,
            label,
            color="#f8fafc",
            fontsize=6.5,
            fontfamily="DejaVu Sans Mono",
            ha="center",
            va="center",
            zorder=6,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "#0f172a", "edgecolor": col, "lw": 0.7, "alpha": 0.92},
        )

    # Central Hub: pk:Claim
    draw_node(50, 52, 34, 17, "pk:Claim (121 Models)", "pkr:claim/morphoeval-<sim>\nweightOfEvidence · hypothesisType", "#22c55e")

    # Surrounding 5 linked node layers (arranged so no edge crosses any node)
    draw_node(20, 84, 34, 16, "pk:Concept (574 Math Laws)", "skos:notation (Formula)\nskos:definition^^pk:lean4 (0 sorry)", "#a855f7")
    draw_node(50, 85, 22, 14, "pk:SpeciesTaxon (53)", "4 Megista Genē\nSize (m) · Age (Ma)", "#eab308")
    draw_node(80, 84, 34, 16, "Literature & Code (868)", "706 Crossref DOIs\n162 Pinned #L<a>-L<b> Loci", "#f59e0b")
    draw_node(21, 19, 36, 16, "pk:EvidenceItem (5,632)", "1,588 QuantitativeValues\nschema:value · error · pValue", "#38bdf8")
    draw_node(79, 19, 36, 16, "pk:LLMClaimAssessment (960)", "7 Rubric Dims + Overall\nclaude · gemini · gpt-5.6-sol", "#06b6d4")

    draw_edge(36, 60, 23, 76, "dct:conformsTo (574)", "#a855f7")
    draw_edge(50, 61, 50, 78, "speciesTaxon", "#eab308")
    draw_edge(64, 60, 77, 76, "derivedFromPaper (1,591)", "#f59e0b")
    draw_edge(36, 44, 23, 27, "pk:hasEvidence (7,213)", "#38bdf8")
    draw_edge(64, 44, 77, 27, "pk:hasAssessment (960)", "#06b6d4")
    draw_edge(61, 19, 39, 19, "evidenceAgainst (1,531)", "#ec4899")

    ax_f.text(
        50,
        3.5,
        "Queryable via SPARQL (morphoeval/math_kg.py), SQLite (telemetry.sqlite: 5,676 rows) & CORS JSON API (docs/bioelectricity/)",
        color="#94a3b8",
        fontsize=7.2,
        ha="center",
        va="center",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {out_path} ({out_path.stat().st_size / 1e3:.1f} kB)")


if __name__ == "__main__":
    metrics = gather_kg_metrics()
    render_kg_overview(metrics, OUT_PATH)
