#!/usr/bin/env python3
"""Build docs/problem-graph.html from the problem-graph canvas data.

Single source of truth is the canvas file
(~/.cursor/projects/<ws>/canvases/problem-graph.canvas.tsx): this script
extracts the AGENTS / NODES / EDGES / DELPHI / CONCORDANCE literals from it
with Node, then injects them as JSON into a self-contained HTML template
(no external dependencies) suitable for GitHub Pages.

Usage: python3 docs/build_problem_graph_html.py [--canvas PATH] [--out PATH]
"""
import argparse, json, subprocess, sys
from pathlib import Path

DEFAULT_CANVAS = Path.home() / ".cursor/projects/Users-jdietz-Documents-GitHub-infinitemorphospace/canvases/problem-graph.canvas.tsx"

def extract_literal(src: str, name: str) -> str:
    i = src.index(f"const {name}")
    i = src.index("=", i) + 1
    i = min(j for j in (src.find("{", i), src.find("[", i)) if j != -1)
    open_c, close_c = ("[", "]") if src[i] == "[" else ("{", "}")
    depth = 0
    for j in range(i, len(src)):
        if src[j] == open_c: depth += 1
        elif src[j] == close_c:
            depth -= 1
            if depth == 0: return src[i:j + 1]
    raise ValueError(name)

def validate(data: dict) -> list[str]:
    """Self-checks so the generator 'knows what it's doing': it refuses to
    emit a page from incoherent data instead of deploying silent nonsense."""
    errs: list[str] = []
    nodes, edges, agents = data["NODES"], data["EDGES"], data["AGENTS"]
    ids = [n["id"] for n in nodes]
    if len(ids) != len(set(ids)):
        errs.append("duplicate node ids")
    for e in edges:
        if e["from"] not in ids or e["to"] not in ids:
            errs.append(f"edge references missing node: {e}")
        if e["agent"] not in agents:
            errs.append(f"edge references unknown agent: {e}")
        if "prob" in e:  # Delphi path-success posterior; never an invented number
            if not (0 <= e["prob"] <= 100):
                errs.append(f"edge {e['from']}->{e['to']}: prob {e['prob']} outside 0-100")
            if e["to"] in data["DELPHI"] and abs(e["prob"] - data["DELPHI"][e["to"]]["pooled"]["posterior"]) > 0.6:
                errs.append(f"edge {e['from']}->{e['to']}: prob {e['prob']} != Delphi pooled posterior "
                            f"{data['DELPHI'][e['to']]['pooled']['posterior']} for its target")
    for n in nodes:
        for np in n.get("nanopubs", []):
            uri = np.get("uri", "")
            if not uri.startswith("https://w3id.org/") or "/RA" not in uri:
                errs.append(f"{n['id']}: nanopub uri not a trusty w3id URI: {uri}")
            if not np.get("label"):
                errs.append(f"{n['id']}: nanopub link without label")
    for pid, prize in data["DELPHI"].items():
        if abs(sum(s["weight"] for s in prize["subgoals"]) - 100) > 1e-9:
            errs.append(f"{pid}: subgoal weights do not sum to 100")
        for agent_key, total_key in (("fable", "fable"), ("gpt", "gpt"), ("gemini", "gemini")):
            weighted = sum(s["weight"] * s[agent_key] for s in prize["subgoals"]) / 100
            if abs(weighted - prize["subgoalTotal"][total_key]) > 0.6:
                errs.append(f"{pid}/{total_key}: weighted subgoals {weighted:.2f} != stated total {prize['subgoalTotal'][total_key]}")
        for row in [*prize["agents"], prize["pooled"]]:
            odds = (row["prior"] / 100) / (1 - row["prior"] / 100) * row["bf"]
            posterior = 100 * odds / (1 + odds)
            if abs(posterior - row["posterior"]) > 0.15:
                errs.append(f"{pid}/{row['agent']}: posterior {row['posterior']} != prior x BF = {posterior:.2f}")
            if abs((row["forDb"] + row["againstDb"]) - row["netDb"]) > 0.51:
                errs.append(f"{pid}/{row['agent']}: forDb + againstDb != netDb")
    for op in OPEN_PROBLEMS:
        if op["node"] not in ids:
            errs.append(f"{op['id']}: references missing graph node {op['node']}")
        if op["delphi"] is not None and op["delphi"] not in data["DELPHI"]:
            errs.append(f"{op['id']}: references missing Delphi prize {op['delphi']}")
    if len({op["id"] for op in OPEN_PROBLEMS}) != len(OPEN_PROBLEMS):
        errs.append("duplicate MorphoSwarm OP ids")
    return errs

# MorphoSwarm work queue: the genuinely open problems, curated against graph
# node ids (validated below). `delphi` links to a Delphi prize for a scored
# viability; without it the problem is honestly NOT SCORED and the swarm's
# first task on it is to Delphi-score it, not to assume a probability.
OPEN_PROBLEMS = [
    {"id": "OP-01", "node": "p1", "delphi": "p1", "title": "Prize 1 — close the transparent-gate loophole (short runs 1≤a_i≤7)",
     "gap": "Defects D1(t)=1 are proved to arrive infinitely often, but not proved to ever land on a closed gate {c_t=1}. Depth-1 right cylinder has positive topological entropy (~0.32 bits/step) on short runs, so no deeper local transducer closes this; it needs a global phase-spreading / mixing invariant.",
     "done": "A proof (ideally Lean-checked) that for all t0, Π and short-run words, some D1(t)=1 lands on {c_t=1} — or a counterexample word, which would be just as valuable."},
    {"id": "OP-02", "node": "p2", "delphi": "p2", "title": "Prize 2 — diagonal-sample decorrelation along j=⌊S/2⌋",
     "gap": "Exact 1/2 density is proved on crystals, but the diagonal sample visits a vanishing 2^-n fraction of each period and no Weyl / exponential-sum discrepancy bound exists along the moving diagonal.",
     "done": "A discrepancy bound for S -> A_S(floor(S/2)) transferring crystal density to the center column, or a proof that such a bound fails."},
    {"id": "OP-03", "node": "p3", "delphi": "p3", "title": "Prize 3 — fixed-seed specialisation transfer",
     "gap": "Multivariate deg(c_T)=2T-1 collapses under specialisation to the unit seed; a uniform Omega(T) bound needs P-vs-EXP-type machinery. Delphi posterior 0.04% — work here should be a barrier result, not another attempt at the bound.",
     "done": "A formal barrier / relativisation result for the fixed-seed problem, or a specified computational model in which the bound is provable."},
    {"id": "OP-04", "node": "t5", "delphi": None, "title": "Prediction beyond the ~50 Myr horizon",
     "gap": "The Thompson prediction engine collapses to a clade prior beyond ~50 Myr and the Monte Bolca test was a null result vs stasis (6/16). What, if anything, carries predictive signal at deep horizons — stem placements, fossil-calibrated priors, non-Gaussian processes?",
     "done": "A pre-registered deep-horizon test (registered before unblinding, as in the FishShapes protocol) that beats the clade-prior baseline beyond 50 Myr, or a sharper null."},
    {"id": "OP-05", "node": "t8", "delphi": None, "title": "Within-species bee P-matrix (the one missing input)",
     "gap": "The why-wings causal decomposition is clean for hornets (within-species P) but weaker for bees, where P is within-tribe. A within-species bee wing dataset would make the hornet/bee comparison exact.",
     "done": "A within-species bee forewing landmark dataset with replicates, run through the registered C1-C8 pipeline without changing its hypotheses."},
    {"id": "OP-06", "node": "f3", "delphi": None, "title": "Flagship structural debt: growing-domain Turing, autonomous Meinhardt, C. elegans voxel calibration",
     "gap": "Pass N+1 left these explicit: the Turing flagship lacks a 2D growing-domain solver; Meinhardt shell parameters are not autonomous; C. elegans lacks true voxel calibration (eggshell 'ASSUMED 50 um').",
     "done": "Each sub-item closed by a self-test that fails on the current code and passes on the fix, with the audit score re-derived, not asserted."},
    {"id": "OP-07", "node": "b2", "delphi": None, "title": "Xenopus neighbour transfer: kill the orientation confound",
     "gap": "Headline blind test was negative (LOOO AUC 0.484); orientation carries the signal (0.833) and the PCA side-split subset is suspect (clean-geometry subset 0.802). Is there any wave-feature signal once geometry is ground-truth?",
     "done": "A v2 pre-registration using only ground-truth-geometry videos, scored by the hash-locked contract, whichever way it comes out."},
    {"id": "OP-08", "node": "b3", "delphi": None, "title": "Pixel v2: image-proximity (mirror) neighbour prediction",
     "gap": "Post-unblinding HTHT mirror signal (AUC 0.61-0.65) is exploratory only and confounded with headward direction because pokes were at the tail. Needs a registered test where mirror and headward predictions separate.",
     "done": "Pre-registered pixel-v2 scoring on videos where poke position separates the mirror and headward hypotheses."},
    {"id": "OP-11", "node": "f5", "delphi": "bos", "title": "Cow Turing follow-ups: independent re-score + migration-wave colonisation variant",
     "gap": "The published negative verdict (static Turing rejected; growing RD vs Eden colonisation p=0.91) rests on a builder-scored Pass 9 assessment — the two blinded Fable audits (54.4, 57.7) predate the Pass 8 scale-matched pipeline. The Eden control is also a minimal interior-founder proxy, while real Holstein white invades from migration fronts; and held-out n=9 limits power.",
     "done": "A blinded independent re-score of the Pass 8+ tree (audit R11), plus a neural-crest migration-wave colonisation variant scored on the same coat-masked, scale-matched statistics — published as a superseding assessment either way."},
    {"id": "OP-10", "node": "t7", "delphi": None, "title": "Portfolio residue: clades where Thompson-7 failed (Leporidae 0/3, Vespinae 0/7 pre-venation)",
     "gap": "45/90 registered multi-clade hypotheses held; the failures are concentrated and unexplained — is it landmark scheme, tree quality, or a real limit of global grids in those clades?",
     "done": "A per-clade failure autopsy in the style of the why-wings decomposition, with at least one failure converted into a correct, pre-registered prediction."},
]

def load_predictions() -> dict | None:
    """Registered outcome+resolvability predictions (predictions/), with the
    resolution evidence merged in. Source of truth stays the registry JSON;
    the graph page is a view of it, never a second copy edited by hand."""
    reg_path = Path(__file__).resolve().parent.parent / "predictions" / "prediction_registry.json"
    if not reg_path.exists():
        print("WARNING: predictions/prediction_registry.json not found — building without the predictions panel")
        return None
    reg = json.loads(reg_path.read_text())
    evidence = {}
    for f in sorted(reg_path.parent.glob("resolutions/*.json")):
        for r in json.loads(f.read_text()).get("resolutions", []):
            evidence[r["prediction_id"]] = r
    preds = []
    for p in reg["predictions"]:
        ev = evidence.get(p["prediction_id"], {})
        preds.append({
            "id": p["prediction_id"], "organism": p["organism"], "mechanism": p["mechanism"],
            "cls": p["class"], "prediction": p["prediction"], "quant": p["quantitative_expectation"],
            "falsifier": p["falsifier"], "test_source": p["test_source"], "datasets": p["datasets"],
            "res_category": p["resolvability"]["predicted_category"],
            "res_confidence": p["resolvability"]["confidence"],
            "expected_verdict": p["resolvability"]["expected_verdict_on_outcome"],
            "outcome": p.get("outcome") or ev.get("outcome_verdict") or "registered_untested",
            "evidence": ev.get("evidence", ""), "resolvability_verdict": ev.get("resolvability_verdict", ""),
        })
    return {"registered_on": reg["registered_on"], "predictions": preds,
            "meta": [{"id": m["meta_id"], "prediction": m["prediction"], "outcome": m.get("outcome") or "registered_untested"}
                     for m in reg.get("meta_resolvability_predictions", [])],
            "tallies": reg.get("scoring", {}).get("tallies", {}),
            "registry_url": "https://github.com/fractastical/morpholib/blob/main/predictions/prediction_registry.json",
            "resolutions_url": "https://github.com/fractastical/morpholib/tree/main/predictions/resolutions"}


PRED_NODE_LABELS = {  # short graph labels; full text lives in the registry/panel
    "ZF-TUR-01": "BMP–Chordin shuttling", "ZF-TUR-02": "Tolloid cleavage → dorsalized",
    "ZF-HH-01": "Shh lipidation gradient", "ZF-CLK-01": "Somite ∝ clock period",
    "ZF-VTX-01": "Jamming at p0*=3.813", "DM-WG-01": "Wg lipidation gates spread",
    "DM-HH-01": "HhN cholesterol range", "DM-DPP-01": "Dpp mature chain 132aa",
    "DM-CLK-NEG-01": "✗ No clock in long-germ fly", "DM-MUR-NEG-01": "✗ Trachea vs Murray=3",
    "XT-BMP-01": "Noggin dorsalizes (BMP bind)", "XT-SHH-01": "Xenopus Shh dual lipidation",
    "XT-DQ-01": "BMP4 accession is a fragment", "XT-CLK-01": "Xenopus somite ∝ period",
}
OUTCOME_STATUS = {"confirmed": "solved", "partially_resolved": "partial", "refuted": "refuted"}
RES_SHORT = {"current_datasets_only": "in-repo", "current_plus_public_sources": "repo+public",
             "partial_only": "partial only"}  # node subs only; full category lives in the panel/detail


def prediction_graph_nodes(pred: dict) -> tuple[list[dict], list[dict]]:
    """Inject the predictions INTO the problem graph as nodes+edges (not just a
    panel): hub -> organism -> prediction, plus cross-links from the existing
    nodes this work builds on (BMP pair-equivalence test, zebrafish and
    Xenopus blind tests). Coordinates extend the canvas downward (new band)."""
    org_slug = {"Danio rerio": "pred-danio", "Drosophila melanogaster": "pred-drosophila",
                "Xenopus tropicalis": "pred-xenopus"}
    org_x = {"Danio rerio": 255, "Drosophila melanogaster": 610, "Xenopus tropicalis": 975}
    nodes = [{
        "id": "pred0", "x": 610, "y": 975, "status": "root", "family": "Registered predictions",
        "label": "Registered predictions", "sub": "outcome + resolvability, registered 2026-10-26 before scoring",
        "time": "2026-10-26", "timeKind": "recorded", "source": "predictions/prediction_registry.json",
        "detail": ("14 predictions for the 3 best-supported organisms (Danio, Drosophila, Xenopus), each with TWO "
                   "registered claims: an outcome prediction and a resolvability prediction (can current data decide "
                   "it?). Scored 2026-10-26: 9 confirmed, 5 partially resolved, 0 refuted; meta-resolvability 2/4.")},
        {"id": "pred-meta", "x": 1215, "y": 1075, "status": "partial", "family": "Registered predictions",
         "label": "Meta-resolvability 2/4", "sub": "solve-rate forecasts: 2 confirmed, 2 refuted",
         "time": "2026-10-26", "timeKind": "recorded", "source": "predictions/prediction_registry.json (META-RES-01..04)",
         "detail": "; ".join(f"{m['id']} [{m['outcome']}]: {m['prediction']}" for m in pred["meta"])},
    ]
    edges = [{"from": "root", "to": "pred0", "agent": "wonder"},
             {"from": "pred0", "to": "pred-meta", "agent": "wonder"},
             {"from": "f4", "to": "pred-danio-x", "agent": "wonder", "label": "BMP evidence base"},
             {"from": "b4", "to": "pred-danio-x", "agent": "wonder", "label": "zebrafish programme"},
             {"from": "b2", "to": "pred-xenopus-x", "agent": "wonder", "label": "xenopus programme"}]
    edges[2]["to"] = edges[3]["to"] = org_slug["Danio rerio"]; edges[4]["to"] = org_slug["Xenopus tropicalis"]
    by_org: dict[str, list[dict]] = {}
    for p in pred["predictions"]:
        by_org.setdefault(p["organism"], []).append(p)
    for org, plist in by_org.items():
        oid = org_slug[org]
        nodes.append({"id": oid, "x": org_x[org], "y": 1075, "status": "partial" if any(
            p["outcome"] == "partially_resolved" for p in plist) else "solved",
            "family": "Registered predictions", "label": org,
            "sub": f"{len(plist)} predictions · {sum(p['outcome'] == 'confirmed' for p in plist)} confirmed · "
                   f"{sum(p['outcome'] == 'partially_resolved' for p in plist)} partial",
            "time": "2026-10-26", "timeKind": "recorded", "source": "predictions/prediction_registry.json",
            "detail": f"Registered prediction organism: {org} — selected for dataset depth, protein-catalog coverage, and a public mutant/literature test atlas."})
        edges.append({"from": "pred0", "to": oid, "agent": "wonder"})
        for i, p in enumerate(plist):
            nodes.append({
                "id": p["id"], "x": org_x[org], "y": 1175 + i * 88,
                "status": OUTCOME_STATUS.get(p["outcome"], "registered"),
                "family": "Registered predictions", "label": p["id"],
                "sub": f"{PRED_NODE_LABELS.get(p['id'], p['mechanism'])} · {p['outcome']} · {RES_SHORT.get(p['res_category'], p['res_category'])} (conf {p['res_confidence']})",
                "time": "2026-10-26", "timeKind": "recorded",
                "source": "predictions/prediction_registry.json + predictions/resolutions/",
                "detail": (f"OUTCOME: {p['prediction']} QUANTITATIVE: {p['quant']} "
                           f"RESOLVABILITY FORECAST: {p['res_category']} (conf {p['res_confidence']}, expected {p['expected_verdict']}). "
                           f"EVIDENCE: {p['evidence'] or '—'}")})
            edges.append({"from": oid, "to": p["id"], "agent": "wonder"})
    return nodes, edges


def load_thompson_claims() -> dict | None:
    """Thompson (1917) claim registry: 59 claims, the browsable unit of the
    Thompson programme. Source of truth thompson/thompson_1917_claims.json."""
    p = Path(__file__).resolve().parent.parent / "thompson" / "thompson_1917_claims.json"
    if not p.exists():
        print("WARNING: thompson/thompson_1917_claims.json not found — building without the claims browser")
        return None
    reg = json.loads(p.read_text())
    claims = [{"id": c["id"], "chapter": c["chapter"], "chapter_title": c["chapter_title"],
               "pages": c["pages"], "figures": c["figures"], "claim": c["claim"], "type": c["type"],
               "status": c["pass2"]["status"], "what_remains": c["pass2"]["what_remains"],
               "method": c["pass2"]["method"], "data_needed": c["pass2"]["data_needed"],
               "effort": f"{c['pass2']['effort_days'][0]}–{c['pass2']['effort_days'][1]}d ({c['pass2']['effort_class']})",
               "priority": c["pass2"]["priority"]} for c in reg["claims"]]
    from collections import OrderedDict
    ch_titles: "OrderedDict[str, None]" = OrderedDict()
    for c in claims:
        ch_titles.setdefault(c["chapter_title"], None)
    return {"n": len(claims), "claims": claims,
            "chapters": [{"title": t, "node": f"tch-{i}"} for i, t in enumerate(ch_titles)],
            "inventory_url": "https://github.com/fractastical/morpholib/blob/main/docs/THOMPSON_1917_CLAIM_INVENTORY.md"}


def thompson_graph_nodes(thomp: dict) -> tuple[list[dict], list[dict]]:
    """Thompson claims enter the graph as a collapsible category subtree:
    t1 -> claims hub -> one node per chapter (category) -> claims are browsed
    in the claims browser / inspector when a chapter node is selected, so all
    59 claims are reachable from the graph without 59 graph nodes."""
    from collections import OrderedDict
    chapters: "OrderedDict[str, list[dict]]" = OrderedDict()
    for c in thomp["claims"]:
        chapters.setdefault(c["chapter_title"], []).append(c)
    nodes = [{"id": "t-claims", "x": 20, "y": 1705, "status": "partial", "family": "Thompson claims",
              "label": "Thompson 1917 claims", "sub": "59 claims · double-click a chapter to browse its claims below",
              "time": "1917 (claims) / registry 2026-10-26", "timeKind": "recorded",
              "source": "thompson/thompson_1917_claims.json",
              "detail": ("Every testable claim in On Growth and Form (1917), cited by chapter/pages/figures and "
                         "assessed: 9 DONE_REPO, 13 PARTIAL_REPO, 13 DONE_ELSEWHERE, 21 UNTESTED, "
                         "3 NOT_TESTABLE_AS_STATED. Select a chapter node, then browse its claims in the "
                         "Thompson claims browser below the graph.")}]
    edges = [{"from": "t1", "to": "t-claims", "agent": "wonder", "label": "59-claim registry"}]
    xs = [20, 255, 490, 735, 975, 1215]
    for i, (title, cs) in enumerate(chapters.items()):
        cid = f"tch-{i}"
        done = sum(c["status"] in ("DONE_REPO", "DONE_ELSEWHERE") for c in cs)
        status = "solved" if done == len(cs) else ("open" if done == 0 and all(c["status"] == "UNTESTED" for c in cs) else "partial")
        nodes.append({"id": cid, "x": xs[i % 6], "y": 1805 + (i // 6) * 88, "status": status,
                      "family": "Thompson claims", "label": f"Ch. {cs[0]['chapter'].split()[0]} · {len(cs)} claims",
                      "sub": f"{title} — {done}/{len(cs)} done",
                      "time": "1917", "timeKind": "recorded", "source": "thompson/thompson_1917_claims.json",
                      "detail": (f"Chapter {cs[0]['chapter']} — {title}: {len(cs)} claims "
                                 f"({', '.join(sorted({c['status'] for c in cs}))}). "
                                 "Its claims are listed in the Thompson claims browser below; "
                                 "click the chapter filter there or a claim row for the full assessment.")})
        edges.append({"from": "t-claims", "to": cid, "agent": "wonder"})
    return nodes, edges


def build_morphoswarm(data: dict, out_dir: Path, pred: dict | None = None) -> None:
    by_id = {n["id"]: n for n in data["NODES"]}
    problems = []
    for op in OPEN_PROBLEMS:
        node = by_id[op["node"]]
        attempted = sorted({e["agent"] for e in data["EDGES"] if e["to"] == op["node"]})
        entry = {**op, "status": node["status"], "graph_node_label": node["label"],
                 "source": node["source"], "attempted_by_agent_ids": attempted,
                 "attempted_by": [data["AGENTS"][a]["name"] for a in attempted],
                 "graph_url": f"https://fractastical.github.io/morpholib/problem-graph.html#{op['node']}"}
        if op["delphi"]:
            prize = data["DELPHI"][op["delphi"]]
            entry["viability"] = {"kind": "delphi_bridge_posterior_percent", "pooled": prize["pooled"]["posterior"],
                                  "panel_range_percent": [r["posterior"] for r in prize["agents"]],
                                  "classification": "possible_pathway" if prize["pooled"]["posterior"] >= 15 else "below_15_percent",
                                  "note": "Classification 'possible_pathway' is >=15% on the pooled posterior, but colour is stricter: green only >70% (light) / >80% (dark), teens are burnt orange. Individual agents may differ (see panel_range_percent)."}
        else:
            entry["viability"] = {"kind": "not_scored", "pooled": None,
                                  "classification": "not_scored",
                                  "note": "No Delphi probability exists for this problem. Do not invent one: the first swarm task is a pre-registered Delphi scoring."}
        problems.append(entry)
    feed = {"name": "MorphoSwarm — open problems work queue", "snapshot": "2026-10-26",
            "source_repo": "https://github.com/fractastical/morpholib",
            "graph_url": "https://fractastical.github.io/morpholib/problem-graph.html",
            "colour_rule": {"possible_pathway": "pooled Delphi bridge posterior >= 15% (classification only — NOT green)", "low_probability": "pooled Delphi bridge posterior < 15%", "grey": "not scored",
                            "shade_bands_percent": {">80_dark_green": "#0B6E3F", "70-80_light_green": "#69B86F", "50-70": "#C9A227", "30-50": "#D98E2B", "15-30": "#C45A24", "5-15": "#B03A2E", "<5": "#7E1F2B"},
                            "note": "Shade bands show the range of opinion. Green is reserved for >70% (light) and >80% (dark); a posterior in the teens is burnt orange. Panel range per problem is in viability.panel_range_percent."},
            "protocol": ["Pick one OP id; do not work two on the same branch without noting it.",
                         "Pre-register the attempt (hypotheses + scoring) before touching held-out data, following the repo's hash-locked blind-test contract.",
                         "Report negative results as results; pruned branches stay in the graph.",
                         "Close by updating the source documents and re-running docs/build_problem_graph_html.py, whose self-checks must pass."],
            "problems": problems}
    (out_dir / "open-problems.json").write_text(json.dumps(feed, indent=2) + "\n")

    def badge(v):
        if v["kind"] == "not_scored":
            return '<span class="badge" style="background:#888">NOT SCORED — Delphi-score first</span>'
        cls = "POSSIBLE PATHWAY" if v["classification"] == "possible_pathway" else "&lt;15% · LOW PROBABILITY"
        p = v["pooled"]  # same shade scale as the problem graph's probColor: green only >70%
        col = "#0B6E3F" if p >= 80 else "#69B86F" if p >= 70 else "#C9A227" if p >= 50 else "#D98E2B" if p >= 30 else "#C45A24" if p >= 15 else "#B03A2E" if p >= 5 else "#7E1F2B"
        cls = "STRONG" if p >= 80 else "PLAUSIBLE" if p >= 70 else "POSSIBLE BUT WEAK" if v["classification"] == "possible_pathway" else "LOW PROBABILITY"
        return f'<span class="badge" style="background:{col}">{v["pooled"]}% · {cls}</span>'
    cards = []
    for p in problems:
        rng = ""
        if p["viability"]["kind"] != "not_scored":
            vals = p["viability"]["panel_range_percent"]
            rng = f'<p class="small mut">Panel range: {min(vals)}% – {max(vals)}% (individual agents can sit in a different shade band from the pooled value — the spread is the range of opinion)</p>'
        cards.append(f"""<div class="card" id="{p['id']}"><div class="row"><b>{p['id']} · {p['title']}</b>{badge(p['viability'])}</div>
<p class="small"><b>The gap:</b> {p['gap']}</p><p class="small"><b>Definition of done:</b> {p['done']}</p>{rng}
<p class="small mut">Attempted so far by: {', '.join(p['attempted_by']) or 'no recorded attempt'} · Source: {p['source']} · <a href="{p['graph_url']}">graph node</a></p></div>""")
    if pred:  # partially-resolved registered predictions are genuine open work for the swarm
        partials = [p for p in pred["predictions"] if p["outcome"] == "partially_resolved"]
        feed["registered_prediction_followups"] = {
            "note": "Registered predictions scored partially_resolved on 2026-10-26: outcome neither confirmed nor refuted; each names the exact missing series. Closing one means supplying that series, not re-arguing the prediction.",
            "items": [{"id": p["id"], "organism": p["organism"], "mechanism": p["mechanism"],
                       "missing_evidence": p["evidence"]} for p in partials],
            "registry_url": pred["registry_url"]}
        (out_dir / "open-problems.json").write_text(json.dumps(feed, indent=2) + "\n")  # rewrite: feed was first written above, before follow-ups existed
        pcards = "".join(
            f'<div class="card" id="{p["id"]}"><div class="row"><b>{p["id"]} · {p["organism"]} — {p["mechanism"]}</b>'
            f'<span class="badge" style="background:#F0A040">PARTIALLY RESOLVED</span></div>'
            f'<p class="small"><b>Registered prediction:</b> {p["prediction"]}</p>'
            f'<p class="small"><b>What is missing (from the resolution record):</b> {p["evidence"]}</p>'
            f'<p class="small mut">Registry: <a href="{pred["registry_url"]}">{p["id"]}</a> · Test source: {p["test_source"]}</p></div>'
            for p in partials)
        followups = ('<h2 style="font-size:18px">Registered-prediction follow-ups — partially resolved</h2>'
                     '<p class="small">From the 2026-10-26 outcome+resolvability registry (9 confirmed, 5 partial, 0 refuted). '
                     'These five are open precisely where the resolution record says the decisive series does not yet exist.</p>' + pcards)
    else:
        followups = ""
    html = MORPHOSWARM_TEMPLATE.replace("/*__CARDS__*/", "\n".join(cards)).replace("/*__PRED_FOLLOWUPS__*/", followups)
    (out_dir / "morphoswarm.html").write_text(html)
    print(f"wrote {out_dir / 'morphoswarm.html'} and {out_dir / 'open-problems.json'} ({len(problems)} open problems)")

MORPHOSWARM_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>MorphoSwarm — Open Problems · Project Delphi</title>
<meta name="description" content="Swarm-facing work queue of genuinely open problems from Project Delphi, with Delphi viability colour coding (amber-to-red below 70%; light green only over 70%, dark green over 80%) and a machine-readable feed.">
<style>
:root{--bg:#f7f7f5;--panel:#fff;--ink:#1c1c1e;--mut:#5c5c60;--line:#d8d8d3;--acc:#2563eb}
*{box-sizing:border-box}body{margin:0;font-family:-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;background:var(--bg);color:var(--ink)}
main{max-width:1000px;margin:0 auto;padding:18px}h1{font-size:24px}.card{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:12px 14px;margin:12px 0}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;justify-content:space-between}.badge{display:inline-block;color:#fff;font-size:10px;font-weight:700;padding:2px 7px;border-radius:10px;white-space:nowrap}
.small{font-size:12.5px;line-height:1.45}.mut{color:var(--mut)}a{color:var(--acc)}code{background:#eee;padding:1px 5px;border-radius:4px}
</style></head><body><main>
<h1>MorphoSwarm — open problems</h1>
<p class="small">The swarm-facing work queue for <a href="https://fractastical.github.io/morpholib/problem-graph.html">Project Delphi's problem graph</a>: every problem that is genuinely still open, what exactly blocks it, what counts as closing it, and who has already attempted it. Colour rule, same as the graph — shade bands show the range of opinion, and green is reserved for high posteriors: <span class="badge" style="background:#0B6E3F">&gt;80% · STRONG</span> <span class="badge" style="background:#69B86F">70–80% · PLAUSIBLE</span> <span class="badge" style="background:#C9A227">50–70% · MIXED</span> <span class="badge" style="background:#D98E2B">30–50%</span> <span class="badge" style="background:#C45A24">&ge;15% · POSSIBLE BUT WEAK</span> <span class="badge" style="background:#B03A2E">5–15%</span> <span class="badge" style="background:#7E1F2B">&lt;5% · LOW</span> <span class="badge" style="background:#888">NOT SCORED</span> — a posterior in the teens is burnt orange, not green; the &ge;15% “possible pathway” classification is recorded in the feed, not painted green.</p>
<div class="card"><b>Swarm protocol</b><ol class="small">
<li>Pick one OP id. Machine-readable feed: <a href="open-problems.json"><code>open-problems.json</code></a> (this page, as data, for agents).</li>
<li>Pre-register the attempt before touching held-out data — the repo's contract is hash-locked predictions, scorer refuses on mismatch.</li>
<li>Negative results are results. Pruned branches stay in the graph; do not quietly retry a refuted branch as if new.</li>
<li>Close by updating the source documents, then re-run <code>docs/build_problem_graph_html.py</code> — its self-checks must pass before anything deploys.</li></ol>
<p class="small mut">Note: this project's name is a work-queue label for this repo's agent swarm. It is unrelated to the robotic-swarm MorphoSwarm of Ben Zion et al. (Science Robotics 2022; doi:10.5281/zenodo.7573761).</p></div>
/*__CARDS__*/
/*__PRED_FOLLOWUPS__*/
<p class="small mut">Snapshot 2026-10-26 · Source repo: <a href="https://github.com/fractastical/morpholib">github.com/fractastical/morpholib</a> · Viability numbers: docs/RULE30_BAYESIAN_DELPHI_ASSESSMENT.md · Regenerate: <code>python3 docs/build_problem_graph_html.py</code></p>
</main></body></html>
"""

def summary_cards(data: dict, pred: dict | None) -> str:
    """Headline counts at the top of the graph, computed from the graph data
    itself (never hand-typed, so they cannot drift from the nodes below).
    'Definitively solved' = graph status solved/proved; the core graph carries
    no calibrated per-node confidence, so no 90% figure is invented for it --
    the only literal >=90% confidences in the record are the registered
    resolvability confidences in predictions/, counted separately."""
    nodes = [n for n in data["NODES"] if n["status"] != "root"]
    attempts = [e for e in data["EDGES"] if e["from"] != "root"]
    solved = [n for n in nodes if n["status"] == "solved"]
    refuted = [n for n in nodes if n["status"] == "refuted"]
    rest = len(nodes) - len(solved) - len(refuted)
    agents = len({e["agent"] for e in attempts})
    pred_html = ""
    if pred:
        conf90 = [p for p in pred["predictions"]
                  if p["outcome"] == "confirmed" and p["res_confidence"] >= 0.90]
        confirmed = sum(p["outcome"] == "confirmed" for p in pred["predictions"])
        ids = ", ".join(f"{p['id']} ({p['res_confidence']:.2f})" for p in conf90)
        pred_html = (f'<p class="small" style="margin-top:8px"><b>Literal ≥90%:</b> '
                     f'{len(conf90)} of {len(pred["predictions"])} registered predictions were confirmed at a '
                     f'registered resolvability confidence ≥0.90 — {ids} '
                     f'({confirmed} confirmed overall). That confidence grades whether current data could '
                     f'<i>decide</i> the prediction, not the biology itself.</p>')
    return f"""<div class="grid4" style="margin-top:10px">
<div class="card"><div class="small mut">Problems attempted</div><div class="stat">{len(nodes)}</div><div class="small mut">graph nodes, umbrella root excluded</div></div>
<div class="card"><div class="small mut">Solution attempts tried</div><div class="stat">{len(attempts)}</div><div class="small mut">recorded agent attempts (edges) by {agents} agents</div></div>
<div class="card"><div class="small mut">Shown definitively</div><div class="stat" style="color:var(--green)">{len(solved)}</div><div class="small mut">status solved / proved in the graph</div></div>
<div class="card"><div class="small mut">Definitively refuted</div><div class="stat" style="color:#C04848">{len(refuted)}</div><div class="small mut">pruned branches, kept not deleted · {rest} partial / open / registered</div></div>
</div>
<div class="card"><b>How to read “shown definitively”.</b> <span class="small">The core graph records no calibrated per-node confidence, so the {len(solved)} above is its <i>solved / proved</i> status bucket — proof-grade or hash-locked-score-grade resolution as documented per node — not a 90% posterior, and no such number is invented here.</span>{pred_html}</div>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canvas", type=Path, default=DEFAULT_CANVAS)
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("problem-graph.html"))
    a = ap.parse_args()
    src = a.canvas.read_text()
    names = ["AGENTS", "NODES", "EDGES", "DELPHI", "CONCORDANCE"]
    js = ";\n".join(f"const {n} = {extract_literal(src, n)}" for n in names)
    js += ";\nconsole.log(JSON.stringify({AGENTS, NODES, EDGES, DELPHI, CONCORDANCE}))"
    out = subprocess.run(["node", "-e", js], capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(out.stderr)
    data = json.loads(out.stdout)
    errs = validate(data)
    if errs:
        sys.exit("REFUSING TO BUILD — data failed self-checks:\n" + "\n".join(f" - {e}" for e in errs))
    print(f"self-checks passed: {len(data['NODES'])} nodes, {len(data['EDGES'])} edges, "
          f"{len(data['DELPHI'])} Delphi prizes (weights=100, weighted totals and prior x BF posteriors verified)")
    pred = load_predictions()
    summary_html = summary_cards(data, pred)  # core-graph counts, before predictions are merged in
    if pred:  # integrate predictions into the graph itself, then re-validate the merged graph
        data["AGENTS"]["wonder"] = {"name": "Muse Wonder", "role": "Protein-state pilot, prediction registry, resolution scoring (this branch of work)"}
        n0, e0 = prediction_graph_nodes(pred)
        data["NODES"] += n0; data["EDGES"] += e0
        errs = validate(data)
        if errs:
            sys.exit("REFUSING TO BUILD — merged graph (with predictions) failed self-checks:\n" + "\n".join(f" - {e}" for e in errs))
        print(f"integrated predictions into graph: +{len(n0)} nodes, +{len(e0)} edges (hub -> 3 organisms -> {len(pred['predictions'])} predictions + meta)")
    thomp = load_thompson_claims()
    if thomp:
        data["AGENTS"].setdefault("wonder", {"name": "Muse Wonder", "role": "Protein-state pilot, prediction registry, claims browsing (this branch of work)"})
        tn, te = thompson_graph_nodes(thomp)
        data["NODES"] += tn; data["EDGES"] += te
        errs = validate(data)
        if errs:
            sys.exit("REFUSING TO BUILD — merged graph (with Thompson claims) failed self-checks:\n" + "\n".join(f" - {e}" for e in errs))
        print(f"integrated Thompson claims into graph: +{len(tn)} nodes (hub + {len(tn) - 1} chapter categories), {thomp['n']} claims browsable")
    html = TEMPLATE.replace("/*__SUMMARY__*/", summary_html)
    html = html.replace("/*__DATA__*/", "const DATA = " + json.dumps(data) + ";")
    html = html.replace("/*__PRED__*/", "const PRED = " + json.dumps(pred) + ";")
    html = html.replace("/*__THOMP__*/", "const THOMP = " + json.dumps(thomp) + ";")
    a.out.write_text(html)
    print(f"wrote {a.out} ({len(html)/1024:.0f} KiB, {len(data['NODES'])} nodes, {len(data['EDGES'])} edges, "
          f"{len(pred['predictions']) if pred else 0} registered predictions)")
    build_morphoswarm(data, a.out.parent, pred)

TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Problem Graph — Project Delphi</title>
<meta name="description" content="Every problem attempted in Project Delphi, with branching paths, probabilistic timestamps, per-agent edge colours, and a double-click Bayesian Delphi inspector. Posterior shades run amber to red below 70%; light green only over 70%, dark green over 80%.">
<style>
:root{--bg:#f7f7f5;--panel:#fff;--ink:#1c1c1e;--mut:#5c5c60;--line:#d8d8d3;--acc:#2563eb;--green:#1F8A65;--orange:#C06028}
*{box-sizing:border-box}body{margin:0;font-family:-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;background:var(--bg);color:var(--ink)}
main{max-width:1460px;margin:0 auto;padding:18px}h1{font-size:24px;margin:2px 0 6px}h2{font-size:17px;margin:14px 0 6px}h3{font-size:14px;margin:10px 0 4px}
p,li{font-size:13.5px;line-height:1.45}.mut{color:var(--mut)}.small{font-size:12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:12px 14px;margin:12px 0}
button{font:inherit;font-size:12.5px;padding:5px 10px;border:1px solid var(--line);border-radius:7px;background:#fff;cursor:pointer}
button.on{background:var(--acc);color:#fff;border-color:var(--acc)}button:hover{border-color:var(--acc)}
.row{display:flex;gap:6px;flex-wrap:wrap;align-items:center}.sw{display:inline-block;width:18px;height:4px;margin-right:6px;vertical-align:middle}
#graphwrap{overflow:hidden;background:#fff;border:1px solid var(--line);border-radius:9px;position:relative}
#graph{cursor:grab;touch-action:none;display:block;width:100%;height:auto}
#graph:active{cursor:grabbing}
.zoomctl{position:absolute;top:10px;right:10px;z-index:20;display:flex;gap:4px;align-items:center;background:rgba(255,255,255,.92);border:1px solid var(--line);border-radius:8px;padding:4px}
.zoomctl button{font-size:15px;font-weight:700;line-height:1;padding:5px 10px}
.zoomctl .zlab{font-size:11.5px;padding:5px 7px}
#zoompct{font-size:12px;font-weight:700;min-width:44px;text-align:center;color:var(--acc)}
.predgrid{display:grid;grid-template-columns:minmax(300px,44%) 1fr;gap:12px}@media(max-width:900px){.predgrid{grid-template-columns:1fr}}
.prow{cursor:pointer;border:1px solid var(--line);border-radius:7px;padding:7px 9px;margin:5px 0;background:#fff}
.prow:hover{border-color:var(--acc)}.prow.sel{border-color:var(--acc);background:#f3f6ff}
.dot{display:inline-block;width:10px;height:10px;clip-path:polygon(0 0,100% 0,0 100%);margin-right:6px;vertical-align:middle}
.tri{display:inline-block;width:11px;height:11px;clip-path:polygon(0 0,100% 0,0 100%);margin-right:4px;vertical-align:middle}
.pchip{display:inline-block;color:#fff;font-size:9.5px;font-weight:700;padding:1px 5px;border-radius:3px;margin-right:2px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px}@media(max-width:900px){.grid2{grid-template-columns:1fr}}
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}@media(max-width:900px){.grid3{grid-template-columns:1fr}}
.grid4{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}@media(max-width:900px){.grid4{grid-template-columns:repeat(2,1fr)}}
.stat{font-size:22px;font-weight:700}.badge{display:inline-block;color:#fff;font-size:10px;font-weight:700;padding:2px 7px;border-radius:10px;white-space:nowrap}
#drawer{position:fixed;top:14px;right:14px;width:660px;max-width:calc(100vw - 28px);max-height:calc(100vh - 28px);overflow-y:auto;z-index:60;background:#fff;border:2px solid var(--acc);border-radius:10px;padding:16px;display:none;box-shadow:0 8px 30px rgba(0,0,0,.18)}
.clickable{cursor:pointer;border:1px solid var(--line);border-radius:7px;padding:7px 9px;margin:5px 0;background:#fff}
.clickable.sel{border-color:var(--acc);background:#f3f6ff}.bar{ display:inline-block;width:110px;height:8px;background:#eee;vertical-align:middle}.bar>i{display:block;height:8px}
a{color:var(--acc)}
.navrow{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:10px 0}
#nodepicker{flex:1;min-width:200px;font:inherit;font-size:14px;padding:10px;border:1px solid var(--line);border-radius:8px;background:#fff}
.viewbtn{padding:10px 14px;font-size:14px}
.nlfam{font-size:11px;font-weight:700;color:#888;text-transform:uppercase;margin:12px 0 4px}
.nlrow{display:flex;width:100%;text-align:left;gap:8px;align-items:center;padding:12px;border:1px solid var(--line);border-radius:8px;background:#fff;margin:5px 0;cursor:pointer;font-size:14px;min-height:48px}
.nlrow:active{border-color:var(--acc)}
@media(max-width:760px){
 main{padding:10px}h1{font-size:19px;line-height:1.25}p,li{font-size:14px}
 .card{padding:10px 11px;margin:8px 0}.stat{font-size:19px}
 #graphwrap{height:62vh}#graph{height:100%;width:100%}
 #drawer{top:0;right:0;width:100%;max-width:100%;height:100dvh;max-height:100dvh;border-radius:0;border:none;padding:14px}
 #drawer .grid4{grid-template-columns:repeat(2,1fr)}#drawer .grid2{grid-template-columns:1fr}
 .zoomctl{top:auto;bottom:10px;right:10px}.zoomctl button{padding:10px 12px}
 button,.clickable{min-height:44px}.clickable{display:flex;align-items:center}
 .bar{width:70px}a{word-break:break-word}#agentbtns button{flex:1 1 40%}
}
</style></head><body><main>
<h1>Problem graph — every problem attempted, including the branches that failed</h1>
<p class="mut">Source: the <a href="https://github.com/fractastical/morpholib">morpholib</a> repo itself (chatlogs, audits, pre-registrations, Delphi assessment, portfolio status), snapshot 2026-10-26. Upper-left corner triangle = outcome. <b>Edge colour = the agent that attempted that step.</b> Dashed / fuzzy nodes have <b>probabilistic, not recorded</b> timing — most importantly the “≈15 h into Rule 30” reckoning. Click a node to select it; <b>double-click a node with downstream nodes to collapse/expand everything below it</b> (categories act as folders), and double-click a leaf node to open its inspector — the Bayesian Delphi panel and Rule 30 prizes open the full per-agent, per-subgoal Delphi ledger. <b>Agents looking for work: <a href="morphoswarm.html">MorphoSwarm — open problems</a></b> (<a href="open-problems.json">JSON feed</a>).</p>

<figure style="margin:12px 0"><img src="media/problem_graph_discovery.gif" alt="Animated tour of the discovery problem graph: problems build up lane by lane, outcomes are tallied (solved, partial, open, refuted-kept), then the camera zooms to the scored registered predictions" style="width:100%;height:auto;border:1px solid var(--line);border-radius:9px" loading="lazy"><figcaption class="small mut" style="margin-top:6px">The graph in ~11 seconds — build-up, outcome tally, the scored registered predictions, and finally the scientific claims shown definitively, positively or negatively (audit, infrastructure and agent-process points excluded from that roll). Still poster: <a href="media/problem_graph_discovery_poster.png">problem_graph_discovery_poster.png</a> · Regenerate: <code>python3 docs/media/make_problem_graph_animation.py</code> (numbers recomputed from this page's data).</figcaption></figure>

/*__SUMMARY__*/

<div class="card"><b>Agent legend — who attempted what</b> <span class="badge" style="background:var(--acc)">edge colour = agent</span>
<div class="row" id="agentbtns" style="margin-top:8px"></div>
<p class="mut small" id="agentnote"></p></div>

<div class="navrow"><select id="nodepicker" aria-label="Find a problem" onchange="if(this.value)pickNode(this.value)"></select><button id="vgraph" class="viewbtn on" onclick="setView('graph')">Graph</button><button id="vlist" class="viewbtn" onclick="setView('list')">☰ List</button></div>
<div id="nodelist" hidden></div>
<div id="graphwrap"><svg id="graph" width="1420" height="2175" aria-label="Problem graph (zoomable)"></svg>
<div class="zoomctl" role="group" aria-label="Graph zoom controls">
<button onclick="zoomStep(0.8)" title="Zoom out (− key)" aria-label="Zoom out">−</button>
<span id="zoompct">100%</span>
<button onclick="zoomStep(1.25)" title="Zoom in (+ key)" aria-label="Zoom in">+</button>
<button class="zlab" onclick="fitView()" title="Fit whole graph in view (F key)">Fit</button>
<button class="zlab" onclick="resetView()" title="Reset to 100% (0 key)">Reset</button>
</div></div>
<p class="mut small" style="margin:4px 0 0">Zoom &amp; pan: mouse-wheel over the graph zooms to the cursor · drag to pan · pinch on touch screens · <b>+</b>/<b>−</b> keys zoom, <b>F</b> fits, <b>0</b> resets · double-click empty graph space zooms in. <b>Collapse / expand: double-click any node that has downstream nodes to hide or reveal everything downstream of it</b> — category nodes (the project root, organism hubs, Thompson chapter hubs) therefore work as folders; a collapsed node (hollow blue top-right corner triangle, dashed outline) shows <b>+N hidden</b>. Leaf nodes still double-click to their inspector, and every node's detail card has Collapse/Expand and Inspect buttons. <b>On phones:</b> the List view is the primary navigation (tap any problem to inspect); in Graph view a single tap on a node opens its inspector full-screen.</p>
<div class="row" style="margin:6px 0"><button onclick="expandAll()">Expand all</button><button onclick="collapseCategories()">Collapse to categories</button><span class="small mut">Categories = the root's direct children, the prediction organism hubs, and the Thompson claims hub — double-click any of them (or Expand all) to drill back in.</span></div>
<div class="row small" style="margin:8px 0">
<span><i class="tri" style="background:#178A5F"></i>solved / proved</span><span><i class="tri" style="background:#E79A28"></i>partial</span><span><i class="tri" style="background:#D64545"></i>still open</span><span><i class="tri" style="background:#7E2A3A"></i>refuted / pruned</span><span><i class="tri" style="background:#3E7FB8"></i>registered, unscored</span><span><i class="tri" style="background:#4B3FA0"></i>umbrella</span>
<span>–– dashed halo = probabilistic timestamp</span>
<span><b>Upper-left corner triangle = outcome</b> (colour above; a triangle, not a dot, so it never sits on top of the title text)</span>
<span><b>Delphi posterior shades (range of opinion, not just pass/fail):</b> <span class="pchip" style="background:#0B6E3F">&gt;80% · dark green</span><span class="pchip" style="background:#69B86F">70–80% · light green</span><span class="pchip" style="background:#C9A227">50–70%</span><span class="pchip" style="background:#D98E2B">30–50%</span><span class="pchip" style="background:#C45A24">15–30%</span><span class="pchip" style="background:#B03A2E">5–15%</span><span class="pchip" style="background:#7E1F2B">&lt;5%</span> — <b>green is reserved for &gt;70%</b>: a posterior in the teens is weak (burnt orange), not a green light; 15% survives only as the recorded “possible pathway” classification, not as a colour</span>
<span><b>Prize-node bottom strip = the 3 per-agent posteriors side by side</b>, each shaded on the same scale — the spread between segments is the panel's range of opinion; the border/“bridge” label is the pooled value</span>
<span><b>Edge % = Delphi path-success posterior</b> (pooled panel, where a panel has scored that route, shaded on the same scale) — edges with no panel show <span class="badge" style="background:#888">NOT SCORED</span> in the inspector, never an invented number</span></div>

<h2>Registered predictions — outcome <i>and</i> resolvability, scored</h2>
<div class="card" id="predpanel"></div>

<h2>Thompson 1917 — browse all 59 claims</h2>
<div class="card" id="claimpanel"></div>

<div class="grid2">
<div class="card" id="detail"></div>
<div><h2>Rule 30 elapsed-time axis — recorded vs probabilistic</h2>
<p class="mut small">X-axis: active hours into the Rule 30 attempt. Solid blocks are phase durations summed in the chatlog (8h 13m at Milestone 18). The dashed band after that is <b>not logged</b>: Turns 35–66 carry no timestamps, so the “≈15 h” reckoning is a probability band (~11–19h, median ≈15h; wall-clock fixed 2026-10-20 18:26 PT), not a point.</p>
<div style="overflow-x:auto;background:#fff;border:1px solid var(--line);border-radius:9px;padding:8px"><svg width="1280" height="170">
<g font-size="10" fill="#888"><line x1="70" y1="20" x2="70" y2="130" stroke="#ddd"/><line x1="338" y1="20" x2="338" y2="130" stroke="#ddd"/><line x1="607" y1="20" x2="607" y2="130" stroke="#ddd"/><line x1="875" y1="20" x2="875" y2="130" stroke="#ddd"/><line x1="1144" y1="20" x2="1144" y2="130" stroke="#ddd"/>
<text x="70" y="148">0h</text><text x="338" y="148">5h</text><text x="607" y="148">10h</text><text x="875" y="148">15h</text><text x="1144" y="148">20h</text></g>
<rect x="70" y="45" width="142" height="34" fill="#2E79B5" opacity=".75"/><rect x="212" y="45" width="120" height="34" fill="#2E79B5" opacity=".65"/><rect x="332" y="45" width="83" height="34" fill="#2E79B5" opacity=".55"/><rect x="415" y="45" width="96" height="34" fill="#2E79B5" opacity=".45"/>
<g font-size="9" font-weight="700"><text x="141" y="66" text-anchor="middle">Phase I · 2.65h</text><text x="272" y="66" text-anchor="middle">Phase II · 2.23h</text><text x="373" y="66" text-anchor="middle">III · 1.55h</text><text x="463" y="66" text-anchor="middle">IV · 1.78h</text></g>
<rect x="511" y="45" width="633" height="34" fill="#eee" stroke="#888" stroke-dasharray="5 4"/>
<text x="827" y="60" text-anchor="middle" font-size="9.5" font-weight="700" fill="#555">Turns 35–66: NO timestamps — elapsed position inferred (Thm 20–47)</text>
<text x="827" y="73" text-anchor="middle" font-size="8.5" fill="#888">8h13m logged at Milestone 18 → next dated turn is Turn 67</text>
<rect x="661" y="92" width="429" height="15" fill="#E8C030" opacity=".55"/><line x1="875" y1="88" x2="875" y2="112" stroke="#F0A040" stroke-width="2.5"/>
<text x="875" y="124" text-anchor="middle" font-size="10" font-weight="700">≈15 h · Turn 67 honest reckoning + Delphi (band ~11–19h)</text>
</svg></div>
<div class="card" style="border-color:#E8C030"><b>How to read the ≈15 h marker.</b> <span class="small">The wall-clock moment is exact; the elapsed-active-time coordinate is not in the record. Anyone quoting “15 hours in” should quote the band, not the point.</span></div>
</div></div>

<h2>What the graph says at a glance</h2>
<div class="grid3">
<div class="card"><b>Closed</b><p class="small">Thompson outline paradox and its ~50 Myr horizon; Rule 30 off-centre theory (lim P<sub>k</sub> = ∞, k&#8325; = 87,867, crystal density, ANF degree); Zhang sorting replication; KG / nanopub infrastructure.</p></div>
<div class="card"><b>Honestly negative</b><p class="small">Box-stretch shortcut, Xenopus neighbour prediction (AUC 0.484), pixel Task B, causal emergence as a practical tool, the overnight Rule 30 loop (true lemmas, wrong boundary object), the BMP pair-equivalence claim (Compara: bmp2b→bmp2 not bmp4, chd→chrd not nog; size prior 0.759 vs 1.283), and the Holstein Turing claim (static RD rejected 6.46 vs ~3.3; growing RD vs melanoblast colonisation indistinguishable, p=0.91 — claim + assessment published as production nanopublications) — kept as pruned branches, not deleted.</p></div>
<div class="card"><b>Still open</b><p class="small">All three Wolfram prizes (Delphi bridge: 11.5% / 2.0% / 0.04%, all in the red/orange bands — even the most optimistic single agent, GPT on Prize 1 at 21.8%, sits far below the 70% green line), diagonal-sample decorrelation, fixed-seed complexity transfer, within-species bee P-matrix, growing-domain Turing, autonomous Meinhardt.</p></div></div>
<p class="mut small">Attribution honesty note: early Rule 30 turns and several pipeline edges are not attributed to a named model anywhere in the repo, so they use the grey “unattributed” colour instead of a guessed agent. Named attributions come from THOMPSON_METHODS.md §1.13, the Delphi panel list, the flagship Pass N+1 referee split, and the causal-emergence guide. Generated by <code>docs/build_problem_graph_html.py</code> from the repo snapshot; underlying claims live in the linked repo documents.</p>
</main>

<div id="drawer"></div>

<script>
/*__DATA__*/
const {AGENTS, NODES, EDGES, DELPHI, CONCORDANCE} = DATA;
const ACOL = {human:'#E8C030', fable:'#7B64B8', gemini:'#2E79B5', gpt:'#1F8A65', sonnet:'#C85898', grok:'#F0A040', wainwright:'#2A9A8A', panel:'#5A6CC0', unattributed:'#8888A8', wonder:'#0E7C7B'};
const SCOL = {solved:'#178A5F', partial:'#E79A28', open:'#D64545', refuted:'#7E2A3A', registered:'#3E7FB8', root:'#4B3FA0'};
const SLAB = {solved:'solved / proved', partial:'partial', open:'OPEN', refuted:'refuted / pruned', registered:'registered, unscored', root:'umbrella'};
const W=150,H=76, byId={}; NODES.forEach(n=>byId[n.id]=n);
const S = {agent:'all', sel:'r5', prize:'p1', agentIdx:3, subgoal:'P1.4', prior:7.07};
const $=id=>document.getElementById(id);
const probColor=p=>p>=80?'#0B6E3F':p>=70?'#69B86F':p>=50?'#C9A227':p>=30?'#D98E2B':p>=15?'#C45A24':p>=5?'#B03A2E':'#7E1F2B'; // green is earned, not default: light green only >70%, dark green >80%; below that amber -> red
const probLabel=p=>p>=80?'STRONG':p>=70?'PLAUSIBLE':p>=50?'MIXED':p>=15?'WEAK':'LOW';
const probBadge=p=>`<span class="badge" style="background:${probColor(p)}">${p}% · ${probLabel(p)}</span>`;
const esc=s=>String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;');

function agentButtons(){ $('agentbtns').innerHTML = `<button class="${S.agent==='all'?'on':''}" onclick="setAgent('all')">All agents</button>` + Object.keys(AGENTS).map(a=>`<button class="${S.agent===a?'on':''}" onclick="setAgent('${a}')"><span class="sw" style="background:${ACOL[a]}"></span>${AGENTS[a].name}</button>`).join('');
 $('agentnote').textContent = S.agent==='all' ? 'Showing every recorded attempt. Multi-agent / pooled edges use indigo; grey edges are work the repo record does not attribute to a named model.' : `${AGENTS[S.agent].name}: ${AGENTS[S.agent].role}. Other agents' edges are dimmed.`; }
function setAgent(a){ S.agent=a; agentButtons(); drawGraph(); }

/* ---- Zoom & pan: the whole graph lives in <g id="viewport">; view state
   {k=scale, tx, ty} is applied as its transform. Wheel zooms to the cursor,
   drag pans, two-pointer pinch zooms, buttons/keys mirror the same ops. ---- */
const VIEW = {k:1, tx:0, ty:0};
let suppressClick = false;
const viewTransform = () => `translate(${VIEW.tx},${VIEW.ty}) scale(${VIEW.k})`;
function applyView(){ const vp=document.getElementById('viewport'); if(vp) vp.setAttribute('transform', viewTransform()); const z=$('zoompct'); if(z) z.textContent = Math.round(VIEW.k*100)+'%'; }
const SW=()=>+$('graph').getAttribute('width'), SH=()=>+$('graph').getAttribute('height');
function svgPoint(evt){ const svg=$('graph').getBoundingClientRect(); return {x:(evt.clientX-svg.left)*SW()/svg.width, y:(evt.clientY-svg.top)*SH()/svg.height}; }
function zoomAt(px, py, factor){ const k2=Math.max(0.15, Math.min(5, VIEW.k*factor)); const f=k2/VIEW.k; VIEW.tx = px-(px-VIEW.tx)*f; VIEW.ty = py-(py-VIEW.ty)*f; VIEW.k=k2; applyView(); }
function zoomStep(f){ zoomAt(SW()/2, SH()/2, f); }
function resetView(){ VIEW.k=1; VIEW.tx=0; VIEW.ty=0; applyView(); }
function fitView(){ const vis=NODES.filter(n=>!hiddenNodes().has(n.id)); if(!vis.length) return; const x0=Math.min(...vis.map(n=>n.x))-30, x1=Math.max(...vis.map(n=>n.x))+W+30, y0=Math.min(...vis.map(n=>n.y))-110, y1=Math.max(...vis.map(n=>n.y))+H+30; VIEW.k=Math.max(0.15, Math.min(2, Math.min(SW()/(x1-x0), SH()/(y1-y0)))); VIEW.tx=(SW()-(x1-x0)*VIEW.k)/2 - x0*VIEW.k; VIEW.ty=(SH()-(y1-y0)*VIEW.k)/2 - y0*VIEW.k; applyView(); }
function initZoom(){
 const svg=$('graph');
 svg.addEventListener('wheel', e=>{ e.preventDefault(); const pt=svgPoint(e); zoomAt(pt.x, pt.y, Math.exp(-e.deltaY*0.0015)); }, {passive:false});
 const pointers=new Map(); let drag=null, pinchD=0;
 svg.addEventListener('pointerdown', e=>{ pointers.set(e.pointerId,{x:e.clientX,y:e.clientY}); svg.setPointerCapture(e.pointerId);
   if(pointers.size===1) drag={x:e.clientX,y:e.clientY,moved:0};
   if(pointers.size===2){ const [a,b]=[...pointers.values()]; pinchD=Math.hypot(a.x-b.x,a.y-b.y); drag=null; } });
 svg.addEventListener('pointermove', e=>{ if(!pointers.has(e.pointerId)) return; pointers.set(e.pointerId,{x:e.clientX,y:e.clientY});
   if(pointers.size===2){ const [a,b]=[...pointers.values()]; const d=Math.hypot(a.x-b.x,a.y-b.y); const r=svg.getBoundingClientRect(); const mid={x:(a.x+b.x)/2,y:(a.y+b.y)/2}; if(pinchD>0) zoomAt((mid.x-r.left)*SW()/r.width,(mid.y-r.top)*SH()/r.height, d/pinchD); pinchD=d; return; }
   if(!drag) return; const dx=e.clientX-drag.x, dy=e.clientY-drag.y; drag.moved+=Math.abs(dx)+Math.abs(dy); if(drag.moved>4) suppressClick=true;
   const r=svg.getBoundingClientRect(); VIEW.tx+=dx*SW()/r.width; VIEW.ty+=dy*SH()/r.height; drag.x=e.clientX; drag.y=e.clientY; applyView(); });
 const end=e=>{ pointers.delete(e.pointerId); if(pointers.size<2) pinchD=0; if(pointers.size===0){ drag=null; setTimeout(()=>suppressClick=false, 50); } };
 svg.addEventListener('pointerup', end); svg.addEventListener('pointercancel', end);
 svg.addEventListener('dblclick', e=>{ if(e.target===svg){ const pt=svgPoint(e); zoomAt(pt.x, pt.y, 1.4); } });
 document.addEventListener('keydown', e=>{ if(e.target.matches('input,textarea,select')) return;
   if(e.key==='+'||e.key==='=') zoomStep(1.25); else if(e.key==='-') zoomStep(0.8);
   else if(e.key==='0') resetView(); else if(e.key.toLowerCase()==='f') fitView(); });
}

/* ---- Collapsible downstream: double-click an internal node to hide/reveal
   ALL nodes downstream of it (following from->to edges). Category hubs
   (root children, organism hubs, Thompson chapter hub) act as folders. ---- */
const CHILDREN={}; EDGES.forEach(e=>{ (CHILDREN[e.from] ??= []).push(e.to); });
const COLLAPSED=new Set();
function descendants(id){ const seen=new Set(), stack=[...(CHILDREN[id]||[])]; while(stack.length){ const c=stack.pop(); if(seen.has(c)) continue; seen.add(c); (CHILDREN[c]||[]).forEach(x=>stack.push(x)); } return seen; }
function hiddenNodes(){ const h=new Set(); COLLAPSED.forEach(id=>descendants(id).forEach(x=>h.add(x))); COLLAPSED.forEach(id=>h.delete(id)); return h; }
function toggleCollapse(id){ COLLAPSED.has(id) ? COLLAPSED.delete(id) : COLLAPSED.add(id); drawGraph(); drawDetail(); }
function expandAll(){ COLLAPSED.clear(); drawGraph(); }
function collapseCategories(){ COLLAPSED.clear(); EDGES.filter(e=>e.from==='root').forEach(e=>COLLAPSED.add(e.to)); ['pred-danio','pred-drosophila','pred-xenopus','t-claims'].forEach(id=>{ if(byId[id]) COLLAPSED.add(id); }); drawGraph(); }

function drawGraph(){
 const HIDDEN=hiddenNodes();
 const svg=$('graph'); const NS='http://www.w3.org/2000/svg'; svg.innerHTML='';
 const vp=document.createElementNS(NS,'g'); vp.setAttribute('id','viewport'); vp.setAttribute('transform', viewTransform()); svg.appendChild(vp);
 const lanes=['Thompson 1917','MorphoEval flagships','Rule 30 — main proof chain','Rule 30 prizes','Levin programme','Bioelectric embryo'], lx=[95,330,565,810,1050,1290];
 lanes.forEach((l,i)=>{const t=document.createElementNS(NS,'text');t.setAttribute('x',lx[i]);t.setAttribute('y',96);t.setAttribute('text-anchor','middle');t.setAttribute('font-size','11');t.setAttribute('font-weight','700');t.setAttribute('fill','#888');t.textContent=l.toUpperCase();vp.appendChild(t);});
 const defs=document.createElementNS(NS,'defs');
 Object.keys(AGENTS).forEach(a=>{const m=document.createElementNS(NS,'marker');m.setAttribute('id','arr-'+a);m.setAttribute('markerWidth','7');m.setAttribute('markerHeight','7');m.setAttribute('refX','6');m.setAttribute('refY','3.5');m.setAttribute('orient','auto');m.innerHTML=`<path d="M0,0 L7,3.5 L0,7 z" fill="${ACOL[a]}"/>`;defs.appendChild(m);});
 svg.appendChild(defs);
 EDGES.forEach((e,i)=>{ const a=byId[e.from],b=byId[e.to]; if(!a||!b||HIDDEN.has(e.from)||HIDDEN.has(e.to))return;
  const x1=a.x+W/2,y1=a.y+H,x2=b.x+W/2,y2=b.y,mid=(y1+y2)/2,off=(i%5-2)*3;
  const p=document.createElementNS(NS,'path');
  p.setAttribute('d', Math.abs(x1-x2)<4?`M${x1+off},${y1} L${x2+off},${y2}`:`M${x1+off},${y1} C${x1+off},${mid} ${x2+off},${mid} ${x2+off},${y2}`);
  p.setAttribute('fill','none'); p.setAttribute('stroke',ACOL[e.agent]);
  const dim = S.agent!=='all'&&S.agent!==e.agent;
  p.setAttribute('stroke-width',dim?'1':'2.2'); p.setAttribute('opacity',dim?'0.10':'0.9');
  p.setAttribute('marker-end',`url(#arr-${e.agent})`); vp.appendChild(p);
  if(e.prob!==undefined){ const t=document.createElementNS(NS,'text'); t.setAttribute('x',(x1+x2)/2+off+4); t.setAttribute('y',mid-3); t.setAttribute('font-size','9.5'); t.setAttribute('font-weight','700'); t.setAttribute('fill',probColor(e.prob)); t.textContent=e.prob+'%'; vp.appendChild(t); } });
 NODES.forEach(n=>{ if(HIDDEN.has(n.id)) return; const g=document.createElementNS(NS,'g'); g.style.cursor='pointer';
  g.addEventListener('click',()=>{ if(suppressClick) return; S.sel=n.id;drawGraph();drawDetail(); if(typeof PRED!=='undefined' && PRED && PRED.predictions.some(p=>p.id===n.id)){ PS.sel=n.id; PS.org='all'; renderPred(); } if(typeof THOMP!=='undefined' && THOMP){ const ch=THOMP.chapters.find(c=>c.node===n.id); if(ch){ CS.chapter=ch.title; renderClaims(); $('claimpanel').scrollIntoView({behavior:'smooth', block:'center'}); } } if(window.innerWidth<=760) openDrawer(n.id); });
  g.addEventListener('dblclick',()=>{ if((CHILDREN[n.id]||[]).length) toggleCollapse(n.id); else openDrawer(n.id); });
  const prob=(n.id==='p1'||n.id==='p2'||n.id==='p3')?DELPHI[n.id].pooled.posterior:undefined;
  const fuzzy=n.timeKind==='probabilistic';
  const nDown=descendants(n.id).size, isCollapsed=COLLAPSED.has(n.id);
  let h=`<title>${esc(n.label)} — click to select${nDown?`, double-click to ${isCollapsed?'expand':'collapse'} ${nDown} downstream nodes`:', double-click to inspect'}</title>`;
  if(isCollapsed) h+=`<rect x="${n.x-4}" y="${n.y-4}" width="${W+8}" height="${H+8}" rx="9" fill="none" stroke="#2563eb" stroke-width="2" stroke-dasharray="7 4"/>`;
  if(nDown) h+=`<path d="M ${n.x+W-14} ${n.y+2} L ${n.x+W-2} ${n.y+2} L ${n.x+W-2} ${n.y+14} Z" fill="${isCollapsed?'#fff':'#2563eb'}" stroke="#2563eb" stroke-width="1.2"><title>${isCollapsed?`collapsed — double-click to expand ${nDown} downstream nodes`:`double-click to collapse ${nDown} downstream nodes`}</title></path>`;
  if(isCollapsed) h+=`<text x="${n.x+W-6}" y="${n.y+H-8}" text-anchor="end" font-size="8" font-weight="700" fill="#2563eb">+${nDown} hidden</text>`;
  if(fuzzy)h+=`<rect x="${n.x-4}" y="${n.y-4}" width="${W+8}" height="${H+8}" rx="9" fill="none" stroke="#999" stroke-dasharray="2 5" opacity=".6"/>`;
  h+=`<rect x="${n.x}" y="${n.y}" width="${W}" height="${H}" rx="7" fill="${n.status==='root'?'#eef2ff':'#fff'}" stroke="${prob!==undefined?probColor(prob):(n.id===S.sel?'#2563eb':'#bbb')}" stroke-width="${prob!==undefined?3:(n.id===S.sel?2.5:1.2)}" ${fuzzy?'stroke-dasharray="6 4"':''}/>`;
  h+=`<path d="M ${n.x+2} ${n.y+2} L ${n.x+14} ${n.y+2} L ${n.x+2} ${n.y+14} Z" fill="${SCOL[n.status]}" stroke="#fff" stroke-width="1"><title>status: ${SLAB[n.status]}</title></path>`;
  if(prob!==undefined){ const ag=DELPHI[n.id].agents; h+=ag.map((r,i)=>`<rect x="${n.x+i*W/3}" y="${n.y+H-12}" width="${W/3}" height="12" fill="${probColor(r.posterior)}"><title>${esc(r.agent)}: ${r.posterior}%</title></rect>`).join('')+`<text x="${n.x+W/2}" y="${n.y+H-3}" text-anchor="middle" font-size="8.5" font-weight="700" fill="#fff">bridge ${prob}% pooled</text>`; }
  if(n.id==='r6')h+=`<rect x="${n.x}" y="${n.y+H-5}" width="${W/3}" height="5" fill="${probColor(DELPHI.p1.pooled.posterior)}"/><rect x="${n.x+W/3}" y="${n.y+H-5}" width="${W/3}" height="5" fill="${probColor(DELPHI.p2.pooled.posterior)}"/><rect x="${n.x+2*W/3}" y="${n.y+H-5}" width="${W/3}" height="5" fill="${probColor(DELPHI.p3.pooled.posterior)}"/>`;
  const labFs=n.label.length>23?9.6:(n.label.length>19?10.4:11.5);
  h+=`<text x="${n.x+W/2}" y="${n.y+19}" text-anchor="middle" font-size="${labFs}" font-weight="700" fill="#1c1c1e">${esc(n.label)}</text>`;
  if(n.nanopubs&&n.nanopubs.length)h+=`<text x="${n.x+6}" y="${n.y+H-8}" font-size="8" font-weight="700" fill="#2563eb">🔗 ${n.nanopubs.length} live nanopub${n.nanopubs.length>1?'s':''}</text>`;
  if(fuzzy)h+=`<text x="${n.x+W-8}" y="${n.y+14}" text-anchor="end" font-size="9" font-weight="700" fill="#888">~prob.</text>`;
  if(n.id==='r6')h+=`<text x="${n.x+W-8}" y="${n.y+H-9}" text-anchor="end" font-size="8" font-weight="700" fill="#2563eb">DOUBLE-CLICK TO INSPECT</text>`;
  h+=`<foreignObject x="${n.x+6}" y="${n.y+23}" width="${W-12}" height="44"><div xmlns="http://www.w3.org/1999/xhtml" style="font-size:9px;line-height:1.2;color:#5c5c60;text-align:center;font-family:inherit">${esc(n.sub)}</div></foreignObject>`;
  g.innerHTML=h; vp.appendChild(g); });
}
function npLive(uri){ return 'https://registry.knowledgepixels.com/np/' + uri.split('/').pop(); }
function npLinks(n){ if(!n.nanopubs||!n.nanopubs.length) return ''; return `<hr style="border:none;border-top:1px solid var(--line)"><b class="small">Live nanopublications</b>` + n.nanopubs.map(np=>`<div class="row"><span class="small">🔗 <a href="${npLive(np.uri)}" target="_blank" rel="noreferrer">${esc(np.label)}</a> <span class="mut">· canonical: <a href="${np.uri}" target="_blank" rel="noreferrer">${esc(np.uri)}</a></span></span></div>`).join('') + `<p class="small mut">Live view = Knowledge Pixels registry (canonical morpholib w3id URIs currently 404 at w3id; registry view verified live).</p>`; }
function edgesList(id){ return EDGES.filter(e=>e.from===id||e.to===id).map(e=>`<div class="row"><span class="sw" style="background:${ACOL[e.agent]}"></span><span class="small">${e.from===id?'→':'←'} ${esc(e.from===id?byId[e.to].label:byId[e.from].label)} — ${AGENTS[e.agent].name}${e.label?` (${esc(e.label)})`:''}</span>${e.prob!==undefined?probBadge(e.prob)+`<span class="small mut">Delphi path-success${e.probNote?` · ${esc(e.probNote)}`:''}</span>`:`<span class="badge" style="background:#888">NOT SCORED</span>`}</div>`).join('') || '<p class="small mut">No recorded edges.</p>'; }
function drawDetail(){ const n=byId[S.sel]; const nd=descendants(n.id).size; $('detail').innerHTML=`<div class="row"><b style="flex:1">${esc(n.label)}</b><span class="badge" style="background:${SCOL[n.status]}">${SLAB[n.status]}</span>${nd?`<button onclick="toggleCollapse('${n.id}')">${COLLAPSED.has(n.id)?'▶ Expand':'▼ Collapse'} downstream (${nd})</button>`:''}<button class="on" onclick="openDrawer('${n.id}')">Inspect</button></div>
<p class="small"><b>Family:</b> ${esc(n.family)} · <b>Timing:</b> ${esc(n.time)}${n.timeKind==='probabilistic'?' (probabilistic — see timeline)':''}</p><p class="small">${esc(n.detail)}</p><p class="small mut">Source: ${esc(n.source)}</p>${npLinks(n)}<hr style="border:none;border-top:1px solid var(--line)"><b class="small">Attempts into / out of this node, by agent</b>${edgesList(n.id)}`; }

function openDrawer(id){ S.sel=id; if(id==='p1'||id==='p2'||id==='p3'){S.prize=id;S.prior=DELPHI[id].pooled.prior;S.agentIdx=3;S.subgoal=DELPHI[id].subgoals[3].id;} if(id==='r6'){S.prize='p1';S.prior=DELPHI.p1.pooled.prior;S.agentIdx=3;S.subgoal='P1.4';} drawGraph(); drawDetail(); renderDrawer(); $('drawer').style.display='block'; }
function closeDrawer(){ $('drawer').style.display='none'; }
function setPrize(p){ S.prize=p; S.prior=DELPHI[p].pooled.prior; S.agentIdx=3; S.subgoal=DELPHI[p].subgoals[3].id; renderDrawer(); }
function bar(v,c){ return `<span class="bar"><i style="width:${Math.max(0,Math.min(100,v))}%;background:${c}"></i></span>`; }

function renderDrawer(){
 const n=byId[S.sel]; const isD = n.id==='r6'||n.id==='p1'||n.id==='p2'||n.id==='p3';
 let h=`<div class="row"><div style="flex:1"><b>${isD?'Inspector — Bayesian Delphi panel':'Inspector — '+esc(n.label)}</b><div class="small mut">${isD?'Every number transcribed from docs/RULE30_BAYESIAN_DELPHI_ASSESSMENT.md (2026-10-20). Click prizes, agents and subgoals.':esc(n.sub)}</div></div><span class="badge" style="background:${SCOL[n.status]}">${SLAB[n.status]}</span><button onclick="closeDrawer()">Close</button></div><hr style="border:none;border-top:1px solid var(--line)">`;
 if(!isD){ h+=`<p class="small"><b>Family:</b> ${esc(n.family)} · <b>Timing:</b> ${esc(n.time)}</p><p class="small">${esc(n.detail)}</p><p class="small mut">Source: ${esc(n.source)}</p>${npLinks(n)}<b class="small">Attempts through this node</b>${edgesList(n.id)}<div class="card" style="background:#f3f6ff">Tip: double-click the Bayesian Delphi panel node (or Prize 1 / 2 / 3) for the deep per-agent, per-subgoal inspector.</div>`; $('drawer').innerHTML=h; return; }
 const d=DELPHI[S.prize], rows=[...d.agents, d.pooled], row=rows[Math.min(S.agentIdx,rows.length-1)], sg=d.subgoals.find(s=>s.id===S.subgoal)||d.subgoals[0];
 const odds=(S.prior/100)/(1-S.prior/100)*d.pooled.bf, post=100*odds/(1+odds);
 h+=`<div class="row">${['p1','p2','p3'].map(p=>`<button class="${S.prize===p?'on':''}" onclick="setPrize('${p}')"><span class="badge" style="background:${probColor(DELPHI[p].pooled.posterior)}">&nbsp;</span> ${DELPHI[p].title} · ${DELPHI[p].pooled.posterior}%</button>`).join('')}</div>`;
 h+=`<p class="small mut">${esc(d.statement)}</p>`;
 h+=`<div class="row"><span class="small"><b>Posterior shade scale:</b></span>${[85,75,60,40,20,3].map(v=>probBadge(v)).join('')}<span class="small">Green is earned: light green only &gt;70%, dark green &gt;80%. Below 70% the shade runs yellow → amber → burnt orange → deep red — a posterior in the teens is WEAK, not green. Per-agent rows below show the full range of opinion behind each pooled shade.</span></div>`;
 h+=`<div class="grid4"><div class="card"><div class="small mut">Proved now</div><div class="stat">0.00%</div><div class="small mut">All 3 prizes open</div></div><div class="card"><div class="small mut">Conjecture truth</div><div class="stat">${d.conjectureTruth}%</div><div class="small mut">Is it true at all?</div></div><div class="card"><div class="small mut">Subgoal coverage</div><div class="stat">${d.subgoalTotal.panel}/100</div><div class="small mut">Weighted rubric</div></div><div class="card"><div class="small mut">Bridge posterior</div><div class="stat" style="color:${probColor(d.pooled.posterior)}">${d.pooled.posterior}%</div>${probBadge(d.pooled.posterior)}<div class="small mut">Closable with current machinery?</div></div></div>`;
 h+=`<div class="card" style="border-color:#E8C030"><b>Do not conflate these.</b> <span class="small">A 99.98% conjecture-truth score is not a proof, and subgoal coverage is not a fraction of the way to a referee-accepted proof. The bridge posterior answers the third, separate question.</span></div>`;
 h+=`<h2>Per-agent bridge ledger — click an agent</h2>`;
 rows.forEach((r,i)=>{ const col=r.agent.includes('Fable')?ACOL.fable:r.agent.includes('GPT')?ACOL.gpt:r.agent.includes('Gemini')?ACOL.gemini:ACOL.panel;
  h+=`<div class="clickable ${i===S.agentIdx?'sel':''}" onclick="S.agentIdx=${i};renderDrawer()"><div class="row"><span class="badge" style="background:${col}">&nbsp;</span><b class="small" style="flex:1">${r.agent}</b><span class="small">${r.prior}% → <b style="color:${probColor(r.posterior)}">${r.posterior}%</b></span>${probBadge(r.posterior)}<span class="small mut">net ${r.netDb>0?'+':''}${r.netDb} dB · BF ${r.bf}</span></div></div>`; });
 h+=`<div class="card"><b class="small">${row.agent} — selected</b><p class="small">Prior ${row.prior}% · for ${row.forDb} dB · against ${row.againstDb} dB · net ${row.netDb>0?'+':''}${row.netDb} dB (BF ${row.bf}) → posterior <b style="color:${probColor(row.posterior)}">${row.posterior}%</b> · confidence ${row.confidence}</p><p class="small mut">Verdict: ${esc(row.verdict)}</p><p class="small mut">Disagreement is data: on Prize 1 the agents span 5.58–21.82% (3.9×) while agreeing exactly on Prize 2's net −4.0 dB. Only GPT rates Prize 1 above 15% at all (21.8% — burnt orange on the shade scale, nowhere near the 70% green line).</p></div>`;
 h+=`<h2>Subgoals — click to inspect</h2>`;
 d.subgoals.forEach(s=>{ h+=`<div class="clickable ${s.id===sg.id?'sel':''}" onclick="S.subgoal='${s.id}';renderDrawer()"><div class="row"><b class="small">${s.id}</b><span class="small" style="flex:1">${esc(s.label)}</span><span class="small mut">w=${s.weight}% · panel ${s.panel}</span></div></div>`; });
 h+=`<div class="card"><b class="small">${sg.id} · weight ${sg.weight}% · ${esc(sg.tier)}</b><p class="small">${esc(sg.label)}</p>` +
   [['Claude Fable 5.1',sg.fable,ACOL.fable],['GPT 5.6 Sol',sg.gpt,ACOL.gpt],['Gemini 4 Argon',sg.gemini,ACOL.gemini],['Panel mean',sg.panel,ACOL.panel]].map(([nm,v,c])=>`<div class="row"><span class="small" style="width:130px">${nm}</span>${bar(v,c)}<b class="small">${v}/100</b></div>`).join('') +
   `<p class="small mut">Agent spread: ${Math.max(sg.fable,sg.gpt,sg.gemini)-Math.min(sg.fable,sg.gpt,sg.gemini)} pts. The highest-weight subgoal is usually the open one.</p></div>`;
 h+=`<div class="grid2"><div><h3>Evidence for (+${d.pooled.forDb} dB pooled)</h3>${d.evidenceFor.map(e=>`<div class="row"><span style="display:inline-block;height:8px;width:${Math.min(70,Math.abs(e.db)*10)}px;background:#1F8A65"></span><span class="small">+${e.db} dB · ${esc(e.text)}</span></div>`).join('')}</div><div><h3>Obstructions (${d.pooled.againstDb} dB pooled)</h3>${d.evidenceAgainst.map(e=>`<div class="row"><span style="display:inline-block;height:8px;width:${Math.min(70,Math.abs(e.db)*10)}px;background:#C06028"></span><span class="small">${e.db} dB · ${esc(e.text)}</span></div>`).join('')}</div></div>`;
 h+=`<h2>Prior sensitivity — try it</h2><p class="small mut">Posterior = prior odds × BF (${d.pooled.bf}) for ${esc(d.title)}.</p><div class="row"><button onclick="adjPrior(-5)">−5 pts</button><button onclick="adjPrior(-1)">−1</button><b style="color:${probColor(post)}">Prior ${S.prior}% → posterior ${post.toFixed(2)}%</b>${probBadge(+post.toFixed(2))}<button onclick="adjPrior(1)">+1</button><button onclick="adjPrior(5)">+5 pts</button><button onclick="S.prior=${d.pooled.prior};renderDrawer()">Reset to pooled prior</button></div><p class="small mut">Watch the shade: it stays yellow/amber/red until the posterior crosses 70% (light green) and only turns dark green over 80%. For reference, merely reaching the old 15% “possible” line needs a prior ≳9.4% (Prize 1), ≳30.7% (Prize 2), ≳82.7% (Prize 3) — no plausible prior rescues Prize 3 on this evidence, and none of the three gets near green.</p>`;
 h+=`<h2>Inter-agent concordance (all 15 subgoals)</h2>` + CONCORDANCE.map(c=>`<div class="row"><b class="small" style="width:150px">${c.pair}</b><span class="small">Spearman ρ ${c.spearman} · Kendall τ ${c.kendall} · Pearson r ${c.pearson} · mean |dev| ${c.mad} pts</span></div>`).join('') + `<p class="small mut">Source: docs/RULE30_BAYESIAN_DELPHI_ASSESSMENT.md §§1–3 · Project Delphi 2.0 · resolvability P1 0.85 · P2 0.65 · P3 0.25.</p>`;
 $('drawer').innerHTML=h;
}
function adjPrior(d){ S.prior=Math.max(0.1,Math.min(90,+(S.prior+d).toFixed(2))); renderDrawer(); }
/* ---- Registered predictions panel (data injected from the registry) ---- */
/*__PRED__*/
const PCOL = {confirmed:'#178A5F', partially_resolved:'#E79A28', refuted:'#7E2A3A', registered_untested:'#3E7FB8', confirmed_meta:'#178A5F'};
const PLAB = {confirmed:'CONFIRMED', partially_resolved:'PARTIALLY RESOLVED', refuted:'REFUTED', registered_untested:'REGISTERED, UNSCORED'};
const RCAT = {current_datasets_only:'solvable in-repo alone', current_plus_public_sources:'solvable in-repo + public sources', partial_only:'partially solvable only'};
const PS = {org:'all', sel:null};
function predList(){ return PRED ? PRED.predictions.filter(p=>PS.org==='all'||p.organism===PS.org) : []; }
function renderPred(){
 const el=$('predpanel'); if(!PRED){ el.innerHTML='<p class="small mut">Prediction registry not available in this build.</p>'; return; }
 const t=PRED.tallies||{}, orgs=[...new Set(PRED.predictions.map(p=>p.organism))];
 if(!PS.sel && PRED.predictions.length) PS.sel=PRED.predictions[0].id;
 const sel=PRED.predictions.find(p=>p.id===PS.sel) || PRED.predictions[0];
 const rows=predList().map(p=>`<div class="prow ${p.id===sel.id?'sel':''}" onclick="PS.sel='${p.id}';renderPred()"><span class="dot" style="background:${PCOL[p.outcome]||'#888'}"></span><b class="small">${p.id}</b> <span class="small">${esc(p.organism)} · ${esc(p.mechanism)}</span><br><span class="badge" style="background:${PCOL[p.outcome]||'#888'}">${PLAB[p.outcome]||p.outcome}</span> <span class="small mut">${RCAT[p.res_category]||p.res_category} (forecast conf ${p.res_confidence}, expected verdict: ${esc(p.expected_verdict)})</span></div>`).join('');
 const metas=PRED.meta.map(m=>`<div class="prow"><span class="dot" style="background:${PCOL[m.outcome]||'#888'}"></span><b class="small">${m.id}</b> <span class="badge" style="background:${PCOL[m.outcome]||'#888'}">${PLAB[m.outcome]||m.outcome}</span><br><span class="small">${esc(m.prediction)}</span></div>`).join('');
 el.innerHTML=`<div class="row"><b style="flex:1">Registered ${PRED.registered_on} — before any scoring. Two independent grades per row: did the biology hold, and could current data even decide it?</b><a class="small" href="${PRED.registry_url}">registry JSON</a><a class="small" href="${PRED.resolutions_url}">resolution records</a></div>
<div class="grid4" style="margin-top:8px">
<div class="card"><div class="small mut">Registered predictions</div><div class="stat">${PRED.predictions.length}</div><div class="small mut">3 organisms · 4 resolution batches</div></div>
<div class="card"><div class="small mut">Outcomes confirmed</div><div class="stat" style="color:#1F8A65">${t.confirmed ?? 0}</div><div class="small mut">${t.refuted ?? 0} refuted</div></div>
<div class="card"><div class="small mut">Partially resolved</div><div class="stat" style="color:#B87700">${t.partially_resolved ?? 0}</div><div class="small mut">each names its missing series — follow-ups on <a href="morphoswarm.html">MorphoSwarm</a></div></div>
<div class="card"><div class="small mut">Meta-resolvability</div><div class="stat">${PRED.meta.filter(m=>m.outcome==='confirmed').length}/${PRED.meta.length}</div><div class="small mut">solve-rate forecasts confirmed (2 refuted — forecast was optimistic by one row)</div></div></div>
<div class="row"><button class="${PS.org==='all'?'on':''}" onclick="PS.org='all';renderPred()">All organisms</button>${orgs.map(o=>`<button class="${PS.org===o?'on':''}" onclick="PS.org='${o}';renderPred()">${o}</button>`).join('')}</div>
<div class="predgrid"><div>${rows}</div>
<div class="card" style="margin:0"><b>${sel.id} · ${esc(sel.organism)} — ${esc(sel.mechanism)}</b> <span class="badge" style="background:${PCOL[sel.outcome]||'#888'}">${PLAB[sel.outcome]||sel.outcome}</span>
<p class="small"><b>Outcome prediction:</b> ${esc(sel.prediction)}</p>
<p class="small"><b>Quantitative expectation:</b> ${esc(sel.quant)}</p>
<p class="small"><b>Resolvability forecast:</b> ${RCAT[sel.res_category]||sel.res_category} (confidence ${sel.res_confidence}; expected verdict ${esc(sel.expected_verdict)}) · <b>Falsifier:</b> ${esc(sel.falsifier)}</p>
<p class="small"><b>Test source:</b> ${esc(sel.test_source)}</p>
${sel.evidence?`<p class="small"><b>Resolution evidence:</b> ${esc(sel.evidence)}</p>`:''}
${sel.resolvability_verdict?`<p class="small"><b>Resolvability verdict:</b> ${esc(sel.resolvability_verdict)}</p>`:''}
<p class="small mut">Datasets: ${sel.datasets.map(esc).join(', ')}</p></div></div>
<h3>Meta-resolvability predictions — grading the solve-rate forecast itself</h3>${metas}`;
}
/* ---- Thompson 1917 claims browser: all 59 claims, browsable from the graph.
   Chapter graph nodes (tch-*) set the chapter filter; every claim row opens
   its full Pass-1 citation + Pass-2 assessment. ---- */
/*__THOMP__*/
const CCOL = {DONE_REPO:'#1F8A65', DONE_ELSEWHERE:'#2A9A8A', PARTIAL_REPO:'#F0A040', UNTESTED:'#C04848', NOT_TESTABLE_AS_STATED:'#888'};
const CS = {chapter:'all', status:'all', q:'', sel:null};
function claimList(){ if(!THOMP) return []; const q=CS.q.trim().toLowerCase();
 return THOMP.claims.filter(c=>(CS.chapter==='all'||c.chapter_title===CS.chapter)&&(CS.status==='all'||c.status===CS.status)&&(!q||(c.id+' '+c.claim+' '+c.chapter_title).toLowerCase().includes(q))); }
function renderClaims(){
 const el=$('claimpanel'); if(!THOMP){ el.innerHTML='<p class="small mut">Thompson claims registry not available in this build.</p>'; return; }
 if(!CS.sel) CS.sel=THOMP.claims[0].id;
 const sel=THOMP.claims.find(c=>c.id===CS.sel)||THOMP.claims[0];
 const list=claimList();
 const rows=list.map(c=>`<div class="prow ${c.id===sel.id?'sel':''}" onclick="CS.sel='${c.id}';renderClaims()"><span class="dot" style="background:${CCOL[c.status]}"></span><b class="small">${c.id}</b> <span class="badge" style="background:${CCOL[c.status]}">${c.status}</span> <span class="small mut">Ch. ${esc(c.chapter)} · pp. ${esc(c.pages)} · Figs. ${esc(c.figures)} · ${esc(c.priority)} · effort ${esc(c.effort)}</span><br><span class="small">${esc(c.claim.slice(0,150))}${c.claim.length>150?'…':''}</span></div>`).join('') || '<p class="small mut">No claims match these filters.</p>';
 el.innerHTML=`<div class="row"><b style="flex:1">All ${THOMP.n} claims from <i>On Growth and Form</i> (1917) — cited, then assessed against this repo</b><a class="small" href="${THOMP.inventory_url}">full inventory</a></div>
<p class="small mut">Status: <span class="badge" style="background:#1F8A65">DONE_REPO 9</span> <span class="badge" style="background:#F0A040">PARTIAL_REPO 13</span> <span class="badge" style="background:#2A9A8A">DONE_ELSEWHERE 13</span> <span class="badge" style="background:#C04848">UNTESTED 21</span> <span class="badge" style="background:#888">NOT_TESTABLE 3</span> · In the graph above, the <b>Thompson 1917 claims</b> hub (bottom band) groups these by chapter — click a chapter node to filter here, double-click it to collapse/expand its place in the graph.</p>
<div class="row"><select onchange="CS.chapter=this.value;renderClaims()" aria-label="Filter by chapter"><option value="all">All chapters (${THOMP.n})</option>${THOMP.chapters.map(ch=>`<option ${CS.chapter===ch.title?'selected':''} value="${esc(ch.title)}">${esc(ch.title)} (${THOMP.claims.filter(c=>c.chapter_title===ch.title).length})</option>`).join('')}</select>
<select onchange="CS.status=this.value;renderClaims()" aria-label="Filter by status"><option value="all">All statuses</option>${Object.keys(CCOL).map(s=>`<option ${CS.status===s?'selected':''}>${s}</option>`).join('')}</select>
<input type="search" placeholder="Search claims…" value="${esc(CS.q)}" oninput="CS.q=this.value;renderClaims()" style="flex:1;min-width:140px;font:inherit;font-size:12.5px;padding:5px 9px;border:1px solid var(--line);border-radius:7px" aria-label="Search claims">
<span class="small mut">showing ${list.length} / ${THOMP.n}</span></div>
<div class="predgrid"><div style="max-height:560px;overflow-y:auto">${rows}</div>
<div class="card" style="margin:0"><b>${sel.id} — Thompson (1917), Ch. ${esc(sel.chapter)} · ${esc(sel.chapter_title)}</b> <span class="badge" style="background:${CCOL[sel.status]}">${sel.status}</span>
<p class="small"><b>Claim (1917, pp. ${esc(sel.pages)}, Figs. ${esc(sel.figures)}):</b> ${esc(sel.claim)}</p>
<p class="small"><b>What remains / verdict:</b> ${esc(sel.what_remains)}</p>
<p class="small"><b>Method that would settle it:</b> ${esc(sel.method)}</p>
<p class="small"><b>Data needed:</b> ${esc(sel.data_needed)} · <b>Effort:</b> ${esc(sel.effort)} · <b>Priority:</b> ${esc(sel.priority)} · <b>Type:</b> ${esc(sel.type)}</p></div></div>`;
}
function renderNav(){ const fams=[...new Set(NODES.map(n=>n.family))];
 $('nodepicker').innerHTML='<option value="">Find a problem… ('+NODES.length+' nodes)</option>'+fams.map(f=>`<optgroup label="${esc(f)}">`+NODES.filter(n=>n.family===f).map(n=>`<option value="${n.id}">${esc(n.label)} — ${SLAB[n.status]}</option>`).join('')+'</optgroup>').join('');
 $('nodelist').innerHTML=fams.map(f=>`<div class="nlfam">${esc(f)}</div>`+NODES.filter(n=>n.family===f).map(n=>`<button class="nlrow" onclick="pickNode('${n.id}')"><span style="flex:1"><b>${esc(n.label)}</b><br><span class="mut small">${esc(n.sub)}</span></span><span class="badge" style="background:${SCOL[n.status]}">${SLAB[n.status]}</span>${n.nanopubs?'<span class="small">🔗</span>':''}</button>`).join('')).join(''); }
function setView(v){ const list=v==='list'; $('nodelist').hidden=!list; $('graphwrap').style.display=list?'none':''; $('vlist').classList.toggle('on',list); $('vgraph').classList.toggle('on',!list); }
function pickNode(id){ if(!byId[id]) return; S.sel=id; const p=$('nodepicker'); if(p) p.value=id; drawGraph(); drawDetail(); openDrawer(id); }
agentButtons(); drawGraph(); drawDetail(); initZoom(); renderPred(); renderClaims(); renderNav();
if(window.matchMedia('(max-width: 760px)').matches) setView('list');
if(location.hash){ const hid = location.hash.slice(1); if(byId[hid]){ S.sel = hid; drawGraph(); drawDetail(); document.getElementById('detail').scrollIntoView({behavior:'smooth', block:'center'}); } }
</script></body></html>
"""

if __name__ == "__main__":
    main()
