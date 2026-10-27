#!/usr/bin/env python3
"""Turing open problems — three-provider Bayesian Delphi panel (supersedes the provisional pass).

Three independent assessors from three providers scored the same blinded packet in
parallel (OpenAI GPT 5.6 Sol, Google Gemini 4 Argon, Anthropic Claude Fable 5.1 in a
fresh context), each on the two axes of morphoeval/build_turing_open_problems.py:

  1. Resolvability — the deterministic Problem-Scout rubric factors; totals are
     recomputed here by problem_scout_100.score_candidate (never taken from the agent).
  2. Viability — a signed deciban ledger from prior 50%; pooled in log-odds across
     assessors (mean net dB, as the Rule 30 Delphi), range reported as part of the result.

The blinded packet and instructions the assessors saw are vendored in the panel dir
together with their raw returns. The provisional single-assessor pass is NOT edited:
it stays in its own audit/KG dir and is superseded (npx:supersedes) by one new
nanopublication per problem minted here.

Inputs : morphoeval/audits/2026-10-27-turing-open-problems-panel/{packet.json,INSTRUCTIONS.md,<assessor>.json}
Outputs: morphoeval/audits/2026-10-27-turing-open-problems-panel.yaml
         docs/turing-open-problems-panel.json                 (merged into the graph by docs/build_problem_graph_html.py)
         docs/TURING_OPEN_PROBLEMS.md                         (panel section appended; provisional section untouched)
         morphoeval/kg/2026-10-27-turing-open-problems-panel/panel-*.trig (unsigned)
Then   : python3 morphoeval/sign_morphoeval.py morphoeval/kg/2026-10-27-turing-open-problems-panel [--publish test]
         python3 docs/build_problem_graph_html.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean

import yaml
from rdflib import Dataset, Literal, Namespace, URIRef
from rdflib.namespace import DCTERMS, PROV, RDF, RDFS, XSD

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from problem_scout_100 import RUBRIC, score_candidate                      # noqa: E402
from build_morphoeval_kg import plain_doubles                              # noqa: E402
from build_turing_open_problems import (AUDIT_ID as PROV_AUDIT_ID, PAPERS, PROBLEMS, PRIOR_PERCENT,  # noqa: E402
                                        ORCID, REPO, posterior, trig_name)

PANEL_ID = "2026-10-27-turing-open-problems-panel"
DATE = "2026-10-27"
CREATED = "2026-10-27T19:40:00Z"   # fixed so signing is deterministic
PANEL_DIR = HERE / "audits" / PANEL_ID
REPORT_URL = f"{REPO}/blob/main/docs/TURING_OPEN_PROBLEMS.md"
ASSESSORS = {  # id -> (label, provider, graph agent key)
    "gpt-5.6-sol": ("GPT 5.6 Sol Medium", "OpenAI", "gpt"),
    "gemini-4-argon": ("Gemini 4 Argon Medium", "Google", "gemini"),
    "claude-fable-5-1": ("Claude Fable 5.1 Thinking High (fresh context)", "Anthropic", "fable"),
}
# Errors in the blinded packet found by the panel. Recorded, not silently patched: the
# provisional nanopubs that carry the wrong text are immutable and are superseded instead.
PACKET_CORRECTIONS = {
    "TOP-06": ("The packet's tool_route said 'no BSW model exists in models/'. False: models/2014-raspopovic-bmp-sox9-wnt-digits.py "
               "(2,066 lines, Pass 3: Bmp-Sox9-Wnt 3-node RD on a growing polar-fan limb-bud mesh, Crampin-Gaffney-Maini dilution) and "
               "models/2012-sheth-hox-turing-polydactyly.py (1,619 lines) already exist. Caught by the Claude Fable panellist, verified "
               "by the aggregator. The provisional tool_fit (9/25) under-priced this problem; the panel scores stand as recorded, and "
               "a re-score with the corrected route is the next action on TOP-06."),
}

PK = Namespace("https://w3id.org/delphi/ontology/"); PKR = Namespace("https://w3id.org/delphi/resource/")
NP = Namespace("http://www.nanopub.org/nschema#"); NPX = Namespace("http://purl.org/nanopub/x/")
NT = Namespace("https://w3id.org/np/o/ntemplate/"); SUB = Namespace("http://purl.org/nanopub/temp/np001/")


def load_returns() -> dict[str, dict]:
    out = {}
    for aid in ASSESSORS:
        f = PANEL_DIR / f"{aid}.json"
        if not f.exists():
            sys.exit(f"missing assessor return {f}")
        d = json.loads(f.read_text())
        ids = [p["id"] for p in d["problems"]]
        if sorted(ids) != sorted(p["id"] for p in PROBLEMS):
            sys.exit(f"{aid}: problem ids {ids} do not match the packet")
        for p in d["problems"]:
            if set(p["factors"]) != set(RUBRIC):
                sys.exit(f"{aid} {p['id']}: factors {sorted(p['factors'])} != rubric")
            net = sum(e["db"] for e in p["evidence_for"] + p["evidence_against"])
            if abs(net) > 12.01:
                sys.exit(f"{aid} {p['id']}: |net dB| {net} exceeds the 12 dB cap given in INSTRUCTIONS.md")
        out[aid] = d
    return out


def aggregate(returns: dict[str, dict]) -> list[dict]:
    prov = {p["id"]: p for p in PROBLEMS}
    rows = []
    for pid in sorted(prov):
        per = {}
        for aid, d in returns.items():
            p = next(q for q in d["problems"] if q["id"] == pid)
            score, detail = score_candidate(p)
            net = round(sum(e["db"] for e in p["evidence_for"] + p["evidence_against"]), 2)
            per[aid] = {"label": ASSESSORS[aid][0], "provider": ASSESSORS[aid][1], "resolvability": score, "band_detail": detail,
                        "factors": p["factors"], "for_db": round(sum(e["db"] for e in p["evidence_for"]), 2),
                        "against_db": round(sum(e["db"] for e in p["evidence_against"]), 2), "net_db": net,
                        "posterior_percent": round(posterior(PRIOR_PERCENT, net), 2), "confidence": p["confidence"],
                        "verdict": p["verdict"], "note": p.get("note", ""),
                        "evidence_for": p["evidence_for"], "evidence_against": p["evidence_against"]}
        nets = [v["net_db"] for v in per.values()]; res = [v["resolvability"] for v in per.values()]
        pooled_net = round(mean(nets), 2); pooled_post = round(posterior(PRIOR_PERCENT, pooled_net), 2)
        res_mean = round(mean(res), 1)
        rows.append({"id": pid, "slug": prov[pid]["slug"], "title": prov[pid]["title"], "assessors": per,
                     "pooled": {"method": "log-odds mean across 3 assessors (equal priors 50%, so pooled net dB = mean net dB)",
                                "prior_percent": PRIOR_PERCENT, "net_db": pooled_net, "bayes_factor": round(10 ** (pooled_net / 10), 4),
                                "posterior_percent": pooled_post, "panel_range_percent": [min(v["posterior_percent"] for v in per.values()), max(v["posterior_percent"] for v in per.values())],
                                "classification": "possible_pathway" if pooled_post >= 15 else "below_15_percent"},
                     "resolvability_panel": {"mean": res_mean, "range": [min(res), max(res)],
                                             "band": "RESOLVE NOW" if res_mean >= 70 else "RESOLVABLE WITH WORK" if res_mean >= 45 else "HARD HERE" if res_mean >= 25 else "NOT WITH OUR TOOLS (yet)",
                                             "factor_means": {k: round(mean(v["factors"][k]["score"] for v in per.values()), 1) for k in RUBRIC}},
                     "packet_correction": PACKET_CORRECTIONS.get(pid)})
    return rows


def build_nanopub(row: dict, prov_manifest: dict) -> str:
    p = next(q for q in PROBLEMS if q["id"] == row["id"])
    ds = Dataset()
    for pfx, ns in (("np", NP), ("npx", NPX), ("nt", NT), ("pk", PK), ("pkr", PKR), ("prov", PROV), ("dct", DCTERMS), ("rdfs", RDFS), ("xsd", XSD), ("sub", SUB)):
        ds.bind(pfx, ns)
    head, asrt, prov_g, pubinfo = (ds.graph(SUB[g]) for g in ("Head", "assertion", "provenance", "pubinfo"))
    this = URIRef(str(SUB)); L = Literal; D = lambda x: Literal(float(x), datatype=XSD.double)
    head.add((this, RDF.type, NP.Nanopublication)); head.add((this, NP.hasAssertion, SUB.assertion))
    head.add((this, NP.hasProvenance, SUB.provenance)); head.add((this, NP.hasPublicationInfo, SUB.pubinfo))

    cid = f"turing-open-problem-{p['id'].lower()}-{p['slug']}"
    claim = PKR[f"claim/{cid}"]; hyp = PKR[f"hypothesis/{cid}-decisive-verdict"]; report = URIRef(REPORT_URL)
    pooled = row["pooled"]; rp = row["resolvability_panel"]
    asrt.add((claim, RDF.type, PK.Claim)); asrt.add((claim, RDFS.label, L(f"{p['id']} — {p['title']}")))
    corr = f" PACKET CORRECTION: {row['packet_correction']}" if row["packet_correction"] else ""
    asrt.add((claim, DCTERMS.description, L(
        f"OPEN PROBLEM in Turing-pattern biology, sourced forward from {p['anchor']}. THE GAP: {p['gap']} DEFINITION OF DONE: {p['done']} "
        f"THREE-PROVIDER DELPHI PANEL ({DATE}; supersedes the provisional single-assessor pass): resolvability panel mean "
        f"{rp['mean']}/100 (range {rp['range'][0]}-{rp['range'][1]}; {rp['band']}) by the deterministic Problem-Scout rubric, and pooled viability "
        f"posterior {pooled['posterior_percent']}% (prior {PRIOR_PERCENT:g}%, pooled net {pooled['net_db']:+g} dB, panel range "
        f"{pooled['panel_range_percent'][0]}-{pooled['panel_range_percent'][1]}%) that the named route reaches a decisive verdict. "
        f"Assessors were blinded to the provisional scores and to each other.{corr} A belief under assessment, not a fact.")))
    asrt.add((claim, DCTERMS.subject, report)); asrt.add((claim, PK.weightOfEvidence, D(round(pooled["posterior_percent"] / 100, 4))))
    asrt.add((claim, PK.context, L(f"tool_route: {p['tool_route']} | deciding_data: {p['deciding_data']} | prior_attempts: {p['prior_attempts']}")))
    for doi in p["papers"]:
        asrt.add((claim, PK.derivedFromPaper, URIRef(f"https://doi.org/{doi}")))
    asrt.add((hyp, RDF.type, PK.TestableHypothesis)); asrt.add((hyp, RDFS.label, L(p["hypothesis"])))
    asrt.add((hyp, PK.hasPriorProbability, D(PRIOR_PERCENT / 100))); asrt.add((hyp, PK.hasResolvability, D(round(rp["mean"] / 100, 4))))

    panel_agent = PKR["agent/turing-open-problems-panel-2026-10-27"]
    asrt.add((panel_agent, RDF.type, PROV.SoftwareAgent))
    asrt.add((panel_agent, RDFS.label, L("Three-provider Bayesian Delphi panel (OpenAI, Google, Anthropic), blinded, pooled in log-odds")))
    a_pool = PKR[f"assessment/{cid}-viability-panel-pooled"]
    asrt.add((claim, PK.hasAssessment, a_pool)); asrt.add((hyp, PK.hasAssessment, a_pool))
    asrt.add((a_pool, RDF.type, PK.BayesianAssessment)); asrt.add((a_pool, PROV.wasAttributedTo, panel_agent))
    asrt.add((a_pool, RDFS.label, L(f"{p['id']} pooled viability {pooled['posterior_percent']}% (3-provider panel, range {pooled['panel_range_percent'][0]}-{pooled['panel_range_percent'][1]}%)")))
    asrt.add((a_pool, PK.calibrationMethod, L(pooled["method"]))); asrt.add((a_pool, PK.evaluatesHypothesis, hyp))
    asrt.add((a_pool, PK.weightOfEvidenceDeciban, D(pooled["net_db"]))); asrt.add((a_pool, PK.bayesFactor, D(pooled["bayes_factor"])))
    asrt.add((a_pool, PK.hasProbability, D(round(pooled["posterior_percent"] / 100, 4))))
    asrt.add((a_pool, PK.hasVerdict, L(f"POOLED_{pooled['classification'].upper()}")))
    for aid, v in row["assessors"].items():
        model = PKR[f"model/{aid}"]
        asrt.add((model, RDF.type, PK.Model)); asrt.add((model, RDFS.label, L(f"{v['label']} ({v['provider']})")))
        asrt.add((model, DCTERMS.identifier, L(aid))); asrt.add((model, PROV.actedOnBehalfOf, ORCID))
        a_v = PKR[f"assessment/{cid}-viability-{aid}"]; a_r = PKR[f"assessment/{cid}-resolvability-{aid}"]
        asrt.add((claim, PK.hasAssessment, a_v)); asrt.add((hyp, PK.hasAssessment, a_v)); asrt.add((claim, PK.hasAssessment, a_r))
        asrt.add((a_v, RDF.type, PK.BayesianAssessment)); asrt.add((a_v, PROV.wasAttributedTo, model))
        asrt.add((a_v, RDFS.label, L(f"{p['id']} viability {v['posterior_percent']}% — {v['label']}")))
        asrt.add((a_v, PK.calibrationMethod, L(f"Blinded signed deciban ledger, prior {PRIOR_PERCENT:g}%, |item| <= 5 dB, |net| <= 12 dB (panel INSTRUCTIONS.md)")))
        asrt.add((a_v, PK.evaluatesHypothesis, hyp))
        for e in v["evidence_for"]:
            asrt.add((a_v, PK.evidenceFor, L(f"[{e['db']:+g} dB] {e['text']}")))
        for e in v["evidence_against"]:
            asrt.add((a_v, PK.evidenceAgainst, L(f"[{e['db']:+g} dB] {e['text']}")))
        asrt.add((a_v, PK.weightOfEvidenceDeciban, D(v["net_db"]))); asrt.add((a_v, PK.bayesFactor, D(round(10 ** (v["net_db"] / 10), 4))))
        asrt.add((a_v, PK.hasProbability, D(round(v["posterior_percent"] / 100, 4)))); asrt.add((a_v, PK.hasConfidence, D(v["confidence"])))
        asrt.add((a_v, PK.hasVerdict, L(v["verdict"])))
        asrt.add((a_r, RDF.type, PK.LLMClaimAssessment)); asrt.add((a_r, PK.assessedBy, model))
        asrt.add((a_r, RDFS.label, L(f"{p['id']} resolvability {v['resolvability']}/100 — {v['label']}")))
        asrt.add((a_r, DCTERMS.date, Literal(DATE, datatype=XSD.date)))
        asrt.add((a_r, DCTERMS.description, L("Deterministic Problem-Scout rubric (morphoeval/problem_scout_100.py RUBRIC); total recomputed from factor scores by the aggregator; an uncalibrated rubric score, not a probability.")))
        asrt.add((a_r, PK.assessmentType, L("problem-scout/1.0/resolvability")))
        asrt.add((a_r, PK.hasScore, D(round(v["resolvability"] / 100, 4)))); asrt.add((a_r, PK.hasVerdict, L(v["band_detail"].split(" — ")[0])))
        for k, (w, _d) in RUBRIC.items():
            asrt.add((a_r, PK.evidenceFor, L(f"{k} {v['factors'][k]['score']}/{w}: {v['factors'][k]['why']}")))

    prov_g.add((SUB.assertion, PROV.wasAttributedTo, ORCID)); prov_g.add((SUB.assertion, PROV.wasAttributedTo, panel_agent))
    for aid in row["assessors"]:
        prov_g.add((SUB.assertion, PROV.wasAttributedTo, PKR[f"model/{aid}"]))
    for doi in p["papers"]:
        u = URIRef(f"https://doi.org/{doi}"); yr, title = PAPERS[doi]
        prov_g.add((SUB.assertion, PROV.wasDerivedFrom, u)); prov_g.add((u, DCTERMS.title, L(title))); prov_g.add((u, DCTERMS.date, Literal(str(yr), datatype=XSD.gYear)))

    pubinfo.add((this, RDFS.label, L(f"Open problem (3-provider panel): {p['id']} {p['title'][:80]}")))
    pubinfo.add((this, DCTERMS.created, Literal(CREATED, datatype=XSD.dateTime))); pubinfo.add((this, DCTERMS.creator, ORCID))
    pubinfo.add((this, DCTERMS.license, URIRef("https://creativecommons.org/licenses/by/4.0/")))
    pubinfo.add((this, NPX.hasNanopubType, PK.Claim)); pubinfo.add((this, NPX.introduces, claim)); pubinfo.add((this, RDFS.seeAlso, report))
    prev = prov_manifest.get(trig_name(p), {}).get("nanopub")
    if prev:
        pubinfo.add((this, NPX.supersedes, URIRef(prev)))
    dom = PKR["domain/morphogenesis"]; pubinfo.add((this, PK.domain, dom)); pubinfo.add((dom, RDFS.label, L("morphogenesis")))
    pubinfo.add((this, NT.wasCreatedFromProvenanceTemplate, URIRef("https://w3id.org/np/RASh_Tn3WNfWht2FgL8XQGseMKF3vVdWi-PRROMGuBu9Y")))
    for t in ("RACJ58Gvyn91LqCKIO9zu1eijDQIeEff28iyDrJgjSJF8", "RAGIjRyQ2tgVqY29I6MKx5hzoIwr1f1kGtQw8EP1tTCPY", "RAukAcWHRDlkqxk7H2XNSegc1WnHI569INvNr-xdptDGI", "RAoTD7udB2KtUuOuAe74tJi1t3VzK0DyWS7rYVAq1GRvw"):
        pubinfo.add((this, NT.wasCreatedFromPubinfoTemplate, URIRef(f"https://w3id.org/np/{t}")))
    pubinfo.add((this, NT.wasCreatedFromTemplate, URIRef("https://w3id.org/np/RA3z187EzQXy4zopVZAmFg2sa_QymD1MKfzMxGzNqdNZM/template")))
    return plain_doubles(ds.serialize(format="trig"))


def panel_markdown(rows: list[dict], returns: dict) -> str:
    prov = {p["id"]: p for p in PROBLEMS}
    L = ["", "## Three-provider Delphi panel (2026-10-27) — supersedes the provisional viabilities above", "",
         "Three assessors scored the same blinded packet in parallel: " + ", ".join(f"{v[0]} ({v[1]})" for v in ASSESSORS.values()) +
         f". Blinding: the packet (`morphoeval/audits/{PANEL_ID}/packet.json`) carried no scores and the assessors were told which files not to open; "
         "exposure disclosures: " + "; ".join(f"{aid}: {d['exposure_disclosure'][:160]}" for aid, d in returns.items()) + ".", "",
         "Resolvability totals are recomputed from each assessor's factor scores; viability is pooled in log-odds (equal priors 50%, so the pooled net dB is the mean). "
         "The spread is part of the result. Raw returns are vendored next to the packet; the signed superseding nanopubs are in "
         f"`morphoeval/kg/{PANEL_ID}/`.", "",
         "| Rank | ID | Problem | Resolvability panel mean (range) | Pooled viability (range) | Pooled net dB | Per-assessor net dB (GPT / Gemini / Fable) |",
         "|---|---|---|---|---|---|---|"]
    ranked = sorted(rows, key=lambda r: -r["resolvability_panel"]["mean"])
    for i, r in enumerate(ranked, 1):
        rp, po = r["resolvability_panel"], r["pooled"]
        nets = " / ".join(f"{r['assessors'][a]['net_db']:+g}" for a in ASSESSORS)
        L.append(f"| {i} | {r['id']} | {prov[r['id']]['title']} | **{rp['mean']}** ({rp['range'][0]}–{rp['range'][1]}) · {rp['band']} | **{po['posterior_percent']}%** ({po['panel_range_percent'][0]}–{po['panel_range_percent'][1]}%) | {po['net_db']:+g} | {nets} |")
    L += ["", "### Packet corrections found by the panel", ""]
    L += [f"- **{pid}**: {text}" for pid, text in PACKET_CORRECTIONS.items()] or ["- none"]
    L += ["", "### Per-assessor verdicts and cruxes", ""]
    for r in ranked:
        L.append(f"**{r['id']}**")
        for aid in ASSESSORS:
            v = r["assessors"][aid]
            L.append(f"- {v['label']}: resolvability {v['resolvability']}, posterior {v['posterior_percent']}% ({v['net_db']:+g} dB, confidence {v['confidence']}), `{v['verdict']}`. {v['note']}")
        L.append("")
    return "\n".join(L)


def main() -> int:
    for req in ("packet.json", "INSTRUCTIONS.md"):
        if not (PANEL_DIR / req).exists():
            sys.exit(f"vendor the blinded packet first: {PANEL_DIR / req} missing")
    returns = load_returns()
    rows = aggregate(returns)
    prov_manifest_p = HERE / "kg" / PROV_AUDIT_ID / "published.json"
    prov_manifest = json.loads(prov_manifest_p.read_text()) if prov_manifest_p.exists() else {}
    ledger = {"audit": {"id": PANEL_ID, "date": DATE, "supersedes": PROV_AUDIT_ID, "prior_percent": PRIOR_PERCENT,
                        "assessors": {aid: {"label": v[0], "provider": v[1], "exposure_disclosure": returns[aid]["exposure_disclosure"]} for aid, v in ASSESSORS.items()},
                        "pooling": "log-odds mean of 3 independent blinded ledgers; resolvability totals recomputed by problem_scout_100.score_candidate",
                        "packet_corrections": PACKET_CORRECTIONS, "problems": rows}}
    (HERE / "audits" / f"{PANEL_ID}.yaml").write_text(yaml.safe_dump(ledger, sort_keys=False, allow_unicode=True, width=120))
    kd = HERE / "kg" / PANEL_ID; kd.mkdir(parents=True, exist_ok=True)
    feed = {"name": "Turing open problems — three-provider Delphi panel", "snapshot": DATE, "supersedes_feed": "docs/turing-open-problems.json (provisional viabilities)",
            "source": f"morphoeval/build_turing_panel_delphi.py · morphoeval/audits/{PANEL_ID}.yaml", "kg_dir": f"morphoeval/kg/{PANEL_ID}",
            "assessors": {aid: {"label": v[0], "provider": v[1], "agent": v[2]} for aid, v in ASSESSORS.items()},
            "prior_percent": PRIOR_PERCENT, "op_threshold_resolvability": 70,
            "problems": [{"id": r["id"], "slug": r["slug"], "pooled": r["pooled"], "resolvability_panel": r["resolvability_panel"],
                          "packet_correction": r["packet_correction"], "trig": f"panel-{r['id'].lower()}-{r['slug']}.trig",
                          "assessors": {aid: {k: v[k] for k in ("label", "provider", "resolvability", "net_db", "posterior_percent", "confidence", "verdict", "note")} for aid, v in r["assessors"].items()}}
                         for r in rows]}
    (ROOT / "docs" / "turing-open-problems-panel.json").write_text(json.dumps(feed, indent=2, ensure_ascii=False) + "\n")
    for r in rows:
        (kd / f"panel-{r['id'].lower()}-{r['slug']}.trig").write_text(build_nanopub(r, prov_manifest))
    md_path = ROOT / "docs" / "TURING_OPEN_PROBLEMS.md"; md = md_path.read_text()
    marker = "\n## Three-provider Delphi panel"
    md = md[:md.index(marker)] if marker in md else md
    md_path.write_text(md.rstrip("\n") + "\n" + panel_markdown(rows, returns) + "\n")
    for r in sorted(rows, key=lambda r: -r["resolvability_panel"]["mean"]):
        rp, po = r["resolvability_panel"], r["pooled"]
        print(f"{r['id']}  resolvability {rp['mean']:5.1f} ({rp['range'][0]}-{rp['range'][1]})  {rp['band']:<22s} viability {po['posterior_percent']:5.1f}% ({po['panel_range_percent'][0]}-{po['panel_range_percent'][1]}%)  net {po['net_db']:+g} dB")
    print(f"wrote audits/{PANEL_ID}.yaml, docs/turing-open-problems-panel.json, TURING_OPEN_PROBLEMS.md panel section, {len(rows)} trig -> {kd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
