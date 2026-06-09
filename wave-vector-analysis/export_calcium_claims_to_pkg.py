#!/usr/bin/env python3
"""
Export scored calcium-imaging claims to morphopkg/spc/1.0 TriG nanopubs.

Reads the verbatim claims (Calcium claims.docx), live verdicts from
wave_laterality.csv + landmark_responses.csv, and emits one nanopublication
per claim in the same probabilistic KG shape used by nanopubs/waves/ and
Probknow (ex:Assertion, ex:EvidenceItem, ex:BayesianAssessment).

Outputs:
  analysis_results/calcium_claims.pkg.trig   — combined TriG (19 nanopubs)
  analysis_results/calcium_claims_graph.json — same graph as JSON (Probknow ingest)

Run:
  python export_calcium_claims_to_pkg.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys_path = str(HERE)
import sys  # noqa: E402

sys.path.insert(0, sys_path)
from claims_doc_parser import DEFAULT_DOCX, parse_claims_doc  # noqa: E402
from generate_claims_inventory_pdf import (  # noqa: E402
    classify, load_landmarks, load_laterality)
from provenance import collect, record_run  # noqa: E402

RESULTS = HERE / "analysis_results"
DEFAULT_LAT = RESULTS / "wave_catalog" / "wave_laterality.csv"
DEFAULT_LM = RESULTS / "landmark_responses.csv"
DEFAULT_GT = RESULTS / "xy_ground_truth.csv"

PROFILE = "https://w3id.org/morphopkg/spc/1.0"
KG = "https://w3id.org/levin-kg"
ACTIVITY = f"{KG}/activity/wave-vector-calcium-v1"
AGENT = f"{KG}/agent/wave-vector-analysis"
DATASET = f"{KG}/dataset/calcium-imaging-box"


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def vovk_sellke_bf(p: float) -> float:
    if not (0 < p < 1):
        return 1.0
    if p > 1 / math.e:
        return 1.0
    return max(1.0, -math.e * p * math.log(p))


def deciban_from_bf(bf: float) -> float:
    return 10.0 * math.log10(max(bf, 1.0))


def deciban_from_rate(success: int, total: int, null: float = 0.5) -> float:
    """Conservative decibans from a binomial rate vs null (Laplace-smoothed)."""
    if total <= 0:
        return 0.0
    s = success + 0.5
    f = (total - success) + 0.5
    rate = s / (s + f)
    bf = max(rate / null, null / rate) if rate not in (0, 1) else 1.0
    return deciban_from_bf(bf)


def claim_slug(idx: int) -> str:
    return f"np-Ca-C{idx:02d}"


def pattern_iri(pattern: str) -> str | None:
    m = re.search(r"[A-Q]", (pattern or "").strip().upper())
    return f"ex:CaPattern-{m.group(0)}" if m else None


def _lat_df(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, comment="#")
    except Exception:
        return None


def _lm_df(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, comment="#")
    except Exception:
        return None


def build_evidence(
    idx: int,
    rec: dict,
    status: str,
    lat: dict | None,
    lat_df: pd.DataFrame | None,
    lm_agg: dict | None,
) -> list[dict]:
    """Return evidence item dicts for one claim."""
    c = rec["claim"].lower()
    items: list[dict] = []

    if status == "NOT YET":
        return items

    if lat and "bidirectional" in c and "within an embryo" in c:
        n = int(lat["n_videos"])
        k = int(round(lat["bidir_pct"] / 100.0 * n))
        items.append({
            "id": f"evidence-Ca-C{idx:02d}-bidir",
            "description": "Stimulated-side waves propagate both ways along embryo axis",
            "countN": k, "totalN": n, "frequency": round(k / n, 3) if n else 0,
            "weightOfEvidence_deciban": deciban_from_rate(k, n),
        })

    if lat and "calcium wave in neighbor" in c:
        n = int(lat["n_videos"])
        k = int(round(lat["neighbor_wave_pct"] / 100.0 * n))
        items.append({
            "id": f"evidence-Ca-C{idx:02d}-neighbor-wave",
            "description": "Distinct wave detected on neighbor embryo after stim side",
            "countN": k, "totalN": n, "frequency": round(k / n, 3) if n else 0,
            "weightOfEvidence_deciban": deciban_from_rate(k, n),
        })

    if lat and "local response in neighbor" in c:
        n = int(lat["n_videos"])
        k = int(lat["neighbor_local_n"])
        items.append({
            "id": f"evidence-Ca-C{idx:02d}-neighbor-local",
            "description": "Neighbor-side front (local) speed measurable where wave exists",
            "countN": k, "totalN": n, "frequency": round(k / n, 3) if n else 0,
            "weightOfEvidence_deciban": deciban_from_rate(k, n),
        })

    if lat and "does not require physical contact" in c:
        nc = int(lat["n_nocontact"])
        cc = int(lat["n_contact"])
        nk = int(round(lat["nocontact_pct"] / 100.0 * nc)) if nc else 0
        ck = int(round(lat["contact_pct"] / 100.0 * cc)) if cc else 0
        items.append({
            "id": f"evidence-Ca-C{idx:02d}-nocontact",
            "description": "Neighbor-side wave in no-contact (WGD) videos",
            "countN": nk, "totalN": nc, "frequency": round(nk / nc, 3) if nc else 0,
            "weightOfEvidence_deciban": deciban_from_rate(nk, nc, null=0.2),
        })
        items.append({
            "id": f"evidence-Ca-C{idx:02d}-contact",
            "description": "Neighbor-side wave in contact videos (comparison)",
            "countN": ck, "totalN": cc, "frequency": round(ck / cc, 3) if cc else 0,
            "weightOfEvidence_deciban": deciban_from_rate(ck, cc, null=0.2),
        })

    if lat and "neighboring embryo" in c and "increase" in c:
        p = lat.get("wp_p", float("nan"))
        if p == p:
            bf = vovk_sellke_bf(p)
            items.append({
                "id": f"evidence-Ca-C{idx:02d}-wound-vs-press",
                "description": "Mann-Whitney U: wound neighbor peak brightness vs pressure",
                "pValue": round(p, 4),
                "bayesFactorVS_MPR": round(bf, 4),
                "weightOfEvidence_deciban": deciban_from_bf(bf),
            })

    if lm_agg and re.search(r"(cement gland|eye|tail) response", c):
        if "tail response in neighbor" in c:
            stat = lm_agg["tail_neighbor"]
            label = "neighbor-embryo tail landmark ΔF/F₀"
        elif "cement gland" in c:
            stat = (lm_agg["cement_press"] if "pressure" in c
                    else lm_agg["cement_wound"])
            label = "cement gland landmark ΔF/F₀"
        elif "eye" in c:
            stat = lm_agg["eye"]
            label = "eye landmark ΔF/F₀"
        else:
            stat = lm_agg["tail_within"]
            label = "stimulated-embryo tail landmark ΔF/F₀"
        n, cor = stat["n"], stat["cor"]
        items.append({
            "id": f"evidence-Ca-C{idx:02d}-organ",
            "description": f"Manual XY landmark corroboration ({label}, threshold ≥ {lm_agg['thr']})",
            "countN": cor, "totalN": n,
            "frequency": round(cor / n, 3) if n else 0,
            "weightOfEvidence_deciban": deciban_from_rate(cor, n, null=0.3),
        })

    return items


def combine_decibans(items: list[dict]) -> tuple[float, float]:
    """Independence-style combine: multiply BFs, sum decibans."""
    if not items:
        return 1.0, 0.0
    bf_prod = 1.0
    for it in items:
        db = it.get("weightOfEvidence_deciban", 0.0)
        bf_prod *= 10 ** (db / 10.0)
    return bf_prod, 10.0 * math.log10(max(bf_prod, 1.0))


def build_nanopub(idx: int, rec: dict, status: str, rationale: str,
                  source: str, evidence: list[dict], created: str,
                  git_short: str) -> str:
    slug = claim_slug(idx)
    base = f"{KG}/np/{slug}"
    claim_id = f"claim-{slug}"
    assess_id = f"assessment-{slug}"
    pat = pattern_iri(rec.get("pattern", ""))

    ev_ids = [it["id"] for it in evidence]
    bf_comb, db_comb = combine_decibans(evidence)

    lines = [
        f"<{base}/Head> {{",
        f"    <{base}/> a np:Nanopublication ;",
        f"        dcterms:conformsTo <{PROFILE}> ;",
        f"        np:hasAssertion <{base}/assertion> ;",
        f"        np:hasProvenance <{base}/provenance> ;",
        f"        np:hasPublicationInfo <{base}/pubinfo> .",
        f"}}",
        "",
        f"<{base}/assertion> {{",
        f"    ex:{claim_id} a ex:Assertion ;",
        f'        dcterms:description "{_esc(rec["claim"])}" ;',
        f'        ex:context "Layer {rec["layer"]}; pattern {rec["pattern"]!r}; '
        f'orientation {rec["orientation"]!r}; poke {rec["poke"]!r}; '
        f'observed {rec["observed"]!r}" ;',
        f"        ex:assessmentStatus \"{status}\" ;",
        f'        ex:assessmentRationale "{_esc(rationale[:500])}" ;',
        f'        ex:dataSource "{_esc(source[:300])}" ;',
    ]
    if pat:
        lines.append(f"        ex:supportsHypothesis {pat} ;")
    if ev_ids:
        lines.append("        ex:hasEvidence " + ",\n            ".join(
            f"ex:{eid}" for eid in ev_ids) + " ;")
    if evidence:
        lines.append(f"        ex:hasAssessment ex:{assess_id} .")
    else:
        lines[-1] = lines[-1].rstrip(" ;") + " ."

    for it in evidence:
        lines.append("")
        lines.append(f"    ex:{it['id']} a ex:EvidenceItem ;")
        lines.append(f'        dcterms:description "{_esc(it["description"])}" ;')
        if "countN" in it:
            lines.append(f"        ex:countN {it['countN']} ;")
            lines.append(f"        ex:totalN {it['totalN']} ;")
            lines.append(f"        ex:frequency {it['frequency']} ;")
        if "pValue" in it:
            lines.append(f"        ex:pValue {it['pValue']} ;")
            lines.append(f"        ex:bayesFactorVS_MPR {it['bayesFactorVS_MPR']} ;")
        lines.append(f"        ex:weightOfEvidence_deciban {it['weightOfEvidence_deciban']:.6f} .")

    lines += ["", "}", ""]

    lines += [
        f"<{base}/provenance> {{",
        f"    <{base}/assertion> prov:wasDerivedFrom <{DATASET}> ;",
        f"    <{base}/assertion> prov:wasGeneratedBy <{ACTIVITY}> .",
        f"}}",
        "",
        f"<{base}/pubinfo> {{",
        f"    <{base}/> dcterms:created \"{created}\"^^xsd:date ;",
        f"        dcterms:source <{DATASET}> .",
    ]
    if evidence:
        lines += [
            f"    ex:{assess_id} a ex:BayesianAssessment ;",
            f'        dcterms:description "Combined from calcium-imaging evidence items (independence assumption)." ;',
            f"        dcterms:created \"{created}\"^^xsd:date ;",
            f'        dcterms:creator "wave-vector-analysis" ;',
            f"        ex:assessmentAgent <{AGENT}> ;",
            f"        prov:wasAttributedTo <{AGENT}> ;",
            f"        ex:bayesFactorCombined {bf_comb:.6f} ;",
            f'        ex:calibrationMethod "Vovk–Sellke for p-values; Laplace-smoothed rates for prevalence/organ corroboration" ;',
            f"        ex:weightOfEvidence_deciban {db_comb:.6f} .",
        ]
    lines += [
        f"    ex:generator a prov:SoftwareAgent ;",
        f'        dcterms:title "export_calcium_claims_to_pkg.py" ;',
        f'        dcterms:description "Calcium imaging claims exporter (git {git_short})." .',
        f"    <{AGENT}> a prov:SoftwareAgent .",
        f"}}",
        "",
    ]
    return "\n".join(lines)


def build_json_graph(claims: list[dict], created: str, git: dict) -> dict:
    return {
        "graph_type": "morphopkg/calcium-imaging",
        "conformsTo": PROFILE,
        "created": created,
        "repository": git,
        "dataset": DATASET,
        "claims": claims,
    }


def export_pkg(
    docx: Path,
    laterality_csv: Path,
    landmarks_csv: Path,
    out_trig: Path,
    out_json: Path,
) -> dict:
    recs = parse_claims_doc(docx)
    lat = load_laterality(laterality_csv)
    lm_agg = load_landmarks(landmarks_csv)
    lat_df = _lat_df(laterality_csv)

    prov = collect(
        "export_calcium_claims_to_pkg.py",
        [out_trig, out_json],
        inputs={
            "claims_docx": docx,
            "laterality_csv": laterality_csv,
            "landmarks_csv": landmarks_csv,
        },
    )
    git_short = prov["repository"].get("commit_short", "?")
    created = dt.date.today().isoformat()

    prefixes = [
        "@prefix cito: <http://purl.org/spar/cito/> .",
        "@prefix dcterms: <http://purl.org/dc/terms/> .",
        "@prefix ex: <https://w3id.org/levin-kg/> .",
        "@prefix np: <http://www.nanopub.org/nschema#> .",
        "@prefix prov: <http://www.w3.org/ns/prov#> .",
        "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
        "",
    ]

    trig_blocks: list[str] = []
    json_claims: list[dict] = []

    for idx, rec in enumerate(recs, 1):
        status, rationale, source = classify(rec, lat, lm_agg)
        evidence = build_evidence(idx, rec, status, lat, lat_df, lm_agg)
        trig_blocks.append(build_nanopub(
            idx, rec, status, rationale, source, evidence, created, git_short))
        bf, db = combine_decibans(evidence)
        json_claims.append({
            "id": claim_slug(idx),
            "layer": rec["layer"],
            "pattern": rec["pattern"],
            "claim": rec["claim"],
            "status": status,
            "rationale": rationale,
            "data_source": source,
            "evidence": evidence,
            "assessment": {
                "bayesFactorCombined": round(bf, 4),
                "weightOfEvidence_deciban": round(db, 4),
            } if evidence else None,
            "hypothesis": pattern_iri(rec.get("pattern", "")),
        })

    out_trig.parent.mkdir(parents=True, exist_ok=True)
    out_trig.write_text("\n".join(prefixes) + "\n".join(trig_blocks), encoding="utf-8")

    graph = build_json_graph(json_claims, created, prov["repository"])
    out_json.write_text(json.dumps(graph, indent=2) + "\n", encoding="utf-8")

    record_run(
        "export_calcium_claims_to_pkg.py", out_trig,
        outputs=[out_trig, out_json],
        inputs={
            "claims_docx": docx,
            "laterality_csv": laterality_csv,
            "landmarks_csv": landmarks_csv,
        },
        extra={
            "n_claims": len(recs),
            "n_tested": sum(1 for c in json_claims if c["status"] == "TESTED"),
            "n_not_yet": sum(1 for c in json_claims if c["status"] == "NOT YET"),
        },
    )
    return {
        "n_claims": len(recs),
        "n_tested": sum(1 for c in json_claims if c["status"] == "TESTED"),
        "n_not_yet": sum(1 for c in json_claims if c["status"] == "NOT YET"),
        "out_trig": str(out_trig),
        "out_json": str(out_json),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docx", default=str(DEFAULT_DOCX))
    ap.add_argument("--laterality-csv", default=str(DEFAULT_LAT))
    ap.add_argument("--landmarks-csv", default=str(DEFAULT_LM))
    ap.add_argument("--out-trig",
                    default=str(RESULTS / "calcium_claims.pkg.trig"))
    ap.add_argument("--out-json",
                    default=str(RESULTS / "calcium_claims_graph.json"))
    args = ap.parse_args()

    summary = export_pkg(
        Path(args.docx), Path(args.laterality_csv), Path(args.landmarks_csv),
        Path(args.out_trig), Path(args.out_json),
    )
    print(f"Wrote {summary['out_trig']}")
    print(f"Wrote {summary['out_json']}")
    print(f"  {summary['n_claims']} claims: "
          f"{summary['n_tested']} TESTED, {summary['n_not_yet']} NOT YET")


if __name__ == "__main__":
    main()
