#!/usr/bin/env python3
"""
build_morphoeval_kg.py — MorphoEval audit -> ProbKnow-conformant nanopublications.

MorphoEval is the morphology-simulation sibling of BioEval / NeuroEval: a
versioned, weighted rubric (morphoeval/rubric.yaml) scored 0-100 per dimension
for each simulation script, so simulations can be checkpointed and re-scored.

The graph is expressed in the ProbKnow / Project Delphi ontology (v1.3,
https://w3id.org/probknow/ontology/, vendored in morphoeval/vendor/) and shaped by
the governed templates of the ProbKnow Space (https://w3id.org/spaces/probknow;
template nanopubs vendored in morphoeval/vendor/templates/), so every nanopub
links the template it was created from, Nanodash renders it with that form, and
the checkpoints live in the same probabilistic knowledge graph as the Levin-lab
claims.  Mapping (rubric scores are pk:hasScore, never pk:hasProbability — see
project-delphi-ontology #12):

    per simulation   pk:Claim (ProbKnow claim template v3)
                       pkr:claim/morphoeval-<sim>: label, description, pk:context
                       (scripts, output, organism, commit, non-DOI sources),
                       pk:derivedFromPaper <doi>, pk:hasAssessment -> the 8
                       assessments, dct:subject each script as a
                       schema:SoftwareSourceCode pinned to the checkpoint commit,
                       pk:weightOfEvidence = overall / 100 (optional unit-interval
                       rescaling of the uncalibrated 0-100 rubric score; omit
                       with --omit-probability)
    per dimension    pk:LLMClaimAssessment (ProbKnow rubric-assessment template)
                       pkr:assessment/morphoeval-<sim>-<dim> (stable across
                       re-scores; "-<role>-<assessor>" suffix for a second assessor):
                       <claim> pk:hasAssessment <it>, dct:subject the dimension
                       concept, pk:assessmentType "morphoeval/<rubric>/<dim>",
                       pk:assessedBy pkr:model/<assessor> a pk:Model,
                       pk:hasScore = score/100 (uncalibrated rubric score on the
                       unit interval, required by the template), pk:hasVerdict
                       band, dct:date, dct:description = rationale,
                       pk:hasEvidence -> pk:EvidenceItem per evidence entry
                       (label, note, prov:wasDerivedFrom the commit-pinned line
                       range) and per morphometric measurement
                       (schema:PropertyValue with its reference value),
                       pk:evidenceAgainst for entries marked `supports: false`
    overall          the same, with the rubric itself as dct:subject;
                       remediation items and open needs as pk:evidenceAgainst
    rubric           pk:Concept with skos:narrower pk:Concept per dimension
                       (ProbKnow rubric template)

Provenance follows the Space's "Extracted from a paper by ProbKnow" template v2
(prov:wasAttributedTo the model and the publisher; prov:wasDerivedFrom each
source — DOI, dataset or repository URL — with its title and year; rdfs:seeAlso
the tracking issue), the rubric the generic "Attributed to" template.
Pubinfo: rdfs:label, dct:created (full UTC timestamp), dct:creator <ORCID>,
dct:license CC BY 4.0, npx:hasNanopubType, npx:introduces, pk:domain, and the
nt:wasCreatedFrom{,Provenance,Pubinfo}Template links (Creator, License, ProbKnow
domain, and Supersedes when superseding).

Every nanopub is one file on the nanopub temp base (http://purl.org/nanopub/temp/np001/),
so it can be sealed with sign_morphoeval.py (or nanopub-java / @nanopub/sign),
which mints the trusty https://w3id.org/np/RA... URI.

Usage
-----
    python3 morphoeval/build_morphoeval_kg.py morphoeval/audits/<audit>.yaml
    ... --check             validate the audit against the rubric only
    ... --conform           offline ProbKnow + template-shape conformance check on the output
    ... --supersede         re-issue: same-named nanopubs supersede the URIs in this dir's published.json
    ... --supersede-from D  reuse/supersede nanopubs published from another audit dir (repeatable)
    ... --no-resolve-heads  skip the Nanopub Query head check before superseding (not recommended)

Before a nanopub supersedes a published URI, the current head of that URI's
version chain is resolved with Nanopub Query's `get-latest-version-of-np` (the
query nanopub-java's `query` command runs); a URI that is no longer the head
has already been superseded elsewhere and is skipped, so chains never fork.

Requires: pyyaml, rdflib.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import yaml
from rdflib import Dataset, Graph, Literal, Namespace, URIRef
from rdflib.namespace import DCTERMS, FOAF, PROV, RDF, RDFS, SKOS, XSD

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUBRIC_PATH = HERE / "rubric.yaml"
ONTOLOGY_PATH = HERE / "vendor" / "probknow-1.3.ttl"
TEMPLATE_DIR = HERE / "vendor" / "templates"

NP = Namespace("http://www.nanopub.org/nschema#")
NPX = Namespace("http://purl.org/nanopub/x/")
NT = Namespace("https://w3id.org/np/o/ntemplate/")
PK = Namespace("https://w3id.org/probknow/ontology/")
PKR = Namespace("https://w3id.org/probknow/resource/")
TEMP = "http://purl.org/nanopub/temp/np001/"             # nanopub temp base (pre-signing)
THIS = URIRef(TEMP)

# Governed templates of the ProbKnow Space (https://w3id.org/spaces/probknow), as published
# by tkuhn/probknow-ontology nanopubs/signed/template-iris.json, plus the ecosystem's Creator,
# License and Supersedes pubinfo templates and the generic "Attributed to" provenance template.
TEMPLATES = {
    "claim": URIRef("https://w3id.org/np/RApE-nGpCMigXqZjeHuYuMJEVTGmTlNTlZl7IcHi_fhdQ/template"),           # claim v3
    "rubric-assessment": URIRef("https://w3id.org/np/RAOAK6Fe6J6JvD4kUCZg53sRpwRkRq7rzFuDtGT7n6Drw/template"),
    "rubric": URIRef("https://w3id.org/np/RAbjeus1yO4SqlPLnpIWNLS6nvW0DGF3879tVduuE0y00/template"),
    "provenance-extraction": URIRef("https://w3id.org/np/RAaYeFYF-Cz2bM5S1wPQsnvs5X6BBxsphL03UkbVZ1lcA"),  # v2
    "provenance-attributed": URIRef("https://w3id.org/np/RA7lSq6MuK_TIC6JMSHvLtee3lpLoZDOqLJCLXevnrPoU"),
    "pubinfo-creator": URIRef("https://w3id.org/np/RAukAcWHRDlkqxk7H2XNSegc1WnHI569INvNr-xdptDGI"),
    "pubinfo-license": URIRef("https://w3id.org/np/RACJ58Gvyn91LqCKIO9zu1eijDQIeEff28iyDrJgjSJF8"),
    "pubinfo-domain": URIRef("https://w3id.org/np/RAtKj9bCnB4gUe0E9k-WhffQFSEWd9Ve2hrVllYoLuOB0"),
    "pubinfo-supersedes": URIRef("https://w3id.org/np/RAoTD7udB2KtUuOuAe74tJi1t3VzK0DyWS7rYVAq1GRvw"),
}
# Templates the 2026-10-13 … 2026-10-19 nanopubs were built against (claim v2, the generic
# LLM-assessment template, provenance v1); still vendored so `--conform` can check those files.
LEGACY_TEMPLATES = {
    "claim-v2": URIRef("https://w3id.org/np/RAIVOEuhWNK4BuCbL_XRChRVlV6LKpFicBzwHpmuRxHWI/template"),
    "llm-assessment": URIRef("https://w3id.org/np/RAku3DINcetdFUnyTLflyV-8ep3QOCuL-IpGdS984vw8s/template"),
    "provenance-extraction-v1": URIRef("https://w3id.org/np/RA6svHudOcFjOo5Vr5kSCv1kVitLgngsx3xNQBeY_0bUg"),
}
SCHEMA = Namespace("https://schema.org/")
LANGUAGE_BY_SUFFIX = {".py": "Python", ".lean": "Lean 4", ".jl": "Julia", ".r": "R", ".R": "R", ".m": "MATLAB"}
LATEST_VERSION_API = ("https://query.knowledgepixels.com/api/"
                      "RAiRsB2YywxjsBMkVRTREJBooXhf2ZOHoUs5lxciEl37I/get-latest-version-of-np")

FORBIDDEN_PREFIXES = ("https://w3id.org/morphopkg", "https://w3id.org/levin-kg",
                      "https://w3id.org/aif", "https://example.org", "http://example.org",
                      "urn:pkg:", "urn:levin-kg:", "https://w3id.org/probknow/ontology/1.0#")


# --------------------------------------------------------------------------- #
# Loading + validation of the audit against the rubric
# --------------------------------------------------------------------------- #
def load(audit_path: Path):
    global PK, PKR
    rubric = yaml.safe_load(RUBRIC_PATH.read_text())
    audit = yaml.safe_load(Path(audit_path).read_text())
    onto = audit["audit"].get("ontology") or {}
    if onto.get("base"):
        PK = Namespace(onto["base"])
    if onto.get("resource_base"):
        PKR = Namespace(onto["resource_base"])
    # optional morphology snapshot (fingerprints + morphometrics) from measure_outputs.py
    side = Path(audit_path).with_name(Path(audit_path).stem + ".measurements.yaml")
    audit["_measurements"] = yaml.safe_load(side.read_text()) if side.exists() else None
    return rubric, audit


HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


def resolve_supersession_chain(audit: dict) -> tuple[list[str], list[str]]:
    """Resolve and validate the fail-closed audit supersession chain in morphoeval/audits/.
    Returns (chain_of_audit_ids_oldest_to_newest, errors)."""
    errs: list[str] = []
    A = audit.get("audit", {})
    cur_id = A.get("id")
    sup = A.get("supersedes") or A.get("supersedes_audit_id")
    if A.get("supersedes") and A.get("supersedes_audit_id") and A["supersedes"] != A["supersedes_audit_id"]:
        errs.append(f"audit.supersedes ({A['supersedes']}) != audit.supersedes_audit_id ({A['supersedes_audit_id']})")
    chain = [cur_id] if cur_id else []
    visited = {cur_id}
    while sup:
        if sup in visited:
            errs.append(f"supersession cycle detected involving audit '{sup}': {' -> '.join(chain + [sup])}")
            break
        sup_path = HERE / "audits" / f"{sup}.yaml"
        if not sup_path.exists():
            errs.append(f"supersedes audit '{sup}' not found at {sup_path.relative_to(ROOT)} (fail-closed chain policy)")
            break
        visited.add(sup)
        chain.append(sup)
        sup_doc = yaml.safe_load(sup_path.read_text()) or {}
        sup_a = sup_doc.get("audit", {})
        sup = sup_a.get("supersedes") or sup_a.get("supersedes_audit_id")
    return list(reversed(chain)), errs


def validate(rubric, audit) -> list[str]:
    errs: list[str] = []
    dims = {d["id"]: d for d in rubric["dimensions"]}
    if (tw := sum(d["weight"] for d in dims.values())) != 100:
        errs.append(f"rubric weights sum to {tw}, expected 100")
    A = audit["audit"]
    if A["rubric_version"] != rubric["version"]:
        errs.append(f"audit targets rubric {A['rubric_version']} but rubric.yaml is {rubric['version']}")
    if not A.get("assessor", {}).get("id"):
        errs.append("audit.assessor.id is required (the pk:Model that produced the scores)")
    _, chain_errs = resolve_supersession_chain(audit)
    errs.extend(chain_errs)
    refs = audit.get("references", {})
    for rid, r in refs.items():
        if not (r.get("doi") or r.get("url")):
            errs.append(f"reference {rid}: needs a doi or url")
    lo, hi = rubric["scale"]["min"], rubric["scale"]["max"]
    for sim in audit["simulations"]:
        sid = sim["id"]
        if sim.get("status") != "evaluated":
            if "scores" in sim:
                errs.append(f"{sid}: unevaluated simulations must not carry scores")
            continue
        if (sim_sup := sim.get("supersedes_audit_id")):
            sup_path = HERE / "audits" / f"{sim_sup}.yaml"
            if not sup_path.exists():
                errs.append(f"{sid}: supersedes_audit_id '{sim_sup}' not found at {sup_path.relative_to(ROOT)}")
        if (prov := sim.get("provenance")):
            for hkey in ("spec_fingerprint", "artifact_sha256"):
                if prov.get(hkey) and not HEX64_RE.match(str(prov[hkey])):
                    errs.append(f"{sid}: provenance.{hkey} is not a 64-hex SHA-256 digest")
            for dfile, dhash in (prov.get("input_dataset_sha256") or {}).items():
                if not HEX64_RE.match(str(dhash)):
                    errs.append(f"{sid}: provenance.input_dataset_sha256[{dfile!r}] is not a 64-hex SHA-256 digest")
        if (cb := sim.get("claim_boundary")):
            for req_tier in ("reported_in_source", "reconstructed_in_script", "does_not_claim"):
                if not cb.get(req_tier) or not isinstance(cb.get(req_tier), list):
                    errs.append(f"{sid}: claim_boundary.{req_tier} must be a non-empty list")
        for rid in sim.get("source_refs", []):
            if rid not in refs:
                errs.append(f"{sid}: unknown reference '{rid}'")
        scores = sim.get("scores", {})
        if (m := set(dims) - set(scores)):
            errs.append(f"{sid}: missing dimensions {sorted(m)}")
        if (x := set(scores) - set(dims)):
            errs.append(f"{sid}: unknown dimensions {sorted(x)}")
        for did, s in scores.items():
            v = s.get("value")
            if not isinstance(v, (int, float)) or not (lo <= v <= hi):
                errs.append(f"{sid}/{did}: value {v!r} outside [{lo},{hi}]")
            if not s.get("rationale"):
                errs.append(f"{sid}/{did}: missing rationale")
            if not (ev := s.get("evidence") or []):
                errs.append(f"{sid}/{did}: no evidence (rubric forbids unevidenced scores)")
            for e in ev:
                if not (ROOT / e["file"]).exists():
                    errs.append(f"{sid}/{did}: evidence file not found: {e['file']}")
                a, b = e["lines"]
                if a > b or a < 0:
                    errs.append(f"{sid}/{did}: bad line range {e['lines']} for {e['file']}")
                if "supports" in e and not isinstance(e["supports"], bool):
                    errs.append(f"{sid}/{did}: evidence `supports` must be true or false")
    return errs


def group_scores(rubric, scores: dict) -> dict:
    out = {}
    for grp in ("faithfulness", "reproducibility", None):
        sel = [d for d in rubric["dimensions"] if grp is None or d["group"] == grp]
        w = sum(d["weight"] for d in sel)
        out[grp or "overall"] = round(sum(d["weight"] * scores[d["id"]]["value"] for d in sel) / w, 1)
    for sub_grp, key in (("model-implementation", "model_implementation"),
                         ("biological-validation", "biological_validation")):
        sel = [d for d in rubric["dimensions"] if d.get("sub_group") == sub_grp]
        if not sel:
            sel = [d for d in rubric["dimensions"]
                   if d["id"] in (("model-fidelity", "parameter-expressivity") if sub_grp == "model-implementation"
                                  else ("specimen-match", "morphometric-validation"))]
        w = sum(d["weight"] for d in sel)
        out[key] = round(sum(d["weight"] * scores[d["id"]]["value"] for d in sel) / w, 1)
        app_sel = [d for d in sel if not scores[d["id"]].get("not_applicable")]
        w_app = sum(d["weight"] for d in app_sel)
        out[f"{key}_applicable"] = round(sum(d["weight"] * scores[d["id"]]["value"] for d in app_sel) / w_app, 1) if w_app else out[key]
    app_dims = [d for d in rubric["dimensions"] if not scores[d["id"]].get("not_applicable")]
    w_app_all = sum(d["weight"] for d in app_dims)
    out["overall_applicable"] = (round(sum(d["weight"] * scores[d["id"]]["value"] for d in app_dims) / w_app_all, 1)
                                 if w_app_all else out["overall"])
    app_faith = [d for d in rubric["dimensions"] if d["group"] == "faithfulness" and not scores[d["id"]].get("not_applicable")]
    w_app_faith = sum(d["weight"] for d in app_faith)
    out["faithfulness_applicable"] = (round(sum(d["weight"] * scores[d["id"]]["value"] for d in app_faith) / w_app_faith, 1)
                                      if w_app_faith else out["faithfulness"])
    out["n_not_applicable"] = sum(1 for d in rubric["dimensions"] if scores[d["id"]].get("not_applicable"))
    return out


def verdict(score_0_100: float) -> str:
    """Short categorical label for pk:hasVerdict (bands on the 0-100 rubric scale)."""
    s = score_0_100
    return ("faithful" if s >= 80 else "mostly faithful" if s >= 60 else
            "partially faithful" if s >= 40 else "unfaithful" if s >= 20 else "not faithful")


def publisher_orcid(audit) -> URIRef:
    o = audit["audit"].get("publisher_orcid")
    if not o:
        prof = Path.home() / ".nanopub" / "profile.yml"
        if prof.exists():
            for line in prof.read_text().splitlines():
                if line.startswith("orcid_id:"):
                    o = line.split(":", 1)[1].strip()
    if not o:
        sys.exit("no publisher ORCID: set audit.publisher_orcid or ~/.nanopub/profile.yml")
    return URIRef(o)


def ref_uri(ref: dict) -> URIRef:
    return URIRef(f"https://doi.org/{ref['doi']}") if ref.get("doi") else URIRef(ref["url"])


YEAR = r"(1[5-9]\d\d|20\d\d)"


def ref_year(ref: dict):
    """Publication year of a reference: an explicit ``year`` field, else the parenthesised
    year in ``venue`` ("Evolution 73:1873-1884 (2019)" -> 2019), else the last year-like
    number in ``venue`` (page ranges and volume numbers come before the year, never after)."""
    if ref.get("year"):
        m = re.search(YEAR, str(ref["year"]))
        return m.group(1) if m else None
    venue = str(ref.get("venue") or "")
    if (m := re.search(r"\(" + YEAR + r"\)", venue)):
        return m.group(1)
    hits = re.findall(r"(?<![\d:/-])" + YEAR + r"(?![\d-])", venue)   # not part of a page range or URL path
    return hits[-1] if hits else None


def unit_interval(x: float) -> Literal:
    """xsd:double in plain decimal form ("0.76"), as the templates' regex ^(0(\\.[0-9]+)?|1(\\.0+)?)$
    requires; rdflib would otherwise serialise it as 7.6e-01."""
    s = repr(round(float(x), 4))
    if s in ("1", "1.0"):
        s = "1.0"
    elif s == "0" or s == "0.0":
        s = "0.0"
    return Literal(s, datatype=XSD.double, normalize=False)


def squash(text) -> str:
    return " ".join(str(text).split())


# --------------------------------------------------------------------------- #
# Nanopub construction (ProbKnow template shapes, temp base, one nanopub per file)
# --------------------------------------------------------------------------- #
def _bind(g):
    for p, n in (("this", Namespace(TEMP)), ("sub", Namespace(TEMP)), ("np", NP), ("npx", NPX), ("nt", NT),
                 ("pk", PK), ("pkr", PKR), ("prov", PROV), ("dct", DCTERMS), ("rdfs", RDFS),
                 ("skos", SKOS), ("foaf", FOAF), ("xsd", XSD), ("schema", SCHEMA)):
        g.bind(p, n)


def slug(text: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^A-Za-z0-9]+", "-", str(text))).strip("-").lower()


class NanopubBuilder:
    def __init__(self, audit, rubric):
        self.A = audit["audit"]
        self.rubric = rubric
        self.orcid = publisher_orcid(audit)
        self.license = URIRef(self.A.get("license", "https://creativecommons.org/licenses/by/4.0/"))
        a = self.A["assessor"]
        self.model = PKR[f"model/{a['id']}"]
        self.model_label = a.get("label", a["id"])
        # Real creation time (UTC, seconds); MORPHOEVAL_CREATED pins it for reproducible rebuilds.
        stamp = os.environ.get("MORPHOEVAL_CREATED") or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.created = Literal(stamp, datatype=XSD.dateTime, normalize=False)
        cp = self.A.get("checkpoint", {})
        self.repo = cp.get("repository")
        self.commit = cp.get("commit")
        self.commit_uri = URIRef(f"{self.repo}/commit/{self.commit}") if self.commit else None
        self.env = cp.get("environment", {})
        self.outputs, self.measurements = {}, {}

    def script_uri(self, script: str) -> URIRef:
        return URIRef(f"{self.repo}/blob/{self.commit}/{script}")

    def set_measurements(self, meas: dict | None):
        self.outputs = {sid: {o["file"]: o for o in lst} for sid, lst in (meas or {}).get("outputs", {}).items()}
        self.measurements = (meas or {}).get("measurements", {})
        self.provenance = (meas or {}).get("provenance", {})

    def sim_provenance_text(self, sim: dict) -> str:
        """Dual semantic (`spec_fingerprint`) + byte (`artifact_sha256`) + input dataset
        provenance summary (adapted from MesmerPrism/rusty-manifold-provenance)."""
        sid = sim["id"]
        prov = dict(self.provenance.get(sid) or {})
        if sim.get("provenance"):
            prov.update({k: v for k, v in sim["provenance"].items() if v is not None})
        parts = []
        if prov.get("spec_fingerprint"):
            parts.append(f"spec_fingerprint: {prov['spec_fingerprint']}")
        if prov.get("artifact_sha256"):
            parts.append(f"artifact_sha256: {prov['artifact_sha256']}")
        if prov.get("input_dataset_sha256"):
            ds_items = ", ".join(f"{k}={v[:16]}..." for k, v in sorted(prov["input_dataset_sha256"].items()))
            parts.append(f"input_dataset_sha256: {{{ds_items}}}")
        return "; ".join(parts)

    def claim_boundary_text(self, sim: dict) -> str:
        cb = sim.get("claim_boundary")
        if not cb:
            return ""
        parts = []
        for k in ("reported_in_source", "reconstructed_in_script", "transfer_boundary", "does_not_claim"):
            if cb.get(k):
                parts.append(f"{k}: " + " | ".join(squash(x) for x in cb[k]))
        return "; ".join(parts)

    # ---- text renderings of the evidence (the templates carry evidence as literals) -- #
    def output_text(self, sim: dict, file: str) -> str:
        fp = self.outputs.get(sim["id"], {}).get(file)
        if not fp:
            return f"{file}: rendered output of {sim['script']}, not present when the audit was snapshotted (unfingerprinted)"
        size = f"{fp['width']}×{fp['height']} px" if fp.get("width") else ""
        frames = f", {fp['frames']} frames" if fp.get("frames", 1) > 1 else ""
        art_sha = fp.get("artifact_sha256") or fp["sha256"]
        spec_fp = fp.get("spec_fingerprint") or (self.provenance.get(sim["id"]) or {}).get("spec_fingerprint")
        spec_clause = f", spec_fingerprint {spec_fp}" if spec_fp else ""
        return (f"{file}: rendered output of {sim['script']} at {self.commit}, {fp['media_type']}, {fp['bytes']} B"
                + (f", {size}{frames}" if size else frames)
                + f", artifact_sha256 {art_sha} (sha256 {fp['sha256']}){spec_clause}")

    def evidence_text(self, sim: dict, e: dict) -> str:
        a, b = e["lines"]
        note = squash(e.get("note", ""))
        if b > 0:
            url = f"{self.repo}/blob/{self.commit}/{e['file']}#L{a}-L{b}"
            return f"{e['file']} L{a}–L{b}" + (f": {note}" if note else "") + f" <{url}>"
        return (f"{e['file']} (generated output, not in repo)" + (f": {note}" if note else "")
                + " — " + self.output_text(sim, e["file"]))

    def measurement_text(self, sim: dict) -> str:
        parts = []
        for m in self.measurements.get(sim["id"], []):
            if m.get("value") is None:
                continue
            s = f"{m['name']} = {m['value']}" + (f" {m['unit']}" if m.get("unit") else "")
            if m.get("reference_value") is not None:
                s += f" (reference {m['reference_value']}" + (f", {m['reference_source']}" if m.get("reference_source") else "") + ")"
            s += f" [{squash(m['method'])}]"
            if m.get("note"):
                s += f" — {squash(m['note'])}"
            parts.append(s)
        return "; ".join(parts)

    # ---- graphs ------------------------------------------------------------ #
    def provenance_extraction(self, agents, papers, see_also=()):
        """'Extracted from a paper by ProbKnow' provenance (v2): attributed to the agents, derived
        from each source (DOI IRI, or the URL of a dataset / repository) with its title and year,
        rdfs:seeAlso the tracking issue."""
        asrt = URIRef(TEMP + "assertion")
        t = [(asrt, PROV.wasAttributedTo, a) for a in agents]
        for r in papers:
            pu = ref_uri(r)
            t += [(asrt, PROV.wasDerivedFrom, pu), (pu, DCTERMS.title, Literal(squash(r["title"])))]
            if (y := ref_year(r)):
                t.append((pu, DCTERMS.date, Literal(y, datatype=XSD.gYear)))
        t += [(asrt, RDFS.seeAlso, s) for s in see_also if s is not None]
        return t

    # ---- ProbKnow resources ----------------------------------------------------- #
    def assessment_uri(self, sim: dict, dim_id: str) -> URIRef:
        """Stable across re-scores (no audit id: the pass is identified by dct:created, the
        commit in the description and the npx:supersedes chain); an independent second
        assessor's assessments are a different resource, suffixed with the role and the
        assessing model (two second assessors of one script must not share an IRI)."""
        role = self.A.get("role")
        suffix = f"-{slug(role)}-{slug(self.A['assessor']['id'])}" if role and role != "primary" else ""
        return PKR[f"assessment/morphoeval-{sim['id']}-{dim_id}{suffix}"]

    def evidence_items(self, sim: dict, au: URIRef, dim_id: str, entries: list) -> list:
        """pk:EvidenceItem nodes (rubric-assessment template, grouped statement 17) for the
        evidence entries that are not marked ``supports: false``: one line label, the note as
        description, prov:wasDerivedFrom the commit-pinned file/line range."""
        t, n = [], 0
        for e in entries:
            if e.get("supports") is False:
                continue
            n += 1
            ev = PKR[f"evidence/morphoeval-{sim['id']}-{dim_id}-{n}"]
            a, b = e["lines"]
            label = f"{e['file']} L{a}–L{b}" if b > 0 else f"{e['file']} (generated output)"
            desc = squash(e.get("note", "")) or None
            if b <= 0:
                desc = (desc + " — " if desc else "") + self.output_text(sim, e["file"])
            t += [(au, PK.hasEvidence, ev), (ev, RDF.type, PK.EvidenceItem), (ev, RDFS.label, Literal(label))]
            if desc:
                t.append((ev, DCTERMS.description, Literal(desc)))
            if b > 0 and self.repo and self.commit:
                t.append((ev, PROV.wasDerivedFrom, URIRef(f"{self.repo}/blob/{self.commit}/{e['file']}#L{a}-L{b}")))
        return t

    def measurement_items(self, sim: dict, au: URIRef) -> list:
        """Morphometric measurements as schema:PropertyValue evidence items with their reference
        values (rubric-assessment template, statements 17f–17n)."""
        t = []
        for m in self.measurements.get(sim["id"], []):
            if m.get("value") is None:
                continue
            ev = PKR[f"evidence/morphoeval-{sim['id']}-measurement-{slug(m['name'])}"]
            label = f"{m['name']} = {m['value']}" + (f" {m['unit']}" if m.get("unit") else "")
            t += [(au, PK.hasEvidence, ev), (ev, RDF.type, PK.EvidenceItem), (ev, RDF.type, SCHEMA.PropertyValue),
                  (ev, RDFS.label, Literal(label)), (ev, SCHEMA.name, Literal(m["name"])),
                  (ev, SCHEMA.value, Literal(str(m["value"])))]
            if m.get("unit"):
                t.append((ev, SCHEMA.unitText, Literal(m["unit"])))
            if m.get("method"):
                t.append((ev, SCHEMA.measurementTechnique, Literal(squash(m["method"]))))
            if m.get("note"):
                t.append((ev, DCTERMS.description, Literal(squash(m["note"]))))
            if m.get("reference_value") is not None:
                ref = URIRef(str(ev) + "-reference")
                t += [(ev, SCHEMA.valueReference, ref), (ref, RDF.type, SCHEMA.PropertyValue),
                      (ref, SCHEMA.value, Literal(str(m["reference_value"])))]
                if m.get("reference_source"):
                    t.append((ref, DCTERMS.source, Literal(squash(m["reference_source"]))))
        return t

    def artifact_triples(self, claim: URIRef, scripts: list) -> list:
        """The assessed scripts as schema:SoftwareSourceCode artifacts of the claim (claim
        template v3, grouped statement 09), pinned to the checkpoint commit."""
        if not (self.repo and self.commit):
            return []
        t = []
        for s in scripts:
            art = self.script_uri(s)
            t += [(claim, DCTERMS.subject, art), (art, RDF.type, SCHEMA.SoftwareSourceCode),
                  (art, RDFS.label, Literal(s)), (art, SCHEMA.codeRepository, URIRef(self.repo)),
                  (art, SCHEMA.version, Literal(self.commit))]
            if (lang := LANGUAGE_BY_SUFFIX.get(Path(s).suffix)):
                t.append((art, SCHEMA.programmingLanguage, Literal(lang)))
        return t

    def model_triples(self) -> list:
        a = self.A["assessor"]
        t = [(self.model, RDF.type, PK.Model), (self.model, RDFS.label, Literal(self.model_label)),
             (self.model, DCTERMS.identifier, Literal(a["id"])), (self.model, PROV.actedOnBehalfOf, self.orcid)]
        if a.get("homepage"):
            t.append((self.model, FOAF.homepage, URIRef(a["homepage"])))
        return t

    def assessment_date(self):
        d = self.A.get("date")
        return Literal(str(d), datatype=XSD.date, normalize=False) if d else None

    def nanopub(self, label: str, nptype: URIRef, introduces: URIRef, domain_slug: str,
                assertion, provenance, template_key: str | None, provenance_key: str, pubinfo_extra=()):
        ds = Dataset(); _bind(ds)
        head = ds.graph(URIRef(TEMP + "Head"))
        asrt, prov, pubi = URIRef(TEMP + "assertion"), URIRef(TEMP + "provenance"), URIRef(TEMP + "pubinfo")
        head.add((THIS, RDF.type, NP.Nanopublication))
        head.add((THIS, NP.hasAssertion, asrt))
        head.add((THIS, NP.hasProvenance, prov))
        head.add((THIS, NP.hasPublicationInfo, pubi))
        g = ds.graph(asrt)
        for t in assertion:
            g.add(t)
        g = ds.graph(prov)
        for t in provenance:
            g.add(t)
        g = ds.graph(pubi)
        dom = PKR[f"domain/{domain_slug}"]
        triples = [(THIS, RDFS.label, Literal(label)),
                   (THIS, DCTERMS.created, self.created),
                   (THIS, DCTERMS.creator, self.orcid),
                   (THIS, DCTERMS.license, self.license),
                   (THIS, NPX.hasNanopubType, nptype),
                   (THIS, NPX.introduces, introduces),
                   (THIS, PK.domain, dom),
                   (dom, RDFS.label, Literal(domain_slug)),
                   (THIS, NT.wasCreatedFromProvenanceTemplate, TEMPLATES[provenance_key]),
                   (THIS, NT.wasCreatedFromPubinfoTemplate, TEMPLATES["pubinfo-creator"]),
                   (THIS, NT.wasCreatedFromPubinfoTemplate, TEMPLATES["pubinfo-license"]),
                   (THIS, NT.wasCreatedFromPubinfoTemplate, TEMPLATES["pubinfo-domain"])]
        if template_key:
            triples.append((THIS, NT.wasCreatedFromTemplate, TEMPLATES[template_key]))
        for t in triples + list(pubinfo_extra):
            g.add(t)
        return ds

    def claim_uri(self, sim):
        return PKR[f"claim/morphoeval-{sim['id']}"]


def add_supersedes(ds: Dataset, old: str):
    g = ds.graph(URIRef(TEMP + "pubinfo"))
    g.add((THIS, NPX.supersedes, URIRef(old)))
    g.add((THIS, NT.wasCreatedFromPubinfoTemplate, TEMPLATES["pubinfo-supersedes"]))


def serialize_trig(ds: Dataset) -> str:
    """Deterministic TriG: shared prefix block, then Head/assertion/provenance/pubinfo in
    fixed order (rdflib's Dataset serializer emits graphs in set order, which makes a
    no-op rebuild show up as a diff in every checkpoint file)."""
    parts, prefixes = [], set()
    for part in ("Head", "assertion", "provenance", "pubinfo"):
        g = ds.graph(URIRef(TEMP + part)); _bind(g)
        body = g.serialize(format="turtle").splitlines()
        prefixes.update(l for l in body if l.startswith("@prefix"))
        triples = [l for l in body if l and not l.startswith("@prefix")]
        parts.append(f"sub:{part} {{\n" + "\n".join("    " + l for l in triples) + "\n}\n")
    return plain_doubles("\n".join(sorted(prefixes)) + "\n\n" + "\n".join(parts))


EXPONENT_DOUBLE = re.compile(r'(?<=[\s,(])(-?\d+(?:\.\d+)?e[+-]\d+)(?=\s*[;,.\]])')
# Turtle string literals (long and short forms), so that text such as "... to 1e-17." is left alone
STRING_LITERAL = re.compile(r'"""(?:[^"\\]|\\.|"(?!""))*"""|"(?:[^"\\\n]|\\.)*"', re.S)


def plain_doubles(trig_text: str) -> str:
    """rdflib writes xsd:double in exponent form (7.6e-01, 1e+00); the templates' 0-1 regex wants
    "0.76", and the lexical form is part of the trusty hash, so it must be written the way it was
    parsed.  Rewrites every bare exponent-form double (outside string literals) as a plain-decimal
    typed literal."""
    out, pos = [], 0
    for m in STRING_LITERAL.finditer(trig_text):
        out.append(EXPONENT_DOUBLE.sub(lambda d: f'"{repr(float(d.group(1)))}"^^xsd:double', trig_text[pos:m.start()]))
        out.append(m.group(0))
        pos = m.end()
    out.append(EXPONENT_DOUBLE.sub(lambda d: f'"{repr(float(d.group(1)))}"^^xsd:double', trig_text[pos:]))
    return "".join(out)


# --------------------------------------------------------------------------- #
# Head resolution (Nanopub Query, the query nanopub-java's `query` command runs)
# --------------------------------------------------------------------------- #
def latest_version(np_uri: str, timeout: float = 30.0) -> str | None:
    """Current head of the version chain of a published nanopub, or None if the query
    returns nothing (the nanopub is not indexed, or the service is unreachable)."""
    url = f"{LATEST_VERSION_API}?np={urllib.parse.quote(np_uri, safe='')}"
    req = urllib.request.Request(url, headers={"Accept": "text/csv", "User-Agent": "morpholib/morphoeval"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            rows = resp.read().decode("utf-8").splitlines()
    except Exception as ex:                                   # noqa: BLE001
        print(f"  head resolution failed for {np_uri}: {ex}")
        return None
    for row in rows[1:]:
        if row.strip():
            return row.split(",")[0].strip()
    return None


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build(rubric, audit, out_dir: Path, supersede: dict | None = None, resolve_heads: bool = True,
          omit_probability: bool = False):
    """supersede: {file name: {"nanopub": published trusty URI, "path": its file, "text": its unsigned TriG}}.
    A nanopub whose content differs from the published one of the same name gets npx:supersedes <old>
    (after confirming <old> is still the head of its chain); one whose content is identical to a nanopub
    published from another audit dir is not re-emitted and is reported in ``build.unchanged``; one
    whose published URI is no longer the head was superseded elsewhere and is reported in
    ``build.skipped``."""
    B = NanopubBuilder(audit, rubric)
    B.set_measurements(audit.get("_measurements"))
    A, refs = audit["audit"], audit.get("references", {})
    is_second_assessor = (A.get("role") == "second_assessor")
    omit_prob = bool(omit_probability or A.get("omit_probability"))
    default_domain = A.get("domain", "morphogenesis")
    issue = URIRef(A["tracked_by"]) if A.get("tracked_by") else None
    rubric_uri = PKR[f"concept/morphoeval-rubric-{rubric['version']}"]
    rubric_src = URIRef(f"{B.repo}/blob/main/morphoeval/rubric.yaml") if B.repo else None

    out_dir.mkdir(parents=True, exist_ok=True)
    merged = Graph(); _bind(merged)
    written, superseded, unchanged, skipped = [], [], [], []
    heads: dict[str, str | None] = {}

    def _content(text):
        """Nanopub text minus the pubinfo lines that differ between otherwise identical builds
        (creation date, supersedes link and its template link)."""
        return "\n".join(l for l in text.splitlines()
                         if not l.lstrip().startswith(("dct:created ", "npx:supersedes "))
                         and str(TEMPLATES["pubinfo-supersedes"]) not in l)

    def _merge(ds):
        for t in ds.graph(URIRef(TEMP + "assertion")):
            merged.add(t)

    def same_dir_prev(prev, p):
        return Path(prev.get("path", "")).resolve() == p.resolve()

    def emit(name, ds):
        p = out_dir / name
        text = serialize_trig(ds)
        prev = supersede.get(name) if supersede else None
        if prev:
            same_dir = same_dir_prev(prev, p)
            if prev.get("text") is not None:
                if _content(prev["text"]) == _content(text) and not same_dir:   # identical to one published from another audit
                    unchanged.append((name, prev["nanopub"]))
                    if p.exists():
                        p.unlink()
                    return
                if prev["text"] == text:        # same dir, nothing changed: just rewrite as-is
                    p.write_text(text); written.append(p); _merge(ds)
                    return
            # prev["text"] is None when the unsigned file was removed on an earlier build because
            # the chain's head had moved to another audit dir; the head check below settles it
            old = prev["nanopub"]
            if resolve_heads:
                head = heads.get(old) or latest_version(old)
                heads[old] = head
                if head is None:
                    sys.exit(f"{name}: cannot resolve the head of {old} on Nanopub Query; refusing to supersede blindly "
                             "(use --no-resolve-heads to override)")
                if head != old:
                    skipped.append((name, old, head))
                    if same_dir_prev(prev, p) and p.exists():
                        p.unlink()              # stale unsigned copy; the chain's head lives in another audit dir
                    return                      # superseded elsewhere already; its head is re-issued from there
            add_supersedes(ds, old)
            text = serialize_trig(ds)
            superseded.append((name, old))
        p.write_text(text); written.append(p); _merge(ds)

    # ---- 1. rubric as a pk:Concept with one narrower concept per dimension (ProbKnow rubric template) -- #
    if not is_second_assessor:
        tr = [(rubric_uri, RDF.type, PK.Concept),
              (rubric_uri, RDF.type, SKOS.Concept),
              (rubric_uri, RDFS.label, Literal(f"{rubric['name']} v{rubric['version']}")),
              (rubric_uri, DCTERMS.description, Literal(
                  "Weighted 0-100 rubric for the faithfulness of a morphology simulation to its organism and "
                  "source model; Overall = weighted mean. Each score is a pk:LLMClaimAssessment (ProbKnow "
                  "rubric-assessment template) of the pk:Claim pkr:claim/morphoeval-<sim> against one dimension "
                  "(dct:subject the dimension concept, pk:assessmentType morphoeval/<version>/<dimension>) with "
                  "pk:hasScore = score/100 — an uncalibrated rubric score, not a probability — and a pk:hasVerdict "
                  "band; the overall score has the rubric itself as dct:subject and is also the "
                  "pk:weightOfEvidence of the claim."))]
        if rubric_src:
            tr.append((rubric_uri, RDFS.seeAlso, rubric_src))
        for d in rubric["dimensions"]:
            du = PKR[f"concept/morphoeval-{rubric['version']}-{d['id']}"]
            tr += [(rubric_uri, SKOS.narrower, du), (du, SKOS.broader, rubric_uri),
                   (du, RDF.type, PK.Concept), (du, RDF.type, SKOS.Concept),
                   (du, RDFS.label, Literal(d["name"])),
                   (du, DCTERMS.description, Literal(f"[{d['group']}, weight {d['weight']}/100, id morphoeval/"
                                                     f"{rubric['version']}/{d['id']}] " + squash(d["question"])))]
        emit("00-rubric.trig", B.nanopub(f"MorphoEval rubric v{rubric['version']}", PK.Concept, rubric_uri, default_domain,
                                         tr, [(URIRef(TEMP + "assertion"), PROV.wasAttributedTo, B.orcid)],
                                         "rubric", "provenance-attributed"))

    rows = []
    for sim in audit["simulations"]:
        if sim.get("status") != "evaluated":
            rows.append((sim, None))            # nothing assessed -> nothing to publish yet
            continue
        dom = sim.get("domain", default_domain)
        claim = B.claim_uri(sim)
        papers = [refs[r] for r in sim.get("source_refs", [])]
        organism = sim.get("organism", "unspecified organism")
        scripts = [sim["script"]] + list(sim.get("also", []))
        gs = group_scores(rubric, sim["scores"])
        claim_label = f"{sim['name']} faithfully reproduces {organism} morphology"
        model_triples = B.model_triples()
        agents = [B.model, B.orcid]
        pub_common = [(THIS, RDFS.seeAlso, issue)] if issue else []
        pub_common += [(THIS, DCTERMS.conformsTo, rubric_uri)]
        assessments = [B.assessment_uri(sim, d["id"]) for d in rubric["dimensions"]]
        overall_au = B.assessment_uri(sim, "overall")
        date = B.assessment_date()

        def assessment_core(au, label, dimension, atype, score, verdict_text, description):
            """The rubric-assessment template's required core: type, label, date, description,
            claim -> assessment, dimension, model, assessment type, verdict, score."""
            t = [(au, RDF.type, PK.LLMClaimAssessment), (au, RDFS.label, Literal(label))]
            if date is not None:
                t.append((au, DCTERMS.date, date))
            t += [(au, DCTERMS.description, Literal(description)),
                  (claim, PK.hasAssessment, au),
                  (au, DCTERMS.subject, dimension),
                  (au, PK.assessedBy, B.model),
                  (au, PK.assessmentType, Literal(atype)),
                  (au, PK.hasVerdict, Literal(verdict_text)),
                  (au, PK.hasScore, unit_interval(score / 100.0))]
            return t + model_triples

        # ---- 2. the faithfulness CLAIM (ProbKnow claim template) ------------------------ #
        if not is_second_assessor:
            non_doi = [r for r in papers if not r.get("doi")]
            context = (f"script(s): {', '.join(B.script_uri(s) for s in scripts)}; output: {sim.get('output', 'n/a')}; "
                       f"organism: {organism}; checkpoint commit: {B.commit_uri or 'n/a'}; "
                       f"environment: {', '.join(f'{k} {v}' for k, v in B.env.items()) or 'n/a'}")
            if (prov_txt := B.sim_provenance_text(sim)):
                context += f"; {prov_txt}"
            if (sup_id := sim.get("supersedes_audit_id") or A.get("supersedes") or A.get("supersedes_audit_id")):
                context += f"; supersedes_audit_id: {sup_id}"
            if (cb_txt := B.claim_boundary_text(sim)):
                context += f"; claim_boundary: [{cb_txt}]"
            if non_doi:
                context += "; other sources: " + ", ".join(f"{squash(r['title'])} <{ref_uri(r)}>" for r in non_doi)
            tr = [(claim, RDF.type, PK.Claim),
                  (claim, RDFS.label, Literal(claim_label)),
                  (claim, DCTERMS.description, Literal(
                      f"The simulation script(s) {', '.join(scripts)} implement the cited source model as published "
                      f"and produce forms that match real {organism} specimens and the source authors' figures. "
                      f"A belief under audit, not a fact: its credibility is the weight of evidence (MorphoEval "
                      f"{rubric['version']} overall {gs['overall']}/100, audit {A['id']}) and the assessments.")),
                  (claim, PK.context, Literal(context))]
            if not omit_prob:
                tr.append((claim, PK.weightOfEvidence, unit_interval(gs["overall"] / 100.0)))
            tr += [(claim, PK.derivedFromPaper, ref_uri(r)) for r in papers if r.get("doi")]
            tr += [(claim, PK.hasAssessment, au) for au in assessments + [overall_au]]
            tr += B.artifact_triples(claim, scripts)
            pub = [(THIS, RDFS.seeAlso, B.script_uri(s)) for s in scripts] + ([(THIS, RDFS.seeAlso, issue)] if issue else [])
            emit(f"{sim['id']}-claim.trig",
                 B.nanopub(f"Claim: {sim['name']} is faithful", PK.Claim, claim, dom, tr,
                           B.provenance_extraction([B.orcid], papers, [issue]), "claim", "provenance-extraction", pub))

        # ---- 3. one pk:LLMClaimAssessment per rubric dimension (ProbKnow rubric-assessment template) -- #
        for d, au in zip(rubric["dimensions"], assessments):
            did, s = d["id"], sim["scores"][d["id"]]
            verdict_text = ("not applicable (neutral baseline)" if s.get("not_applicable") else verdict(s["value"]))
            tr = assessment_core(au, f"{sim['name']} — {d['name']}: {s['value']}/100",
                                 PKR[f"concept/morphoeval-{rubric['version']}-{did}"],
                                 f"morphoeval/{rubric['version']}/{did}", s["value"], verdict_text,
                                 squash(s["rationale"]))
            tr += [(au, PK.evidenceAgainst, Literal(B.evidence_text(sim, e)))
                   for e in s.get("evidence", []) if e.get("supports") is False]
            tr += B.evidence_items(sim, au, did, s.get("evidence", []))
            if did == "morphometric-validation":
                tr += B.measurement_items(sim, au)
            emit(f"{sim['id']}-{did}.trig",
                 B.nanopub(f"{sim['name']} — {d['name']} {s['value']}/100", PK.LLMClaimAssessment, au, dom, tr,
                           B.provenance_extraction(agents, papers, [issue]), "rubric-assessment",
                           "provenance-extraction", pub_common))

        # ---- 4. overall checkpoint assessment (same template, dct:subject the rubric itself) -- #
        env = ", ".join(f"{k} {v}" for k, v in B.env.items())
        sub_note = (f"; model-implementation {gs['model_implementation']}/100, biological-validation "
                    f"{gs['biological_validation']}/100, applicable-only overall {gs['overall_applicable']}/100; "
                    f"uncalibrated rubric score, not a calibrated probability"
                    if is_second_assessor else "")
        reasoning = (squash(sim["summary"]) +
                     f" Overall {gs['overall']}/100 (faithfulness {gs['faithfulness']}, reproducibility "
                     f"{gs['reproducibility']}{sub_note}); rubric morphoeval {rubric['version']}; checkpoint commit {B.commit}; "
                     f"environment: {env}. Dimension assessments: " + ", ".join(
                         f"{d['name']} {sim['scores'][d['id']]['value']}" for d in rubric["dimensions"]) + ".")
        if sim.get("output"):
            reasoning += " Output judged: " + B.output_text(sim, sim["output"]) + "."
        if (prov_txt := B.sim_provenance_text(sim)):
            reasoning += f" Dual provenance: {prov_txt}."
        if (sup_id := sim.get("supersedes_audit_id") or A.get("supersedes") or A.get("supersedes_audit_id")):
            reasoning += f" Supersedes audit: {sup_id}."
        if (cb_txt := B.claim_boundary_text(sim)):
            reasoning += f" Claim boundary: {cb_txt}."
        tr = assessment_core(overall_au, f"{sim['name']} — MorphoEval overall {gs['overall']}/100", rubric_uri,
                             f"morphoeval/{rubric['version']}/overall", gs["overall"], verdict(gs["overall"]), reasoning)
        tr += [(overall_au, PK.evidenceAgainst, Literal(f"remediation needed: {squash(fix)}"))
               for fix in sim.get("remediation", [])]
        for nd in sim.get("needs", []):
            if nd.get("status", "open") != "resolved":
                n_val = float(nd.get("novelty_n", 1.0))
                c_val = float(nd.get("contradiction_c", 0.0))
                d_val = float(nd.get("depth_d", 1.0))
                a_val = float(nd.get("age_days_a", 0.0))
                import math as _math
                p_score = round(2.0 * n_val + c_val + 0.5 * d_val + 0.2 * _math.log(1.0 + max(0.0, a_val)), 3)
                tr.append((overall_au, PK.evidenceAgainst, Literal(
                    f"open need [{nd.get('need_type')}/{nd.get('target_dimension')}, pressure P={p_score}]: {squash(nd.get('description', ''))}"
                )))
        pub = pub_common + [(THIS, RDFS.seeAlso, claim)]
        emit(f"{sim['id']}-overall.trig",
             B.nanopub(f"{sim['name']} — MorphoEval overall {gs['overall']}/100", PK.LLMClaimAssessment, overall_au,
                       dom, tr, B.provenance_extraction(agents, papers, [issue]), "rubric-assessment",
                       "provenance-extraction", pub))
        try:
            from morphoeval.math_kg import add_math_triples_for_simulation
        except ImportError:
            from math_kg import add_math_triples_for_simulation  # type: ignore[no-redef]
        add_math_triples_for_simulation(
            g=merged,
            sim_id=sim["id"],
            audit_id=A["id"],
            sim=sim,
            measurements=B.measurements.get(sim["id"], []),
            references=papers,
            repo=B.repo or "https://github.com/fractastical/morpholib",
            commit=B.commit or "HEAD",
            overall_score=gs["overall"],
        )
        rows.append((sim, gs))

    merged_path = out_dir / f"morphoeval_{A['id']}.ttl"
    merged_path.write_text(merged.serialize(format="turtle"))
    report_path = out_dir / "REPORT.md"
    report_path.write_text(report(rubric, audit, rows))
    build.unchanged, build.skipped = unchanged, skipped
    return written, merged_path, report_path, merged, superseded


# --------------------------------------------------------------------------- #
# Offline conformance: ProbKnow terms + the shape of each graph against its template
# --------------------------------------------------------------------------- #
class Template:
    """A nanopub template parsed from its (vendored) nanopub: statements with subject /
    predicate / object patterns and the placeholders' constraints."""

    def __init__(self, path: Path):
        ds = Dataset(); ds.parse(path, format="trig")
        g = Graph()
        for ctx in ds.graphs():
            if str(ctx.identifier).endswith("assertion"):
                for t in ctx:
                    g.add(t)
        self.g = g
        root = next(iter(list(g.subjects(RDF.type, NT.AssertionTemplate)) + list(g.subjects(RDF.type, NT.ProvenanceTemplate))
                         + list(g.subjects(RDF.type, NT.PubinfoTemplate))))
        self.kind = next(g.objects(root, RDF.type))
        self.label = str(g.value(root, RDFS.label) or path.name)
        sts = list(g.objects(root, NT.hasStatement))
        if not sts and (root, RDF.predicate, None) in g:
            sts = [root]                                  # single-statement template (Creator)
        by_id = {}
        for st in sts:
            inner = list(g.objects(st, NT.hasStatement))  # grouped statement: its parts inherit its flags
            for s in (inner or [st]):
                if (s, RDF.predicate, None) not in g:
                    continue
                types = set(g.objects(s, RDF.type)) | (set(g.objects(st, RDF.type)) if inner else set())
                sid = str(s).split("/")[-1].split("#")[-1]
                cur = by_id.setdefault(sid, {"id": sid, "s": g.value(s, RDF.subject), "p": g.value(s, RDF.predicate),
                                             "o": g.value(s, RDF.object), "optional": False, "repeatable": False})
                cur["optional"] |= NT.OptionalStatement in types
                cur["repeatable"] |= NT.RepeatableStatement in types
                order = g.value(s, NT.statementOrder)
                cur["order"] = int(str(order)) if order is not None and str(order).isdigit() else 10 ** 6
        # the template author's order: statements about the introduced resource come first, so a
        # triple is matched against them before a same-namespace placeholder statement gets a go
        self.statements = sorted(by_id.values(), key=lambda st: (st["order"], st["id"]))
        self.placeholders = {}
        for s, _, kind in g.triples((None, RDF.type, None)):
            if str(kind).startswith(str(NT)) and str(kind).endswith(("Placeholder", "IntroducedResource")):
                self.placeholders.setdefault(s, {"kinds": set()})["kinds"].add(kind)
                self.placeholders[s]["prefix"] = g.value(s, NT.hasPrefix)
                self.placeholders[s]["datatype"] = g.value(s, NT.hasDatatype)
                self.placeholders[s]["regex"] = g.value(s, NT.hasRegex)

    def _is_var(self, node):
        return node in self.placeholders or str(node) in (str(NT.ASSERTION), str(NT.NANOPUB), str(NT.CREATOR)) \
            or "~~ARTIFACTCODE~~" in str(node)

    def _match_node(self, pattern, value, bindings, special):
        key = str(pattern)
        if key in special:
            return value == special[key]
        introduced = special.get("introduced")
        if "~~ARTIFACTCODE~~" in key:                       # introduced resource: same namespace, own local id
            if introduced is not None:
                return value == introduced
            return isinstance(value, URIRef) and str(value).startswith(key.split("~~")[0])
        if pattern in self.placeholders:
            ph = self.placeholders[pattern]
            if introduced is not None and value == introduced and "~~ARTIFACTCODE~~" not in key:
                return False                                # a placeholder never stands for the introduced resource
            kinds = {str(k).split("/")[-1] for k in ph["kinds"]}
            literal_kind = bool(kinds & {"LiteralPlaceholder", "LongLiteralPlaceholder", "RestrictedChoicePlaceholder"})
            if literal_kind != isinstance(value, Literal) and "RestrictedChoicePlaceholder" not in kinds:
                return False
            if isinstance(value, Literal):
                if ph["datatype"] is not None and value.datatype != ph["datatype"]:
                    return False
                if ph["regex"] is not None and not re.fullmatch(str(ph["regex"]), str(value)):
                    return False
            elif ph["prefix"] is not None and not str(value).startswith(str(ph["prefix"])):
                return False
            bound = bindings.get(pattern)
            if bound is not None and bound != value and not ph.get("_multi"):
                return False
            return True
        return pattern == value

    def check(self, graph: Graph, special: dict) -> list[str]:
        """Every triple of the graph must match a statement; required statements must be present;
        non-repeatable statements at most once.  `special` binds nt:ASSERTION / nt:NANOPUB / nt:CREATOR."""
        probs, counts, bindings = [], {}, {}
        for st in self.statements:                      # object placeholders of repeatable statements may vary
            if st["repeatable"] and st["o"] in self.placeholders:
                self.placeholders[st["o"]]["_multi"] = True
        for s, p, o in graph:
            hit = None
            for st in self.statements:
                if st["p"] != p:
                    continue
                if self._match_node(st["s"], s, bindings, special) and self._match_node(st["o"], o, bindings, special):
                    hit = st
                    break
            if hit is None:
                probs.append(f"triple not in template '{self.label}': {s.n3()} {p.n3()} {o.n3()[:80]}")
                continue
            counts[hit["id"]] = counts.get(hit["id"], 0) + 1
            for pat, val in ((hit["s"], s), (hit["o"], o)):
                if pat in self.placeholders and not self.placeholders[pat].get("_multi"):
                    bindings[pat] = val
        for st in self.statements:
            n = counts.get(st["id"], 0)
            if n == 0 and not st["optional"]:
                probs.append(f"required statement {st['id']} ({st['p'].n3()}) of '{self.label}' missing")
            if n > 1 and not st["repeatable"]:
                probs.append(f"statement {st['id']} ({st['p'].n3()}) of '{self.label}' used {n} times, not repeatable")
        return probs


def load_templates() -> dict[str, Template]:
    out = {}
    for key, uri in {**LEGACY_TEMPLATES, **TEMPLATES}.items():
        code = str(uri).split("/np/")[1][:8]
        matches = list(TEMPLATE_DIR.glob(f"*.{code}.trig"))
        if matches:
            out[str(uri)] = Template(matches[0])
    return out


def conformance(files: list[Path]) -> list[str]:
    onto = Graph().parse(ONTOLOGY_PATH, format="turtle")
    onto_iri = next(onto.subjects(RDF.type, URIRef("http://www.w3.org/2002/07/owl#Ontology")))
    onto_base = str(onto_iri).rstrip("/") + "/"
    declared = {str(PK) + str(s)[len(onto_base):] for s in onto.subjects() if str(s).startswith(onto_base)}
    templates = load_templates()
    probs: list[str] = []
    for f in files:
        raw = f.read_text()
        if (m := EXPONENT_DOUBLE.search(STRING_LITERAL.sub('""', raw))):
            probs.append(f"{f.name}: exponent-form double {m.group(0)} (templates' 0-1 regex wants plain decimals)")
        ds = Dataset(); ds.parse(f, format="trig")
        # works on unsigned (temp-namespace) and signed (trusty-URI) files alike
        this = next(iter(set(s for g in ds.graphs() for s in g.subjects(RDF.type, NP.Nanopublication))), None)
        if this is None:
            probs.append(f"{f.name}: no np:Nanopublication in any graph"); continue
        G = {"Head": next((g.identifier for g in ds.graphs() if (this, RDF.type, NP.Nanopublication) in g), None)}
        head = ds.graph(G["Head"])
        G["assertion"], G["provenance"], G["pubinfo"] = (head.value(this, NP.hasAssertion), head.value(this, NP.hasProvenance), head.value(this, NP.hasPublicationInfo))
        names = {str(g.identifier) for g in ds.graphs()}
        for part, gid in G.items():
            if gid is None or str(gid) not in names:
                probs.append(f"{f.name}: missing graph {part}")
        if any(v is None for v in G.values()):
            continue
        THIS, pub, asrt, prov = this, ds.graph(G["pubinfo"]), ds.graph(G["assertion"]), ds.graph(G["provenance"])
        nodes = set()
        for g in ds.graphs():
            for s, p, o in g:
                nodes.update(x for x in (s, p, o) if isinstance(x, URIRef))
        for n in nodes:
            ns = str(n)
            if ns.startswith(FORBIDDEN_PREFIXES):
                probs.append(f"{f.name}: forbidden legacy IRI {ns}")
            if ns.startswith(str(PK)) and ns not in declared:
                probs.append(f"{f.name}: undeclared pk: term {ns}")
        for p in (DCTERMS.creator, DCTERMS.license, NPX.hasNanopubType, NPX.introduces, PK.domain, DCTERMS.created):
            if (THIS, p, None) not in pub:
                probs.append(f"{f.name}: pubinfo lacks {p.n3(ds.namespace_manager)}")
        created = pub.value(THIS, DCTERMS.created)
        if created is not None and (created.datatype != XSD.dateTime or "T00:00:00" in str(created)):
            probs.append(f"{f.name}: dct:created {created} is not a full xsd:dateTime with a real time")
        special = {str(NT.ASSERTION): G["assertion"], str(NT.NANOPUB): THIS,
                   str(NT.CREATOR): pub.value(THIS, DCTERMS.creator),
                   "introduced": pub.value(THIS, NPX.introduces)}
        # assertion against its template
        tmpl = [str(o) for o in pub.objects(THIS, NT.wasCreatedFromTemplate)]
        if tmpl:
            t = templates.get(tmpl[0])
            if t is None:
                probs.append(f"{f.name}: template {tmpl[0]} not vendored in {TEMPLATE_DIR.name}/")
            else:
                probs += [f"{f.name}: {m}" for m in t.check(asrt, special)]
                target = t.g.value(next(t.g.subjects(RDF.type, NT.AssertionTemplate)), NT.hasTargetNanopubType)
                if target is not None and (THIS, NPX.hasNanopubType, target) not in pub:
                    probs.append(f"{f.name}: npx:hasNanopubType is not the template's target type {target}")
        elif any((None, RDF.type, c) in asrt for c in (PK.Claim, PK.LLMClaimAssessment)):
            probs.append(f"{f.name}: a ProbKnow claim/assessment without nt:wasCreatedFromTemplate")
        # provenance against its template
        for pt in pub.objects(THIS, NT.wasCreatedFromProvenanceTemplate):
            t = templates.get(str(pt))
            if t is None:
                probs.append(f"{f.name}: provenance template {pt} not vendored")
            else:
                probs += [f"{f.name}: {m}" for m in t.check(prov, special)]
        # pubinfo templates: each linked one must be satisfied (extras in pubinfo are allowed)
        for pt in pub.objects(THIS, NT.wasCreatedFromPubinfoTemplate):
            t = templates.get(str(pt))
            if t is None:
                probs.append(f"{f.name}: pubinfo template {pt} not vendored")
                continue
            for st in t.statements:
                if not st["optional"] and (THIS, st["p"], None) not in pub:
                    probs.append(f"{f.name}: pubinfo lacks {st['p'].n3()} required by '{t.label}'")
        if (THIS, NPX.supersedes, None) in pub and TEMPLATES["pubinfo-supersedes"] not in set(pub.objects(THIS, NT.wasCreatedFromPubinfoTemplate)):
            probs.append(f"{f.name}: npx:supersedes without the Supersedes pubinfo template link")
        for s, p, o in asrt:
            if p in (PK.weightOfEvidence, PK.hasConfidence, PK.hasProbability, PK.hasScore):
                if not (0.0 <= float(o) <= 1.0):
                    probs.append(f"{f.name}: {p.n3(ds.namespace_manager)} {o} outside [0,1]")
            if p == PK.hasProbability and (s, RDF.type, PK.LLMClaimAssessment) in asrt \
                    and (s, PK.assessmentType, None) in asrt:
                # project-delphi-ontology #12: a rubric score is a pk:hasScore, not a probability.
                probs.append(f"{f.name}: rubric assessment {s.n3(ds.namespace_manager)} carries pk:hasProbability; "
                             "use pk:hasScore (rubric-assessment template)")
            if p == PK.hasProbability:
                # Issue #9 regression guard for the legacy (pre-rubric-template) files: pk:hasProbability
                # on an assessment needs dct:conformsTo the MorphoEval rubric concept in pubinfo so the
                # unit-interval value cannot be mistaken for an unanchored probability.
                conforms = [str(c) for c in pub.objects(THIS, DCTERMS.conformsTo)]
                if not any("morphoeval-rubric-" in c for c in conforms):
                    probs.append(f"{f.name}: pk:hasProbability used without dct:conformsTo morphoeval-rubric link in pubinfo")
            if p == PK.hasScore and (s, DCTERMS.subject, None) not in asrt:
                probs.append(f"{f.name}: pk:hasScore on {s.n3(ds.namespace_manager)} without dct:subject <rubric dimension>")
    return probs


# --------------------------------------------------------------------------- #
# Scoreboard
# --------------------------------------------------------------------------- #
def report(rubric, audit, rows) -> str:
    A = audit["audit"]; dims = rubric["dimensions"]
    chain, _ = resolve_supersession_chain(audit)
    chain_note = f" · supersession chain `{' -> '.join(chain)}`" if len(chain) > 1 else ""
    L = [f"# MorphoEval report — audit `{A['id']}`", "",
         f"Rubric **{rubric['version']}** · checkpoint `{A['checkpoint']['commit']}` · "
         f"assessor `pkr:model/{A['assessor']['id']}` (pk:LLMClaimAssessment) · "
         f"tracked in {A.get('tracked_by', '—')}{chain_note}", "",
         "Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; "
         "`pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings "
         "for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** "
         "(weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` "
         "+ `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** "
         "renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** "
         "substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.", "",
         "| Simulation | " + " | ".join(f"{d['id']} ({d['weight']})" for d in dims)
         + " | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |",
         "|---|" + "---:|" * (len(dims) + 5) + "---|"]
    for sim, gs in rows:
        if gs is None:
            L.append(f"| {sim['name']} (`{sim['script']}`) | " + " | ".join("—" for _ in dims)
                     + " | — | — | — | — | *unevaluated* | — |")
            continue
        cells = " | ".join(
            f"{sim['scores'][d['id']]['value']}*" if sim["scores"][d["id"]].get("not_applicable")
            else str(sim["scores"][d["id"]]["value"])
            for d in dims
        )
        L.append(f"| {sim['name']} (`{sim['script']}`) | {cells} | {gs['model_implementation']} | "
                 f"{gs['biological_validation']} | {gs['reproducibility']} | {gs['overall_applicable']} | "
                 f"**{gs['overall']}** | {verdict(gs['overall'])} |")
    L += ["", "## Findings and remediation", ""]
    for sim, gs in rows:
        if gs is None:
            continue
        L += [f"### {sim['name']}  —  overall {gs['overall']} ({verdict(gs['overall'])})", "",
              squash(sim["summary"]), ""]
        for d in dims:
            s = sim["scores"][d["id"]]
            ev = "; ".join(f"`{e['file']}` L{e['lines'][0]}–{e['lines'][1]}" for e in s["evidence"])
            L.append(f"- **{d['name']}: {s['value']}** — {squash(s['rationale'])}  \n  evidence: {ev}")
        if sim.get("remediation"):
            L += ["", "Remediation:"] + [f"- {r}" for r in sim["remediation"]]
        if sim.get("needs"):
            import math as _math
            L += ["", "Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\\ln(1+a)$):"]
            for nd in sim["needs"]:
                p_sc = round(2.0 * float(nd.get("novelty_n", 1.0)) + float(nd.get("contradiction_c", 0.0))
                             + 0.5 * float(nd.get("depth_d", 1.0)) + 0.2 * _math.log(1.0 + max(0.0, float(nd.get("age_days_a", 0.0)))), 3)
                L.append(f"- **[{nd.get('status', 'open').upper()} · P={p_sc}]** `{nd.get('need_type')}` → `{nd.get('target_dimension')}`: {squash(nd.get('description', ''))}")
        L.append("")
    ms = audit.get("_measurements")
    if ms:
        L += ["## Morphology snapshot", "",
              "The binaries stay out of git; the sha256 pins the exact file that was scored, and the "
              "morphometrics record the form itself (written into the reasoning of the "
              "morphometric-validation assessment).", ""]
        for sim, gs in rows:
            if gs is None:
                continue
            outs = ms.get("outputs", {}).get(sim["id"], [])
            meas = ms.get("measurements", {}).get(sim["id"], [])
            miss = [m["file"] for m in ms.get("missing", []) if m["simulation"] == sim["id"]]
            if not (outs or meas or miss):
                continue
            L.append(f"### {sim['name']}")
            prov = ms.get("provenance", {}).get(sim["id"], {})
            if prov.get("spec_fingerprint"):
                L.append(f"- **spec_fingerprint** (semantic model + inputs): `{prov['spec_fingerprint']}`")
            if prov.get("input_dataset_sha256"):
                ds_str = ", ".join(f"`{k}` (`{v[:16]}…`)" for k, v in sorted(prov["input_dataset_sha256"].items()))
                L.append(f"- **input_dataset_sha256**: {ds_str}")
            for o in outs:
                size = f"{o['width']}×{o['height']} px" if o.get("width") else ""
                fr = f", {o['frames']} frames" if o.get("frames", 1) > 1 else ""
                art_sha = o.get("artifact_sha256") or o["sha256"]
                L.append(f"- `{o['file']}` — artifact_sha256 `{art_sha[:16]}…`, {o['bytes']:,} B, {size}{fr}")
            for f in miss:
                L.append(f"- `{f}` — **not on disk at snapshot time** (no hash, no measurements)")
            if meas:
                L += ["", "| measurement | value | reference | method |", "|---|---:|---|---|"]
                for m in meas:
                    ref = m.get("reference_value")
                    ref = "—" if ref is None else str(ref)
                    L.append(f"| {m['name']} | {m['value']} {m.get('unit') or ''} | {ref} | {m['method']}"
                             + (f" *({m['note']})*" if m.get("note") else "") + " |")
            L.append("")
    L += ["## Environment at checkpoint", ""]
    L += [f"- {k}: {v}" for k, v in A["checkpoint"].get("environment", {}).items()]
    if A["checkpoint"].get("note"):
        L += ["", squash(A["checkpoint"]["note"])]
    return "\n".join(L) + "\n"


DEMO_QUERY = """
PREFIX pk:   <%s>
PREFIX dct:  <http://purl.org/dc/terms/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?claim ?score ?verdict ?model WHERE {
  ?c pk:hasAssessment ?a ; rdfs:label ?claim .
  ?a a pk:LLMClaimAssessment ; pk:assessmentType ?code ; pk:hasScore ?score ; pk:hasVerdict ?verdict ; pk:assessedBy ?m .
  ?m rdfs:label ?model .
  FILTER(STRENDS(?code, "/overall"))
} ORDER BY ?score
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audit", nargs="?", default=None)
    ap.add_argument("--out", default=None, help="output dir (default morphoeval/kg/<audit-id>)")
    ap.add_argument("--check", action="store_true", help="validate the audit only")
    ap.add_argument("--conform", action="store_true", help="run the offline ProbKnow + template-shape conformance check")
    ap.add_argument("--query", action="store_true", help="run a demo SPARQL on the merged graph")
    ap.add_argument("--supersede", action="store_true",
                    help="nanopubs whose content changed get npx:supersedes <URI recorded in kg/<audit>/published.json>")
    ap.add_argument("--supersede-from", metavar="KG_DIR", action="append", default=[],
                    help="another audit's kg dir: same-named nanopubs supersede the URIs in its published.json; "
                         "identical ones are not re-emitted (repeatable; this dir's own manifest wins)")
    ap.add_argument("--rubric-from", metavar="KG_DIR", action="append", default=[],
                    help="the audit dir the rubric nanopub is published from: 00-rubric.trig is not re-emitted here "
                         "when identical to that dir's (use when this audit shares no simulation with it)")
    ap.add_argument("--no-resolve-heads", action="store_true",
                    help="do not check on Nanopub Query that a superseded URI is still the head of its chain")
    ap.add_argument("--omit-probability", action="store_true",
                    help="omit pk:weightOfEvidence on the claim (optional in the claim template); pk:hasScore on the "
                         "assessments is required by the rubric-assessment template and is always written")
    ap.add_argument("--list-needs", action="store_true",
                    help="list pressure-ranked open remediation needs across all morphoeval/audits/*.yaml and exit")
    args = ap.parse_args()

    if args.list_needs:
        import math as _math
        items = []
        for yml in sorted((HERE / "audits").glob("*.yaml")):
            if yml.name.endswith(".measurements.yaml"):
                continue
            doc = yaml.safe_load(yml.read_text()) or {}
            if (doc.get("audit") or {}).get("role") == "second_assessor":
                continue
            aid = (doc.get("audit") or {}).get("id", yml.stem)
            for sim in doc.get("simulations") or []:
                if sim.get("status") != "evaluated":
                    continue
                sid = sim["id"]
                explicit_needs = sim.get("needs") or []
                if explicit_needs:
                    for nd in explicit_needs:
                        if nd.get("status", "open") == "resolved":
                            continue
                        p_sc = round(2.0 * float(nd.get("novelty_n", 1.0)) + float(nd.get("contradiction_c", 0.0))
                                     + 0.5 * float(nd.get("depth_d", 1.0)) + 0.2 * _math.log(1.0 + max(0.0, float(nd.get("age_days_a", 0.0)))), 3)
                        items.append((p_sc, aid, sid, nd.get("need_type", "remediation"), nd.get("target_dimension", "overall"), squash(nd.get("description", ""))))
                else:
                    # Synthesize pressure-ranked needs from remediation items + lowest-scoring dimension
                    scores = sim.get("scores") or {}
                    min_dim = min(scores, key=lambda k: scores[k].get("value", 100)) if scores else "overall"
                    min_val = scores.get(min_dim, {}).get("value", 75)
                    n_val = round((100.0 - float(min_val)) / 50.0, 3)
                    for idx, fix in enumerate(sim.get("remediation") or []):
                        p_sc = round(2.0 * n_val + 0.5 * (1.0 + 0.2 * idx), 3)
                        items.append((p_sc, aid, sid, "remediation_gap", min_dim, squash(fix)))
        items.sort(key=lambda x: (-x[0], x[2]))
        print(f"Open MorphoEval Needs ({len(items)} items ranked by pressure P = 2n + c + 0.5d + 0.2*ln(1+a)):")
        for rank, (psc, aid, sid, ntype, tdim, desc) in enumerate(items[:20], 1):
            print(f"  {rank:2d}. [P={psc:5.3f}] {sid} ({aid}) -> {tdim} [{ntype}]: {desc[:110]}")
        return

    rubric, audit = load(Path(args.audit))
    if errs := validate(rubric, audit):
        print("INVALID audit:"); [print("  -", e) for e in errs]; sys.exit(1)
    print(f"audit {audit['audit']['id']} valid against rubric {rubric['version']}")
    if args.check:
        return
    out_dir = Path(args.out) if args.out else HERE / "kg" / audit["audit"]["id"]
    supersede = None
    if args.supersede or args.supersede_from or args.rubric_from:
        supersede = {}
        # (dir, only these names or None); later dirs override earlier ones, own dir last
        dirs = [(Path(d), {"00-rubric.trig"}) for d in args.rubric_from]
        dirs += [(Path(d), None) for d in args.supersede_from] + ([(out_dir, None)] if args.supersede else [])
        for prev_dir, only in dirs:
            mp = prev_dir / "published.json"
            if not mp.exists():
                sys.exit(f"--supersede: no {mp}")
            for n, m in json.loads(mp.read_text()).items():
                if not m.get("published") or (only and n not in only):
                    continue
                prev_file = prev_dir / n
                supersede[n] = {"nanopub": m["nanopub"], "path": str(prev_file),
                                "text": prev_file.read_text() if prev_file.exists() else None}
    written, merged_path, report_path, merged, superseded = build(rubric, audit, out_dir, supersede,
                                                                  resolve_heads=not args.no_resolve_heads,
                                                                  omit_probability=args.omit_probability)
    if getattr(build, "unchanged", None):
        print(f"  {len(build.unchanged)} nanopub(s) identical to already-published ones, not re-emitted:")
        for n, old in build.unchanged:
            print(f"    {n}  = {old}")
    if getattr(build, "skipped", None):
        print(f"  {len(build.skipped)} nanopub(s) whose published URI is no longer the head of its chain, skipped:")
        for n, old, head in build.skipped:
            print(f"    {n}  {old}  -> head {head}")
    print(f"wrote {len(written)} nanopubs, {merged_path.name} ({len(merged)} triples), {report_path.name}")
    print(f"  -> {out_dir}")
    if audit.get("_measurements"):
        ms = audit["_measurements"]
        print(f"  morphology snapshot: {sum(len(v) for v in ms.get('outputs', {}).values())} fingerprinted output(s), "
              f"{sum(len(v) for v in ms.get('measurements', {}).values())} measurement(s)"
              + (f", missing: {[m['file'] for m in ms.get('missing', [])]}" if ms.get("missing") else ""))
    if superseded:
        print(f"  {len(superseded)} nanopub(s) changed -> npx:supersedes added (heads confirmed on Nanopub Query):")
        for n, old in superseded:
            print(f"    {n}  supersedes  {old}")
    if args.conform:
        probs = conformance(written)
        if probs:
            print(f"\nProbKnow conformance: {len(probs)} problem(s)"); [print("  -", p) for p in probs]
            sys.exit(2)
        print(f"ProbKnow conformance: OK ({len(written)} nanopubs; pk: terms all declared in {ONTOLOGY_PATH.name}; "
              f"assertions, provenance and pubinfo match their linked templates; full timestamps; no legacy IRIs)")
    if args.query:
        print("\nSPARQL: overall assessments")
        for r in merged.query(DEMO_QUERY % PK):
            print(f"  {float(r.score):.3f}  {str(r.verdict):20s} {r.claim}   [assessed by {r.model}]")


if __name__ == "__main__":
    main()
