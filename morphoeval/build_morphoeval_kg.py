#!/usr/bin/env python3
"""
build_morphoeval_kg.py — MorphoEval audit -> ProbKnow-conformant nanopublications.

MorphoEval is the morphology-simulation sibling of BioEval / NeuroEval: a
versioned, weighted rubric (morphoeval/rubric.yaml) scored 0-100 per dimension
for each simulation script, so simulations can be checkpointed and re-scored.

The graph is expressed in the ProbKnow / Project Delphi ontology (v1.1,
https://w3id.org/probknow/ontology/, vendored in morphoeval/vendor/) so the
checkpoints live in the same probabilistic knowledge graph as the Levin-lab
claims and are queryable with the existing ProbKnow SPARQL.  Mapping:

    per simulation      pk:Claim      "script S faithfully reproduces the morphology
                                        of organism O under source model P"
                                        (a belief, not a fact)  — pk:derivedFromPaper DOI
    per dimension       pk:LLMClaimAssessment  pk:assessedBy pk:Model (the agent
                                        that scored it), pk:assessmentType
                                        "morphoeval/<rubric>/<dimension>",
                                        pk:weightOfEvidence = score/100,
                                        pk:hasVerdict band, dct:description rationale
    per evidence locus  pk:EvidenceItem  prov:wasDerivedFrom commit-pinned GitHub
                                        line range
    overall/checkpoint  pk:LLMClaimAssessment  "morphoeval/<rubric>/overall",
                                        prov:wasDerivedFrom the 7 dimension assessments
                                        + the commit; environment in description
    rubric              pk:Concept with skos:narrower pk:Concept per dimension
    source papers       https://doi.org/... a pk:Paper (real DOI IRIs as subjects)

Every nanopub is one file on the nanopub temp base (http://purl.org/nanopub/temp/np001/),
exactly like project-delphi-ontology/nanopubs/src/*.trig, so it can be sealed with
    java -jar nanopub-*.jar sign <file>        (or @nanopub/nanopub-js)
which mints the trusty https://w3id.org/np/RA... URI.  Pubinfo carries
dct:creator <ORCID>, CC BY 4.0, npx:hasNanopubType, npx:introduces, pk:domain.

Usage
-----
    python3 morphoeval/build_morphoeval_kg.py morphoeval/audits/2026-10-13-morphology-audit.yaml
    ... --check     validate the audit against the rubric only
    ... --conform   also run the offline ProbKnow conformance check on the output
    ... --query     also run a demo SPARQL over the merged graph

Requires: pyyaml, rdflib.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from rdflib import Dataset, Graph, Literal, Namespace, URIRef
from rdflib.namespace import DCTERMS, FOAF, PROV, RDF, RDFS, SKOS, XSD

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUBRIC_PATH = HERE / "rubric.yaml"
ONTOLOGY_PATH = HERE / "vendor" / "probknow-1.1.ttl"

NP = Namespace("http://www.nanopub.org/nschema#")
NPX = Namespace("http://purl.org/nanopub/x/")
PK = Namespace("https://w3id.org/probknow/ontology/")
PKR = Namespace("https://w3id.org/probknow/resource/")
TEMP = "http://purl.org/nanopub/temp/np001/"             # nanopub temp base (pre-signing)
THIS = URIRef(TEMP)

FORBIDDEN_PREFIXES = ("https://w3id.org/morphopkg", "https://w3id.org/levin-kg",
                      "https://w3id.org/aif", "https://example.org", "http://example.org",
                      "urn:pkg:", "urn:levin-kg:", "https://w3id.org/probknow/ontology/1.0#")


# --------------------------------------------------------------------------- #
# Loading + validation of the audit against the rubric
# --------------------------------------------------------------------------- #
def load(audit_path: Path):
    return yaml.safe_load(RUBRIC_PATH.read_text()), yaml.safe_load(Path(audit_path).read_text())


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
    return errs


def group_scores(rubric, scores: dict) -> dict:
    out = {}
    for grp in ("faithfulness", "reproducibility", None):
        sel = [d for d in rubric["dimensions"] if grp is None or d["group"] == grp]
        w = sum(d["weight"] for d in sel)
        out[grp or "overall"] = round(sum(d["weight"] * scores[d["id"]]["value"] for d in sel) / w, 1)
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


# --------------------------------------------------------------------------- #
# Nanopub construction (ProbKnow shape, temp base, one nanopub per file)
# --------------------------------------------------------------------------- #
def _bind(g):
    for p, n in (("this", Namespace(TEMP)), ("sub", Namespace(TEMP)), ("np", NP), ("npx", NPX),
                 ("pk", PK), ("pkr", PKR), ("prov", PROV), ("dct", DCTERMS), ("rdfs", RDFS),
                 ("skos", SKOS), ("foaf", FOAF), ("xsd", XSD)):
        g.bind(p, n)


class NanopubBuilder:
    def __init__(self, audit, rubric):
        self.A = audit["audit"]
        self.rubric = rubric
        self.orcid = publisher_orcid(audit)
        self.license = URIRef(self.A.get("license", "https://creativecommons.org/licenses/by/4.0/"))
        a = self.A["assessor"]
        self.model = PKR[f"model/{a['id']}"]
        self.model_label = a.get("label", a["id"])
        self.model_home = URIRef(a["homepage"]) if a.get("homepage") else None
        self.created = Literal(f"{self.A['date']}T00:00:00Z", datatype=XSD.dateTime)
        cp = self.A.get("checkpoint", {})
        self.repo = cp.get("repository")
        self.commit = cp.get("commit")
        self.commit_uri = URIRef(f"{self.repo}/commit/{self.commit}") if self.commit else None
        self.env = cp.get("environment", {})

    def nanopub(self, label: str, nptype: URIRef, introduces: URIRef, domain_slug: str,
                assertion, provenance=(), pubinfo_extra=()):
        ds = Dataset(); _bind(ds)
        head = ds.graph(URIRef(TEMP + "Head"))
        asrt, prov = URIRef(TEMP + "assertion"), URIRef(TEMP + "provenance")
        pubi = URIRef(TEMP + "pubinfo")
        head.add((THIS, RDF.type, NP.Nanopublication))
        head.add((THIS, NP.hasAssertion, asrt))
        head.add((THIS, NP.hasProvenance, prov))
        head.add((THIS, NP.hasPublicationInfo, pubi))
        g = ds.graph(asrt)
        for t in assertion:
            g.add(t)
        g = ds.graph(prov)
        for t in provenance:
            g.add((asrt if t[0] is None else t[0], t[1], t[2]))
        g = ds.graph(pubi)
        dom = PKR[f"domain/{domain_slug}"]
        for t in [(THIS, RDFS.label, Literal(label)),
                  (THIS, DCTERMS.created, self.created),
                  (THIS, DCTERMS.creator, self.orcid),
                  (THIS, DCTERMS.license, self.license),
                  (THIS, NPX.hasNanopubType, nptype),
                  (THIS, NPX.introduces, introduces),
                  (THIS, PK.domain, dom),
                  (dom, RDFS.label, Literal(domain_slug))] + list(pubinfo_extra):
            g.add(t)
        return ds

    # ---- shared bits ------------------------------------------------------ #
    def model_triples(self):
        t = [(self.model, RDF.type, PK.Model),
             (self.model, RDFS.label, Literal(self.model_label)),
             (self.model, DCTERMS.identifier, Literal(self.A["assessor"]["id"])),
             (self.model, PROV.actedOnBehalfOf, self.orcid)]
        if self.model_home:
            t.append((self.model, FOAF.homepage, self.model_home))
        return t

    def paper_triples(self, ref: dict):
        pu = ref_uri(ref)
        t = [(pu, RDF.type, PK.Paper),
             (pu, DCTERMS.title, Literal(ref["title"])),
             (pu, DCTERMS.creator, Literal(ref["authors"])),
             (pu, DCTERMS.bibliographicCitation, Literal(ref["venue"]))]
        if ref.get("doi"):
            t.append((pu, DCTERMS.identifier, Literal(f"doi:{ref['doi']}")))
        if ref.get("pmid"):
            t.append((pu, DCTERMS.identifier, Literal(f"pmid:{ref['pmid']}")))
        if ref.get("code"):
            t.append((pu, RDFS.seeAlso, URIRef(ref["code"])))
        return t

    def claim_uri(self, sim):
        return PKR[f"claim/morphoeval-{sim['id']}"]


def ref_uri(ref: dict) -> URIRef:
    return URIRef(f"https://doi.org/{ref['doi']}") if ref.get("doi") else URIRef(ref["url"])


def build(rubric, audit, out_dir: Path):
    B = NanopubBuilder(audit, rubric)
    A, refs = audit["audit"], audit.get("references", {})
    default_domain = A.get("domain", "morphogenesis")
    issue = URIRef(A["tracked_by"]) if A.get("tracked_by") else None
    rubric_uri = PKR[f"concept/morphoeval-rubric-{rubric['version']}"]
    # The rubric is versioned by its own `version` field and lands in the same commit as the
    # KG it describes, so it cannot be pinned to the checkpoint commit; point at the branch.
    rubric_src = URIRef(f"{B.repo}/blob/main/morphoeval/rubric.yaml") if B.repo else None

    out_dir.mkdir(parents=True, exist_ok=True)
    merged = Graph(); _bind(merged)
    written = []

    def emit(name, ds):
        p = out_dir / name
        p.write_text(ds.serialize(format="trig"))
        written.append(p)
        for ctx in ds.graphs():
            if str(ctx.identifier) == TEMP + "assertion":
                for t in ctx:
                    merged.add(t)

    # ---- 1. rubric as a pk:Concept with one narrower concept per dimension -- #
    tr = [(rubric_uri, RDF.type, PK.Concept),
          (rubric_uri, RDF.type, SKOS.Concept),
          (rubric_uri, RDFS.label, Literal(f"{rubric['name']} v{rubric['version']}")),
          (rubric_uri, DCTERMS.description, Literal(
              "Weighted 0-100 rubric for the faithfulness of a morphology simulation to its "
              "organism and source model; Overall = weighted mean. Scores are emitted as "
              "pk:LLMClaimAssessment with pk:weightOfEvidence = score/100."))]
    if rubric_src:
        tr.append((rubric_uri, RDFS.seeAlso, rubric_src))
    for d in rubric["dimensions"]:
        du = PKR[f"concept/morphoeval-{rubric['version']}-{d['id']}"]
        tr += [(rubric_uri, SKOS.narrower, du), (du, SKOS.broader, rubric_uri),
               (du, RDF.type, PK.Concept), (du, RDF.type, SKOS.Concept),
               (du, RDFS.label, Literal(d["name"])),
               (du, DCTERMS.description, Literal(
                   f"[{d['group']}, weight {d['weight']}/100] " + " ".join(d["question"].split())))]
    emit("00-rubric.trig", B.nanopub(f"MorphoEval rubric v{rubric['version']}", PK.Concept, rubric_uri,
                                     default_domain, tr, [(None, PROV.wasAttributedTo, B.orcid)]))

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
        context = (f"script(s): {', '.join(scripts)}; output: {sim.get('output', 'n/a')}; "
                   f"organism: {organism}; checkpoint commit: {B.commit or 'n/a'}")

        # ---- 2. the faithfulness CLAIM (a belief, not a fact) --------------- #
        tr = [(claim, RDF.type, PK.Claim),
              (claim, RDFS.label, Literal(f"{sim['name']} faithfully reproduces {organism} morphology")),
              (claim, DCTERMS.description, Literal(
                  f"The simulation script(s) {', '.join(scripts)} implement the cited source model as "
                  f"published and produce forms that match real {organism} specimens and the source "
                  f"authors' figures.")),
              (claim, PK.context, Literal(context))]
        prov = [(None, PROV.wasAttributedTo, B.orcid)]
        for r in papers:
            tr.append((claim, PK.derivedFromPaper, ref_uri(r)))
            tr += B.paper_triples(r)
            prov.append((None, PROV.wasDerivedFrom, ref_uri(r)))
        if B.commit_uri:
            prov.append((None, PROV.wasDerivedFrom, B.commit_uri))
        emit(f"{sim['id']}-claim.trig",
             B.nanopub(f"Claim: {sim['name']} is faithful", PK.Claim, claim, dom, tr, prov))

        # ---- 3. one pk:LLMClaimAssessment per rubric dimension ------------- #
        dim_assessments = []
        for d in rubric["dimensions"]:
            did, s = d["id"], sim["scores"][d["id"]]
            au = PKR[f"assessment/{A['id']}-{sim['id']}-{did}"]
            dim_assessments.append(au)
            w = round(s["value"] / 100.0, 4)
            tr = [(claim, PK.hasAssessment, au),
                  (au, RDF.type, PK.LLMClaimAssessment),
                  (au, RDFS.label, Literal(f"{sim['name']} — {d['name']}: {s['value']}/100")),
                  (au, PK.assessmentType, Literal(f"morphoeval/{rubric['version']}/{did}")),
                  (au, PK.assessedBy, B.model),
                  (au, PK.weightOfEvidence, Literal(w, datatype=XSD.double)),
                  (au, PK.hasVerdict, Literal(verdict(s["value"]))),
                  (au, DCTERMS.description, Literal(" ".join(str(s["rationale"]).split()))),
                  (au, DCTERMS.date, Literal(str(A["date"]), datatype=XSD.date)),
                  (au, DCTERMS.subject, PKR[f"concept/morphoeval-{rubric['version']}-{did}"])]
            if s.get("not_applicable"):
                tr.append((au, PK.hasVerdict, Literal("not applicable (neutral baseline)")))
            tr += B.model_triples()
            for i, e in enumerate(s.get("evidence", [])):
                eu = PKR[f"evidence/{A['id']}-{sim['id']}-{did}-{i}"]
                a, b = e["lines"]
                is_code = b > 0
                loc = f"{e['file']} L{a}–L{b}" if is_code else f"{e['file']} (generated output, not in repo)"
                tr += [(au, PK.hasEvidence, eu),
                       (eu, RDF.type, PK.EvidenceItem),
                       (eu, RDFS.label, Literal(loc)),
                       (eu, DCTERMS.description, Literal(e.get("note", loc)))]
                if is_code and B.commit:        # source-code evidence: commit-pinned line range
                    tr.append((eu, PROV.wasDerivedFrom,
                               URIRef(f"{B.repo}/blob/{B.commit}/{e['file']}#L{a}-L{b}")))
                elif B.commit:                  # rendered artefact: derived from the script at the commit
                    tr.append((eu, PROV.wasGeneratedBy,
                               URIRef(f"{B.repo}/blob/{B.commit}/{sim['script']}")))
            prov = [(None, PROV.wasAttributedTo, B.model),
                    (None, PROV.wasDerivedFrom, rubric_uri)]
            for r in papers:                       # what the assessment validates against
                prov.append((None, PROV.wasDerivedFrom, ref_uri(r)))
            if B.commit_uri:
                prov.append((None, PROV.wasDerivedFrom, B.commit_uri))
            if issue:
                prov.append((None, RDFS.seeAlso, issue))
            emit(f"{sim['id']}-{did}.trig",
                 B.nanopub(f"{sim['name']} — {d['name']} {s['value']}/100", PK.LLMClaimAssessment,
                           au, dom, tr, prov))

        # ---- 4. overall checkpoint assessment ----------------------------- #
        gs = group_scores(rubric, sim["scores"])
        au = PKR[f"assessment/{A['id']}-{sim['id']}-overall"]
        env = ", ".join(f"{k} {v}" for k, v in B.env.items())
        desc = (" ".join(sim["summary"].split()) +
                f" Overall {gs['overall']}/100 (faithfulness {gs['faithfulness']}, "
                f"reproducibility {gs['reproducibility']}); rubric morphoeval {rubric['version']}; "
                f"checkpoint commit {B.commit}; environment: {env}.")
        tr = [(claim, PK.hasAssessment, au),
              (au, RDF.type, PK.LLMClaimAssessment),
              (au, RDFS.label, Literal(f"{sim['name']} — MorphoEval overall {gs['overall']}/100")),
              (au, PK.assessmentType, Literal(f"morphoeval/{rubric['version']}/overall")),
              (au, PK.assessedBy, B.model),
              (au, PK.weightOfEvidence, Literal(round(gs["overall"] / 100.0, 4), datatype=XSD.double)),
              (au, PK.hasVerdict, Literal(verdict(gs["overall"]))),
              (au, DCTERMS.description, Literal(desc)),
              (au, DCTERMS.date, Literal(str(A["date"]), datatype=XSD.date)),
              (au, DCTERMS.subject, rubric_uri)]
        tr += B.model_triples()
        for i, fix in enumerate(sim.get("remediation", [])):
            tr.append((au, PK.evidenceAgainst, Literal(f"remediation needed: {fix}")))
        prov = [(None, PROV.wasAttributedTo, B.model)] + \
               [(None, PROV.wasDerivedFrom, x) for x in dim_assessments]
        if B.commit_uri:
            prov.append((None, PROV.wasDerivedFrom, B.commit_uri))
        if issue:
            prov.append((None, RDFS.seeAlso, issue))
        emit(f"{sim['id']}-overall.trig",
             B.nanopub(f"{sim['name']} — MorphoEval overall {gs['overall']}/100",
                       PK.LLMClaimAssessment, au, dom, tr, prov))
        rows.append((sim, gs))

    merged_path = out_dir / f"morphoeval_{A['id']}.ttl"
    merged_path.write_text(merged.serialize(format="turtle"))
    report_path = out_dir / "REPORT.md"
    report_path.write_text(report(rubric, audit, rows))
    return written, merged_path, report_path, merged


# --------------------------------------------------------------------------- #
# Offline ProbKnow conformance check (mirrors project-delphi-ontology/tests)
# --------------------------------------------------------------------------- #
def conformance(files: list[Path]) -> list[str]:
    onto = Graph().parse(ONTOLOGY_PATH, format="turtle")
    declared = {str(s) for s in onto.subjects() if str(s).startswith(str(PK))}
    probs: list[str] = []
    for f in files:
        ds = Dataset(); ds.parse(f, format="trig")
        names = {str(g.identifier) for g in ds.graphs()}
        for part in ("Head", "assertion", "provenance", "pubinfo"):
            if TEMP + part not in names:
                probs.append(f"{f.name}: missing graph {part}")
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
        pub = ds.graph(URIRef(TEMP + "pubinfo"))
        for p in (DCTERMS.creator, DCTERMS.license, NPX.hasNanopubType, NPX.introduces, PK.domain):
            if (THIS, p, None) not in pub:
                probs.append(f"{f.name}: pubinfo lacks {p.n3(ds.namespace_manager)}")
        asrt = ds.graph(URIRef(TEMP + "assertion"))
        for a in asrt.subjects(RDF.type, PK.LLMClaimAssessment):
            m = asrt.value(a, PK.assessedBy)
            if m is None or (m, RDF.type, PK.Model) not in asrt:
                probs.append(f"{f.name}: LLMClaimAssessment without pk:assessedBy a typed pk:Model")
        for s, p, o in asrt:
            if p in (PK.weightOfEvidence, PK.hasConfidence, PK.hasProbability):
                if not (0.0 <= float(o) <= 1.0):
                    probs.append(f"{f.name}: {p.n3(ds.namespace_manager)} {o} outside [0,1]")
        for c in asrt.subjects(RDF.type, PK.Claim):
            if (c, PK.derivedFromPaper, None) not in asrt and (c, PK.hasAssessment, None) not in asrt:
                probs.append(f"{f.name}: claim {c} has neither derivedFromPaper nor hasAssessment")
    return probs


# --------------------------------------------------------------------------- #
# Scoreboard
# --------------------------------------------------------------------------- #
def report(rubric, audit, rows) -> str:
    A = audit["audit"]; dims = rubric["dimensions"]
    L = [f"# MorphoEval report — audit `{A['id']}`", "",
         f"Rubric **{rubric['version']}** · checkpoint `{A['checkpoint']['commit']}` · "
         f"assessor `pkr:model/{A['assessor']['id']}` (pk:LLMClaimAssessment) · "
         f"tracked in {A.get('tracked_by', '—')}", "",
         "Scores are 0–100 (emitted as `pk:weightOfEvidence` = score/100); Overall is the "
         "rubric-weighted mean. Unevaluated scripts carry a claim but no assessments.", "",
         "| Simulation | " + " | ".join(f"{d['id']} ({d['weight']})" for d in dims)
         + " | Faithfulness | Reproducibility | **Overall** | verdict |",
         "|---|" + "---:|" * (len(dims) + 3) + "---|"]
    for sim, gs in rows:
        if gs is None:
            L.append(f"| {sim['name']} (`{sim['script']}`) | " + " | ".join("—" for _ in dims)
                     + " | — | — | *unevaluated* | — |")
            continue
        cells = " | ".join(str(sim["scores"][d["id"]]["value"]) for d in dims)
        L.append(f"| {sim['name']} (`{sim['script']}`) | {cells} | {gs['faithfulness']} | "
                 f"{gs['reproducibility']} | **{gs['overall']}** | {verdict(gs['overall'])} |")
    L += ["", "## Findings and remediation", ""]
    for sim, gs in rows:
        if gs is None:
            continue
        L += [f"### {sim['name']}  —  overall {gs['overall']} ({verdict(gs['overall'])})", "",
              " ".join(sim["summary"].split()), ""]
        for d in dims:
            s = sim["scores"][d["id"]]
            ev = "; ".join(f"`{e['file']}` L{e['lines'][0]}–{e['lines'][1]}" for e in s["evidence"])
            L.append(f"- **{d['name']}: {s['value']}** — {' '.join(str(s['rationale']).split())}  \n  evidence: {ev}")
        if sim.get("remediation"):
            L += ["", "Remediation:"] + [f"- {r}" for r in sim["remediation"]]
        L.append("")
    L += ["## Environment at checkpoint", ""]
    L += [f"- {k}: {v}" for k, v in A["checkpoint"].get("environment", {}).items()]
    if A["checkpoint"].get("note"):
        L += ["", " ".join(A["checkpoint"]["note"].split())]
    return "\n".join(L) + "\n"


DEMO_QUERY = """
PREFIX pk:   <https://w3id.org/probknow/ontology/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?claim ?woe ?verdict ?model WHERE {
  ?c a pk:Claim ; rdfs:label ?claim ; pk:hasAssessment ?a .
  ?a pk:assessmentType ?t ; pk:weightOfEvidence ?woe ; pk:hasVerdict ?verdict ; pk:assessedBy ?m .
  ?m rdfs:label ?model .
  FILTER(STRENDS(?t, "/overall"))
} ORDER BY ?woe
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audit")
    ap.add_argument("--out", default=None, help="output dir (default morphoeval/kg/<audit-id>)")
    ap.add_argument("--check", action="store_true", help="validate the audit only")
    ap.add_argument("--conform", action="store_true", help="run the offline ProbKnow conformance check")
    ap.add_argument("--query", action="store_true", help="run a demo SPARQL on the merged graph")
    args = ap.parse_args()

    rubric, audit = load(Path(args.audit))
    if errs := validate(rubric, audit):
        print("INVALID audit:"); [print("  -", e) for e in errs]; sys.exit(1)
    print(f"audit {audit['audit']['id']} valid against rubric {rubric['version']}")
    if args.check:
        return
    out_dir = Path(args.out) if args.out else HERE / "kg" / audit["audit"]["id"]
    written, merged_path, report_path, merged = build(rubric, audit, out_dir)
    print(f"wrote {len(written)} nanopubs, {merged_path.name} ({len(merged)} triples), {report_path.name}")
    print(f"  -> {out_dir}")
    if args.conform:
        probs = conformance(written)
        if probs:
            print(f"\nProbKnow conformance: {len(probs)} problem(s)"); [print("  -", p) for p in probs]
            sys.exit(2)
        print(f"ProbKnow conformance: OK ({len(written)} nanopubs; pk: terms all declared in "
              f"{ONTOLOGY_PATH.name}; no legacy IRIs; pubinfo complete; LLM assessments attributed)")
    if args.query:
        print("\nSPARQL: overall assessments")
        for r in merged.query(DEMO_QUERY):
            print(f"  {float(r.woe):.3f}  {str(r.verdict):20s} {r.claim}   [assessed by {r.model}]")


if __name__ == "__main__":
    main()
