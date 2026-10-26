#!/usr/bin/env python3
"""
Record the Thompson (1917) claim-validation PROCESS as nanopublications.

Why a process nanopub, not just result nanopubs?
  The repo already records *results* (nanopubs/thompson_predictions/*.trig).
  This script records *how a dated 1917 claim travels through the pipeline*,
  as a PROV chain that any later result nanopub can point back to:

    Step 1  CLAIM CITATION (Pass 1)
            Thompson (1917, 1st ed.) claim extracted with chapter/printed
            pages/figures and date_made=1917 into
            thompson/thompson_1917_claims.json (builder/checker:
            thompson/build_1917_claim_inventory.py, OCR-verified against the
            pinned Internet Archive full text, SHA-256 in the registry).
    Step 2  VALIDATION ASSESSMENT (Pass 2)
            Status vs repo artifacts, verdict if any, method, data needed,
            effort (person-days) -- the `pass2` block of each registry row.
    Step 3  DATASET-EXISTENCE SCOUT
            thompson/scout_1917_datasets.py probes (a) local repo paths,
            (b) Dryad API v2 live search (title-overlap filtered, throttled
            and cached -- Dryad returns HTTP 429 under rapid querying, and a
            failed probe is recorded as UNPROBED, never as 'no dataset'),
            (c) Wikimedia Commons, (d) fallback: digitise Thompson's own
            1917 plates from the pinned IA scan. Output:
            thompson/thompson_1917_dataset_audit.json with an existence
            class and an automation tier per claim.
    Step 4  LOOP QUEUEING
            Every FULL_LOOP / LOOP_WITH_SEED claim becomes a
            campaign-compatible slot in thompson/thompson_1917_loop_queue.yaml
            and a full 12-step MorphoEval cloud-loop prompt (reusing
            morphoeval/cloud_orchestrator.build_cloud_agent_prompt) with a
            claim override: Step 1 becomes dataset *verification*, a
            DATASET_MISMATCH stop replaces silent analogue-fitting, and the
            verdict is written to datasets/thompson_1917/<claim>/result.json
            for human merge-back into the registry.
    Step 5  LOOP EXECUTION + AUDIT (per claim, not yet done for the example)
            The 12-step loop fits, self-audits (morphoeval/rubric.yaml),
            is independently measured, and its result nanopub (future)
            must link back to the process-instance nanopub below.

Nanopubs written (nanopubs/thompson_predictions/):
  np_thompson_process-1917-claim-validation-workflow.trig
      The workflow itself (prov:Plan + 5 step activities + counts).
  np_thompson_process-example-t1917-051-crocodile-skulls.trig
      Worked example: claim T1917-051 (Thompson 1917, Ch. XVII, pp. 752-753,
      Fig. 383, crocodile skulls) carried through Steps 1-4 on 2026-10-26:
      dataset scouted PUBLIC_SCRIPTABLE (Dryad 10.5061/dryad.14fn1), loop
      slot queued, prompt generated (SHA-256 pinned), Step 5 pending.
  np_thompson_process-example-t1917-046-scarus-pomacanthus.trig
      Completed example: T1917-046 (Thompson 1917, p. 749, Figs. 375-376)
      carried through all steps before this registry existed; verdict
      recorded (literal conformal reading refuted; see audit link).

Every file the process produced is pinned by SHA-256 in the assertion, so
the nanopub stays verifiable even before the artifacts are committed.

Usage (use the repo venv: rdflib/nanopub live there):
    venv/bin/python nanopubs/thompson_1917_process_to_nanopub.py
    venv/bin/python nanopubs/thompson_1917_process_to_nanopub.py --sign
    venv/bin/python nanopubs/thompson_1917_process_to_nanopub.py --publish test
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from rdflib import Dataset, Graph, Literal, Namespace, URIRef
from rdflib.namespace import DCTERMS, PROV, RDF, RDFS, XSD

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "nanopubs" / "thompson_predictions"
PROFILE = Path.home() / ".nanopub" / "profile.yml"
PROD_REGISTRY = "https://registry.knowledgepixels.com/np/"

BASE = "https://w3id.org/morpholib/thompson/np/"
MLT = Namespace("https://w3id.org/morpholib/thompson/")
NP = Namespace("http://www.nanopub.org/nschema#")
THOMPSON_1917 = URIRef("https://archive.org/details/ongrowthform1917thom")
IA_TEXT_SHA256 = "1ba78d3708fba8c8cf1b819cf951dad72ccb0ac34fb3dd13e5c23745b68f9ff0"

GH = "https://github.com/fractastical/morpholib/blob/main/"
ARTIFACTS = {
    "claim_registry": "thompson/thompson_1917_claims.json",
    "claim_inventory_doc": "docs/THOMPSON_1917_CLAIM_INVENTORY.md",
    "registry_builder": "thompson/build_1917_claim_inventory.py",
    "dataset_audit": "thompson/thompson_1917_dataset_audit.json",
    "dataset_existence_doc": "docs/THOMPSON_1917_DATASET_EXISTENCE.md",
    "dataset_scout": "thompson/scout_1917_datasets.py",
    "loop_queue": "thompson/thompson_1917_loop_queue.yaml",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def read_profile() -> dict:
    if not PROFILE.exists():
        return {}
    out = {}
    for line in PROFILE.read_text().splitlines():
        if ":" in line and not line.startswith("#"):
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip() or None
    return out


def new_dataset(np_id: str):
    base = URIRef(BASE + np_id + "/")
    ds = Dataset()
    for pfx, ns in [("np", NP), ("mlt", MLT), ("dcterms", DCTERMS), ("prov", PROV), ("xsd", XSD), ("rdfs", RDFS)]:
        ds.bind(pfx, ns)
    head = ds.graph(URIRef(base + "Head"))
    assertion = ds.graph(URIRef(base + "assertion"))
    provenance = ds.graph(URIRef(base + "provenance"))
    pubinfo = ds.graph(URIRef(base + "pubinfo"))
    head.add((base, RDF.type, NP.Nanopublication))
    head.add((base, NP.hasAssertion, assertion.identifier))
    head.add((base, NP.hasProvenance, provenance.identifier))
    head.add((base, NP.hasPublicationInfo, pubinfo.identifier))
    return ds, base, assertion, provenance, pubinfo


def artifact_node(g: Graph, key: str) -> URIRef:
    rel = ARTIFACTS[key]
    node = URIRef(GH + rel)
    p = REPO_ROOT / rel
    g.add((node, RDF.type, PROV.Entity))
    g.add((node, RDFS.label, Literal(f"{rel} (repo artifact)")))
    if p.exists():
        g.add((node, MLT.sha256, Literal(sha256(p))))
        g.add((node, MLT.bytes, Literal(len(p.read_bytes()), datatype=XSD.integer)))
    return node


def add_provenance(P: Graph, base: URIRef, assertion_uri: URIRef, commit: str, today: str, script_rel: str):
    activity = URIRef(f"https://github.com/fractastical/morpholib/commit/{commit}")
    script = URIRef(GH + script_rel)
    P.add((assertion_uri, PROV.wasGeneratedBy, activity))
    P.add((activity, RDF.type, PROV.Activity))
    P.add((activity, RDFS.label, Literal(f"Thompson 1917 process-nanopub generation at morpholib commit {commit[:12]} (working-tree artifacts pinned by SHA-256 in the assertion)")))
    P.add((activity, PROV.used, script))
    P.add((activity, PROV.used, THOMPSON_1917))
    P.add((activity, PROV.endedAtTime, Literal(today, datatype=XSD.date)))
    P.add((assertion_uri, PROV.wasAttributedTo, script))
    P.add((script, RDF.type, PROV.SoftwareAgent))
    P.add((script, RDFS.label, Literal("Thompson 1917 claim-validation process: claim registry -> validation assessment -> dataset-existence scout -> loop queue -> MorphoEval loop")))


def add_pubinfo(I: Graph, base: URIRef, today: str, orcid, name):
    I.add((base, DCTERMS.created, Literal(today, datatype=XSD.date)))
    I.add((base, DCTERMS.license, URIRef("https://creativecommons.org/publicdomain/zero/1.0/")))
    I.add((base, DCTERMS.isPartOf, URIRef(GH + "docs/THOMPSON_1917_DATASET_EXISTENCE.md")))
    if orcid:
        I.add((base, DCTERMS.creator, URIRef(orcid)))
        if name:
            I.add((URIRef(orcid), RDFS.label, Literal(name)))


STEPS = [
    ("step1-claim-citation", "Step 1 — Claim citation (Pass 1)", "Extract each distinct testable assertion from Thompson (1917, 1st ed.) with chapter, printed pages (IA leaf - 20) and figures; date_made=1917 on every row; OCR-verify chapter/figure anchors against the pinned IA full text.", "claim_registry"),
    ("step2-validation-assessment", "Step 2 — Validation assessment (Pass 2)", "Score each claim against repo artifacts: status (DONE_REPO/PARTIAL_REPO/DONE_ELSEWHERE/UNTESTED/NOT_TESTABLE_AS_STATED), verdict where one exists, method, data needed, blockers, effort in person-days.", "claim_registry"),
    ("step3-dataset-existence-scout", "Step 3 — Dataset-existence scout", "Probe local repo paths, Dryad API v2 (throttled, cached; failures recorded as UNPROBED, never as 'no dataset'), Wikimedia Commons, and the digitise-Thompson's-own-1917-plates fallback. Emit existence class + automation tier per claim.", "dataset_audit"),
    ("step4-loop-queueing", "Step 4 — Loop queueing", "Turn every FULL_LOOP/LOOP_WITH_SEED claim into a campaign-compatible slot plus a 12-step cloud-loop prompt with the claim override (dataset verification, DATASET_MISMATCH stop, result.json merge-back).", "loop_queue"),
    ("step5-loop-execution", "Step 5 — Loop execution and audit (per claim)", "MorphoEval 12-step loop: fetch+SHA-pin dataset, fit, --self-test, rubric audit, independent measurement, KG/nanopub of the RESULT, linking back to the claim's process-instance nanopub.", None),
]


def workflow_nanopub(counts: dict, commit: str, today: str, orcid, name):
    np_id = "process-1917-claim-validation-workflow"
    ds, base, A, P, I = new_dataset(np_id)
    wf = URIRef(MLT + "process/1917-claim-validation-workflow")
    A.add((wf, RDF.type, MLT.ClaimValidationWorkflow))
    A.add((wf, RDF.type, PROV.Plan))
    A.add((wf, RDFS.label, Literal("Process for validating every claim in Thompson (1917) On Growth and Form: citation -> validation assessment -> dataset-existence scout -> loop queue -> MorphoEval loop")))
    A.add((wf, MLT.dateOfClaimsValidated, Literal("1917", datatype=XSD.gYear)))
    A.add((wf, MLT.sourceEdition, Literal("Thompson, D.W. (1917) On Growth and Form, 1st edition, Cambridge University Press (not the 1942 2nd edition or 1961 abridgement)")))
    A.add((wf, MLT.sourceInternetArchiveId, Literal("ongrowthform1917thom")))
    A.add((wf, MLT.sourceFullTextSha256, Literal(IA_TEXT_SHA256)))
    A.add((wf, MLT.leafToPrintedPageRule, Literal("printed_page = IA leaf - 20 (checked: leaf 769 = p. 749)")))
    prev = None
    for step_id, label, desc, artifact_key in STEPS:
        node = URIRef(MLT + f"process/{step_id}")
        A.add((wf, MLT.hasProcessStep, node))
        A.add((node, RDF.type, MLT.ProcessStep))
        A.add((node, RDFS.label, Literal(label)))
        A.add((node, DCTERMS.description, Literal(desc)))
        if prev is not None:
            A.add((node, MLT.followsStep, prev))
        if artifact_key:
            A.add((node, MLT.producesArtifact, artifact_node(A, artifact_key)))
        prev = node
    for k, v in counts.items():
        A.add((wf, MLT[k], Literal(v, datatype=XSD.integer)))
    A.add((wf, MLT.failureMode, Literal("DATASET_MISMATCH: if the scouted dataset does not match the claim's species/variables, the loop stops and reports mismatch instead of fitting an analogue and claiming a verdict. Analogues must be labelled as such.")))
    A.add((wf, MLT.failureMode, Literal("UNPROBED is not 'no dataset': a failed/rate-limited probe (e.g. Dryad HTTP 429) is recorded as UNPROBED and re-probed by the loop with throttling and caching.")))
    A.add((wf, MLT.mergeBackRule, Literal("Loops write only new files and a per-claim result.json; a human merges verdicts back into thompson/thompson_1917_claims.json. Loops must not edit the registry.")))
    add_provenance(P, base, A.identifier, commit, today, "nanopubs/thompson_1917_process_to_nanopub.py")
    add_pubinfo(I, base, today, orcid, name)
    return np_id, ds


def previous_test_uri(np_id: str) -> str | None:
    """Last test-registry trusty URI published for this nanopub file, if any."""
    manifest = OUT_DIR / "published_manifest_test.txt"
    if not manifest.exists():
        return None
    fname, found = f"np_thompson_{np_id}.trig", None
    for line in manifest.read_text().splitlines():
        parts = line.split("|")
        if len(parts) >= 2 and parts[0] == fname:
            found = parts[1]
    return found


def claim_process_nanopub(row: dict, audit_row: dict, np_suffix: str, step5: str, commit: str, today: str, orcid, name, extra: list[tuple] | None = None, lit: dict | None = None, prev_uri: str | None = None):
    np_id = f"process-example-{np_suffix}"
    ds, base, A, P, I = new_dataset(np_id)
    subj = URIRef(MLT + f"process-instance/{np_suffix}")
    A.add((subj, RDF.type, MLT.ClaimValidationProcessInstance))
    A.add((subj, MLT.instantiatesWorkflow, URIRef(MLT + "process/1917-claim-validation-workflow")))
    A.add((subj, RDFS.label, Literal(f"Process instance: {row['id']} — Thompson (1917), Ch. {row['chapter']}, pp. {row['pages']}, Figs. {row['figures']}")))
    A.add((subj, MLT.claimId, Literal(row["id"])))
    A.add((subj, MLT.dateMade, Literal("1917", datatype=XSD.gYear)))
    A.add((subj, MLT.citation, Literal(f"Thompson, D.W. (1917) On Growth and Form. Cambridge University Press. Ch. {row['chapter']}, pp. {row['pages']}, Figs. {row['figures']}.")))
    A.add((subj, MLT.claim, Literal(row["claim"])))
    if row.get("quote"):
        A.add((subj, MLT.verbatimQuote, Literal(row["quote"])))
    A.add((subj, MLT.pass2Status, Literal(row["pass2"]["status"])))
    A.add((subj, MLT.effortDaysLow, Literal(row["pass2"]["effort_days"][0], datatype=XSD.integer)))
    A.add((subj, MLT.effortDaysHigh, Literal(row["pass2"]["effort_days"][1], datatype=XSD.integer)))
    if audit_row:
        A.add((subj, MLT.datasetExistence, Literal(audit_row["existence"])))
        A.add((subj, MLT.automationTier, Literal(audit_row["automation"])))
        A.add((subj, MLT.datasetAuditReason, Literal(audit_row["reason"])))
        A.add((subj, MLT.dataKindNeeded, Literal(audit_row["data_kind_needed"] or "")))
        for hit in audit_row.get("dryad_hits") or []:
            if hit.get("identifier"):
                A.add((subj, MLT.candidateDataset, URIRef("https://doi.org/" + hit["identifier"].replace("doi:", ""))))
                A.add((subj, MLT.candidateDatasetTitle, Literal(hit["title"])))
    if lit:
        A.add((subj, MLT.literatureRelation, Literal(lit["relation"])))
        A.add((subj, MLT.literatureWhoAndWhen, Literal(lit["who_and_when"])))
        A.add((subj, MLT.literatureCitation, Literal(lit["literature_citation"])))
        A.add((subj, MLT.counterclaim, Literal(lit["counterclaim"])))
        if lit.get("quantitative_margin"):
            A.add((subj, MLT.quantitativeMargin, Literal(lit["quantitative_margin"])))
        A.add((subj, MLT.literatureArtifact, URIRef(GH + "thompson/thompson_1917_counterclaims.json")))
    if prev_uri:
        A.add((subj, PROV.wasRevisionOf, URIRef(prev_uri)))
    A.add((subj, MLT.step5Status, Literal(step5)))
    A.add((subj, MLT.usesArtifact, artifact_node(A, "claim_registry")))
    if audit_row:
        A.add((subj, MLT.usesArtifact, artifact_node(A, "dataset_audit")))
        A.add((subj, MLT.usesArtifact, artifact_node(A, "loop_queue")))
    for pred, val in (extra or []):
        A.add((subj, pred, val))
    add_provenance(P, base, A.identifier, commit, today, "nanopubs/thompson_1917_process_to_nanopub.py")
    add_pubinfo(I, base, today, orcid, name)
    return np_id, ds


def prompt_sha256(claim_id: str) -> str | None:
    """Generate the claim's loop prompt (queue must exist) and pin its hash, not its text."""
    try:
        import yaml  # venv only

        doc = yaml.safe_load((REPO_ROOT / ARTIFACTS["loop_queue"]).read_text())
        entry = next((s for s in doc["slots"] if s["claim_id"] == claim_id), None)
        if entry is None:
            return None
        sys.path.insert(0, str(REPO_ROOT / "morphoeval"))
        import cloud_orchestrator

        prompt = cloud_orchestrator.build_cloud_agent_prompt(entry)
        return hashlib.sha256(prompt.encode()).hexdigest()
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sign", action="store_true", help="also write signed.<name>.trig using ~/.nanopub/profile.yml")
    ap.add_argument("--publish", choices=["test", "prod"], help="sign + publish (test registry first is the sane default)")
    ap.add_argument("--all", action="store_true", help="also write a process-instance nanopub for EVERY claim (roll-out mode; default is workflow + 2 examples). Sign/publish only with an explicit decision -- 59 nanopubs is a batch.")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    reg = json.loads((REPO_ROOT / ARTIFACTS["claim_registry"]).read_text())
    audit = json.loads((REPO_ROOT / ARTIFACTS["dataset_audit"]).read_text())
    cc_path = REPO_ROOT / "thompson" / "thompson_1917_counterclaims.json"
    counterclaims = json.loads(cc_path.read_text())["claims"] if cc_path.exists() else {}
    claims = {c["id"]: c for c in reg["claims"]}
    audit_rows = {r["id"]: r for r in audit["rows"]}
    from collections import Counter

    ex = Counter(r["existence"] for r in audit["rows"])
    au = Counter(r["automation"] for r in audit["rows"])
    counts = {
        "nClaims": len(reg["claims"]),
        "nDoneRepo": sum(1 for c in reg["claims"] if c["pass2"]["status"] == "DONE_REPO"),
        "nPartialRepo": sum(1 for c in reg["claims"] if c["pass2"]["status"] == "PARTIAL_REPO"),
        "nUntested": sum(1 for c in reg["claims"] if c["pass2"]["status"] == "UNTESTED"),
        "nInRepoDatasets": ex.get("IN_REPO", 0),
        "nPublicScriptable": ex.get("PUBLIC_SCRIPTABLE", 0),
        "nFullLoop": au.get("FULL_LOOP", 0),
        "nLoopWithSeed": au.get("LOOP_WITH_SEED", 0),
        "nHumanRequired": au.get("HUMAN_REQUIRED", 0),
        "nLoopQueueSlots": au.get("FULL_LOOP", 0) + au.get("LOOP_WITH_SEED", 0),
        "nLitNoDirectTest": sum(1 for v in counterclaims.values() if v["relation"] == "NO_DIRECT_TEST"),
        "nLitPartiallyRefuted": sum(1 for v in counterclaims.values() if v["relation"] == "PARTIALLY_REFUTED"),
        "nLitSuperseded": sum(1 for v in counterclaims.values() if v["relation"] == "SUPERSEDED"),
        "nLitSupported": sum(1 for v in counterclaims.values() if v["relation"] == "SUPPORTED"),
    }
    commit, today = git_commit(), dt.date.today().isoformat()
    prof = read_profile()
    orcid, name = prof.get("orcid_id"), prof.get("name")
    args.out.mkdir(parents=True, exist_ok=True)

    written = [workflow_nanopub(counts, commit, today, orcid, name)]

    p051 = prompt_sha256("T1917-051")
    extra051 = [(MLT.loopQueueSlot, Literal("thompson1917-t1917-051 (slot 241 in thompson/thompson_1917_loop_queue.yaml)"))]
    if p051:
        extra051.append((MLT.loopPromptSha256, Literal(p051)))
    written.append(claim_process_nanopub(
        claims["T1917-051"], audit_rows.get("T1917-051"), "t1917-051-crocodile-skulls",
        "PENDING: Steps 1-4 completed 2026-10-26 (claim cited, assessment UNTESTED, dataset PUBLIC_SCRIPTABLE via Dryad, loop slot queued and prompt generated). Step 5 (loop execution) not yet dispatched.",
        commit, today, orcid, name, extra051, lit=counterclaims.get("T1917-051"),
        prev_uri=previous_test_uri("process-example-t1917-051-crocodile-skulls")))

    if args.all:
        import re as _re

        for cid, crow in claims.items():
            if cid in ("T1917-051", "T1917-046"):
                continue
            suffix = cid.lower() + "-" + _re.sub(r"[^a-z0-9]+", "-", crow["chapter_title"].lower()).strip("-")[:36]
            arow = audit_rows.get(cid)
            if crow["pass2"]["status"] == "DONE_REPO":
                s5 = "COMPLETED (see registry pass2 + repo evidence); this instance records the process retrospectively."
            elif arow and arow["automation"] in ("FULL_LOOP", "LOOP_WITH_SEED"):
                s5 = f"PENDING Step 5: Steps 1-4 completed (dataset {arow['existence']}, tier {arow['automation']}); loop slot queued in thompson/thompson_1917_loop_queue.yaml."
            elif arow and arow["automation"] == "HUMAN_REQUIRED":
                s5 = f"BLOCKED before Step 5: HUMAN_REQUIRED -- {arow['human_blocker']}"
            else:
                s5 = "NOT QUEUED: not testable as stated, analytic, or unprobed; see dataset audit."
            written.append(claim_process_nanopub(crow, arow, suffix, s5, commit, today, orcid, name,
                                                 lit=counterclaims.get(cid),
                                                 prev_uri=previous_test_uri(f"process-example-{suffix}")))

    written.append(claim_process_nanopub(
        claims["T1917-046"], audit_rows.get("T1917-046"), "t1917-046-scarus-pomacanthus",
        "COMPLETED (pre-registry, 2026-10-16/17): all steps executed by thompson/1917-thompson-scarus-pomacanthus.py and the MorphoEval audit 2026-10-17 (76.6/100). Verdict: literal conformal (Mobius) coaxial-circle reading REFUTED (explains ~1%; conformal maps cannot change depth/length 0.377->0.587); generous oblique-coaxial reading explains 32% at genus-mean landmarks, fails leave-one-landmark-out (0.0371 > 0.0312), and on real outlines is worse than no deformation (IoU 0.618 vs similarity 0.662).",
        commit, today, orcid, name,
        [(MLT.auditReport, URIRef(GH + "morphoeval/kg/2026-10-17-thompson-audit/REPORT.md")),
         (MLT.verdict, Literal("refuted-as-stated (conformal reading); bounded/weak in the generous oblique reading"))],
        lit=counterclaims.get("T1917-046"),
        prev_uri=previous_test_uri("process-example-t1917-046-scarus-pomacanthus")))

    for np_id, ds in written:
        path = args.out / f"np_thompson_{np_id}.trig"
        ds.serialize(destination=str(path), format="trig")
        print(f"wrote {path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path}")
    print(f"counts pinned in workflow nanopub: {counts}")

    if args.sign or args.publish:
        from nanopub import Nanopub, NanopubConf, Profile

        if not prof:
            print("no ~/.nanopub/profile.yml; cannot sign")
            return 1
        profile = Profile(orcid_id=prof["orcid_id"], name=prof["name"], private_key=Path(prof["private_key"]),
                          public_key=Path(prof["public_key"]), introduction_nanopub_uri=prof.get("introduction_nanopub_uri"))
        conf = NanopubConf(profile=profile, use_test_server=(args.publish == "test"),
                           add_prov_generated_time=False, add_pubinfo_generated_time=False,
                           attribute_publication_to_profile=False)
        if args.publish == "prod":
            conf.use_server = PROD_REGISTRY
        ok = pub = 0
        for np_id, _ in written:
            src = args.out / f"np_thompson_{np_id}.trig"
            try:
                npub = Nanopub(rdf=src, conf=conf)
                npub.sign()
                if not npub.has_valid_signature:
                    raise RuntimeError("signature did not verify")
                signed = args.out / f"signed.np_thompson_{np_id}.trig"
                npub.rdf.serialize(destination=str(signed), format="trig")
                ok += 1
                print(f"signed {signed.name} (valid signature)")
                if args.publish:
                    npub.publish()
                    pub += 1
                    manifest = args.out / f"published_manifest_{args.publish}.txt"
                    with manifest.open("a") as fh:
                        fh.write(f"{src.name}|{npub.source_uri}|{dt.datetime.now().isoformat()}|{args.publish}\n")
                    print(f"  published -> {npub.source_uri}")
            except Exception as e:  # noqa: BLE001
                print(f"  failed for {src.name}: {e}")
        print(f"signed {ok}" + (f", published {pub} ({args.publish})" if args.publish else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
