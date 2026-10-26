#!/usr/bin/env python3
"""
Check the logged claim that the zebrafish and Xenopus BMP pairs are
"orthologs of the same [activator-inhibitor] pair" (sequences/README.md).

Logged claim under test (2026-10-26):
  *Xenopus* bmp4/nog (noggin) give orthologs of the same pair [as zebrafish
  bmp2b/chd] in two more species, so diffusion priors can be compared.

Two falsifiable sub-claims:
  A. ORTHOLOGY   -- bmp2b's Xenopus ortholog is the catalog's bmp4 gene, and
                    chd's Xenopus ortholog is the catalog's nog gene.
                    (Online: Ensembl Compara REST. Offline: UNRESOLVED.)
  B. EQUIVALENCE -- both pairs satisfy the classical Turing size-prior
                    ordering D_inhibitor > D_activator under the repo's own
                    prior D ~ L^(-1/3), computed on the SHA-256-pinned
                    precursor sequences in sequences/sequence_links.json.
                    (Offline, hermetic.)

Verdict REFUTED if either checked sub-claim fails. Exit 1 = REFUTED,
0 = SUPPORTED, 2 = UNRESOLVED (e.g. online evidence unavailable and the
offline half passes). The full evidence is written as a JSON log.

Usage:
  python3 sequences/check_pair_equivalence.py                 # offline B only
  python3 sequences/check_pair_equivalence.py --online        # A + B
  python3 sequences/check_pair_equivalence.py --online --log sequences/pair_equivalence_log.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
LINKS = HERE / "sequence_links.json"
CACHE = HERE / "cache"
DEFAULT_LOG = HERE / "pair_equivalence_log.json"
ENSEMBL = "https://rest.ensembl.org"
UA = {"User-Agent": "infinitemorphospace/pair-equivalence-check (research; stdlib urllib)",
      "Accept": "application/json"}

CLAIM = ("Xenopus bmp4/nog give orthologs of the same pair as zebrafish "
         "bmp2b/chd, so diffusion priors can be compared across species.")

# BMP activator/antagonist pairs as cast in catalog.json (same species,
# morphogen_role gradient_source vs inhibitor, pathway BMP/BMP-antagonist).
PAIRS = [("danio_rerio", "bmp2b", "chd"),
         ("xenopus_tropicalis", "bmp4", "nog")]

# Catalog Ensembl IDs the claim (A) predicts as ortholog targets in Xenopus.
CATALOG_XENOPUS = {"bmp4": "ENSXETG00000008620", "nog": "ENSXETG00000043161"}


def diffusion_ratio(activator_aa: int, inhibitor_aa: int) -> float:
    """D_inh/D_act under D ~ L^(-1/3); >1 = inhibitor predicted faster."""
    return (activator_aa / inhibitor_aa) ** (1.0 / 3.0)


def load_resolved() -> dict:
    data = json.loads(LINKS.read_text())
    return {(r["species"], r["gene_symbol"]): r for r in data["resolved"]}


def verify_hashes(resolved: dict) -> list:
    """Every pinned protein FASTA must hash to its recorded SHA-256."""
    rows = []
    for (sp, sym), r in sorted(resolved.items()):
        p = HERE / r["protein_fasta"]
        seq = "".join(ln.strip() for ln in p.read_text().splitlines()
                      if not ln.startswith(">"))
        ok = (hashlib.sha256(seq.encode()).hexdigest() == r["protein_sha256"]
              and len(seq) == r["protein_length_aa"])
        rows.append({"species": sp, "gene_symbol": sym, "protein_aa": len(seq),
                     "sha256_verified": ok})
    return rows


def check_equivalence(resolved: dict) -> dict:
    rows = []
    for sp, act, inh in PAIRS:
        a, i = resolved[(sp, act)], resolved[(sp, inh)]
        ratio = diffusion_ratio(a["protein_length_aa"], i["protein_length_aa"])
        rows.append({
            "species": sp, "activator": act, "inhibitor": inh,
            "activator_aa": a["protein_length_aa"],
            "inhibitor_aa": i["protein_length_aa"],
            "activator_accession": a.get("ensembl_gene_id") or a.get("uniprot"),
            "inhibitor_accession": i.get("ensembl_gene_id") or i.get("uniprot"),
            "D_inhibitor_over_D_activator": round(ratio, 4),
            "supports_faster_inhibitor": ratio > 1.0,
        })
    return {"pairs": rows,
            "all_pairs_pass": all(r["supports_faster_inhibitor"] for r in rows)}


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310 fixed API
        return json.loads(r.read().decode("utf-8"))


def _lookup(gene_id: str) -> str:
    try:
        return _get(f"{ENSEMBL}/lookup/id/{gene_id}").get("display_name", "?")
    except Exception:
        return "?"


def _homology_targets(species: str, gene_id: str, target_species: str) -> list:
    d = _get(f"{ENSEMBL}/homology/id/{species}/{gene_id}?target_species={target_species}")
    out = []
    for h in d.get("data", [{}])[0].get("homologies", []):
        tid = h.get("target", {}).get("id")
        out.append({"type": h.get("type"), "target_id": tid,
                    "target_symbol": _lookup(tid) if tid else None})
    return out


def check_orthology(resolved: dict) -> dict:
    """Claim A: bmp2b -> Xenopus bmp4 (catalog) and chd -> Xenopus nog (catalog)."""
    danio_bmp = resolved[("danio_rerio", "bmp2b")]["ensembl_gene_id"]
    tests = [{
        "gene": "danio_rerio bmp2b", "gene_id": danio_bmp,
        "predicted_xenopus_ortholog": CATALOG_XENOPUS["bmp4"],
        "predicted_symbol": "bmp4",
        "actual_homologies": _homology_targets("danio_rerio", danio_bmp, "xenopus_tropicalis"),
    }]
    tests[-1]["claim_holds"] = any(
        h["target_id"] == CATALOG_XENOPUS["bmp4"] for h in tests[-1]["actual_homologies"])
    # chd has no Ensembl gene under that symbol (UniProt O57472 fallback in
    # fetch_sequences.py); query Compara by symbol instead.
    try:
        d = _get(f"{ENSEMBL}/homology/symbol/danio_rerio/chd?target_species=xenopus_tropicalis")
        hs = [{"type": h.get("type"), "target_id": h.get("target", {}).get("id"),
               "target_symbol": _lookup(h.get("target", {}).get("id"))}
              for h in d.get("data", [{}])[0].get("homologies", [])]
    except Exception as exc:
        hs, tests_err = [], str(exc)[:120]
    else:
        tests_err = None
    tests.append({
        "gene": "danio_rerio chd", "gene_id": None,
        "predicted_xenopus_ortholog": CATALOG_XENOPUS["nog"],
        "predicted_symbol": "nog",
        "actual_homologies": hs, "query_error": tests_err,
        "claim_holds": any(h["target_id"] == CATALOG_XENOPUS["nog"] for h in hs),
    })
    return {"tests": tests, "claim_holds": all(t["claim_holds"] for t in tests)}


def run(online: bool = False) -> dict:
    resolved = load_resolved()
    hashes = verify_hashes(resolved)
    equiv = check_equivalence(resolved)
    log = {
        "schema_version": "pair-equivalence-log-1.0",
        "logged_at": "2026-10-26",
        "claim_under_test": CLAIM,
        "claim_source": "sequences/README.md, 'The curated panel' (Turing / reaction-diffusion pairs)",
        "evidence_basis": {
            "sequences": "sequences/sequence_links.json + sequences/cache/*.fasta (SHA-256 pinned)",
            "taxon_ids": {"danio_rerio": 7955, "xenopus_tropicalis": 8364},
            "orthology_source": ("Ensembl Compara REST (live)" if online
                                 else "not queried (offline run)"),
        },
        "hash_verification": {"all_verified": all(h["sha256_verified"] for h in hashes),
                              "n": len(hashes)},
        "subclaim_B_diffusion_equivalence": equiv,
        "subclaim_A_orthology": (check_orthology(resolved) if online
                                 else {"status": "UNRESOLVED (run with --online)"}),
    }
    b_fail = not equiv["all_pairs_pass"]
    a_fail = online and not log["subclaim_A_orthology"].get("claim_holds", False)
    if b_fail or a_fail:
        log["verdict"] = "REFUTED"
        reasons = []
        if b_fail:
            bad = [r["species"] for r in equiv["pairs"] if not r["supports_faster_inhibitor"]]
            reasons.append(f"diffusion ordering fails for: {', '.join(bad)} "
                           "(D_inhibitor/D_activator <= 1 under the repo size prior)")
        if a_fail:
            reasons.append("Ensembl Compara ortholog targets are not the catalog "
                           "bmp4/nog genes (paralog / non-homolog substitution)")
        log["refutation_reasons"] = reasons
    elif not online:
        log["verdict"] = "UNRESOLVED"
        log["refutation_reasons"] = []
    else:
        log["verdict"] = "SUPPORTED"
        log["refutation_reasons"] = []
    return log


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Logged test: BMP pair-equivalence claim")
    ap.add_argument("--online", action="store_true", help="also query Ensembl Compara (sub-claim A)")
    ap.add_argument("--log", default=str(DEFAULT_LOG), help="write JSON evidence log here")
    args = ap.parse_args(argv)
    log = run(online=args.online)
    Path(args.log).write_text(json.dumps(log, indent=2) + "\n")
    print(f"Claim: {CLAIM}")
    print(f"Hashes: {log['hash_verification']['n']} pinned protein FASTAs, "
          f"all verified = {log['hash_verification']['all_verified']}")
    for r in log["subclaim_B_diffusion_equivalence"]["pairs"]:
        print(f"  {r['species']:20s} {r['activator']}({r['activator_aa']} aa) / "
              f"{r['inhibitor']}({r['inhibitor_aa']} aa)  "
              f"D_inh/D_act = {r['D_inhibitor_over_D_activator']}  "
              f"faster-inhibitor: {r['supports_faster_inhibitor']}")
    if args.online:
        for t in log["subclaim_A_orthology"]["tests"]:
            actual = [(h["target_symbol"], h["target_id"], h["type"]) for h in t["actual_homologies"]]
            print(f"  orthology {t['gene']}: predicted {t['predicted_symbol']} "
                  f"{t['predicted_xenopus_ortholog']} | actual {actual} | holds: {t['claim_holds']}")
    else:
        print("  orthology: UNRESOLVED (offline; rerun with --online)")
    print(f"VERDICT: {log['verdict']}")
    for reason in log["refutation_reasons"]:
        print(f"  - {reason}")
    print(f"Log written: {args.log}")
    return {"REFUTED": 1, "SUPPORTED": 0, "UNRESOLVED": 2}[log["verdict"]]


if __name__ == "__main__":
    raise SystemExit(main())
