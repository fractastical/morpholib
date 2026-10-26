#!/usr/bin/env python3
"""
Link known DNA/RNA/protein sequences to this repo's morphogenesis simulations.

Resolves every gene in `sequences/catalog.json` LIVE against public, stable,
accession-based databases -- no sequence is hand-copied into the repo:

  * Ensembl REST   https://rest.ensembl.org/lookup/symbol/{species}/{symbol}
                   https://rest.ensembl.org/sequence/id/{transcript}?type=cds
                   https://rest.ensembl.org/sequence/id/{transcript}?type=protein
  * UniProt REST   https://rest.uniprot.org/uniprotkb/search?query=gene:{g}+AND+organism_id:{taxid}
  * NCBI E-utils   (fallback / cross-reference; see --ncbi in the README)

What you get per gene (written to `sequences/sequence_links.json`, FASTA cached
under `sequences/cache/` and SHA-256 pinned there, exactly as
`models/1952-turing-morpho.py` pins its specimen photographs):

  gene/transcript/protein accessions, CDS + protein sequence, length, GC%,
  protein mass proxy, and a RELATIVE diffusion prior D_rel = (M_ref/M)^(1/3).

HONEST BOUNDARY (also in catalog.json): a sequence identifies the molecule and
ranks its size/diffusion against other morphogens. It does NOT give absolute
diffusion coefficients, production/degradation rates or expression patterns --
those stay literature parameters in the model scripts.

Usage:
  python3 sequences/fetch_sequences.py              # resolve all, write sequence_links.json
  python3 sequences/fetch_sequences.py --only bmp2b,dpp
  python3 sequences/fetch_sequences.py --offline    # report cache only, no network
  python3 sequences/fetch_sequences.py --self-test  # offline logic tests, no network
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CATALOG = HERE / "catalog.json"
LINKS_OUT = HERE / "sequence_links.json"
CACHE = HERE / "cache"
ENSEMBL = "https://rest.ensembl.org"
UNIPROT = "https://rest.uniprot.org"
UA = {"User-Agent": "infinitemorphospace/sequence-linker (research; stdlib urllib)"}

# Approx. average amino-acid residue mass (Da) -- only used for RELATIVE ranking.
RESIDUE_DA = 110.0

TAXON = {"danio_rerio": 7955, "xenopus_tropicalis": 8364, "drosophila_melanogaster": 7227,
         "caenorhabditis_elegans": 6239, "mus_musculus": 10090, "bos_taurus": 9913}


def _get(url: str, accept: str) -> str:
    req = urllib.request.Request(url, headers={**UA, "Accept": accept})
    with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310 (fixed public APIs)
        return r.read().decode("utf-8")


def gc_fraction(seq: str) -> float:
    s = seq.upper()
    return (s.count("G") + s.count("C")) / len(s) if s else 0.0


def diffusion_prior(protein_len: int, ref_len: int) -> float:
    """Relative diffusion prior D ~ MW^(-1/3) (Stokes-Einstein), ref = 1.0.

    A ranking prior for emulation (bigger ligand -> slower), NOT a measured D.
    """
    return round((ref_len / protein_len) ** (1.0 / 3.0), 4) if protein_len else 0.0


def parse_fasta(text: str) -> tuple[str, str]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    header = lines[0].lstrip(">") if lines and lines[0].startswith(">") else ""
    return header, "".join(lines[1:] if header else lines).upper()


def uniprot_fallback(entry: dict, why_failed: str) -> dict:
    """Ensembl has no record under this symbol (e.g. zebrafish chd): fall back to
    the reviewed UniProt protein, labelled as such -- never silently substituted."""
    sp, sym = entry["species"], entry["gene_symbol"]
    out = {k: entry[k] for k in ("species", "gene_symbol", "pathway", "morphogen_role", "usable_in", "why", "expression_atlas")}
    q = urllib.parse.quote(f"gene_exact:{sym} AND organism_id:{TAXON[sp]} AND reviewed:true")
    hits = json.loads(_get(f"{UNIPROT}/uniprotkb/search?query={q}&fields=accession,length&format=json&size=1", "application/json"))["results"]
    if not hits:
        raise RuntimeError(f"Ensembl: {why_failed}; UniProt reviewed: no hit")
    acc = hits[0]["primaryAccession"]
    try:
        out["protein_state"] = fetch_uniprot_state(acc)
    except Exception as exc:  # state is additive; never lose the sequence record
        out["protein_state_error"] = str(exc)[:120]
    hdr, pro = parse_fasta(_get(f"{UNIPROT}/uniprotkb/{acc}?format=fasta", "text/x-fasta"))
    CACHE.mkdir(exist_ok=True)
    p = CACHE / f"{sp}_{sym}_protein.fasta"
    p.write_text(f">{hdr}\n{pro}\n", encoding="utf-8")
    out.update(sequence_source="UniProt (reviewed) -- Ensembl symbol lookup failed, see ensembl_error",
               ensembl_error=why_failed[:160], uniprot=acc, protein_fasta=f"cache/{p.name}",
               protein_sha256=hashlib.sha256(pro.encode()).hexdigest(),
               cds_length_bp=None, cds_gc=None, protein_length_aa=len(pro),
               protein_mass_kda=round(len(pro) * RESIDUE_DA / 1000, 1),
               links={"uniprot": f"https://www.uniprot.org/uniprotkb/{acc}",
                      "ncbi_gene_search": f"https://www.ncbi.nlm.nih.gov/gene/?term={sym}[Gene]+AND+{sp.replace('_', '+')}[Organism]"})
    return out


def _span(feature: dict) -> dict:
    loc = feature.get("location", {})
    start, end = loc.get("start", {}).get("value"), loc.get("end", {}).get("value")
    return {"start": start, "end": end,
            "length_aa": (end - start + 1) if isinstance(start, int) and isinstance(end, int) else None,
            "description": feature.get("description", "")}


def parse_uniprot_state(record: dict) -> dict:
    """Protein STATE (not just identity) from a UniProtKB JSON record.

    Captures what can change a morphogenetic model: signal peptide, mature
    chain(s), lipidation / glycosylation / modified residues, cleavage sites,
    domains, and subcellular location. Pure function -- offline-testable.
    Kinetic facts (half-life, affinity, absolute D) are NOT in UniProt and are
    deliberately not invented here; see catalog.json claim_boundary.
    """
    feats = record.get("features", [])
    by_type = lambda t: [_span(f) for f in feats if f.get("type") == t]  # noqa: E731
    locations, ptm, function = [], [], []
    for c in record.get("comments", []):
        if c.get("commentType") == "SUBCELLULAR LOCATION":
            locations += [s["location"]["value"] for s in c.get("subcellularLocations", []) if s.get("location")]
        elif c.get("commentType") == "PTM":
            ptm += [t["value"] for t in c.get("texts", []) if t.get("value")]
        elif c.get("commentType") == "FUNCTION":
            function += [t["value"] for t in c.get("texts", []) if t.get("value")]
    name = (record.get("proteinDescription", {}).get("recommendedName", {})
            .get("fullName", {}).get("value"))
    return {
        "source": "UniProtKB feature table + comments (live REST, not hand-copied)",
        "accession": record.get("primaryAccession"),
        "protein_name": name,
        "precursor_length_aa": record.get("sequence", {}).get("length"),
        "sequence_mass_kda": round(record["sequence"]["molWeight"] / 1000, 1) if record.get("sequence", {}).get("molWeight") else None,
        "signal_peptide": (by_type("Signal") or [None])[0],
        "propeptides": by_type("Propeptide"),
        "chains": by_type("Chain"),
        "domains": by_type("Domain"),
        "lipidation": by_type("Lipidation"),
        "glycosylation": by_type("Glycosylation"),
        "modified_residues": by_type("Modified residue"),
        "cleavage_sites": [s for s in by_type("Site") if "leavage" in s["description"]],
        "disulfide_bond_count": len(by_type("Disulfide bond")),
        "subcellular_location": locations,
        "ptm_comments": ptm,
        "function_comments": function,
        "keywords": [k["name"] for k in record.get("keywords", [])],
    }


def fetch_uniprot_state(accession: str) -> dict:
    state = parse_uniprot_state(json.loads(_get(f"{UNIPROT}/uniprotkb/{accession}?format=json", "application/json")))
    state["url"] = f"https://www.uniprot.org/uniprotkb/{accession}"
    return state


def resolve_entry(entry: dict) -> dict:
    sp, sym = entry["ensembl_species"], entry["gene_symbol"]
    out = {k: entry[k] for k in ("species", "gene_symbol", "pathway", "morphogen_role", "usable_in", "why", "expression_atlas")}
    try:
        if entry.get("ensembl_id"):  # e.g. WormBase WBGene IDs for C. elegans
            gene = json.loads(_get(f"{ENSEMBL}/lookup/id/{entry['ensembl_id']}", "application/json"))
        else:
            gene = json.loads(_get(f"{ENSEMBL}/lookup/symbol/{sp}/{urllib.parse.quote(sym)}?expand=1", "application/json"))
    except Exception as exc:
        return uniprot_fallback(entry, str(exc))
    out["sequence_source"] = "Ensembl REST (canonical transcript CDS + protein)"
    out["ensembl_gene_id"] = gene["id"]
    out["assembly"] = gene.get("assembly_name")
    tx_raw = gene.get("canonical_transcript", "").rstrip(".")  # Ensembl appends a bare '.' when unversioned
    tx_id = tx_raw.split(".")[0] if tx_raw.startswith("ENS") else tx_raw  # worm tx IDs need their suffix (E01A2.3.1)
    out["canonical_transcript"] = tx_id

    cds_h, cds = parse_fasta(_get(f"{ENSEMBL}/sequence/id/{tx_id}?type=cds", "text/x-fasta"))
    pro_h, pro = parse_fasta(_get(f"{ENSEMBL}/sequence/id/{tx_id}?type=protein", "text/x-fasta"))
    CACHE.mkdir(exist_ok=True)
    for kind, hdr, seq in (("cds", cds_h, cds), ("protein", pro_h, pro)):
        p = CACHE / f"{sp}_{sym}_{kind}.fasta"
        p.write_text(f">{hdr}\n{seq}\n", encoding="utf-8")
        out[f"{kind}_fasta"] = f"cache/{p.name}"
        out[f"{kind}_sha256"] = hashlib.sha256(seq.encode()).hexdigest()
    out.update(cds_length_bp=len(cds), cds_gc=round(gc_fraction(cds), 4),
               protein_length_aa=len(pro), protein_mass_kda=round(len(pro) * RESIDUE_DA / 1000, 1))
    try:  # UniProt cross-reference is best-effort; Ensembl record stands alone.
        q = urllib.parse.quote(f"gene_exact:{sym} AND organism_id:{TAXON[entry['species']]}")
        up = json.loads(_get(f"{UNIPROT}/uniprotkb/search?query={q}&fields=accession,id,length&format=json&size=1", "application/json"))
        if up.get("results"):
            out["uniprot"] = up["results"][0]["primaryAccession"]
            out["protein_state"] = fetch_uniprot_state(out["uniprot"])
    except Exception as exc:  # never let a cross-ref failure lose the sequence
        out["uniprot_error"] = str(exc)[:120]
    out["links"] = {
        "ensembl_gene": f"https://www.ensembl.org/{sp.capitalize()}/Gene/Summary?g={gene['id']}",
        "ensembl_rest_cds": f"{ENSEMBL}/sequence/id/{tx_id}?type=cds",
        "uniprot_search": f"https://www.uniprot.org/uniprotkb?query=gene:{sym}",
        "ncbi_gene_search": f"https://www.ncbi.nlm.nih.gov/gene/?term={urllib.parse.quote(sym)}[{urllib.parse.quote('Gene Name')}]+AND+{sp.replace('_', '+')}[Organism]",
    }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", help="comma-separated gene symbols to resolve")
    ap.add_argument("--offline", action="store_true", help="report cache only, no network")
    ap.add_argument("--self-test", action="store_true", help="offline logic tests, no network")
    args = ap.parse_args(argv)

    if args.self_test:
        assert gc_fraction("ATGC") == 0.5 and gc_fraction("") == 0.0
        assert parse_fasta(">x\nAC\ngt\n") == ("x", "ACGT")
        assert diffusion_prior(100, 100) == 1.0 and diffusion_prior(800, 100) == 0.5
        fixture = {"primaryAccession": "TEST1", "sequence": {"length": 100, "molWeight": 11000},
                   "proteinDescription": {"recommendedName": {"fullName": {"value": "Test morphogen"}}},
                   "features": [{"type": "Signal", "location": {"start": {"value": 1}, "end": {"value": 20}}},
                                {"type": "Chain", "description": "Mature", "location": {"start": {"value": 21}, "end": {"value": 100}}},
                                {"type": "Lipidation", "description": "N-palmitoyl cysteine", "location": {"start": {"value": 21}, "end": {"value": 21}}}],
                   "comments": [{"commentType": "SUBCELLULAR LOCATION", "subcellularLocations": [{"location": {"value": "Secreted"}}]}],
                   "keywords": [{"name": "Secreted"}]}
        st = parse_uniprot_state(fixture)
        assert st["chains"][0]["length_aa"] == 80 and st["signal_peptide"]["length_aa"] == 20
        assert st["lipidation"][0]["description"] == "N-palmitoyl cysteine" and st["subcellular_location"] == ["Secreted"]
        cat = json.loads(CATALOG.read_text())
        assert cat["entries"] and all({"species", "gene_symbol", "morphogen_role", "usable_in"} <= set(e) for e in cat["entries"])
        print(f"[PASS] self-test: GC/FASTA/diffusion-prior logic + catalog schema ({len(cat['entries'])} entries)")
        return 0

    entries = json.loads(CATALOG.read_text())["entries"]
    if args.only:
        want = {s.strip() for s in args.only.split(",")}
        entries = [e for e in entries if e["gene_symbol"] in want]

    if args.offline:
        for e in entries:
            cached = list(CACHE.glob(f"{e['species']}_{e['gene_symbol']}_*.fasta")) if CACHE.exists() else []
            print(f"{e['species']:26s} {e['gene_symbol']:8s} cached={len(cached)}")
        return 0

    resolved, failed = [], []
    for e in entries:
        try:
            r = resolve_entry(e)
            resolved.append(r)
            print(f"[OK]   {r['species']:26s} {r['gene_symbol']:8s} {r.get('ensembl_gene_id') or r.get('uniprot')}  CDS {r['cds_length_bp']} bp GC={r['cds_gc']}  protein {r['protein_length_aa']} aa")
        except Exception as exc:
            failed.append({"species": e["species"], "gene_symbol": e["gene_symbol"], "error": str(exc)[:200]})
            print(f"[FAIL] {e['species']:26s} {e['gene_symbol']:8s} {exc}", file=sys.stderr)

    if resolved:  # diffusion priors are comparable only within one resolved set
        ref = min(r["protein_length_aa"] for r in resolved)
        for r in resolved:
            r["diffusion_prior_vs_smallest"] = diffusion_prior(r["protein_length_aa"], ref)
    LINKS_OUT.write_text(json.dumps({"schema_version": "morphogen-sequence-links-1.0",
                                     "resolved": resolved, "failed": failed}, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {LINKS_OUT} ({len(resolved)} resolved, {len(failed)} failed)")
    return 0 if resolved else 1


if __name__ == "__main__":
    raise SystemExit(main())
