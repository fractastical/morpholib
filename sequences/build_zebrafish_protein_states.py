#!/usr/bin/env python3
"""
Zebrafish pilot: capture PROTEIN STATES and wire them into morphogenetic states.

Reads the Danio rerio entries resolved in `sequence_links.json` (identity +
sequence, Ensembl/UniProt), attaches the UniProt protein state fetched by
`fetch_sequences.fetch_uniprot_state` (signal peptide, mature chain, lipidation,
glycosylation, cleavage, location), derives the few model-facing quantities
those states legitimately give, and writes sidecars next to two datasets:

  datasets/danio_rerio_turing_wavelength/protein_states.json   (bmp2b, chd, shha, fgf8a)
  datasets/danio_rerio_clock_wavefront/protein_states.json     (her1, fgf8a)
  sequences/zebrafish_protein_states.json                      (master pilot record)

HONEST BOUNDARY (same as catalog.json): a protein state gives the diffusible
species (mature chain, not precursor), its RELATIVE size/diffusion ranking
D ~ MW^(-1/3), and qualitative tethering (dual lipidation of ShhNp). It does
NOT give absolute D, production/degradation rates, clock period, or wavefront
speed -- those remain literature parameters, labelled literature_required.

Usage:
  python3 sequences/build_zebrafish_protein_states.py            # fetch states live, write sidecars
  python3 sequences/build_zebrafish_protein_states.py --offline  # only if links already carry protein_state
  python3 sequences/build_zebrafish_protein_states.py --self-test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import fetch_sequences as fs  # noqa: E402

LINKS = HERE / "sequence_links.json"
MASTER_OUT = HERE / "zebrafish_protein_states.json"
SPECIES = "danio_rerio"

CLAIM_BOUNDARY = {
    "protein_state_gives_you": (
        "Which molecular species actually diffuses (mature/signalling chain after signal-peptide "
        "and proprotein cleavage, not the precursor), a RELATIVE diffusion ranking "
        "D_rel ~ MW^(-1/3) between those species, and qualitative membrane tethering "
        "when lipidation is annotated (ShhNp: N-palmitate + C-cholesterol)."),
    "protein_state_does_not_give_you": (
        "Absolute diffusion coefficients, production/degradation rates, Her1 half-life or "
        "segmentation-clock period, wavefront speed, receptor affinities. Those are kinetic "
        "facts for the cited literature / ZFIN, labelled literature_required below."),
    "therefore": (
        "Every morphospace prediction here is DIRECTIONAL or a ratio under a stated "
        "held-fixed assumption (e.g. lambda ~ sqrt(D/k) at fixed k), never an absolute fit."),
}


def diffusible_species(gene: str, state: dict) -> dict:
    """Pick the species that actually moves, from the annotated state.

    Returns {basis, length_aa, note}. Basis is explicit so a proxy is never
    silently dressed up as an annotated mature chain.
    """
    chains = state.get("chains") or []
    if gene == "shha":  # annotated signalling N-product, dually lipidated
        np_chain = next((c for c in chains if "N-product" in c["description"]), None)
        if np_chain:
            return {"basis": "annotated_chain", "description": np_chain["description"],
                    "length_aa": np_chain["length_aa"],
                    "note": "Dually lipidated (N-palmitoyl Cys24, cholesteryl Gly197): size prior overstates free diffusion; membrane tethering dominates."}
    if gene == "bmp2b":  # UniProt annotates one chain; mature ligand = TGF-beta domain (proxy)
        tgfb = next((d for d in state.get("domains") or [] if "TGF-beta" in d["description"]), None)
        if tgfb:
            return {"basis": "domain_proxy_not_annotated_mature_chain", "description": tgfb["description"],
                    "length_aa": tgfb["length_aa"],
                    "note": "UniProt annotates a single 24-411 chain plus 'cleavage on pair of basic residues'; the C-terminal TGF-beta domain is used as the mature-ligand proxy and labelled as such."}
    if chains:
        longest = max(chains, key=lambda c: c["length_aa"] or 0)
        return {"basis": "annotated_chain", "description": longest["description"] or "Chain",
                "length_aa": longest["length_aa"], "note": "Secreted chain after signal-peptide cleavage."}
    return {"basis": "precursor_only", "description": "Precursor (no chain annotated)",
            "length_aa": state.get("precursor_length_aa"), "note": "No mature chain annotated in UniProt."}


def d_rel(length_aa: int, ref_aa: int) -> float:
    """D_rel = (M_ref/M)^(1/3), Stokes-Einstein size ranking only."""
    return round((ref_aa / length_aa) ** (1 / 3), 4)


def build(offline: bool) -> dict:
    links = json.loads(LINKS.read_text())
    entries = [r for r in links["resolved"] if r["species"] == SPECIES]
    if not entries:
        raise SystemExit("No danio_rerio entries in sequence_links.json -- run fetch_sequences.py first")
    proteins = {}
    for r in entries:
        state = r.get("protein_state")
        if not state:
            if offline:
                raise SystemExit(f"{r['gene_symbol']}: no protein_state in links and --offline set; rerun without --offline")
            state = fs.fetch_uniprot_state(r["uniprot"])
        proteins[r["gene_symbol"]] = {
            "gene_symbol": r["gene_symbol"], "pathway": r["pathway"],
            "morphogen_role": r["morphogen_role"], "uniprot": r.get("uniprot"),
            "ensembl_gene_id": r.get("ensembl_gene_id"), "expression_atlas": r["expression_atlas"],
            "why": r["why"], "protein_state": state,
        }

    # --- Turing side: diffusible species + relative diffusion ranking -------
    ref_gene, ref_species = "bmp2b", None
    ref_species = diffusible_species("bmp2b", proteins["bmp2b"]["protein_state"])
    ref_aa = ref_species["length_aa"]
    turing_proteins = []
    for gene in ("bmp2b", "chd", "shha", "fgf8a"):
        p = dict(proteins[gene])
        sp = diffusible_species(gene, p["protein_state"])
        sp["D_rel_vs_mature_bmp2b"] = d_rel(sp["length_aa"], ref_aa)
        p["diffusible_species"] = sp
        p["status"] = "measured_state_extant_species"
        turing_proteins.append(p)

    by_gene = {p["gene_symbol"]: p for p in turing_proteins}
    d_act = by_gene["bmp2b"]["diffusible_species"]["D_rel_vs_mature_bmp2b"]
    d_inh = by_gene["chd"]["diffusible_species"]["D_rel_vs_mature_bmp2b"]
    precursor_aa = proteins["bmp2b"]["protein_state"]["precursor_length_aa"]
    turing_effect = {
        "poson_law": "Turing Wavenumber k_c; lambda_c ~ sqrt(D_eff/k) at fixed k",
        "reference_species": {"gene": ref_gene, **ref_species},
        "activator_inhibitor_size_check": {
            "D_activator_bmp2b_rel": d_act, "D_inhibitor_chordin_rel": d_inh,
            "D_inhibitor_over_activator": round(d_inh / d_act, 4),
            "naive_turing_requirement": "inhibitor must diffuse FASTER than activator (D_inh/D_act > 1)",
            "verdict": (
                "FAILS on size prior alone: chordin (921 aa secreted chain) is larger and slower "
                "than the mature BMP ligand. The real BMP-Chordin system patterns by shuttling / "
                "facilitated transport and tolloid cleavage (annotated on chordin), not by free "
                "differential diffusion -- so this pair must not be dropped into a naive two-species "
                "Turing solver. The protein state is what exposes that."),
        },
        "worked_sensitivity_precursor_vs_mature": {
            "question": "What error if the 411 aa bmp2b precursor is used as the diffusing species instead of the mature ligand proxy?",
            "D_precursor_over_mature": d_rel(precursor_aa, ref_aa),
            "lambda_ratio_sqrt_D_at_fixed_k": round(d_rel(precursor_aa, ref_aa) ** 0.5, 4),
            "reading": "Using the precursor understates D by ~33% and lambda_c by ~18% at fixed k. Directional prior only, not a measured D.",
        },
        "shha_tethering": {
            "state": "ShhNp dually lipidated (see shha protein_state.lipidation)",
            "prediction": "Effective free diffusion << size prior (D_rel 0.89 by mass alone); expect a shorter-range, membrane-associated gradient than BMP/FGF at equal production. Qualitative only -- no tethered-D number is claimed.",
        },
    }

    # --- Clock-wavefront side: Her1 is nuclear, FGF8a sets the wavefront ----
    her1 = dict(proteins["her1"]); her1["status"] = "measured_state_extant_species"
    her1["diffusible_species"] = {"basis": "not_applicable", "length_aa": None,
                                  "note": "Nuclear bHLH transcriptional repressor (UniProt location: Nucleus). Intracellular oscillator, not a diffusible morphogen; D_rel is not defined for it."}
    fgf_clock = next(p for p in turing_proteins if p["gene_symbol"] == "fgf8a")
    clock_effect = {
        "poson_law": "Somite Length S = u * T_clock",
        "her1_state_moves": {
            "parameter": "T_clock (period), via Her/Hes autorepression delay + protein/mRNA degradation",
            "captured_from_uniprot": "Identity, 328 aa nuclear bHLH repressor, domains (bHLH 12-69, Orange 118-152). No modified-residue / degradation annotation in UniProt.",
            "literature_required": "Her1 half-life, transcriptional/translational delays, and the resulting ~30 min zebrafish clock period must come from the somitogenesis literature / ZFIN, not from sequence or UniProt state.",
        },
        "fgf8a_state_moves": {
            "parameter": "u (wavefront position/speed), via the posterior FGF gradient cells exit as it regresses",
            "captured_from_uniprot": fgf_clock["diffusible_species"],
        },
        "worked_sensitivity": {
            "formula": "dS/S = dT/T at fixed u (equivalently dS/S = du/u at fixed T)",
            "S_multiplier_for_T_change": {"-20%": 0.8, "-10%": 0.9, "+10%": 1.1, "+20%": 1.2},
            "reading": "A protein-state change that altered Her1 degradation would move somite length in direct proportion to the period change -- but the size of that period change is literature_required, so no numeric S prediction is claimed here.",
        },
    }

    def sidecar(dataset_dir: str, genes: list, effect: dict) -> dict:
        name = Path(dataset_dir).name
        ref = json.loads((ROOT / dataset_dir / f"{name}_reference.json").read_text())
        return {
            "schema_version": "protein-state-sidecar-1.0-pilot",
            "pilot": "zebrafish (Danio rerio) -- extant species, states measured from UniProt, not inferred",
            "dataset": dataset_dir, "sim_id": ref.get("sim_id"), "organism": ref.get("organism"),
            "poson_law": ref.get("aristotelian_categories", {}).get("poson_law"),
            "generated_from": ["sequences/sequence_links.json", "UniProtKB REST feature tables (live)"],
            "claim_boundary": CLAIM_BOUNDARY,
            "protein_states": [next(p for p in (turing_proteins + [her1]) if p["gene_symbol"] == g) for g in genes],
            "effect_on_morphogenetic_state": effect,
        }

    turing = sidecar("datasets/danio_rerio_turing_wavelength", ["bmp2b", "chd", "shha", "fgf8a"], turing_effect)
    clock = sidecar("datasets/danio_rerio_clock_wavefront", ["her1", "fgf8a"], clock_effect)
    (ROOT / "datasets/danio_rerio_turing_wavelength/protein_states.json").write_text(json.dumps(turing, indent=2) + "\n")
    (ROOT / "datasets/danio_rerio_clock_wavefront/protein_states.json").write_text(json.dumps(clock, indent=2) + "\n")

    master = {"schema_version": "zebrafish-protein-state-pilot-1.0", "species": SPECIES,
              "claim_boundary": CLAIM_BOUNDARY,
              "proteins": proteins,
              "sidecars": {"turing_wavelength": "datasets/danio_rerio_turing_wavelength/protein_states.json",
                           "clock_wavefront": "datasets/danio_rerio_clock_wavefront/protein_states.json"},
              "headline_findings": [turing_effect["activator_inhibitor_size_check"]["verdict"],
                                    turing_effect["worked_sensitivity_precursor_vs_mature"]["reading"],
                                    clock_effect["her1_state_moves"]["literature_required"]]}
    MASTER_OUT.write_text(json.dumps(master, indent=2) + "\n")
    return master


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--offline", action="store_true", help="require protein_state already in sequence_links.json")
    ap.add_argument("--self-test", action="store_true", help="offline logic tests, no network, no writes")
    args = ap.parse_args(argv)
    if args.self_test:
        assert d_rel(100, 100) == 1.0 and d_rel(800, 100) == 0.5
        fake = {"chains": [{"description": "Sonic hedgehog protein A N-product", "start": 24, "end": 197, "length_aa": 174}],
                "domains": [], "precursor_length_aa": 418}
        sp = diffusible_species("shha", fake)
        assert sp["basis"] == "annotated_chain" and sp["length_aa"] == 174
        fake_bmp = {"chains": [{"description": "", "start": 24, "end": 411, "length_aa": 388}],
                    "domains": [{"description": "TGF-beta family profile", "start": 290, "end": 411, "length_aa": 122}],
                    "precursor_length_aa": 411}
        sp = diffusible_species("bmp2b", fake_bmp)
        assert sp["basis"].startswith("domain_proxy") and sp["length_aa"] == 122
        print("[PASS] self-test: D_rel math + diffusible-species selection (annotated chain vs labelled domain proxy)")
        return 0
    master = build(offline=args.offline)
    for gene, p in master["proteins"].items():
        st = p["protein_state"]
        print(f"[OK] {gene:6s} {p['uniprot']}  precursor {st['precursor_length_aa']} aa  chains={len(st['chains'])} lipidation={len(st['lipidation'])} location={st['subcellular_location']}")
    print(f"\nWrote {MASTER_OUT}")
    print("Wrote datasets/danio_rerio_turing_wavelength/protein_states.json")
    print("Wrote datasets/danio_rerio_clock_wavefront/protein_states.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
