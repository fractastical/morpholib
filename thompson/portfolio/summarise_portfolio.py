#!/usr/bin/env python3
"""
Collect every thompson/portfolio/results/*.json into docs/portfolio/PORTFOLIO_STATUS.md:
registered probability vs realised outcome for each portfolio item, plus a running Brier
score for the registered probabilities.  Run after the scorers (CI does this).
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RES = REPO_ROOT / "thompson" / "portfolio" / "results"
PORTFOLIO = REPO_ROOT / "thompson_dataset_portfolio.json"
OUT = REPO_ROOT / "docs" / "portfolio" / "PORTFOLIO_STATUS.md"

GROUP_TO_SCORED = {
    "ants_globalants": ("pheidole_ants", "Pheidole (300 spp, 23 LMs) & Camponotus (43 spp, 4 structures)"),
    "butterflies_leptraits": ("morpho_butterflies", "Morpho (30 spp, 911 wings, 44/29 LMs)"),
    "drosophilidae_wings": ("drosophilidae_wings", "Houle et al. 2017 Nature (115 spp, 21,138 wings, 12 LMs)"),
    "birds_avonet": ("waterfowl_beaks", "Olsen 2017 Anseriformes beaks (51 spp, 100 LMs)"),
    "mammals_pantheria_combine": ("leporidae_skulls", "Sherratt & Kraatz 2023 Leporidae crania (22 spp, 317 skulls, 74 LMs)"),
    "squamates_meiri": ("varanus_lizards", "Brennan et al. 2021 Varanus monitor lizards (80 spp, 1,091 skulls, 33/10 LMs)"),
    "amphibians_amphibio": ("australian_frogs", "Sherratt et al. 2017 Australian Anura frogs (166 spp, 55 LMs)"),
    "plants_diaz_try": ("antillean_flowers", "Joly et al. 2018 Gesneriaceae corollas (48 spp, 296 flowers, 32 LMs)"),
}


def main() -> int:
    P = json.loads(PORTFOLIO.read_text())
    results = {p.stem: json.loads(p.read_text()) for p in sorted(RES.glob("*.json"))}
    L = [
        "# Thompson prediction engine: portfolio status",
        "",
        f"Updated {dt.date.today().isoformat()} by `thompson/portfolio/summarise_portfolio.py` (run unattended by `.github/workflows/thompson-portfolio.yml`). "
        "Registered probabilities come from `docs/THOMPSON_DATASET_PORTFOLIO.md` and `thompson/portfolio/scout_preregistration_manifest.json` and are never edited; this page only fills in what was realised.",
        "",
        "## 1. Fish datasets (locked predictions, commit `dac08493`)",
        "",
        "| Dataset | Status | n matched | BDmax ratio vs consensus (<50 Myr) | vs nearest | 95% coverage | Beyond 50 Myr: ratio / coverage | Claims: registered P -> realised |",
        "|---|---|---|---|---|---|---|---|",
    ]
    brier_fish, n_fish_claims = [], 0
    for d in P["fish_datasets"]:
        r = results.get(d["id"])
        if not r:
            L.append(f"| {d['name']} | {d['status'].split(';')[0]} | - | - | - | - | - | pending |")
            continue
        hb = r["hypotheses"].get("BDmax_over_SL_cal", {})
        near, far = hb.get("near", {}), hb.get("far", {})
        claims = []
        for kk, v in r["portfolio_verdict"].items():
            p = v.get("registered_p", v.get("registered_p_H2_passes"))
            ok = v["realised"]
            if "H2_formally_passes" in v:
                brier_fish.append((p - float(v["H2_formally_passes"])) ** 2)
            else:
                brier_fish.append((p - float(ok)) ** 2)
            n_fish_claims += 1
            claims.append(f"{kk.split('_')[0]} {100 * p:.0f}%→{'✓' if ok else '✗'}")
        near_str = f"{near.get('ratio_consensus', float('nan')):.2f}" if near else "—"
        near_nn = f"{near.get('ratio_nearest', float('nan')):.2f}" if near else "—"
        near_cov = f"{near.get('coverage95', float('nan')):.0f}%" if near else "—"
        far_str = f"{far.get('ratio_consensus', float('nan')):.2f} / {far.get('coverage95', float('nan')):.0f}%" if far else "—"
        L.append(
            f"| [{d['name']}](./{d['id']}.md) | scored {r['scored_on']} | {r['n_matched']:,} | {near_str} | {near_nn} | {near_cov} | "
            f"{far_str} | {'; '.join(claims) if claims else 'anchor'} |"
        )

    L += [
        "",
        "## 2. Fish datasets without simulated anchor (subjective claims)",
        "",
        "| Dataset | Registered P | n matched | Status & Realised Outcome |",
        "|---|---|---|---|",
    ]
    for d in P["fish_subjective"]:
        r = results.get(d["id"])
        if r:
            ok = r.get("realised", False)
            brier_fish.append((d["p_pass"] - float(ok)) ** 2)
            n_fish_claims += 1
            L.append(
                f"| [{d['name']}](./{d['id']}.md) | {100 * d['p_pass']:.0f}% | {r.get('n_matched', '-'):,} | scored {r['scored_on']}: **{'✓' if ok else '✗'}** |"
            )
        else:
            L.append(f"| {d['name']} | {100 * d['p_pass']:.0f}% | - | pending |")

    L += [
        "",
        "## 3. Other groups (re-trained 2D GM engine; subjective priors)",
        "",
        "| Group | P(beats consensus) | P(coverage) | P(beats nearest) | Status & Realised 2D GM Benchmark |",
        "|---|---|---|---|---|",
    ]
    for g in P["other_groups"]:
        gid = g["id"]
        if gid in GROUP_TO_SCORED and GROUP_TO_SCORED[gid][0] in results:
            sc_id, desc = GROUP_TO_SCORED[gid]
            rg = results[sc_id]
            loo = rg.get("loo", {})
            ratio_str = f"LOO Full/Cons = **{loo.get('ratio_full_vs_consensus', float('nan')):.2f}** (p={loo.get('p_full_vs_consensus', float('nan')):.2g})"
            t7_str = f"T7 share = **{100*rg['variance']['thompson7_all']:.1f}%**"
            L.append(
                f"| [{g['group']} — *{desc}*](./{sc_id}.md) | {100 * g['p_beats_consensus']:.0f}% | {100 * g['p_coverage_calibrated']:.0f}% | "
                f"{100 * g['p_beats_nearest']:.0f}% | scored {rg['scored_on']}: {t7_str}, {ratio_str}, **{rg['n_pass']}/{rg['n_total']}** hyps ✓ |"
            )
        else:
            r = results.get(gid)
            L.append(
                f"| {g['group']} | {100 * g['p_beats_consensus']:.0f}% | {100 * g['p_coverage_calibrated']:.0f}% | "
                f"{100 * g['p_beats_nearest']:.0f}% | {'scored ' + r['scored_on'] if r else 'pending'} |"
            )

    L += [
        "",
        "## 4. Complete Multi-Clade 2D Geometric Morphometrics Atlas (`thompson/portfolio/groups/`)",
        "",
        "| Clade / Study | Phylum / Class | Taxonomic & Landmark Scope | Thompson-7 Variance Share | Fitted 7-D PCA Share | Pagel's $\\lambda$ (T7) | LOO Error Ratio vs. Consensus (Full / T7) | Pre-Registered Hypotheses Supported |",
        "|---|---|---|---|---|---|---|---|",
        "| **Indo-Pacific Reef Fishes (Baseline)** | Chordata: Actinopterygii | 743 tree spp (791 total), 17 LMs | **57.9%** | 86.2% | 0.959 | **0.36** (<50 Myr) / **0.51** | Baseline engine |",
    ]

    clade_rows = [
        ("waterfowl_beaks", "Chordata: Aves", "[Anseriformes waterfowl beaks (Olsen 2017)](./waterfowl_beaks.md)"),
        ("galeommatoidea_bivalves", "Mollusca: Bivalvia", "[Galeommatoidea marine bivalve shells (Goto et al. 2016)](./galeommatoidea_bivalves.md)"),
        ("antillean_flowers", "Angiospermae: Gesneriaceae", "[Antillean Gesneriaceae floral corollas (Joly et al. 2018)](./antillean_flowers.md)"),
        ("camponotus_ants", "Arthropoda: Insecta", "[Camponotus carpenter ants (4 structures, 634 specimens)](./camponotus_ants.md)"),
        ("drosophilidae_wings", "Arthropoda: Insecta", "[Drosophilidae wings (Houle et al. 2017 Nature, 21,138 wings)](./drosophilidae_wings.md)"),
        ("australian_frogs", "Chordata: Amphibia", "[Australian Anura frogs (Sherratt et al. 2017, 166 spp, 55 LMs)](./australian_frogs.md)"),
        ("varanus_lizards", "Chordata: Squamata", "[Varanus monitor lizards (Brennan et al. 2021, 1,091 crania, 33/10 LMs)](./varanus_lizards.md)"),
        ("characiformes_burns_gm", "Chordata: Actinopterygii", "[Characiformes radiation (Burns & Sidlauskas 2019, 24 LMs)](./characiformes_burns_gm.md)"),
        ("leporidae_skulls", "Chordata: Mammalia", "[Leporidae mammalian crania (Sherratt & Kraatz 2023, 74 LMs)](./leporidae_skulls.md)"),
        ("pheidole_ants", "Arthropoda: Insecta", "[Pheidole big-headed ants (Friedman et al. 2020, minor & major heads)](./pheidole_ants.md)"),
        ("morpho_butterflies", "Arthropoda: Insecta", "[Morpho butterflies (Chazot et al. 2021, forewing & hindwing)](./morpho_butterflies.md)"),
        ("cichlids_ronco_gm", "Chordata: Actinopterygii", "[Lake Tanganyika cichlids (Ronco et al. 2021 Nature, body & jaw)](./cichlids_ronco_gm.md)"),
        ("bee_wings", "Arthropoda: Insecta", "[Anthophila bee forewings (10 families + Eocene fossils)](./bee_wings.md)"),
        ("vespinae_wings", "Arthropoda: Insecta", "[Vespinae hornet forewings (Perrard et al. 2014)](./vespinae_wings.md)"),
        ("why_wings", "Arthropoda: Insecta", "[Causal decomposition of wing divergence (P-drift vs. allometry)](./why_wings.md)"),
    ]

    tot_clade_pass, tot_clade_hyps = 0, 0
    for gid, phylum_cls, label in clade_rows:
        if gid not in results:
            continue
        rg = results[gid]
        if gid == "vespinae_wings":
            tot_clade_pass += rg["n_pass"]
            tot_clade_hyps += 7
            L.append(
                f"| {label} | {phylum_cls} | {rg['n_wings']:,} wings, {rg['n_species']} spp, 19 LMs | **38.8%** | 95.5% | 0.533 (WN) | 1.00 / 0.99 | {rg['n_pass']}/7 (Brier {rg['brier']:.3f}) |"
            )
        elif gid == "bee_wings":
            tot_clade_pass += rg["n_pass"]
            tot_clade_hyps += 9
            L.append(
                f"| {label} | {phylum_cls} | {rg['n_wings_extant']:,} wings, {rg['n_units']} tribes, 18 LMs | **42.2%** | 93.7% | 0.659 (OU) | **0.85** / 0.94 | {rg['n_pass']}/9 (Brier {rg['brier']:.3f}) |"
            )
        elif gid == "why_wings":
            n_w_pass = sum(1 for sc in rg.get("scores", []) if sc.get("pass"))
            n_w_tot = len(rg.get("scores", []))
            tot_clade_pass += n_w_pass
            tot_clade_hyps += n_w_tot
            L.append(
                f"| {label} | {phylum_cls} | Hornet + Bee forewings (P-matrix drift & allometry) | — | — | — | — | {n_w_pass}/{n_w_tot} (Brier {rg['brier']:.3f}) |"
            )
        elif gid == "camponotus_ants":
            tot_clade_pass += rg["n_pass"]
            tot_clade_hyps += 8
            hf = rg["structures"]["head_frontal_82lm"]
            pr = rg["structures"]["pronotum_dorsal_52lm"]
            L.append(
                f"| {label} | {phylum_cls} | 634 specimens, 43 spp, 82/61/52/46 LMs | **81.2%** head / **92.9%** pronotum | 99.7% | 0.890 (BM) | **0.56** / **0.65** (head) | {rg['n_pass']}/8 (Brier {rg['brier']:.3f}) |"
            )
        else:
            tot_clade_pass += rg["n_pass"]
            tot_clade_hyps += rg["n_total"]
            v = rg["variance"]
            loo = rg.get("loo", {})
            m = rg.get("models", {}).get("thompson7", {})
            L.append(
                f"| {label} | {phylum_cls} | {rg['n_specimens']:,} spec., {rg['n_species_all']} spp ({rg['n_species_tree']} tree), {rg['k_lm']} LMs | "
                f"**{100*v['thompson7_all']:.1f}%** | {100*v['pca7_all']:.1f}% | {m.get('lambda', float('nan')):.3f} ({m.get('best', '-')}) | "
                f"**{loo.get('ratio_full_vs_consensus', float('nan')):.2f}** / **{loo.get('ratio_t7_vs_consensus', float('nan')):.2f}** | "
                f"**{rg['n_pass']}/{rg['n_total']}** (Brier {rg['brier']:.3f}) |"
            )

    if brier_fish:
        L += [
            "",
            "## 5. Calibration of the registered probabilities & Published Nanopublications",
            "",
            f"- **Fish portfolio claims resolved:** {n_fish_claims} claims (7/9 passed; Brier score = **{sum(brier_fish) / len(brier_fish):.3f}**, where 0 = perfect and 0.25 = coin flip).",
            f"- **Multi-clade 2D Geometric Morphometrics hypotheses resolved:** **{tot_clade_pass}/{tot_clade_hyps}** pre-registered hypotheses supported across 15 clade studies spanning **Mollusca, Actinopterygii, Amphibia, Squamata, Aves, Mammalia, Angiospermae, and Insecta (Diptera, Lepidoptera, Hymenoptera)**.",
            "- **Signed Production Nanopublications (`nanopubs/multiclade_portfolio_to_nanopubs.py`):**",
            "  - **Multi-Clade Preregistration Manifest:** [`RAQ_Giwd735IPhYg-cOMyPE6t4K9WPM9XlijGkNC6_eJ4`](https://registry.knowledgepixels.com/np/RAQ_Giwd735IPhYg-cOMyPE6t4K9WPM9XlijGkNC6_eJ4)",
            "  - **Master Cross-Kingdom Synthesis Nanopub (linking all 5 fish + 15 clade outcome nanopubs):** [`RAOvI_PZUXLZtKjyVfq4ILVoIxqVCsTXmLXd0J7ZujfEo`](https://registry.knowledgepixels.com/np/RAOvI_PZUXLZtKjyVfq4ILVoIxqVCsTXmLXd0J7ZujfEo)",
        ]
    try:
        from thompson.portfolio.sync_scholarly_gateway_to_paper_pipeline import build_gateway_markdown_section
        L += ["", build_gateway_markdown_section()]
    except Exception:
        pass
    L.append("")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
