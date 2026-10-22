#!/usr/bin/env python3
"""
Dated tribe-level tree for the De Meulemeester et al. (2014) bee forewing dataset, assembled
by hand from published phylogenies BEFORE any wing-shape analysis.  Node ages in Ma.

Sources (topology / ages):
  Apoidea backbone .... Branstetter MG et al. 2017 Curr Biol 27:1019; Sann M et al. 2018 BMC Evol Biol 18:71
                        (Ampulicidae sister to rest; Sphecidae; Crabronidae paraphyletic with Philanthinae sister to bees)
  Bee families ........ Cardinal S & Danforth BN 2013 Proc R Soc B 280:20122686; Branstetter 2017
                        (Melittidae sister to all other bees; long-tongued Apidae+Megachilidae; short-tongued
                        Andrenidae + (Halictidae + (Colletidae + Stenotritidae)))
  Apidae .............. Bossert S et al. 2019 Mol Phylogenet Evol 130:121 (Xylocopinae sister to Nomadinae s.l. + Apinae;
                        cleptoparasitic tribes united in Nomadinae s.l.); corbiculates Romiguier J et al. 2016 / Bossert 2017
  Halictidae .......... Danforth BN et al. 2004 Syst Biol 53:309; Gibbs J et al. 2012 Mol Phylogenet Evol 65:926
  Colletidae .......... Almeida EAB et al. 2012 Mol Phylogenet Evol 65:526
  Andrenidae .......... Bossert S et al. 2022 Mol Phylogenet Evol 170:107151
  Fossil tribes ....... Engel MS 2001 Bull Am Mus Nat Hist 259 (Electrapini, Melikertini: Baltic amber corbiculates, ~45 Ma,
                        nearest extant lineage Apini); not placed as tips - used as time-slice targets.
Ages are rounded to the nearest 5 Myr where the literature disagrees; tribes with no resolved
position are placed as polytomies at their subfamily crown.  This is an a-priori scaffold, not a
new phylogenetic estimate.
"""
import json
from pathlib import Path

# (age_Ma, children) for internal nodes; strings are tips
T = (135, [
    "Ampulicini",
    (125, [
        (60, ["Ammophilinae", "Sphecinae"]),                       # Sphecidae
        (118, [
            "Larrinae",
            (115, [
                "Philanthinae",
                (110, [                                           # Anthophila crown ~110
                    (85, ["Meganomiinae", "Melittini"]),          # Melittidae
                    (108, [
                        (100, [                                   # long-tongued
                            "Fideliini",                          # Megachilidae
                            (90, [                                # Apidae crown
                                (60, ["Xylocopini", "Ceratinini"]),
                                (85, [
                                    (75, [                        # Nomadinae s.l.
                                        (50, ["Nomadini", (45, ["Epeolini", "Brachynomadini"])]),
                                        (65, [(55, ["Melectini", "Ericrocidini"]), (50, ["Osirini", (35, ["Protepeolini", "Isepeolini"])])]),
                                    ]),
                                    (80, [                        # Apinae s.s.
                                        "Anthophorini",
                                        (75, [
                                            (70, [(65, ["Eucerini", (60, ["Emphorini", (55, ["Exomalopsini", (45, ["Ancylaini", "Tapinotaspidini"])])])]), "Tetrapediini"]),
                                            (72, ["Centridini", (68, ["Euglossini", (60, ["Apini", "Bombini"])])]),   # corbiculates
                                        ]),
                                    ]),
                                ]),
                            ]),
                        ]),
                        (105, [                                   # short-tongued
                            (80, [(65, ["Andreninae", "Oxaeinae"]), (45, ["Melitturgini", "Protandrenini"])]),   # Andrenidae
                            (100, [
                                (75, [                            # Halictidae
                                    "Rophitinae",
                                    (65, ["Nomiinae", (58, ["Nomioidinae", (50, ["Augochlorini", (42, ["Thrinchostomatini", (35, ["Caenohalictini", (25, ["Halictini", "Sphecodini"])])])])])]),
                                ]),
                                (85, [
                                    "Stenotritidae",
                                    (70, [(40, ["Caupolicanini", "Diphaglossini", "Dissoglotini"]), (50, ["Colletini", "Paracolletini"])]),   # Colletidae
                                ]),
                            ]),
                        ]),
                    ]),
                ]),
            ]),
        ]),
    ]),
])

FOSSIL_TARGETS = {
    "Electrapini": {"age_ma": 45, "lineage_tip": "Apini", "alt_lineage_tip": "Bombini", "source": "Engel 2001; Baltic amber, Lutetian ~44-48 Ma"},
    "Melikertini": {"age_ma": 45, "lineage_tip": "Apini", "alt_lineage_tip": "Bombini", "source": "Engel 2001; Baltic amber, Lutetian ~44-48 Ma"},
}


def newick(node, parent_age=None):
    if isinstance(node, str):
        return f"{node}:{parent_age}"
    age, kids = node
    inner = ",".join(newick(k, age) for k in kids)
    return f"({inner})" + ("" if parent_age is None else f":{parent_age - age}")


def tips(node):
    return [node] if isinstance(node, str) else [t for k in node[1] for t in tips(k)]


def check(node):
    if isinstance(node, str):
        return
    age, kids = node
    for k in kids:
        if not isinstance(k, str):
            assert k[0] < age, (k[0], age)
        check(k)


if __name__ == "__main__":
    check(T)
    out = {"newick": newick(T) + ";", "root_age_ma": T[0], "n_tips": len(tips(T)), "tips": tips(T), "fossil_targets": FOSSIL_TARGETS, "note": __doc__.strip()}
    Path(__file__).with_name("bee_tribe_tree_dated.json").write_text(json.dumps(out, indent=1))
    print(f"{out['n_tips']} tips, root {out['root_age_ma']} Ma")
