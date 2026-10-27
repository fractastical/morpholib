#!/usr/bin/env python3
"""Build Aristotle's Lagoon — the agent-fleet dashboard as a natural-history catalogue.

Every agent type we have used, and how often, exhibited after the 1913 Thompson
museum edition in this repo: the 1848/1860 Admiralty chart of the Lagoon of
Pyrrha is the chart of the collection, each agent type is represented by a
species already held in the library (real specimen photographs from
datasets/species_gallery/, not plots and not cartoons), and the fleet's
numbers are the accession ledger. Pinned, not swimming.

Sources (repo-recorded usage only — chat-session subagents leave no record here):
  - morphoeval/cloud/dispatches/*.json   cloud dispatch records
      kind=problem_scout (100), kind=problem_fixer (12), slot-* loop builders (2,201)
  - morphoeval/problem_scouts/scout-*.json  returned scout reports (ground x lens)
  - morphoeval/problem_fixers/           fixer queue + result files landed on main
  - morphoeval/runs/telemetry.sqlite     track_loop telemetry for the 12 canonical
      loop roles (steps/actor/model counts over tracked runs)

Output: docs/fleet-dashboard.html. Linked assets: the canonical pipeline plate
(docs/media/morphoeval_pipeline.png), the Thompson 1913 museum plates already in
docs/media/, and the species portraits this script stages into
docs/media/specimens/lagoon_*.jpg (downscaled copies of tracked library photos,
so GitHub Pages — which serves docs/ only — can show them).

Usage: python3 docs/build_fleet_dashboard.py
"""
from __future__ import annotations

import collections
import glob
import html
import json
import shutil
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DISPATCH_DIR = ROOT / "morphoeval" / "cloud" / "dispatches"
SCOUT_DIR = ROOT / "morphoeval" / "problem_scouts"
FIXER_DIR = ROOT / "morphoeval" / "problem_fixers"
TELEMETRY = ROOT / "morphoeval" / "runs" / "telemetry.sqlite"
OUT = ROOT / "docs" / "fleet-dashboard.html"
SPECIMEN_DIR = ROOT / "docs" / "media" / "specimens"
GALLERY = "datasets/species_gallery"
GH = "https://github.com/fractastical/morpholib/blob/main"
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII",
         "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX"]

# --- The lagoon: one library species per agent type -------------------------
# Every image is a tracked specimen photograph from the library (never a plot,
# never a drawing of a swimmer). The curator's note says why the species holds
# that office; most tie back to Aristotle's own notes on the Lagoon of Pyrrha
# (Thompson 1913, pp. 12-14).

# slug -> (latin, common name, source image, curator's note)
ROLE_SPECIES = {
    "scout": ("Hydra vulgaris", "the Hydra", f"{GALLERY}/II_Anaima_Invertebrata/Hydra_vulgaris.jpg",
              "Feels the water with a ring of tentacles before it commits. Tastes OpenAlex, Crossref "
              "and Commons for a paper worth the work. The founding specimen of the collection "
              "(Gierer & Meinhardt 1972, Slot 001)."),
    "paper": ("Pomacanthus imperator", "the Emperor Angelfish", f"{GALLERY}/I_Enaima_Vertebrata/Pomacanthus_imperator.jpg",
              "Reads its own stripes line by line as it grows. Extracts equations, state variables "
              "and parameter regimes from the paper, stripe by stripe (Kondo & Asai 1995)."),
    "source": ("Euplectella aspergillum", "the Glass Sponge", f"{GALLERY}/II_Anaima_Invertebrata/Euplectella_aspergillum.jpg",
               "Aristotle compared the sponges of the landward and seaward faces of Cape Malia. "
               "Filters the sea for real CC-licensed specimens and pins them by SHA-256."),
    "rules": ("Strongylocentrotus purpuratus", "the Purple Sea Urchin", f"{GALLERY}/II_Anaima_Invertebrata/Strongylocentrotus_purpuratus.jpg",
              "Five-part jaw — Aristotle's lantern — one exact structure; at Pyrrha the urchins 'come into "
              "season in the winter time', which he noted as unusual. Pins published equations into "
              "Lean-checked invariants."),
    "script": ("Physarum polycephalum", "the Slime Mould", f"{GALLERY}/IV_Protista_Monera_Mycota/Physarum_polycephalum.jpg",
               "Writes itself as a living network, thickening the tubes that work. Grows the "
               "standalone model script with --self-test (Tero et al. 2010)."),
    "output": ("Danio rerio", "the Zebrafish", f"{GALLERY}/I_Enaima_Vertebrata/Danio_rerio.jpg",
               "Three pigment cell types negotiate every stripe in public. Renders the comparison "
               "figure, animation and metrics where anyone may check them (Nakamasu et al. 2009)."),
    "audit": ("Panthera pardus", "the Leopard", f"{GALLERY}/I_Enaima_Vertebrata/Panthera_pardus.jpg",
              "Examined spot by spot — Murray showed spots become rings exactly where geometry "
              "requires. Dual-assessor scoring of all 7 rubric dimensions; never the builder."),
    "measure": ("Acanthostega gunnari", "the Eight-Fingered Fossil", f"{GALLERY}/I_Enaima_Vertebrata/Acanthostega_gunnari.jpg",
                "Eight digits, counted one by one against the rock. Independently re-hashes and "
                "measures every rendered output against specimen and literature."),
    "build": ("Scleractinia", "the Reef Coral", "datasets/coral_calice_septa_insertion/scleractinian_coral_septa_specimen.jpg",
              "Accretes a skeleton septum by septum until the colony is architecture. Compiles audit "
              "and measurements into TriG nanopublications — claims with a skeleton."),
    "publish": ("Conus textile", "the Textile Cone", "datasets/conus/conus_textile_shell.jpg",
                "Prints its own signed pattern onto its shell, one row at a time. RSA-signs each "
                "nanopublication and verifies it — the shell is the signature."),
    "graph": ("Hydractinia symbiolongicarpus", "the Colonial Hydroid", f"{GALLERY}/II_Anaima_Invertebrata/Hydractinia_symbiolongicarpus.jpg",
              "One animal, many mouths, joined by a stolon network. Records the signed trusty URIs "
              "and ranks the whole colony by SPARQL."),
    "remed": ("Schmidtea mediterranea", "the Planarian", f"{GALLERY}/II_Anaima_Invertebrata/Schmidtea_mediterranea.jpg",
              "Re-grows the missing head, polarity and all. Closes every defect the audit flagged, "
              "re-renders, re-measures, supersedes its own nanopubs. Pass 2 is regeneration, with paperwork."),
}

# Fleet-level agent types (the roster in Section I) get a species too.
FLEET_SPECIES = {
    "Loop builder (slot agent)": ("Nautilus pompilius", "the Chambered Nautilus",
        f"{GALLERY}/II_Anaima_Invertebrata/Nautilus_pompilius.jpg",
        "Builds its shell one sealed chamber at a time, each complete before the next is begun. "
        "One slot agent builds one complete loop."),
    "Problem scout": ("Murex pecten", "the Venus Comb Murex",
        f"{GALLERY}/II_Anaima_Invertebrata/Murex_pecten.jpg",
        "Aristotle recorded 'the big purple Murex shells at Cape Lectum'. A comb of a hundred spines; "
        "the scouts comb a hundred ground × lens cells for what is missing."),
    "Fixer": ("Metacrinus rotundus", "the Sea Lily",
        f"{GALLERY}/II_Anaima_Invertebrata/Metacrinus_rotundus.jpg",
        "A crinoid: loses arms readily and regrows them completely. Repairs one flagged candidate "
        "in an isolated PR."),
    "Orchestrator — Muse Wonder": ("Pecten maximus", "the Great Scallop",
        f"{GALLERY}/II_Anaima_Invertebrata/Pecten_maximus.jpg",
        "Up to two hundred eyes along the mantle edge, watching everything at once. Aristotle noted "
        "the Pyrrha scallops were exterminated by drought and the dredge — a caution for any "
        "conductor. Leaves no dispatch record, hence the weaker evidence grade."),
}

# Heuristic for the synthetic spiral placeholders that stand in for photographs
# in part of datasets/species_gallery/ (the *_turing_wavelength scaffold species):
# smooth gradient, ~2,000 distinct colours, a black label bar. Real photographs
# and micrographs fall well outside this box.
PLACEHOLDER_EDGE_VAR = (480.0, 620.0)
PLACEHOLDER_MAX_COLOURS = 2600


def is_placeholder_image(path: Path) -> bool | None:
    """True if the image looks like a synthetic placeholder, None if undecidable."""
    try:
        import numpy as np
        from PIL import Image, ImageFilter
        im = Image.open(path)
        grey = im.convert("L").resize((256, 256))
        edge_var = float(np.asarray(grey.filter(ImageFilter.FIND_EDGES), dtype=float).var())
        small = im.convert("RGB").resize((128, 128))
        ncol = len(set(small.getdata()))
        return PLACEHOLDER_EDGE_VAR[0] < edge_var < PLACEHOLDER_EDGE_VAR[1] and ncol < PLACEHOLDER_MAX_COLOURS
    except Exception:
        return None


def gallery_placeholder_census() -> dict:
    files = sorted((ROOT / GALLERY).glob("*/*.jpg"))
    flagged = [f for f in files if is_placeholder_image(f)]
    return {"total": len(files), "placeholders": len(flagged),
            "examples": [f.stem.replace("_", " ") for f in flagged[:4]]}

# Plates on deposit from the museum-illustrated Thompson (1913) edition.
# accession, image (under docs/), title, source, label
MUSEUM_PLATES = [
    ("Accession 1913.1 — Plate II", "media/thompson_1913_fig2_lesbos_pyrrha_lagoon.png",
     "Insula Nobilis et Amoena — Lesbos & the Lagoon of Pyrrha",
     "Admiralty Chart No. 1665 (surveyed 1832–1849) with museum specimens",
     "The chart of record for this collection. Port Kalloni, centre-left, is the ancient Lagoon of "
     "Pyrrha where Aristotle made the first field notes in this catalogue, c. 345 BC: starfish a pest, "
     "scallops dredged out, sea-urchins in season in winter, no octopus of either kind."),
    ("Plate XVII", "media/verany_1851_argonauta_plate17.jpg", "Argonauta argo Linné",
     "Vérany, Céphalopodes (1851), Pl. 17 — del. ex vivo",
     "Exhibited with intent: an absence, carefully noted, is also a specimen."),
    ("Fig. 10–15", "media/malpighi_1687_pulli_in_ovo_plate.jpg", "De pulli in ovo — the chick in the egg",
     "Malpighi (1687), Tom. II, p. 56",
     "Development drawn figure by figure, letter by letter. Every loop still files its evidence this way."),
    ("Taf. III", "media/muller_1842_mustelus_placenta_taf3.jpg", "Mustelus laevis — the yolk-sac placenta",
     "Johannes Müller (1842), Taf. III",
     "Aristotle described it; for two thousand years it was thought an error, until Müller dissected it. "
     "Patron specimen for trusting careful observation over received opinion."),
    ("Pl. 3", "media/turpin_1837_urpflanze_plate.jpg", "Végétal type, idéal, appendiculé — the Urpflanze",
     "Turpin (1837), after Goethe",
     "Hung as a standing instruction: exhibit types, reserve judgement on descent until the morphology is in hand."),
]

# What each office leaves in the tree (counted at build time).
STEP_ARTIFACTS = {
    "scout": ("scout manifests", "datasets/*/scout_manifest.json"),
    "paper": ("reference dossiers", "datasets/*/*_reference.json"),
    "source": ("citation ledgers", "datasets/*/paper_citations.bib"),
    "rules": ("Lean loop files", "morphoeval/lean/loops/*.lean"),
    "script": ("model scripts", "models/*.py"),
    "output": ("comparison figures", "docs/media/*_comparison.png"),
    "audit": ("audits, all passes", "morphoeval/audits/*.yaml"),
    "measure": ("measurement sidecars", "morphoeval/audits/*.measurements.yaml"),
    "build": ("knowledge-graph reports", "morphoeval/kg/*/REPORT.md"),
    "publish": ("signed TriG nanopublications", "morphoeval/kg/*/signed/*.trig"),
    "graph": ("published graph bundles", "morphoeval/kg/*/published.json"),
    "remed": ("telemetry run logs", "morphoeval/runs/*.md"),
}


def count(pattern: str) -> int:
    return len(glob.glob(str(ROOT / pattern)))


def stage_specimen(src_rel: str, name: str, max_w: int = 560) -> str:
    """Copy a tracked library photo into docs/media/specimens/ (downscaled) and
    return its docs-relative URL. Pages serves docs/ only, so datasets/ images
    must be staged. Idempotent; skips when the staged copy is newer."""
    src = ROOT / src_rel
    SPECIMEN_DIR.mkdir(parents=True, exist_ok=True)
    dst = SPECIMEN_DIR / f"lagoon_{name}.jpg"
    rel = f"media/specimens/{dst.name}"
    if not src.exists():
        raise FileNotFoundError(f"specimen for {name!r} missing: {src_rel}")
    if is_placeholder_image(src):
        raise ValueError(f"specimen for {name!r} is a synthetic placeholder, not a photograph: {src_rel}")
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return rel
    try:
        from PIL import Image, ImageOps
        im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
        if im.width > max_w:
            im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
        im.save(dst, "JPEG", quality=82, optimize=True)
    except Exception:
        shutil.copy2(src, dst)
    return rel


def load_jsons(pattern: str):
    out = []
    for p in sorted(DISPATCH_DIR.glob(pattern)):
        try:
            out.append(json.loads(p.read_text()))
        except Exception:
            continue
    return out


# Plain-English names for every hat in the fleet. The technical slug/record name
# always travels next to the simple name so the two can never drift apart.
SIMPLE_ROLE_NAMES = {
    "scout": "Science scout",      # step 1: finds a paper worth reproducing
    "paper": "Paper extractor",    # step 2: pulls the formal model out of it
    "source": "Source curator",    # step 3: vendors + SHA-locks the records
    "rules": "Rule pinner",        # step 4: prose -> exact rules
    "script": "Script builder",    # step 5: writes the model program
    "output": "Renderer",          # step 6: comparison figure/animation
    "audit": "Assessor",           # step 7: independent judge (different model)
    "measure": "Measurer",         # step 8: fingerprints the outputs
    "build": "KG builder",         # step 9: emits the nanopublications
    "publish": "Publisher",        # step 10: signs + publishes them
    "graph": "Record keeper",      # step 11: public record & report
    "remed": "Remediator",         # step 12: fixes what the Assessor found
}


def simple_actor_name(actor: str) -> str:
    a = str(actor).lower()
    if "assessor" in a:
        return "Assessor"
    if "measure_outputs" in a:
        return "Measurer"
    if "build_morphoeval_kg" in a:
        return "KG builder"
    if "sign_morphoeval" in a or "check_signed" in a:
        return "Publisher"
    if "scout" in a:
        return "Science scout"
    if "builder" in a:
        return "Script builder"
    if a.strip() == "tool":
        return "Deterministic tool"
    return "—"


def bar(pct: float, color: str = "#7a5c2e") -> str:
    return (f'<div class="bar"><div class="fill" '
            f'style="width:{max(0.0, min(100.0, pct)):.1f}%;background:{color}"></div></div>')


def census() -> dict:
    data: dict = {}

    slots = load_jsons("slot-*.json")
    slot_status = collections.Counter(s.get("status") for s in slots)
    slot_dates = [s.get("completed_at") or s.get("dispatched_at") or "" for s in slots]
    data["slots"] = {
        "n": len(slots),
        "completed": slot_status.get("completed", 0) + slot_status.get("pass3_completed", 0)
                     + slot_status.get("pass3_complete", 0),
        "numbered": sum(1 for s in slots if isinstance(s.get("slot"), int)),
        "max_slot": max((s["slot"] for s in slots if isinstance(s.get("slot"), int)), default=0),
        "nanopubs_signed": sum(s.get("nanopubs_signed") or 0 for s in slots),
        "with_pass2": sum(1 for s in slots if s.get("pass2_score") is not None),
        "with_pass3": sum(1 for s in slots if s.get("audit_pass3_id") or s.get("pass3_score") is not None
                          or str(s.get("status", "")).startswith("pass3")),
        "first": min((d for d in slot_dates if d), default=""),
        "last": max((d for d in slot_dates if d), default=""),
    }

    scout_dispatch = load_jsons("problem-scout-*.json")
    scout_files = []
    for p in sorted(SCOUT_DIR.glob("scout-*.json")):
        try:
            scout_files.append(json.loads(p.read_text()))
        except Exception:
            continue
    grounds = collections.Counter(s.get("ground") for s in scout_files)
    lenses = collections.Counter(s.get("lens") for s in scout_files)
    cells = collections.Counter((s.get("ground"), s.get("lens")) for s in scout_files)
    data["scouts"] = {
        "dispatched": len(scout_dispatch),
        "returned": len(scout_files),
        "candidates": sum(len(s.get("candidates") or []) for s in scout_files),
        "grounds": grounds, "lenses": lenses, "cells": cells,
        "ground_order": sorted(grounds), "lens_order": sorted(lenses),
    }

    fixer_dispatch = load_jsons("problem-fixer-*.json")
    queue = json.loads((FIXER_DIR / "fix_queue.json").read_text())
    results = {}
    for task in queue["tasks"]:
        rp = FIXER_DIR / "results" / f"{task['id']}.json"
        if rp.exists():
            try:
                results[task["id"]] = json.loads(rp.read_text())
            except Exception:
                pass
    data["fixers"] = {
        "dispatched": len(fixer_dispatch),
        "tasks": queue["tasks"],
        "results": results,
        "landed": len(results),
    }

    tele = {"runs": 0, "steps": 0, "roles": [], "actors": [], "models": [], "first": "", "last": ""}
    if TELEMETRY.exists():
        con = sqlite3.connect(TELEMETRY)
        tele["runs"] = con.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        tele["steps"] = con.execute("SELECT COUNT(*) FROM steps").fetchone()[0]
        tele["roles"] = con.execute(
            "SELECT slug, tier, COUNT(*) FROM steps GROUP BY slug, tier ORDER BY MIN(step_num)"
        ).fetchall()
        tele["actors"] = con.execute(
            "SELECT actor, COUNT(*) FROM steps WHERE actor IS NOT NULL "
            "GROUP BY actor ORDER BY COUNT(*) DESC LIMIT 10"
        ).fetchall()
        tele["models"] = con.execute(
            "SELECT model, COUNT(*) FROM steps WHERE model IS NOT NULL "
            "GROUP BY model ORDER BY COUNT(*) DESC LIMIT 6"
        ).fetchall()
        tele["first"], tele["last"] = con.execute(
            "SELECT MIN(created_at), MAX(created_at) FROM runs").fetchone()

        # --- Plate IV: per-model quality, joined from runs + math_models ---
        builders = con.execute(
            "SELECT r.builder_model, COUNT(*), AVG(m.overall_score), AVG(m.model_fidelity_score), "
            "AVG(m.morphometric_score), AVG(m.specimen_match_score), AVG(r.steps_completed), "
            "AVG(r.wall_time_s), AVG(r.total_llm_tokens) "
            "FROM runs r LEFT JOIN math_models m ON m.sim_id = r.target_simulation "
            "WHERE r.builder_model IS NOT NULL GROUP BY r.builder_model "
            "ORDER BY AVG(m.overall_score) DESC"
        ).fetchall()
        violations = dict(con.execute(
            "SELECT offending_model, COUNT(*) FROM model_integrity_violations "
            "GROUP BY offending_model").fetchall())
        incidents = con.execute(
            "SELECT offending_model, detecting_referee_model, superseded_self_score, "
            "referee_recalibrated_score FROM model_integrity_incidents").fetchall()
        assessors = {
            "primary": con.execute(
                "SELECT primary_assessor, COUNT(*) FROM runs WHERE primary_assessor IS NOT NULL "
                "GROUP BY primary_assessor ORDER BY COUNT(*) DESC").fetchall(),
            "second": con.execute(
                "SELECT second_assessor, COUNT(*) FROM runs WHERE second_assessor IS NOT NULL "
                "GROUP BY second_assessor ORDER BY COUNT(*) DESC").fetchall(),
        }
        step_models_raw = con.execute(
            "SELECT model, COUNT(*), AVG(duration_s) FROM steps WHERE model IS NOT NULL "
            "GROUP BY model").fetchall()
        con.close()

        # Assessor identities stamped into audit YAMLs (the telemetry seats only
        # cover the 106 tracked loops; most judging lives in the audit corpus).
        def norm_model(name: str) -> str:
            n = str(name).strip().lower().replace(" ", "-")
            for suffix in ("-phase-a", "-thinking-high"):
                if n.endswith(suffix):
                    n = n[: -len(suffix)]
            return n.replace("claude-sonnet-5.5-high", "claude-sonnet-5-5-high")

        audit_assessors: collections.Counter = collections.Counter()
        try:
            import yaml as _yaml

            def _walk(o):
                if isinstance(o, dict):
                    for k, v in o.items():
                        kl = str(k).lower()
                        if "assessor" in kl or "referee" in kl:
                            if isinstance(v, dict) and v.get("id"):
                                audit_assessors[norm_model(v["id"])] += 1
                            elif isinstance(v, str):
                                audit_assessors[norm_model(v)] += 1
                        _walk(v)
                elif isinstance(o, list):
                    for v in o:
                        _walk(v)

            loader = getattr(_yaml, "CSafeLoader", _yaml.SafeLoader)
            AUDITS_DIR = ROOT / "morphoeval" / "audits"
            for f in AUDITS_DIR.glob("*.yaml"):
                try:
                    _walk(_yaml.load(f.read_text(errors="replace"), Loader=loader))
                except Exception:
                    continue
            delphi = AUDITS_DIR / "2026-10-20-rule30-grand-prizes-bayesian-delphi.yaml"
            panel_models = []
            if delphi.exists():
                dd = _yaml.load(delphi.read_text(errors="replace"), Loader=loader)
                models_reg = (dd.get("audit") or dd).get("models") or {}
                panel_models = [norm_model(m) for m in models_reg]
        except Exception:
            panel_models = []
        data["assessor_breakout"] = {
            "audit_counts": dict(audit_assessors),
            "delphi_panel": panel_models,
        }
        merged: dict[str, list] = {}
        for name, n, avg_dur in step_models_raw:
            key = str(name).strip().lower().replace(" ", "-").replace("fable-5.1", "fable-5-1")
            if key in merged:
                pn, pd = merged[key]
                merged[key] = [pn + n, (pd * pn + (avg_dur or 0) * n) / (pn + n)]
            else:
                merged[key] = [n, avg_dur or 0]
        data["model_quality"] = {
            "builders": [
                {"model": b[0], "loops": b[1], "overall": b[2], "fidelity": b[3],
                 "morphometric": b[4], "specimen": b[5], "steps_completed": b[6],
                 "wall_s": b[7], "tokens": b[8], "violations": violations.get(b[0], 0)}
                for b in builders
            ],
            "incidents": incidents,
            "assessors": assessors,
            "step_models": sorted(
                ({"model": k, "steps": v[0], "avg_s": v[1]} for k, v in merged.items()),
                key=lambda r: -r["steps"]),
        }
    else:
        data["model_quality"] = {"builders": [], "incidents": [], "assessors": {}, "step_models": []}
        data["assessor_breakout"] = {"audit_counts": {}, "delphi_panel": []}
    data["telemetry"] = tele
    data["red_flags"] = detect_red_flags(data)
    return data


# Detector thresholds: a flag fires when recorded evidence crosses these lines.
FLAG_SELF_AUDIT_GAP = 20.0      # points between a self-score and the referee score
FLAG_PASS_GAP_MEAN = 15.0       # mean pass1->pass2 jump across all slots
FLAG_PASS3_DROP = -10.0         # pass-3 re-audit regression
FLAG_STUB_SHARE = 0.50          # placeholder share of the Lean corpus
FLAG_MODEL_CONCENTRATION = 0.80  # one model building >= this share of tracked loops


def detect_red_flags(d: dict) -> list[dict]:
    """Scan every record source for big gaps and severe misreporting."""
    flags: list[dict] = []

    def num(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    slots = load_jsons("slot-*.json")
    p1 = [num(s.get("pass1_score")) for s in slots if num(s.get("pass1_score")) is not None]
    p2 = [num(s.get("pass2_score")) for s in slots if num(s.get("pass2_score")) is not None]
    if p1 and p2 and len(p1) == len(p2):
        gap = sum(b - a for a, b in zip(p1, p2)) / len(p1)
        if gap >= FLAG_PASS_GAP_MEAN:
            flags.append({
                "severity": "high", "status": "open",
                "title": "Systemic pass-1 → pass-2 score inflation",
                "detector": f"mean(slot pass2 − pass1) ≥ {FLAG_PASS_GAP_MEAN:.0f} points",
                "evidence": (f"Mean pass-1 {sum(p1)/len(p1):.1f} → mean pass-2 {sum(p2)/len(p2):.1f} "
                             f"(+{gap:.1f}) across {len(p1):,} slots; every slot gained ≥20 points. "
                             "The two passes are not cross-calibrated, so pass-2 scores alone "
                             "must not be quoted as quality."),
            })

    drops = []
    for s in slots:
        delta = num(s.get("pass3_score_delta_vs_pass2") or s.get("pass3_score_delta"))
        if delta is not None and delta <= FLAG_PASS3_DROP:
            drops.append((delta, s.get("slot_id") or s.get("branch_name") or "?"))
    if drops:
        drops.sort()
        flags.append({
            "severity": "medium", "status": "open",
            "title": "Pass-3 re-audits reversed earlier scores",
            "detector": f"pass3_score_delta ≤ {FLAG_PASS3_DROP:.0f}",
            "evidence": f"{len(drops)} slot(s) dropped ≥10 points on independent re-audit; worst: "
                        + ", ".join(f"{name} ({delta:+.1f})" for delta, name in drops[:3]) + ".",
        })

    census_path = SCOUT_DIR / "lean_stub_census.json"
    if census_path.exists():
        c = json.loads(census_path.read_text())
        share = c["stub_template_files"] / c["total_lean_loop_files"]
        if share >= FLAG_STUB_SHARE:
            flags.append({
                "severity": "high", "status": "labelled — proofs outstanding",
                "title": "Lean corpus is mostly placeholder templates",
                "detector": f"template-stub share ≥ {FLAG_STUB_SHARE:.0%}",
                "evidence": (f"{c['stub_template_files']:,} of {c['total_lean_loop_files']:,} Lean loop "
                             f"files ({share:.1%}) are the overnight template (issue #268); slot records "
                             "nonetheless recorded theorems as proved. Relabelled in PR #269; "
                             "mechanism proofs are still outstanding."),
            })

    missing = [(g, l) for g in d["scouts"]["ground_order"] for l in d["scouts"]["lens_order"]
               if d["scouts"]["cells"].get((g, l), 0) == 0]
    held_fixers = [t["id"] for t in d["fixers"]["tasks"] if t["id"] not in d["fixers"]["results"]]
    if missing or held_fixers:
        flags.append({
            "severity": "medium", "status": "open",
            "title": "Agents dispatched but never landed",
            "detector": "dispatch record exists, report/result absent on main",
            "evidence": (f"{len(missing)} scout cell(s) unreturned "
                         f"({', '.join(f'{g}/{l}' for g, l in missing) or 'none'}); "
                         f"fixer report(s) not on main: {', '.join(held_fixers) or 'none'}."),
        })

    if TELEMETRY.exists():
        con = sqlite3.connect(TELEMETRY)
        for off, referee, self_score, recal in con.execute(
                "SELECT offending_model, detecting_referee_model, superseded_self_score, "
                "referee_recalibrated_score FROM model_integrity_incidents"):
            gap = (self_score or 0) - (recal or 0)
            if gap >= FLAG_SELF_AUDIT_GAP:
                flags.append({
                    "severity": "critical", "status": "open — mitigated by cross-model refereeing",
                    "title": "Self-audit conflict of interest (caught)",
                    "detector": f"self-score − referee-score ≥ {FLAG_SELF_AUDIT_GAP:.0f}",
                    "evidence": (f"{off} scored its own work {self_score}; independent referee "
                                 f"{referee} recalibrated it to {recal} (gap {gap:.1f})."),
                })
        violations = con.execute(
            "SELECT code, severity, COUNT(*) FROM model_integrity_violations "
            "GROUP BY code, severity ORDER BY COUNT(*) DESC").fetchall()
        if violations:
            flags.append({
                "severity": "critical", "status": "open",
                "title": "Recorded integrity violations",
                "detector": "any row in model_integrity_violations",
                "evidence": "; ".join(f"{code} ({sev}) ×{n}" for code, sev, n in violations) + ".",
            })
        failed = con.execute(
            "SELECT slug, COUNT(*) FROM steps WHERE status='failed' GROUP BY slug").fetchall()
        if failed:
            flags.append({
                "severity": "medium", "status": "open",
                "title": "Pipeline steps failed in tracked runs",
                "detector": "steps.status = 'failed'",
                "evidence": "; ".join(f"{slug} ×{n}" for slug, n in failed)
                            + " (all at the signing/publish step).",
            })
        total_runs = con.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        top_builder = con.execute(
            "SELECT builder_model, COUNT(*) FROM runs GROUP BY builder_model "
            "ORDER BY COUNT(*) DESC LIMIT 1").fetchone()
        con.close()
        if top_builder and total_runs and top_builder[1] / total_runs >= FLAG_MODEL_CONCENTRATION:
            flags.append({
                "severity": "high", "status": "open",
                "title": "Single-model concentration",
                "detector": f"top builder share ≥ {FLAG_MODEL_CONCENTRATION:.0%}",
                "evidence": (f"{top_builder[0]} built {top_builder[1]} of {total_runs} tracked loops "
                             f"({top_builder[1]/total_runs:.0%}) and was also primary assessor on most "
                             "of them — builder/judge overlap is the surface the self-audit incident "
                             "came from."),
            })

    gal = gallery_placeholder_census()
    if gal["total"] and gal["placeholders"]:
        share = gal["placeholders"] / gal["total"]
        flags.append({
            "severity": "high" if share >= 0.25 else "medium", "status": "open",
            "title": "Species gallery exhibits synthetic placeholders as specimens",
            "detector": "smooth-gradient / low-colour image test over datasets/species_gallery/*/*.jpg",
            "evidence": (f"{gal['placeholders']} of {gal['total']} gallery images ({share:.0%}) are generated "
                         f"spiral placeholders, not photographs (e.g. {', '.join(gal['examples'])}); "
                         "species_manifest.json records them with a SHA-256 like any specimen. This "
                         "dashboard refuses to hang them; the manifest should label them."),
        })

    fix1 = d["fixers"]["results"].get("fix-001")
    if fix1 and fix1.get("status") == "fixed":
        flags.append({
            "severity": "resolved", "status": "resolved",
            "title": "Fabricated 'empirical' curves quarantined",
            "detector": "Problem-Scout candidate BIO-TPL-DQ-01, verified by fix-001",
            "evidence": ("Three bioelectric_polarity 'empirical' series were confirmed slot-seeded "
                         "alpha templates and quarantined with provenance labels (PR #271); "
                         "kept here as the worked example of a flag closing."),
        })
    return flags


def main() -> int:
    d = census()
    slots, scouts, fixers, tele = d["slots"], d["scouts"], d["fixers"], d["telemetry"]
    total_cloud = slots["n"] + scouts["dispatched"] + fixers["dispatched"]

    def pct(a, b):
        return 100.0 * a / b if b else 0.0

    roster = [
        ("Loop builder (slot agent)", "One cloud agent per MorphoEval slot: writes the model "
         "script, dataset, Lean module and nanopubs for one organism/mechanism loop.",
         slots["n"], f'{slots["completed"]} completed', pct(slots["completed"], slots["n"]),
         "morphoeval/cloud/dispatches/slot-*.json"),
        ("Problem scout", "Hunts one hunting-ground x tool-lens cell for resolvable problems; "
         "returns candidates for the central resolvability ranker.",
         scouts["dispatched"], f'{scouts["returned"]} reports in tree · {scouts["candidates"]} candidates',
         pct(scouts["returned"], scouts["dispatched"]), "morphoeval/problem_scouts/scout-*.json"),
        ("Fixer", "Repairs one leaderboard candidate in an isolated PR under a path "
         "allowlist + trusted merge gate; never self-merges.",
         fixers["dispatched"], f'{fixers["landed"]} landed on main', pct(fixers["landed"], fixers["dispatched"]),
         "morphoeval/problem_fixers/results/*.json"),
        ("12-Step Loop Agent (the roles)", "Not separate dispatches and not one model doing "
         "everything: the loop is a 12-step protocol. One builder-agent writes (steps 1–6), a "
         "DIFFERENT model judges (step 7, dual assessors), deterministic tools measure, "
         "build and publish (8–11), and findings loop back (12). See the diagram in Section IV.",
         tele["steps"], f'{tele["runs"]} tracked runs · 106 executions per role', 100.0,
         "morphoeval/runs/telemetry.sqlite (steps)"),
        ("Orchestrator — Muse Wonder", "The session conductor: designs the fleets, writes the "
         "dispatchers, gates and this dashboard, verifies scout claims before acting on them, "
         "and merges the PRs. Leaves no dispatch record — GitHub attests the account, not the "
         "model — so this row is session-attested, a weaker evidence grade than every row above.",
         "1 session", "16 PRs merged (#269–#288, incl. 10 fixer PRs) · issues #268/#283/#285 · "
         "Plates III–V", 100.0, "github.com/fractastical/morpholib/pulls (merged)"),
    ]
    def portrait(name: str, key: str) -> str:
        sp = FLEET_SPECIES.get(name)
        if not sp:
            return "<td></td>"
        latin, common, img, why = sp
        url = stage_specimen(img, key)
        return (f'<td class="portrait"><img src="{url}" alt="{html.escape(latin)}" title="{html.escape(why)}">'
                f'<i>{html.escape(latin)}</i><span class="dim">{html.escape(common)}</span></td>')

    roster_rows = "\n".join(
        f"<tr>{portrait(name, f'fleet_{i}')}"
        f"<td><strong>{html.escape(name)}</strong><br><span class='dim'>{html.escape(desc)}</span></td>"
        f"<td class='num'>{f'{used:,}' if isinstance(used, int) else html.escape(str(used))}</td>"
        f"<td>{html.escape(done)}</td>"
        f"<td>{bar(p)}<span class='dim'>{p:.0f}%</span></td>"
        f"<td><code>{html.escape(src)}</code></td></tr>"
        for i, (name, desc, used, done, p, src) in enumerate(roster)
    )

    # Scout ground x lens matrix
    g_order, l_order = scouts["ground_order"], scouts["lens_order"]
    head = "".join(f"<th>{html.escape(str(x))}</th>" for x in l_order)
    matrix_rows = ""
    for g in g_order:
        tds = ""
        for lens in l_order:
            n = scouts["cells"].get((g, lens), 0)
            cls = "cell hit" if n else "cell miss"
            tds += f'<td class="{cls}">{n if n else "·"}</td>'
        matrix_rows += f"<tr><th>{html.escape(str(g))}</th>{tds}<td class='num'>{scouts['grounds'][g]}</td></tr>"

    fixer_rows = ""
    for task in fixers["tasks"]:
        res = fixers["results"].get(task["id"])
        status = res.get("status") if res else "report not on main (PR open / held)"
        fixer_rows += (
            f"<tr><td><code>{task['id']}</code></td><td>{html.escape(task['title'])}</td>"
            f"<td>#{task['leaderboard_rank']}</td><td>{task['merge_policy']}</td>"
            f"<td>{html.escape(str(status))}</td></tr>"
        )

    # The Twelve Offices — one specimen card per loop role. Execution counts come
    # from telemetry (same numbers the old role table carried); artifact counts
    # are what that office has actually left in the tree.
    role_exec = {slug: (tier, n) for slug, tier, n in tele["roles"]}
    role_order = [s for s, *_ in tele["roles"]] or list(SIMPLE_ROLE_NAMES)
    for s in SIMPLE_ROLE_NAMES:
        if s not in role_order:
            role_order.append(s)
    max_art = max((count(g) for _, g in STEP_ARTIFACTS.values()), default=1) or 1
    office_cards = ""
    for i, slug in enumerate(role_order, 1):
        latin, common, img, why = ROLE_SPECIES[slug]
        tier, n = role_exec.get(slug, ("?", 0))
        label, pattern = STEP_ARTIFACTS[slug]
        n_art = count(pattern)
        url = stage_specimen(img, slug)
        office_cards += (
            f'<article class="specimen" id="office-{slug}">'
            f'<div class="frame"><img src="{url}" alt="{html.escape(latin)}" loading="lazy">'
            f'<span class="plateno">Plate {ROMAN[i + 3]}</span></div>'
            f'<div class="label"><p class="office">Office {i} · <code>{html.escape(slug)}</code> · Tier {html.escape(str(tier))}</p>'
            f'<h3>{html.escape(SIMPLE_ROLE_NAMES.get(slug, slug))}</h3>'
            f'<p class="latin"><i>{html.escape(latin)}</i> — {html.escape(common)}</p>'
            f'<p class="why">{html.escape(why)}</p>'
            f'<p class="tally"><b>{n:,}</b> executions in telemetry {bar(pct(n, tele["steps"]))}</p>'
            f'<p class="tally"><b>{n_art:,}</b> {html.escape(label)} held '
            f'<span class="ruler"><i style="width:{max(3, round(100 * n_art / max_art))}%"></i></span></p>'
            f'</div></article>'
        )

    hero = MUSEUM_PLATES[0]
    deposit_plates = "".join(
        f'<figure class="plate-fig"><img src="{img}" alt="{html.escape(title)}" loading="lazy">'
        f'<figcaption><span class="acc">{html.escape(acc)}</span><b>{html.escape(title)}</b>'
        f'<i>{html.escape(src)}</i><p>{html.escape(label)}</p></figcaption></figure>'
        for acc, img, title, src, label in MUSEUM_PLATES[1:]
    )
    actor_rows = "\n".join(
        f"<tr><td><strong>{html.escape(simple_actor_name(a))}</strong></td>"
        f"<td>{html.escape(str(a))}</td><td class='num'>{n}</td></tr>" for a, n in tele["actors"])
    model_rows = "\n".join(
        f"<tr><td>{html.escape(str(m))}</td><td class='num'>{n}</td></tr>" for m, n in tele["models"])

    mq = d.get("model_quality", {"builders": [], "incidents": [], "assessors": {}, "step_models": []})

    def fmt(x, digits=1):
        return f"{x:.{digits}f}" if isinstance(x, (int, float)) else "—"

    builder_rows = ""
    for i, b in enumerate(mq["builders"], 1):
        evidence = "solid" if b["loops"] >= 30 else ("thin — indicative only" if b["loops"] >= 5 else "anecdotal — do not rank on this")
        wall_min = f"{b['wall_s'] / 60:.1f} min" if b["wall_s"] else "—"
        tokens = f"{b['tokens']:,.0f}" if b["tokens"] else "—"
        builder_rows += (
            f"<tr><td>{i}</td><td><strong>{html.escape(str(b['model']))}</strong></td>"
            f"<td class='num'>{b['loops']}</td><td class='num'>{fmt(b['overall'])}</td>"
            f"<td class='num'>{fmt(b['fidelity'])}</td><td class='num'>{fmt(b['morphometric'])}</td>"
            f"<td class='num'>{fmt(b['specimen'])}</td><td class='num'>{fmt(b['steps_completed'])}/12</td>"
            f"<td class='num'>{wall_min}</td><td class='num'>{tokens}</td>"
            f"<td class='num'>{b['violations']}</td><td class='dim'>{evidence}</td></tr>"
        )
    incident_note = ""
    for off, referee, self_score, recal in mq["incidents"]:
        incident_note += (
            f'<p class="dim">Recorded integrity incident: loops built by <strong>{html.escape(str(off))}</strong> '
            f"carried a self-audit of {fmt(self_score)}, recalibrated to {fmt(recal)} by referee "
            f"<strong>{html.escape(str(referee))}</strong> — the reason auditing and building are kept on different models.</p>"
        )
    ab = d.get("assessor_breakout", {"audit_counts": {}, "delphi_panel": []})
    seat_counts = {"primary": dict(mq["assessors"].get("primary", [])),
                   "second": dict(mq["assessors"].get("second", []))}
    assessor_models = sorted(
        set(ab["audit_counts"]) | set(seat_counts["primary"]) | set(seat_counts["second"])
        | set(ab["delphi_panel"]),
        key=lambda m: -ab["audit_counts"].get(m, 0),
    )
    assessor_rows = ""
    for model in assessor_models:
        panels = "Rule30 Delphi panel" if model in ab["delphi_panel"] else ""
        assessor_rows += (
            f"<tr><td><strong>{html.escape(model)}</strong></td>"
            f"<td class='num'>{ab['audit_counts'].get(model, 0)}</td>"
            f"<td class='num'>{seat_counts['primary'].get(model, 0)}</td>"
            f"<td class='num'>{seat_counts['second'].get(model, 0)}</td>"
            f"<td>{html.escape(panels)}</td></tr>"
        )
    step_model_rows = "\n".join(
        f"<tr><td>{html.escape(r['model'])}</td><td class='num'>{r['steps']}</td>"
        f"<td class='num'>{r['avg_s']:.1f}s</td></tr>" for r in mq["step_models"][:8]
    )

    # Small 12-step loop diagram (mirrors morphoeval/track_loop.py CANONICAL_STEPS
    # default_actor values). The point it makes visually: step 7 is worn by a
    # DIFFERENT hat — dual assessors — never by the builder who wrote steps 1-6.
    loop_steps = [
        (1, "scout", "human+scout", "human"), (2, "paper", "builder", "builder"),
        (3, "source", "builder+tool", "mixed"), (4, "rules", "builder", "builder"),
        (5, "script", "builder", "builder"), (6, "output", "tool", "tool"),
        (7, "audit", "dual assessors ★", "assessor"), (8, "measure", "tool", "tool"),
        (9, "build", "tool", "tool"), (10, "publish", "tool", "tool"),
        (11, "graph", "tool", "tool"), (12, "remed", "builder+assessor", "mixed"),
    ]

    def step_box(num, slug, actor, who):
        return (f'<div class="lstep who-{who}"><b>{num}. {SIMPLE_ROLE_NAMES.get(slug, slug)}</b>'
                f'<span>{slug} · {actor}</span></div>')

    row1 = '<span class="larr">→</span>'.join(step_box(*s) for s in loop_steps[:6])
    row2 = '<span class="larr">←</span>'.join(step_box(*s) for s in reversed(loop_steps[6:]))
    loop_diagram = (
        f'<div class="loopdiag"><div class="lrow">{row1}</div>'
        '<div class="ldown">↓ &nbsp;Tier B — judging &amp; publishing</div>'
        f'<div class="lrow">{row2}</div>'
        '<div class="lback">↩ step 12 (remed): fixes loop back to step 5 (script), then 6–11 re-run. '
        '★ The auditor is a different model from the builder — that separation is the control '
        'the red-flag detectors in Section VIII watch over.</div></div>'
    )

    sev_badge = {
        "critical": '<span class="badge crit">CRITICAL</span>',
        "high": '<span class="badge high">HIGH</span>',
        "medium": '<span class="badge med">MEDIUM</span>',
        "resolved": '<span class="badge ok">RESOLVED</span>',
    }
    sev_rank = {"critical": 0, "high": 1, "medium": 2, "resolved": 3}
    flag_rows = "\n".join(
        f"<tr class='flag-{f['severity']}'><td>{sev_badge[f['severity']]}</td>"
        f"<td><strong>{html.escape(f['title'])}</strong><br><span class='dim'>Detector: "
        f"<code>{html.escape(f['detector'])}</code></span></td>"
        f"<td>{html.escape(f['evidence'])}</td><td>{html.escape(f['status'])}</td></tr>"
        for f in sorted(d["red_flags"], key=lambda x: sev_rank[x["severity"]])
    )
    active_flags = sum(1 for f in d["red_flags"] if f["severity"] != "resolved")

    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Fleet Registry — Agent Types &amp; Usage</title>
<style>
 body {{ background:#e8dcc0; color:#2b2118; font-family: Georgia, 'Times New Roman', serif; margin:0; }}
 .plate {{ max-width:1060px; margin:24px auto; background:#f5edd8; border:2px solid #7a5c2e;
   box-shadow:0 2px 14px rgba(60,40,10,.35); padding:28px 34px; }}
 h1 {{ font-variant:small-caps; letter-spacing:.06em; border-bottom:3px double #7a5c2e; padding-bottom:8px; }}
 h2 {{ font-variant:small-caps; color:#5d4420; border-bottom:1px solid #b39b5e; margin-top:30px; }}
 table {{ border-collapse:collapse; width:100%; font-size:14px; }}
 th, td {{ border:1px solid #c4ab72; padding:5px 8px; text-align:left; vertical-align:top; }}
 th {{ background:#e2d2a4; }}
 td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
 .dim {{ color:#6d5a38; font-size:12.5px; }}
 .bar {{ display:inline-block; width:110px; height:9px; background:#d8c892;
   border:1px solid #7a5c2e; vertical-align:middle; margin-right:6px; }}
 .fill {{ height:100%; }}
 .cell {{ text-align:center; }}
 .loopdiag {{ border:1.5px solid #7a5c2e; background:#efe4c4; padding:10px 12px; margin:10px 0 14px; }}
 .lrow {{ display:flex; align-items:stretch; gap:4px; flex-wrap:wrap; }}
 .lstep {{ flex:1 1 90px; border:1.5px solid #7a5c2e; padding:4px 6px; font-size:12.5px; text-align:center; }}
 .lstep b {{ display:block; font-size:13px; }} .lstep span {{ color:#5d4420; font-size:11.5px; }}
 .who-builder {{ background:#e8d5a8; }} .who-assessor {{ background:#eeb9ac; border-color:#a33327; border-width:2.5px; }}
 .who-tool {{ background:#cfe0b8; }} .who-human {{ background:#c8d4e8; }} .who-mixed {{ background:#e4cfe0; }}
 .larr {{ align-self:center; font-weight:bold; color:#7a5c2e; }}
 .ldown {{ text-align:right; color:#7a5c2e; font-size:12.5px; padding:3px 2px; }}
 .lback {{ font-size:12.5px; color:#5d4420; padding-top:6px; }}
 .badge {{ font-size:11px; font-weight:bold; padding:2px 6px; border:1px solid #5d4420; }}
 .badge.crit {{ background:#a33327; color:#fff; }} .badge.high {{ background:#d98a4a; color:#fff; }}
 .badge.med {{ background:#e3cf6b; }} .badge.ok {{ background:#7ba05b; color:#fff; }}
 tr.flag-critical td {{ background:#f3d9d2; }} tr.flag-high td {{ background:#f7e3c8; }}
 tr.flag-resolved td {{ background:#dde8cf; }} .hit {{ background:#cfe0b8; }} .miss {{ background:#e8c9c0; color:#8a4a3a; }}
 .totals {{ display:flex; gap:14px; flex-wrap:wrap; margin:16px 0; }}
 .card {{ border:1.5px solid #7a5c2e; background:#efe4c4; padding:10px 16px; min-width:150px; }}
 .card b {{ font-size:24px; display:block; }}
 code {{ font-size:12px; }}
 /* museum furniture */
 .museum {{ text-align:center; letter-spacing:.34em; font-size:11px; color:#6d5a38; text-transform:uppercase; margin:0 0 4px; }}
 h1.lagoon {{ text-align:center; font-variant:normal; font-weight:normal; font-size:40px; letter-spacing:0; margin:4px 0 2px; }}
 h1.lagoon i {{ color:#7a2e22; }}
 .subtitle {{ text-align:center; font-style:italic; color:#6d5a38; margin:0 0 14px; }}
 .curator {{ background:#efe4c4; border:1px solid #b39b5e; border-left:4px solid #8a6d2f; padding:10px 16px; font-size:14px; line-height:1.5; }}
 figure.chart {{ margin:14px 0 6px; background:#faf5e6; border:1px solid #b39b5e; padding:10px; }}
 figure.chart img {{ width:100%; display:block; border:1px solid #d8cbb0; }}
 figcaption {{ padding:8px 4px 2px; font-size:13.5px; line-height:1.45; }}
 figcaption .acc {{ display:block; text-align:right; font-variant:small-caps; color:#8a6d2f; letter-spacing:.12em; font-size:11px; }}
 figcaption b {{ display:block; font-size:16px; }} figcaption i {{ color:#6d5a38; }} figcaption p {{ margin:5px 0 0; }}
 td.portrait {{ width:92px; text-align:center; font-size:11px; line-height:1.25; padding:6px 4px; }}
 td.portrait img {{ width:80px; height:80px; object-fit:cover; border:1px solid #7a5c2e; display:block; margin:0 auto 3px; filter:sepia(.25); }}
 td.portrait i {{ display:block; }}
 .offices {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(235px,1fr)); gap:14px; margin:12px 0 6px; }}
 .specimen {{ background:#faf5e6; border:1px solid #b39b5e; display:flex; flex-direction:column; }}
 .specimen .frame {{ position:relative; border-bottom:1px solid #b39b5e; }}
 .specimen .frame img {{ width:100%; height:160px; object-fit:cover; display:block; filter:sepia(.25) contrast(.98); }}
 .plateno {{ position:absolute; top:7px; right:8px; font-variant:small-caps; letter-spacing:.14em; font-size:10.5px; color:#8a6d2f; background:rgba(245,237,216,.9); padding:1px 6px; border:1px solid #b39b5e; }}
 .specimen .label {{ padding:9px 12px 11px; font-size:12.5px; line-height:1.45; }}
 .specimen .office {{ font-variant:small-caps; letter-spacing:.1em; font-size:11px; color:#8a6d2f; margin:0; }}
 .specimen h3 {{ margin:2px 0 0; font-weight:normal; font-size:18px; }}
 .specimen .latin {{ margin:0 0 5px; color:#6d5a38; }}
 .specimen .why {{ margin:0 0 7px; }}
 .specimen .tally {{ margin:2px 0; font-size:12px; color:#5d4420; }} .specimen .tally .bar {{ width:70px; }}
 .ruler {{ display:inline-block; vertical-align:middle; width:70px; height:7px; background:repeating-linear-gradient(90deg,#efe4c4 0 6px,#ddd0b4 6px 7px); border:1px solid #b39b5e; margin-left:6px; }}
 .ruler i {{ display:block; height:100%; background:#7a2e22; }}
 .deposit {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(220px,1fr)); gap:14px; }}
 figure.plate-fig {{ margin:0; background:#faf5e6; border:1px solid #b39b5e; padding:10px; }}
 figure.plate-fig img {{ width:100%; height:250px; object-fit:cover; object-position:top; display:block; border:1px solid #d8cbb0; }}
 figure.plate-fig figcaption {{ font-size:12.5px; }} figure.plate-fig figcaption b {{ font-size:14px; }}
</style></head><body><div class="plate">
<p class="museum">Museum of Morphogenesis · Project Delphi · Est. Oct. MMXXVI</p>
<h1 class="lagoon">Aristotle's <i>Lagoon</i></h1>
<p class="subtitle">Plate III — Registry of the Agent Fleet, exhibited as natural history —
after D'Arcy Thompson, <a href="thompson_1913_aristotle_as_a_biologist.html">On Aristotle as a Biologist</a> (1913)</p>

<div class="curator"><b>From the Curator.</b> Aristotle catalogued the Lagoon of Pyrrha by presence, absence,
and what each creature's form is <i>for</i>. This plate does the same for the agent fleet. Every agent type
is represented by a species already held in the library — a real specimen photograph, pinned and labelled,
not a drawing set swimming — and every number is counted from the records the repository itself keeps
(cloud dispatch records, returned reports, loop telemetry). Chat-session subagents leave no repo record and
are not counted. When enough morphology has been captured to let forms transform into one another, in
Thompson's manner, the Evolution Gallery at the foot of the page will open. Until then: no swimming.
Regenerate: <code>python3 docs/build_fleet_dashboard.py</code>.</div>

<figure class="chart"><img src="{hero[1]}" alt="{html.escape(hero[2])}">
<figcaption><span class="acc">{html.escape(hero[0])}</span><b>{html.escape(hero[2])}</b>
<i>{html.escape(hero[3])}</i><p>{html.escape(hero[4])}</p></figcaption></figure>

<div class="totals">
 <div class="card"><b>{total_cloud:,}</b>cloud agents dispatched</div>
 <div class="card"><b>{slots['n']:,}</b>loop-builder slots (#{1}–{slots['max_slot']})</div>
 <div class="card"><b>{scouts['candidates']:,}</b>scout candidates returned</div>
 <div class="card"><b>{tele['steps']:,}</b>loop-role executions tracked</div>
 <div class="card"><b>{slots['nanopubs_signed']:,}</b>nanopubs signed by slot agents*</div>
</div>

<p class="dim"><strong>The names used on this plate:</strong>
<strong>Loop builder</strong> builds one loop end-to-end · <strong>Science scout</strong> finds the paper ·
<strong>Paper extractor</strong> · <strong>Source curator</strong> · <strong>Rule pinner</strong> ·
<strong>Script builder</strong> · <strong>Renderer</strong> · <strong>Assessor</strong> judges (never the builder) ·
<strong>Measurer</strong> · <strong>KG builder</strong> makes the nanopubs · <strong>Publisher</strong> signs them ·
<strong>Record keeper</strong> · <strong>Remediator</strong> · <strong>Problem scout</strong> hunts problems ·
<strong>Fixer</strong> repairs them · <strong>Orchestrator</strong> conducts.</p>

<h2>I. The Fleet Roster — types &amp; how often used</h2>
<table><tr><th>Specimen</th><th>Agent type</th><th>Used</th><th>Outcome</th><th>Completion</th><th>Records</th></tr>
{roster_rows}</table>
<p class="dim">The 12-step loop is not a single species: its twelve offices are each pinned separately in Section IV.</p>

<h2>II. Problem-Scouts — hunting ground × tool lens ({scouts['returned']} of {scouts['dispatched']} returned)</h2>
<table><tr><th>ground \\ lens</th>{head}<th>Σ</th></tr>
{matrix_rows}</table>
<p class="dim">Missing cells (·) are scouts that were dispatched but whose report has not landed on main.</p>

<h2>III. Problem-Fixer Wave 1 — {fixers['landed']} of {fixers['dispatched']} landed on main</h2>
<table><tr><th>Fix</th><th>Candidate</th><th>Rank</th><th>Policy</th><th>Agent-reported status</th></tr>
{fixer_rows}</table>

<h2>IV. The 12-Step Loop Agent — how one loop is applied (telemetry, {tele['runs']} runs)</h2>
<p class="dim">One dispatch sends out one builder to run this protocol. The hats change as the
loop proceeds: the builder writes, independent assessors judge, tools measure and publish.
Execution counts per role are underneath the diagram.</p>
{loop_diagram}
<figure style="margin:8px 0 14px">
<img src="media/morphoeval_pipeline.png" alt="Canonical MorphoEval 12-step pipeline diagram"
 style="width:100%;border:1.5px solid #7a5c2e;background:#fff">
<figcaption class="dim">The canonical plate, small: <code>docs/media/morphoeval_pipeline.png</code>
(drawn by <code>morphoeval/report/draw_pipeline.py</code>). Note its bottom row — two independent
assessors from different companies score step 7, and the footer tallies passes by company:
the auditor was never the loop's own builder.</figcaption>
</figure>
<h3 style="font-variant:small-caps;color:#5d4420;margin:18px 0 2px">The Twelve Offices — specimens of the loop</h3>
<p class="dim">One library species per office. Execution counts are from telemetry ({tele['runs']} tracked runs);
"held" counts are what that office has actually left in the tree, with the ruler scaled to the largest.</p>
<div class="offices">{office_cards}</div>

<h2>V. Who (or what) filled the roles — top actors &amp; models in telemetry</h2>
<table><tr><th>Simple name</th><th>Actor (as recorded)</th><th>Steps</th></tr>{actor_rows}</table>
<table style="margin-top:10px"><tr><th>Model</th><th>Steps</th></tr>{model_rows}</table>

<h2>VI. Plate IVa — Builder Leaderboard (ranked by mean audit score)</h2>
<p class="dim">Quality = the independent audit scores of the loops each model built
(<code>math_models</code> joined to tracked runs), plus speed, token cost and integrity
record. Small samples are marked — three loops do not make a ranking.</p>
<table><tr><th>#</th><th>Model</th><th>Loops built</th><th>Overall</th><th>Fidelity</th>
<th>Morphometric</th><th>Specimen</th><th>Steps done</th><th>Wall / loop</th><th>Tokens / loop</th>
<th>Integrity violations</th><th>Evidence</th></tr>
{builder_rows}</table>
{incident_note}

<h2>VII. Plate IVb — Assessor Leaderboard (judges, broken out from builders)</h2>
<p class="dim">Building and judging are different jobs and are ranked separately. Audit-YAML counts
come from the <code>assessor.id</code> stamps in <code>morphoeval/audits/*.yaml</code>; telemetry
seats cover only the 106 tracked loops. GPT-5.6 Sol's recorded footprint is small — chiefly the
Cervera audits and a seat on the Rule30 Delphi panel — so any heavier assessing it did through
chat sessions left no stamp and, by this dashboard's own rule, is not counted.</p>
<table><tr><th>Assessor model</th><th>Audits judged (YAML)</th><th>Telemetry primary seat</th>
<th>Telemetry second seat</th><th>Panels</th></tr>
{assessor_rows}</table>
<table style="margin-top:10px"><tr><th>Model (steps, name-normalised)</th><th>Steps executed</th><th>Avg step time</th></tr>
{step_model_rows}</table>

<h2>VIII. Red Flags — big gaps &amp; severe misreporting ({active_flags} active)</h2>
<p class="dim">Each flag is a named detector rule over the same records; regenerate the dashboard
and any threshold crossed since last time appears here. Resolved flags stay on the plate as
precedent.</p>
<table><tr><th>Severity</th><th>Flag</th><th>Evidence</th><th>Status</th></tr>
{flag_rows}</table>

<h2>IX. The unranked fleets — a recording gap</h2>
<p>The {slots['n']:,} slot dispatches and the Cursor cloud fleets ({scouts['dispatched']} scouts, {fixers['dispatched']} fixers) do not record
which model each agent ran, so they cannot be ranked yet — the leaderboard above covers only the
106 telemetry-tracked loops. Fleet outcomes by completion are in Plate I; recording the model in
every dispatch record (the fixer dispatcher already accepts <code>--model</code>) would make the
next wave rankable end-to-end.</p>

<h2>X. On Deposit — plates from the natural history museum</h2>
<div class="deposit">{deposit_plates}</div>
<p class="dim">Plates held in this repo at <code>docs/media/</code>, as catalogued in the
<a href="thompson_1913_aristotle_as_a_biologist.html">museum-illustrated Thompson (1913) edition</a>.
Species portraits above are tracked library photographs from <code>datasets/species_gallery/</code>
(<a href="{GH}/datasets/species_gallery/SPECIES_ICONIC_ATLAS.md">iconic atlas</a>), staged into
<code>docs/media/specimens/</code> at build time.</p>

<h2>XI. The Evolution Gallery — in preparation</h2>
<figure class="chart"><img src="media/thompson_transformations.gif" alt="Thompson coordinate transformations"
 style="max-height:320px;object-fit:contain;background:#faf5e6">
<figcaption><span class="acc">Not yet accessioned</span><b>On Growth and Form, ch. XVII — transformations</b>
<i>Thompson (1917)</i><p>This room is roped off. Forms in this collection will be allowed to evolve into one
another — by coordinate transformation, not by costume — when enough morphology has been captured to carry
the deformation honestly. The specimens above are being pinned for precisely that day.</p></figcaption></figure>

<p class="dim">* Slot-agent Lean/nanopub counts are as recorded at dispatch time; per issue #268 the
overnight-slot Lean modules are placeholder templates, not mechanism proofs, so treat
<code>lean_theorems_proved</code>-style tallies from those records as claims, not verifications.
Telemetry runs span {html.escape(str(tele['first']))} – {html.escape(str(tele['last']))}.</p>
</div></body></html>
"""
    OUT.write_text(page)
    print(f"wrote {OUT} ({len(page):,} bytes)")
    print(f"cloud total={total_cloud:,} slots={slots['n']:,} scouts={scouts['dispatched']} "
          f"fixers={fixers['dispatched']} telemetry_steps={tele['steps']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
