#!/usr/bin/env python3
"""
Dataset-existence scout for the Thompson (1917) claim registry.

Question answered, per claim in thompson/thompson_1917_claims.json:
  **Does a usable dataset exist, where, and can the MorphoEval loop get it
  without a human?**

Existence classes (field `existence`):
  IN_REPO            data already on disk (datasets/, .thompson_cache/, portfolio
                     caches) and verified by path/keyword scan at scout time
  PUBLIC_SCRIPTABLE  a public dataset was found *live* by the Dryad API
                     (api/v2/search) with a title matching the claim's query,
                     or is Thompson's own 1917 figures, fetchable from the
                     pinned Internet Archive scan (DIGITIZE_1917)
  PUBLIC_CANDIDATE   public search returned hits but relevance is unvetted --
                     the loop's Step 1 (morphoeval/scout_literature.py) vets
                     them; one human glance may be cheaper
  NO_DATA_NEEDED     analytic/theorem claim (geometry, scaling law): the
                     'dataset' is a proof or a textbook table
  NOT_TESTABLE       no dataset can settle it as stated
  NO_PUBLIC_FOUND    no local data and no relevant public hit at scout time;
                     human must source data (CT access, fossils, experiment)

Automation tiers (field `automation`):
  FULL_LOOP      loop can run scout->fetch->fit->audit unaided
  LOOP_WITH_SEED loop can run once a human (or a first loop pass) picks the
                 dataset / species list -- a single decision, then automated
  HUMAN_REQUIRED dataset creation itself needs a human (new CT, wet lab,
                 fossil preparation, licence negotiation)
  DONE           already validated in repo; loop would only re-audit
  NA             not testable / analytic

Outputs (all regenerated, never hand-edited):
  thompson/thompson_1917_dataset_audit.json   full probe results + verdicts
  docs/THOMPSON_1917_DATASET_EXISTENCE.md     human-readable table
  thompson/thompson_1917_loop_queue.yaml      campaign-compatible slots for
                                              every FULL_LOOP / LOOP_WITH_SEED
                                              claim (feeds
                                              morphoeval/cloud_orchestrator.py's
                                              build_cloud_agent_prompt)

Usage:
    python3 thompson/scout_1917_datasets.py                 # local + live probes
    python3 thompson/scout_1917_datasets.py --offline       # local only, fast
    python3 thompson/scout_1917_datasets.py --only T1917-051,T1917-056
    venv/bin/python thompson/scout_1917_datasets.py --prompt T1917-051   # print loop prompt (venv: pyyaml)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = Path(__file__).with_name("thompson_1917_claims.json")
AUDIT_JSON = Path(__file__).with_name("thompson_1917_dataset_audit.json")
QUEUE_YAML = Path(__file__).with_name("thompson_1917_loop_queue.yaml")
OUT_MD = ROOT / "docs" / "THOMPSON_1917_DATASET_EXISTENCE.md"
UA = {"User-Agent": "infinitemorphospace/1.0 (Thompson 1917 dataset scout)", "Accept": "application/json"}

# ---------------------------------------------------------------------------
# Per-claim scouting seeds. query = Dryad dataset search; commons = Wikimedia
# Commons specimen search; local = keywords matched against repo path names;
# kind = data the test actually consumes; human = blocker if HUMAN_REQUIRED.
# Claims not listed fall back to a query derived from the claim text.
# ---------------------------------------------------------------------------
SEEDS: dict[str, dict] = {
    "T1917-001": {"kind": "none (programme)", "automation": "NA"},
    "T1917-002": {"kind": "roll-up of descendants", "automation": "NA"},
    "T1917-003": {"kind": "analytic + size tables", "automation": "FULL_LOOP", "query": "body size scaling surface volume allometry"},
    "T1917-004": {"kind": "bone dimensions + mass table", "automation": "LOOP_WITH_SEED", "query": "limb bone scaling body mass mammals", "local": ["bone", "limb"]},
    "T1917-005": {"kind": "organism size/physics table (literature)", "automation": "FULL_LOOP", "query": "body size locomotion Reynolds number"},
    "T1917-006": {"kind": "cell size vs body size table", "automation": "LOOP_WITH_SEED", "query": "cell size body size neuron"},
    "T1917-007": {"kind": "growth time series (his figures + modern)", "automation": "FULL_LOOP", "query": "growth curve von Bertalanffy length weight time series", "local": ["growth"]},
    "T1917-008": {"kind": "organ/part size vs body size", "automation": "FULL_LOOP", "query": "relative growth allometry organ size", "local": ["antscan", "pheidole", "allometr"]},
    "T1917-009": {"kind": "growth vs temperature/season series", "automation": "FULL_LOOP", "query": "tree ring growth climate time series dendrochronology"},
    "T1917-010": {"kind": "regeneration time-series morphometrics", "automation": "LOOP_WITH_SEED", "query": "regeneration tail morphometrics growth", "local": ["planarian"]},
    "T1917-011": {"kind": "spindle imaging (mechanism, not shape)", "automation": "HUMAN_REQUIRED", "human": "Mechanistic claim; needs imaging + perturbation, not a shape dataset"},
    "T1917-012": {"kind": "embryo cleavage imaging", "automation": "LOOP_WITH_SEED", "query": "embryo cleavage geometry cell division", "local": ["embryo"]},
    "T1917-013": {"kind": "isolated cell shape images", "automation": "LOOP_WITH_SEED", "query": "cell shape morphology segmentation", "local": ["vertex_packing"]},
    "T1917-014": {"kind": "protist/foram outlines (his figures + modern images)", "automation": "FULL_LOOP", "query": "foraminifera shell morphology outline", "commons": "Lagena foraminifera", "local": ["foram"]},
    "T1917-015": {"kind": "images of beading strands (analytic criterion)", "automation": "FULL_LOOP", "query": "protoplasm strand droplet breakup"},
    "T1917-016": {"kind": "cell shape vs osmolarity", "automation": "HUMAN_REQUIRED", "human": "Needs the osmolarity experiment or a very specific public replicate"},
    "T1917-017": {"kind": "none (too general)", "automation": "NA"},
    "T1917-018": {"kind": "segmented epithelium images", "automation": "LOOP_WITH_SEED", "query": "epithelium cell segmentation junction angle", "local": ["vertex_packing", "drosophila"]},
    "T1917-019": {"kind": "segmented epithelium images (same as T1917-018)", "automation": "LOOP_WITH_SEED", "query": "epithelial cell packing polygon", "local": ["vertex_packing", "drosophila"]},
    "T1917-020": {"kind": "plant meristem imaging/segmentation", "automation": "LOOP_WITH_SEED", "query": "shoot apical meristem cell division segmentation"},
    "T1917-021": {"kind": "analytic (geometry) + comb photos", "automation": "FULL_LOOP", "commons": "honeycomb Apis mellifera"},
    "T1917-022": {"kind": "per-structure images (split first)", "automation": "LOOP_WITH_SEED", "query": "radiolarian skeleton morphology", "commons": "Radiolaria Haeckel"},
    "T1917-023": {"kind": "spherite/concretion images (his figures + modern)", "automation": "LOOP_WITH_SEED", "query": "concretion spherulite radial growth", "local": ["biomineral"]},
    "T1917-024": {"kind": "spicule images/SEM with ray angles", "automation": "LOOP_WITH_SEED", "query": "sponge spicule morphology", "commons": "sponge spicule"},
    "T1917-025": {"kind": "radiolarian lattice images (Haeckel, public domain)", "automation": "LOOP_WITH_SEED", "commons": "Radiolaria Haeckel Aulonia", "local": ["acanthometra", "radiolarian"]},
    "T1917-026": {"kind": "one 3D surface + curve traces", "automation": "HUMAN_REQUIRED", "human": "Needs a chosen 3D system with surface curves; no generic dataset"},
    "T1917-027": {"kind": "shell outlines (Commons/museum + his figures)", "automation": "FULL_LOOP", "query": "mollusc shell shape geometric morphometrics", "commons": "Nautilus pompilius shell section", "local": ["raup", "nautilus", "conus"]},
    "T1917-028": {"kind": "Nautilus shell section image", "automation": "FULL_LOOP", "commons": "Nautilus shell section septa", "local": ["nautilus"]},
    "T1917-029": {"kind": "foram chamber measurements (Brady plates + modern)", "automation": "FULL_LOOP", "query": "foraminifera chamber shape morphometrics", "commons": "Foraminifera Brady Challenger", "local": ["foram"]},
    "T1917-030": {"kind": "Lagena outlines (bundle T1917-014)", "automation": "FULL_LOOP", "commons": "Lagena foraminifera", "local": ["foram"]},
    "T1917-031": {"kind": "horn 3D centrelines (CT/photos)", "automation": "LOOP_WITH_SEED", "query": "bovid horn shape morphology", "commons": "ram horn Ovis", "local": ["horn"]},
    "T1917-032": {"kind": "tusk/tooth shapes (bundle T1917-031)", "automation": "LOOP_WITH_SEED", "query": "tusk tooth shape morphometrics", "local": ["tusk", "tooth"]},
    "T1917-033": {"kind": "capitulum/stem images (measure divergence)", "automation": "FULL_LOOP", "query": "phyllotaxis divergence angle sunflower", "commons": "sunflower capitulum phyllotaxis", "local": ["phyllotaxis", "vogel", "douady"]},
    "T1917-034": {"kind": "egg outline images", "automation": "FULL_LOOP", "query": "bird egg shape morphometrics", "commons": "bird egg museum"},
    "T1917-035": {"kind": "urchin test profiles (split first)", "automation": "LOOP_WITH_SEED", "query": "sea urchin test shape morphology", "commons": "sea urchin test"},
    "T1917-036": {"kind": "analytic (beam theory)", "automation": "FULL_LOOP"},
    "T1917-037": {"kind": "femur (micro-)CT + FE stress field", "automation": "HUMAN_REQUIRED", "human": "Needs licensed femur CT + a pre-specified FE model; MorphoSource download is scriptable but specimen/model choice is a human seed decision that changes the verdict", "query": "femur trabecular bone micro-CT"},
    "T1917-038": {"kind": "vascular/tree branching measurements", "automation": "LOOP_WITH_SEED", "query": "vascular branching Murray law diameter"},
    "T1917-039": {"kind": "roll-up of T1917-040..058", "automation": "NA"},
    "T1917-040": {"kind": "head profile landmarks + his Durer grids", "automation": "LOOP_WITH_SEED", "query": "human facial profile anthropometric landmarks"},
    "T1917-041": {"kind": "leaf outline images (herbaria/Commons)", "automation": "FULL_LOOP", "query": "leaf shape outline morphometrics", "commons": "leaf outline lanceolate cordate", "local": ["leaf", "antillean"]},
    "T1917-042": {"kind": "IN_REPO (drawings) + copepod specimen images", "automation": "FULL_LOOP", "commons": "Oithona copepod", "local": ["crabs", "copepod"]},
    "T1917-043": {"kind": "IN_REPO (drawings + crab cache) + carapace scans", "automation": "FULL_LOOP", "query": "crab carapace geometric morphometrics", "local": ["crabs"]},
    "T1917-044": {"kind": "IN_REPO (1917 plate) + amphipod images", "automation": "LOOP_WITH_SEED", "commons": "Hyperia amphipod", "local": ["crabs"]},
    "T1917-045": {"kind": "IN_REPO", "automation": "FULL_LOOP", "local": ["thompson_1917_exact_species", "trimmed_tps"]},
    "T1917-046": {"kind": "IN_REPO", "automation": "DONE", "local": ["trimmed_tps"]},
    "T1917-047": {"kind": "IN_REPO (existing fits; re-run saving constants)", "automation": "FULL_LOOP", "local": ["trimmed_tps"]},
    "T1917-048": {"kind": "IN_REPO", "automation": "FULL_LOOP", "local": ["thompson_1917_exact_species"]},
    "T1917-049": {"kind": "IN_REPO", "automation": "FULL_LOOP", "local": ["thompson_1917_exact_species"]},
    "T1917-050": {"kind": "IN_REPO (extension, not a 1917 claim)", "automation": "DONE", "local": ["trimmed_tps", "fishshapes"]},
    "T1917-051": {"kind": "crocodylian cranial landmarks", "automation": "FULL_LOOP", "query": "crocodylian cranial ontogeny geometric morphometrics", "known_doi": "10.5061/dryad.14fn1", "local": ["crocod"]},
    "T1917-052": {"kind": "fossil girdle figures/reconstructions", "automation": "HUMAN_REQUIRED", "human": "Fossil reconstruction uncertainty; data must be assembled and judged specimen-by-specimen"},
    "T1917-053": {"kind": "pterosaur skull landmarks (needs clade frame)", "automation": "LOOP_WITH_SEED", "query": "pterosaur skull geometric morphometrics"},
    "T1917-054": {"kind": "bird pelvis landmarks (literature assembly)", "automation": "LOOP_WITH_SEED", "query": "bird pelvis shape morphometrics Archaeopteryx"},
    "T1917-055": {"kind": "ungulate skull landmarks", "automation": "LOOP_WITH_SEED", "query": "rhinoceros ungulate skull geometric morphometrics", "local": ["leporidae"]},
    "T1917-056": {"kind": "equid skull landmark series incl. fossils", "automation": "LOOP_WITH_SEED", "query": "horse skull evolution geometric morphometrics", "known_query_title": "horses deviate"},
    "T1917-057": {"kind": "scapula outlines", "automation": "LOOP_WITH_SEED", "query": "scapula shape morphometrics human"},
    "T1917-058": {"kind": "primate cranial landmarks", "automation": "LOOP_WITH_SEED", "query": "primate skull cranial landmarks geometric morphometrics", "local": ["leporidae"]},
    "T1917-059": {"kind": "none (restatement)", "automation": "NA"},
}

STOP = {"the", "a", "an", "of", "and", "or", "in", "on", "to", "from", "data", "for"}


CACHE_PATH = ROOT / ".thompson_cache" / "claim_inventory" / "dataset_scout_cache.json"
_CACHE: dict = {}


def load_cache() -> None:
    global _CACHE
    if CACHE_PATH.exists():
        try:
            _CACHE = json.loads(CACHE_PATH.read_text())
        except Exception:
            _CACHE = {}


def save_cache() -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(_CACHE, indent=1) + "\n")


def http_json(url: str, timeout: float = 15.0) -> dict | None:
    """None means UNPROBED (network/rate-limit failure) -- never 'no results'.

    Dryad rate-limits aggressive clients (observed: HTTP 429 after ~50 rapid
    queries), so callers must cache and throttle; see dryad_search().
    """
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", errors="replace"))
    except Exception:
        return None  # includes Dryad HTTP 429: report UNPROBED, let the loop re-probe later


def dryad_search(query: str, per_page: int = 5) -> list[dict]:
    """Dryad API v2 dataset search; keep only title-relevant hits.

    Dryad ANDs long queries to zero, so simplify progressively:
    full query -> drop generic morphometrics words -> first 3 content words.
    """
    import time

    ckey = f"dryad::{query}"
    if ckey in _CACHE:
        return _CACHE[ckey]
    generic = {"geometric", "morphometrics", "morphometric", "landmarks", "landmark", "shape", "analysis"}
    words_q = [w for w in re.findall(r"[A-Za-z]{3,}", query) if w.lower() not in STOP]
    attempts = [query,
                " ".join(w for w in words_q if w.lower() not in generic),
                " ".join(words_q[:3])]
    d = None
    probed = False
    for q in dict.fromkeys(a for a in attempts if a.strip()):
        url = "https://datadryad.org/api/v2/search?q=" + urllib.parse.quote(q) + f"&per_page={per_page}"
        d = http_json(url)
        time.sleep(1.2)  # Dryad throttling: ~50 rapid queries earned a 429 (observed 2026-10-26)
        if d is not None:
            probed = True
            if (d.get("_embedded", {}) or {}).get("stash:datasets"):
                break
    if not probed:
        return None  # UNPROBED: do not cache, do not call it 'no dataset'
    words = {w for w in re.findall(r"[a-z]{4,}", query.lower()) if w not in STOP}
    out = []
    for ds in (d.get("_embedded", {}) or {}).get("stash:datasets", []) or []:
        title = ds.get("title", "")
        tw = {w for w in re.findall(r"[a-z]{4,}", title.lower()) if w not in STOP}
        overlap = len(words & tw)
        out.append({"title": title, "identifier": ds.get("identifier"), "title_overlap": overlap})
    _CACHE[ckey] = out
    return out


def commons_search(taxon: str, limit: int = 4) -> list[dict] | None:
    ckey = f"commons::{taxon}"
    if ckey in _CACHE:
        return _CACHE[ckey]
    q = urllib.parse.quote(f"{taxon} filetype:bitmap")
    url = (f"https://commons.wikimedia.org/w/api.php?action=query&generator=search&gsrsearch={q}"
           f"&gsrnamespace=6&gsrlimit={limit}&prop=imageinfo&iiprop=url|extmetadata&format=json")
    d = http_json(url)
    if d is None:
        return None  # UNPROBED
    out = []
    for pg in ((d.get("query", {}) or {}).get("pages", {}) or {}).values():
        ii = (pg.get("imageinfo") or [{}])[0]
        if ii.get("url"):
            lic = ((ii.get("extmetadata") or {}).get("LicenseShortName") or {}).get("value", "?")
            out.append({"title": pg.get("title"), "url": ii["url"], "license": lic})
    _CACHE[ckey] = out
    return out


def local_index() -> list[str]:
    paths: list[str] = []
    for base in ("datasets", ".thompson_cache", "thompson/portfolio", "models"):
        p = ROOT / base
        if p.exists():
            paths += [str(x.relative_to(ROOT)).lower() for x in p.rglob("*") if x.is_file()]
    return paths


def local_hits(keywords: list[str], index: list[str]) -> list[str]:
    hits = []
    for kw in keywords:
        hits += [h for h in index if kw.lower() in h][:3]
    return sorted(set(hits))[:8]


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:48]


def classify(claim: dict, seed: dict, local: list[str], dryad: list[dict] | None, commons: list[dict] | None) -> tuple[str, str, str]:
    """Returns (existence, automation, reason). Live evidence beats priors.

    dryad/commons = None means UNPROBED (network or Dryad 429 rate-limit):
    that must never be reported as 'no dataset exists'.
    """
    unprobed = dryad is None or commons is None
    dryad, commons = dryad or [], commons or []
    status = claim["pass2"]["status"]
    auto = seed.get("automation", "LOOP_WITH_SEED")
    if auto == "NA" or status == "NOT_TESTABLE_AS_STATED":
        return "NOT_TESTABLE", "NA", "Not testable as stated; no dataset settles it (operationalise first)."
    if seed.get("human"):
        return ("PUBLIC_CANDIDATE" if dryad else "NO_PUBLIC_FOUND"), "HUMAN_REQUIRED", seed["human"]
    if status == "DONE_REPO":
        # DONE only where the seed marks the claim itself closed (T1917-046/050);
        # DONE_REPO claims whose seed is FULL_LOOP still have loop-doable
        # remaining work (real-specimen outlines), so queue them.
        if auto == "DONE":
            return "IN_REPO", "DONE", "Already validated in repo on data on disk; closed."
        return "IN_REPO", "FULL_LOOP", "Core test done in repo; remaining upgrade (real-specimen outlines) is loop-doable on in-repo data."
    relevant = [d for d in dryad if d["title_overlap"] >= 1]
    if local and status in ("PARTIAL_REPO", "UNTESTED", "DONE_ELSEWHERE"):
        # local material exists but is scout/analogue material, not the claim's own test
        if auto == "FULL_LOOP":
            return "IN_REPO", "FULL_LOOP", "Usable material already on disk (scout cache / portfolio / models); loop can ingest it directly."
    if relevant:
        return "PUBLIC_SCRIPTABLE", auto, f"Dryad live search returned {len(relevant)} title-relevant dataset(s), top: {relevant[0]['identifier']}."
    if dryad or commons:
        return "PUBLIC_CANDIDATE", "LOOP_WITH_SEED", "Public hits exist but title relevance is weak; loop Step 1 must vet species/variables before fetch."
    if "his figures" in seed.get("kind", "") or "1917" in seed.get("kind", "") or commons:
        return "PUBLIC_SCRIPTABLE", auto, "Thompson's own 1917 figures / public-domain plates are fetchable from the pinned IA scan; loop digitises them."
    if "analytic" in seed.get("kind", ""):
        return "NO_DATA_NEEDED", "FULL_LOOP", "Analytic claim: proof/textbook check, no empirical dataset required."
    if commons:
        return "PUBLIC_SCRIPTABLE", auto, "Wikimedia Commons specimen images found live (CC/public domain)."
    if unprobed:
        return "UNPROBED", "LOOP_WITH_SEED", "Live probe failed or Dryad rate-limited (HTTP 429) at scout time -- existence UNKNOWN, not absent. Loop Step 1 re-probes with throttling/cache; do not quote this row as 'no dataset'."
    return "NO_PUBLIC_FOUND", ("HUMAN_REQUIRED" if auto == "HUMAN_REQUIRED" else "LOOP_WITH_SEED"), "No local data and no relevant Dryad hit at scout time; loop Step 1 (OpenAlex/Commons) is the next probe, else human sourcing."


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline", action="store_true", help="local scan only, no Dryad/Commons probes")
    ap.add_argument("--only", type=str, default=None, help="comma-separated claim IDs")
    ap.add_argument("--prompt", type=str, default=None, help="print the cloud-loop prompt for one claim ID and exit")
    args = ap.parse_args()

    reg = json.loads(REGISTRY.read_text())
    claims = reg["claims"]
    if args.only:
        want = {x.strip() for x in args.only.split(",")}
        claims = [c for c in claims if c["id"] in want]

    if args.prompt:
        return print_prompt(args.prompt)

    load_cache()
    index = local_index()
    rows = []
    for c in claims:
        seed = SEEDS.get(c["id"], {"kind": c["pass2"]["data_needed"], "query": c["claim"][:80]})
        local = local_hits(seed.get("local", []), index)
        dryad: list | None = None  # None = UNPROBED (offline, or probe failure)
        commons: list | None = None
        if not args.offline and seed.get("automation") not in ("NA", "DONE"):
            if seed.get("query"):
                dryad = dryad_search(seed["query"])
            if seed.get("commons"):
                commons = commons_search(seed["commons"])
        existence, automation, reason = classify(c, seed, local, dryad, commons)
        # Thompson's figures are always a fallback dataset for figure claims
        digitize = bool(re.search(r"\d", c["figures"]))
        rows.append({
            "id": c["id"], "date_made": "1917",
            "citation": f"Thompson (1917), Ch. {c['chapter']}, pp. {c['pages']}, Figs. {c['figures']}",
            "claim": c["claim"], "pass2_status": c["pass2"]["status"],
            "data_kind_needed": seed.get("kind"), "existence": existence, "automation": automation,
            "reason": reason, "local_hits": local,
            "dryad_hits": dryad, "commons_hits": commons,
            "digitize_1917_figures": digitize,
            "seed_query": seed.get("query"), "human_blocker": seed.get("human"),
            "effort_days": c["pass2"]["effort_days"],
        })
        print(f"{c['id']}: {existence:<18} {automation:<15} dryad={len(dryad) if dryad is not None else 'UNPROBED'} commons={len(commons) if commons is not None else 'UNPROBED'} local={len(local)}")

    audit = {
        "scouted_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "offline": args.offline,
        "source_registry": "thompson/thompson_1917_claims.json (all claims: Thompson 1917, 1st ed.)",
        "probes": ["local path scan of datasets/, .thompson_cache/, thompson/portfolio/, models/",
                   "Dryad API v2 https://datadryad.org/api/v2/search (live, title-overlap filtered)",
                   "Wikimedia Commons API (live, where a commons seed is set)",
                   "fallback for every figure claim: digitise Thompson's own 1917 plates from the pinned IA scan"],
        "rows": rows,
    }
    save_cache()
    AUDIT_JSON.write_text(json.dumps(audit, indent=2) + "\n")
    write_queue(rows, reg)
    OUT_MD.write_text(render_md(rows, audit))
    from collections import Counter
    print("\nexistence:", dict(Counter(r["existence"] for r in rows)))
    print("automation:", dict(Counter(r["automation"] for r in rows)))
    print(f"wrote {AUDIT_JSON.relative_to(ROOT)}, {QUEUE_YAML.relative_to(ROOT)}, {OUT_MD.relative_to(ROOT)}")
    return 0


def queue_entry(i: int, r: dict, claim: dict) -> dict:
    slug = f"t1917-{r['id'].split('-')[1]}-" + slugify(claim["chapter_title"])[:30]
    return {
        "slot": 200 + i,
        "slot_id": f"thompson1917-{r['id'].lower()}",
        "claim_id": r["id"],
        "date_made": "1917",
        "domain": f"Thompson 1917 claim validation — Ch. {claim['chapter']}",
        "organism": (r.get("seed_query") or claim["chapter_title"]),
        "math_family": claim["type"],
        "search_query": r.get("seed_query") or claim["claim"][:100],
        "seed_paper": f"Thompson, D.W. (1917) On Growth and Form, Ch. {claim['chapter']}, pp. {claim['pages']}, Figs. {claim['figures']}",
        "seed_doi": None,
        "governing_mechanism": claim["claim"],
        "dataset_existence": r["existence"],
        "dataset_audit_reason": r["reason"],
        "proposed_slug": slug,
        "proposed_sim_id": f"thompson1917-{r['id'].lower()}",
        "proposed_script": f"thompson/claims/{r['id'].lower()}_{slugify(claim['chapter_title'])}.py",
        "proposed_dataset_dir": f"datasets/thompson_1917/{r['id'].lower()}",
        "proposed_lean_module": f"morphoeval/lean/loops/Thompson1917_{r['id'].replace('-', '_')}.lean",
    }


def write_queue(rows: list[dict], reg: dict) -> None:
    try:
        import yaml
    except ImportError:
        QUEUE_YAML.write_text("# pyyaml unavailable; queue not written\n")
        return
    by_id = {c["id"]: c for c in reg["claims"]}
    entries = [queue_entry(i, r, by_id[r["id"]]) for i, r in enumerate(
        [r for r in rows if r["automation"] in ("FULL_LOOP", "LOOP_WITH_SEED")], start=1)]
    doc = {
        "schema_version": "thompson-1917-loop-queue-1.0",
        "source": "Thompson (1917) On Growth and Form, 1st edition — every claim dated 1917",
        "registry": "thompson/thompson_1917_claims.json",
        "dataset_audit": "thompson/thompson_1917_dataset_audit.json",
        "usage": "Feed one entry to morphoeval/cloud_orchestrator.build_cloud_agent_prompt(entry), or run: venv/bin/python thompson/scout_1917_datasets.py --prompt T1917-051",
        "isolation": "Loops write ONLY new files: thompson/claims/<claim>.py, datasets/thompson_1917/<claim>/, morphoeval audits/kg for that claim. They must NOT edit thompson_1917_claims.json or the generated docs; a human merges verdicts back into the registry.",
        "total_slots": len(entries),
        "slots": entries,
    }
    QUEUE_YAML.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))


def print_prompt(claim_id: str) -> int:
    try:
        import yaml
    except ImportError:
        print("pyyaml required for --prompt", file=sys.stderr)
        return 1
    if not QUEUE_YAML.exists():
        print("queue not generated yet; run without --prompt first", file=sys.stderr)
        return 1
    doc = yaml.safe_load(QUEUE_YAML.read_text())
    entry = next((s for s in doc["slots"] if s["claim_id"] == claim_id), None)
    if entry is None:
        print(f"{claim_id} is not loop-automatable (see dataset audit) — no prompt generated", file=sys.stderr)
        return 1
    sys.path.insert(0, str(ROOT / "morphoeval"))
    import cloud_orchestrator  # noqa: E402

    prompt = cloud_orchestrator.build_cloud_agent_prompt(entry)
    prompt += f"""

---

## THOMPSON (1917) CLAIM OVERRIDE — READ FIRST
This slot is NOT a new-model scout. It validates one dated claim:
**{entry['seed_paper']}** — "{entry['governing_mechanism']}"
Dataset existence was pre-scouted as **{entry['dataset_existence']}**: {entry['dataset_audit_reason']}
- Step 1 is therefore a *verification*, not a search: fetch exactly the dataset named in
  `thompson/thompson_1917_dataset_audit.json` for {claim_id} (or Thompson's own 1917 figures
  from the pinned IA scan `ongrowthform1917thom`, printed page = leaf - 20), SHA-256-pin it,
  and record in your scout manifest why it does or does not match the claim's species/variables.
- If the dataset does not match the claim, STOP and report `DATASET_MISMATCH` in your result
  JSON instead of fitting an analogue and claiming a verdict. Analogues must be labelled as such.
- Write your verdict to `datasets/thompson_1917/{claim_id.lower()}/result.json` with fields:
  claim_id, date_made="1917", dataset (name/DOI/SHA-256), verdict (supported|refuted|mixed|untestable),
  headline_numbers, script, figure. Do NOT edit `thompson/thompson_1917_claims.json`.
"""
    print(prompt)
    return 0


def render_md(rows: list[dict], audit: dict) -> str:
    from collections import Counter

    ex, au = Counter(r["existence"] for r in rows), Counter(r["automation"] for r in rows)
    auto = [r for r in rows if r["automation"] in ("FULL_LOOP", "LOOP_WITH_SEED")]
    human = [r for r in rows if r["automation"] == "HUMAN_REQUIRED"]
    lines = [
        "# Thompson (1917) — Do the Datasets Exist, and Can the Loop Run Them?",
        "",
        "Companion to [`THOMPSON_1917_CLAIM_INVENTORY.md`](THOMPSON_1917_CLAIM_INVENTORY.md). Every claim is",
        "**Thompson (1917)**, *On Growth and Form*, 1st edition. Generated by",
        "[`thompson/scout_1917_datasets.py`](../thompson/scout_1917_datasets.py) from",
        "[`thompson/thompson_1917_claims.json`](../thompson/thompson_1917_claims.json); full probe output in",
        "[`thompson/thompson_1917_dataset_audit.json`](../thompson/thompson_1917_dataset_audit.json).",
        f"Scouted {audit['scouted_at']} (offline={audit['offline']}). Probes: local repo scan, Dryad API v2 live search",
        "(title-overlap filtered), Wikimedia Commons live search, and — for every figure claim — Thompson's own",
        "1917 plates from the pinned Internet Archive scan as a digitisable fallback dataset.",
        "",
        f"**Existence:** " + ", ".join(f"{k} {v}" for k, v in sorted(ex.items())),
        f"**Automation:** " + ", ".join(f"{k} {v}" for k, v in sorted(au.items())),
        "",
        f"**Bottom line: {len(auto)}/{len(rows)} claims can be fed to the loop now** "
        f"({au['FULL_LOOP']} fully unaided, {au['LOOP_WITH_SEED']} after one dataset/species pick), "
        f"{au['DONE']} are already done, and only {len(human)} genuinely need a human first: "
        + ", ".join(r["id"] for r in human) + ".",
        "For every figure claim there is *always* at least the DIGITIZE_1917 fallback: his plates are public domain,",
        "already fetched for Ch. XVII (`.thompson_cache/ongrowthform1917thom_*.jp2`), and the repo already has the",
        "segmentation/landmarking pipeline that turned them into the fish/arthropod tests. 'No dataset' therefore",
        "usually means 'no *modern* dataset' — the 1917-as-data test is automatable today.",
        "",
        "## How the loop consumes this",
        "",
        "```bash",
        "python3 thompson/scout_1917_datasets.py            # refresh existence audit + queue",
        "venv/bin/python thompson/scout_1917_datasets.py --prompt T1917-051   # full 12-step cloud-loop prompt (venv python: pyyaml)",
        "# queue for batch dispatch: thompson/thompson_1917_loop_queue.yaml (campaign-compatible slots)",
        "```",
        "",
        "Each queue slot carries the 1917 citation, the pre-scouted dataset verdict, and an override that (a) makes",
        "Step 1 a verification of the named dataset rather than an open search, (b) forces a `DATASET_MISMATCH` stop",
        "instead of silently fitting an analogue, and (c) writes the verdict to",
        "`datasets/thompson_1917/<claim>/result.json` for human merge-back into the registry.",
        "",
        "## Per-claim dataset existence",
        "",
        "| ID | Claim (Thompson 1917) | Data needed | Exists? | Loop? | Best evidence found |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        ev = (r["dryad_hits"][0]["identifier"] if r["dryad_hits"] else None) or \
             (r["local_hits"][0] if r["local_hits"] else None) or \
             (r["commons_hits"][0]["title"] if r["commons_hits"] else None) or \
             ("1917 plates (IA scan)" if r["digitize_1917_figures"] else "—")
        lines.append(f"| {r['id']} | {r['citation']} | {r['data_kind_needed']} | `{r['existence']}` | `{r['automation']}` | {ev} — {r['reason'][:120]} |")
    lines += [
        "",
        "## What still needs a human, and why",
        "",
    ]
    for r in human:
        lines.append(f"- **{r['id']}** ({r['citation']}): {r['human_blocker']}")
    lines += [
        "",
        "## Honesty notes",
        "",
        "- `PUBLIC_SCRIPTABLE` means a *title-relevant* Dryad dataset was returned live at scout time; the loop still",
        "  verifies species/variable match before fitting (the `DATASET_MISMATCH` stop). It does not mean 'guaranteed fit'.",
        "- `PUBLIC_CANDIDATE` / weak hits are deliberately not upgraded: keyword search alone is not dataset existence.",
        "- Dryad file downloads can sit behind a browser challenge for some records (see `datasets/thompson/README.md`);",
        "  where that happens the loop must record it and fall back to the 1917-plates test rather than stalling.",
        "- Effort per claim is in the claim inventory; dataset existence removes the *sourcing* risk, not the fitting/audit work.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
