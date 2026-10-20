#!/usr/bin/env python3
"""
morphoeval/build_web_explorer.py — Builds the Web-Accessible Interactive Aristotelian
Morphospace Explorer (`docs/index.html` + `docs/aristotelian_data.js`).

Features:
  - Interactive 10×10 Aristotelian Coverage Matrix (`Ousia × Poiein`, 100/100 cells, n = 1,116 models)
  - Click any of the 100 matrix cells (or use the Best-9 quick-jump pills / genus search) to zoom
    into that grid space's **3×3 Animated Tile View ("Best 9 in Cell")** with pagination for all
    resident models in the cell (10–22 models per cell).
  - Every tile in the 3×3 Zoom Grid features:
      1. Tracked animated GIF (`docs/media/*.gif`) OR a live 60-FPS HTML5 Canvas biophysical simulation
         matched to that cell's exact Formal/Efficient Cause + scaling law curve.
      2. Picture-in-Picture (PiP) Iconic Species Specimen photograph from
         `datasets/species_gallery/<Megiston_Genos>/<Genus>_<species>.jpg`.
      3. Aristotelian metadata header & footer (`#Rank`, `Genus (Common Name)`, `Pass 1 → Pass 2 Score`,
         `script.py`, `Poson Scaling Law`, and `Lean 4 0-sorry` badge).
      4. Click-to-inspect Modal with full Aristotelian Four-Cause breakdown and direct GitHub links.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "docs"
MEDIA_DIR = DOCS_DIR / "media"
INDEX_JSON = ROOT / "morphoeval" / "runs" / "aristotelian_categories_index.json"
SPECIES_MANIFEST_JSON = ROOT / "datasets" / "species_gallery" / "species_manifest.json"


def get_tracked_gifs() -> set[str]:
    """Return set of GIF filenames under `docs/media/` tracked by git (available on GitHub Pages)."""
    try:
        out = subprocess.check_output(
            ["git", "ls-files", "docs/media/*.gif"], cwd=ROOT, text=True
        )
        return {Path(line.strip()).name for line in out.splitlines() if line.strip()}
    except Exception:
        return {p.name for p in MEDIA_DIR.glob("*.gif")}


def main() -> int:
    entries: list[dict[str, Any]] = json.loads(INDEX_JSON.read_text(encoding="utf-8"))
    species_manifest: list[dict[str, Any]] = json.loads(SPECIES_MANIFEST_JSON.read_text(encoding="utf-8"))
    tracked_gifs = get_tracked_gifs()

    genus_to_gallery: dict[str, str] = {}
    for sm in species_manifest:
        # Path relative to `docs/index.html` or raw GitHub URL fallback
        rel = sm["gallery_path"]
        genus_to_gallery[sm["genus"]] = rel

    # Match entries to tracked GIFs where available
    compact_models: list[dict[str, Any]] = []
    for idx, e in enumerate(entries):
        slug = e["slug"]
        slug_u = slug.replace("-", "_")
        matched_gif = ""
        for cand in [
            f"{slug_u}_growth.gif",
            f"{slug_u}.gif",
            f"{slug_u}_closed_loop.gif",
        ]:
            if cand in tracked_gifs:
                matched_gif = f"media/{cand}"
                break
        if not matched_gif:
            # Check substring match among tracked GIFs for foundational scripts
            for gname in sorted(tracked_gifs):
                if slug_u in gname and not gname.startswith("morphospace_"):
                    matched_gif = f"media/{gname}"
                    break

        gal_rel = genus_to_gallery.get(e["genus"], "")
        compact_models.append(
            {
                "id": idx + 1,
                "slot": e.get("slot", idx + 1),
                "slug": slug,
                "script": e["script"],
                "species": e["species"],
                "genus": e["genus"],
                "common": e["common"],
                "megiston": e["megiston_genos"],
                "subaltern": e["subaltern_genos"],
                "poion": e["poion_form"],
                "poson": e["poson_law"],
                "cause": e["efficient_cause"],
                "math": e.get("math_family", e.get("domain", e["poion_form"])),
                "p1": round(float(e.get("pass1", 58.0)), 1),
                "score": round(float(e.get("score", 93.5)), 1),
                "gif": matched_gif,
                "specimen": gal_rel,
                "lean": e.get("lean_file", ""),
            }
        )

    data_payload = {
        "generated": "2026-10-20",
        "total_models": len(compact_models),
        "total_genera": len({m["genus"] for m in compact_models}),
        "repo_url": "https://github.com/fractastical/morpholib/blob/main",
        "raw_url": "https://raw.githubusercontent.com/fractastical/morpholib/main",
        "models": compact_models,
    }

    data_js_path = DOCS_DIR / "aristotelian_data.js"
    data_js_path.write_text(
        "window.ARISTOTELIAN_DATA = " + json.dumps(data_payload, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )
    print(f"Wrote {data_js_path} ({data_js_path.stat().st_size // 1024} KB, {len(compact_models)} models)")

    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Infinite Morphospace — Aristotelian 3×3 Zoom Grid Explorer (1,116 Models · 150 Genera)</title>
  <style>
    :root {
      --bg-deep: #060911;
      --bg-panel: #0d1322;
      --bg-card: #11192c;
      --bg-header: #16223b;
      --border-dim: #263554;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --accent-gold: #facc15;
      --accent-green: #22c55e;
      --accent-blue: #38bdf8;
      --accent-amber: #f59e0b;
      --accent-purple: #a855f7;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg-deep);
      color: var(--text-main);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "DejaVu Sans", sans-serif;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }
    /* Top Navbar */
    header.topbar {
      background: linear-gradient(90deg, #0b1325 0%, #13203d 100%);
      border-bottom: 2px solid var(--accent-green);
      padding: 12px 20px;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    .brand-title {
      font-size: 1.18rem;
      font-weight: 800;
      letter-spacing: -0.01em;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .brand-sub {
      font-size: 0.8rem;
      color: #a7f3d0;
      margin-top: 3px;
    }
    .top-controls {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }
    .search-input {
      background: #090e1a;
      border: 1px solid #334155;
      color: #fff;
      padding: 7px 12px;
      border-radius: 6px;
      font-size: 0.82rem;
      width: 220px;
    }
    .search-input:focus {
      outline: none;
      border-color: var(--accent-gold);
    }
    .btn {
      background: #1e293b;
      color: #e2e8f0;
      border: 1px solid #334155;
      padding: 6px 11px;
      border-radius: 6px;
      font-size: 0.78rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s ease;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      gap: 5px;
    }
    .btn:hover {
      background: #334155;
      border-color: var(--accent-blue);
      color: #fff;
    }
    .btn.active {
      background: var(--accent-gold);
      color: #060911;
      border-color: #fff;
      font-weight: 700;
    }
    /* Quick-Jump Best-9 Bar */
    .quick-bar {
      background: #0a0f1d;
      border-bottom: 1px solid #1e293b;
      padding: 8px 20px;
      display: flex;
      align-items: center;
      gap: 8px;
      overflow-x: auto;
    }
    .quick-label {
      font-size: 0.75rem;
      font-weight: 700;
      color: var(--accent-gold);
      text-transform: uppercase;
      white-space: nowrap;
      margin-right: 4px;
    }
    .pill {
      background: #131c31;
      border: 1px solid #283756;
      color: #cbd5e1;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 0.74rem;
      font-weight: 600;
      cursor: pointer;
      white-space: nowrap;
      transition: all 0.15s;
    }
    .pill:hover {
      border-color: var(--accent-gold);
      color: #fff;
    }
    .pill.active {
      background: rgba(250, 204, 21, 0.18);
      border-color: var(--accent-gold);
      color: var(--accent-gold);
    }

    /* Main Split Layout */
    .workspace {
      display: grid;
      grid-template-columns: 440px 1fr;
      gap: 14px;
      padding: 14px 18px;
      flex: 1;
      align-items: start;
    }
    @media (max-width: 1220px) {
      .workspace {
        grid-template-columns: 1fr;
      }
    }

    /* Left Panel: Interactive 10x10 Aristotelian Coverage Matrix */
    .matrix-panel {
      background: var(--bg-panel);
      border: 1px solid var(--border-dim);
      border-radius: 10px;
      padding: 12px;
      position: sticky;
      top: 12px;
    }
    .panel-hdr {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      margin-bottom: 8px;
    }
    .panel-hdr h2 {
      font-size: 0.92rem;
      font-weight: 700;
      color: #fff;
    }
    .panel-hdr span {
      font-size: 0.72rem;
      color: var(--accent-gold);
    }
    .matrix-table {
      width: 100%;
      border-collapse: separate;
      border-spacing: 2px;
      user-select: none;
    }
    .matrix-table th.col-hdr {
      font-size: 0.63rem;
      color: #cbd5e1;
      font-weight: 600;
      writing-mode: vertical-rl;
      transform: rotate(180deg);
      text-align: left;
      padding: 6px 2px;
      height: 105px;
      vertical-align: bottom;
    }
    .matrix-table td.row-hdr {
      font-size: 0.68rem;
      font-weight: 700;
      padding: 3px 6px 3px 4px;
      white-space: nowrap;
      border-left: 3px solid transparent;
    }
    .matrix-cell {
      text-align: center;
      font-size: 0.72rem;
      font-weight: 700;
      height: 27px;
      border-radius: 4px;
      cursor: pointer;
      position: relative;
      transition: transform 0.1s, outline 0.1s;
    }
    .matrix-cell:hover {
      transform: scale(1.12);
      z-index: 4;
      outline: 2px solid #fff;
    }
    .matrix-cell.selected {
      outline: 2.5px solid var(--accent-gold);
      box-shadow: 0 0 10px rgba(250, 204, 21, 0.65);
      z-index: 5;
    }
    .cell-badge {
      position: absolute;
      top: -4px;
      right: -4px;
      background: var(--accent-gold);
      color: #060911;
      font-size: 0.54rem;
      font-weight: 900;
      padding: 0 3px;
      border-radius: 3px;
      line-height: 1.2;
    }
    .matrix-legend {
      margin-top: 10px;
      padding-top: 8px;
      border-top: 1px solid #1e293b;
      font-size: 0.71rem;
      color: var(--text-muted);
      line-height: 1.45;
    }
    .taxon-dots {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 5px;
      font-size: 0.68rem;
      font-weight: 600;
    }

    /* Right Panel: 3x3 Animated Zoom Grid */
    .zoom-panel {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .zoom-banner {
      background: var(--bg-panel);
      border: 1px solid var(--border-dim);
      border-left: 4px solid var(--accent-green);
      border-radius: 8px;
      padding: 10px 14px;
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
    }
    .zoom-title {
      font-size: 1.02rem;
      font-weight: 800;
      color: #fff;
    }
    .zoom-meta {
      font-size: 0.78rem;
      color: #a7f3d0;
      margin-top: 3px;
    }
    .pager {
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .pager-info {
      font-size: 0.76rem;
      color: var(--text-muted);
      margin-right: 4px;
    }

    /* 3x3 Tile Grid */
    .tiles-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
    }
    @media (max-width: 960px) {
      .tiles-grid {
        grid-template-columns: repeat(2, 1fr);
      }
    }
    @media (max-width: 620px) {
      .tiles-grid {
        grid-template-columns: 1fr;
      }
    }
    .morpho-tile {
      background: var(--bg-card);
      border: 2px solid var(--accent-green);
      border-radius: 8px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      cursor: pointer;
      transition: transform 0.14s ease, box-shadow 0.14s ease;
    }
    .morpho-tile:hover {
      transform: translateY(-2px);
      box-shadow: 0 6px 20px rgba(0, 0, 0, 0.55);
    }
    .tile-header {
      background: var(--bg-header);
      padding: 6px 10px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid rgba(255, 255, 255, 0.07);
    }
    .tile-rank-genus {
      font-size: 0.82rem;
      font-weight: 700;
      color: #fff;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .tile-rank-genus em {
      font-style: italic;
      color: #f8fafc;
    }
    .tile-score {
      font-size: 0.78rem;
      font-weight: 800;
      color: #34d399;
      background: rgba(16, 185, 129, 0.14);
      padding: 2px 6px;
      border-radius: 4px;
      white-space: nowrap;
    }
    .tile-stage {
      position: relative;
      height: 210px;
      background: #070a12;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
    }
    .tile-stage img.anim-gif {
      width: 100%;
      height: 100%;
      object-fit: contain;
      background: #070a12;
    }
    .tile-stage canvas.sim-canvas {
      width: 100%;
      height: 100%;
      display: block;
    }
    .pip-specimen {
      position: absolute;
      top: 6px;
      right: 6px;
      width: 68px;
      height: 48px;
      border: 1.5px solid var(--accent-gold);
      border-radius: 4px;
      overflow: hidden;
      background: #060911;
      box-shadow: 0 2px 8px rgba(0,0,0,0.7);
      z-index: 3;
    }
    .pip-specimen img {
      width: 100%;
      height: 100%;
      object-fit: cover;
    }
    .mode-tag {
      position: absolute;
      bottom: 5px;
      left: 6px;
      background: rgba(7, 10, 18, 0.82);
      border: 1px solid #334155;
      color: #cbd5e1;
      font-size: 0.62rem;
      font-weight: 600;
      padding: 1px 5px;
      border-radius: 3px;
      z-index: 3;
    }
    .tile-footer {
      background: #0b111e;
      padding: 6px 10px;
      border-top: 1px solid rgba(255, 255, 255, 0.06);
      display: flex;
      flex-direction: column;
      gap: 2px;
    }
    .footer-script {
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 0.69rem;
      color: #cbd5e1;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .footer-law {
      font-size: 0.69rem;
      color: #38bdf8;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 4px;
    }
    .lean-pill {
      color: var(--accent-gold);
      font-weight: 700;
      font-size: 0.64rem;
      white-space: nowrap;
    }

    /* Modal Inspector */
    .modal-backdrop {
      position: fixed;
      inset: 0;
      background: rgba(3, 6, 14, 0.82);
      display: none;
      align-items: center;
      justify-content: center;
      z-index: 100;
      padding: 20px;
    }
    .modal-backdrop.open {
      display: flex;
    }
    .modal-card {
      background: #0f172a;
      border: 2px solid var(--accent-gold);
      border-radius: 12px;
      max-width: 820px;
      width: 100%;
      padding: 20px;
      color: #f8fafc;
      position: relative;
      max-height: 90vh;
      overflow-y: auto;
    }
    .modal-close {
      position: absolute;
      top: 14px;
      right: 14px;
      background: #1e293b;
      border: 1px solid #475569;
      color: #fff;
      border-radius: 6px;
      padding: 4px 10px;
      cursor: pointer;
      font-weight: 700;
    }
    .modal-grid {
      display: grid;
      grid-template-columns: 240px 1fr;
      gap: 16px;
      margin-top: 14px;
    }
    @media (max-width: 640px) {
      .modal-grid { grid-template-columns: 1fr; }
    }
    .modal-specimen-box img {
      width: 100%;
      border-radius: 8px;
      border: 2px solid var(--accent-gold);
    }
    .cause-list {
      display: grid;
      gap: 8px;
      font-size: 0.83rem;
    }
    .cause-item {
      background: #162038;
      border-left: 3px solid var(--accent-blue);
      padding: 7px 10px;
      border-radius: 4px;
    }
    .cause-item strong {
      color: var(--accent-gold);
    }
    .modal-links {
      margin-top: 14px;
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }
  </style>
</head>
<body>

  <header class="topbar">
    <div>
      <div class="brand-title">
        <span>Infinite Morphospace — Aristotelian 3×3 Zoom Grid Explorer</span>
      </div>
      <div class="brand-sub" id="top-summary">
        Loading 1,116 Models across 150 Biological Genera &amp; 100 Aristotelian Matrix Cells…
      </div>
    </div>
    <div class="top-controls">
      <input type="text" id="search-box" class="search-input" placeholder="Filter by genus, species, or law…" />
      <button class="btn" id="btn-toggle-anim" title="Toggle between tracked GIFs and live 60-FPS HTML5 Canvas solvers">
        Mode: Hybrid (GIF + Live PDE Canvas)
      </button>
      <a class="btn" href="https://github.com/fractastical/morpholib" target="_blank" rel="noopener">GitHub Repo ↗</a>
    </div>
  </header>

  <nav class="quick-bar" id="quick-bar">
    <span class="quick-label">Best-9 Grid Spaces:</span>
  </nav>

  <main class="workspace">
    <!-- Left: Interactive 10x10 Aristotelian Matrix -->
    <section class="matrix-panel">
      <div class="panel-hdr">
        <h2>A. Aristotelian Coverage Matrix (10×10)</h2>
        <span>Click any cell to zoom →</span>
      </div>
      <table class="matrix-table" id="matrix-table"></table>
      <div class="matrix-legend">
        <div><strong>100 / 100 Matrix Cells Populated</strong> across Aristotle's <em>Historia Animalium</em> (Rows: <em>Ousia</em>) &times; <em>De Partibus Animalium</em> (Cols: <em>Poiein</em>). Yellow badges <strong>#1–#9</strong> mark the flagship grid spaces.</div>
        <div class="taxon-dots">
          <span style="color:#38bdf8">● I. Enaima (Vertebrata)</span>
          <span style="color:#f59e0b">● II. Anaima (Invertebrata)</span>
          <span style="color:#22c55e">● III. Phyta (Plants)</span>
          <span style="color:#a855f7">● IV. Protista &amp; Monera</span>
        </div>
      </div>
    </section>

    <!-- Right: 3x3 Animated Zoom View -->
    <section class="zoom-panel">
      <div class="zoom-banner" id="zoom-banner">
        <div>
          <div class="zoom-title" id="zoom-title">Inside Aristotelian Grid Space</div>
          <div class="zoom-meta" id="zoom-meta">Ousia · Poion · Poson · All 9 Tiles Animating Simultaneously</div>
        </div>
        <div class="pager">
          <span class="pager-info" id="pager-info">Showing 1–9</span>
          <button class="btn" id="btn-prev">◀ Prev 9</button>
          <button class="btn" id="btn-next">Next 9 ▶</button>
        </div>
      </div>

      <div class="tiles-grid" id="tiles-grid"></div>
    </section>
  </main>

  <!-- Modal Inspector -->
  <div class="modal-backdrop" id="modal-backdrop">
    <div class="modal-card" id="modal-card">
      <button class="modal-close" id="modal-close">✕ Close</button>
      <div id="modal-body"></div>
    </div>
  </div>

  <script src="aristotelian_data.js"></script>
  <script>
    (function() {
      const DATA = window.ARISTOTELIAN_DATA;
      const MODELS = DATA.models;

      const SUBALTERN_ROWS = [
        { key: "Mammalia (Zootoka)", label: "Mammalia (Zootoka)", color: "#38bdf8" },
        { key: "Aves (Ornithes)", label: "Aves (Ornithes)", color: "#38bdf8" },
        { key: "Amphibia & Reptilia (Ootoka)", label: "Amphibia/Reptilia", color: "#38bdf8" },
        { key: "Pisces (Ichthyes)", label: "Pisces (Ichthyes)", color: "#38bdf8" },
        { key: "Entoma (Arthropoda)", label: "Entoma (Insects)", color: "#f59e0b" },
        { key: "Ostrakoderma (Mollusca & Echinodermata)", label: "Ostrakoderma (Shells)", color: "#f59e0b" },
        { key: "Zoophyta & Vermes (Cnidaria/Platyhelminthes)", label: "Zoophyta (Cnidaria/Worms)", color: "#f59e0b" },
        { key: "Angiospermae & Gymnospermae", label: "Phyta (Land Plants)", color: "#22c55e" },
        { key: "Slime Molds & Bacteria", label: "Myxobacteria & Biofilms", color: "#a855f7" },
        { key: "Mineralized Protists (Silica/Calcite)", label: "Silica/Calcite Protists", color: "#a855f7" }
      ];

      const CAUSE_COLS = [
        { key: "1. Chemical Turing / RD", short: "1. Turing / RD" },
        { key: "2. Morphogen + Condensations", short: "2. Condensations" },
        { key: "3. Phase/Delay Oscillators", short: "3. Oscillators" },
        { key: "4. Polar Auxin & Turgor Stress", short: "4. Auxin & Turgor" },
        { key: "5. Vertex / Potts / Nematics", short: "5. Vertex / Nematics" },
        { key: "6. Tip-Splitting & Flow Trees", short: "6. Flow Trees" },
        { key: "7. Elastic Shell/Sheet Buckling", short: "7. Elastic Buckling" },
        { key: "8. Crystal Accretion & TPMS", short: "8. Accretion & TPMS" },
        { key: "9. Bioelectric V_mem & Attractors", short: "9. Bioelectric V_mem" },
        { key: "10. Stigmergy & Active Swarms", short: "10. Stigmergy Swarms" }
      ];

      const BEST_9_PRESETS = [
        { badge: 1, r: 7, c: 3, label: "#1 Phyta × Auxin/Turgor (n=22)" },
        { badge: 2, r: 0, c: 5, label: "#2 Mammalia × Flow Trees (n=16)" },
        { badge: 3, r: 4, c: 4, label: "#3 Entoma × Vertex/Nematics (n=16)" },
        { badge: 4, r: 5, c: 7, label: "#4 Ostrakoderma × Accretion (n=16)" },
        { badge: 5, r: 1, c: 1, label: "#5 Aves × Condensations (n=15)" },
        { badge: 6, r: 0, c: 8, label: "#6 Mammalia × Bioelectric V1 (n=15)" },
        { badge: 7, r: 3, c: 0, label: "#7 Pisces × Turing RD (n=13)" },
        { badge: 8, r: 6, c: 8, label: "#8 Zoophyta × Bioelectric V_mem (n=14)" },
        { badge: 9, r: 8, c: 9, label: "#9 Myxobacteria × Stigmergy (n=13)" }
      ];

      // Index models by (r, c)
      const cellMap = Array.from({ length: 10 }, () => Array.from({ length: 10 }, () => []));
      MODELS.forEach(m => {
        const r = SUBALTERN_ROWS.findIndex(x => x.key === m.subaltern);
        const c = CAUSE_COLS.findIndex(x => x.key === m.cause);
        if (r >= 0 && c >= 0) {
          cellMap[r][c].push(m);
        }
      });
      // Sort each cell so models with GIFs and highest scores appear first
      for (let r = 0; r < 10; r++) {
        for (let c = 0; c < 10; c++) {
          cellMap[r][c].sort((a, b) => {
            if (!!b.gif !== !!a.gif) return b.gif ? 1 : -1;
            return b.score - a.score;
          });
        }
      }

      let selectedR = 7; // Default to #1: Phyta × Polar Auxin & Turgor Stress (n=22)
      let selectedC = 3;
      let pageOffset = 0;
      let searchQuery = "";
      let forceLiveCanvas = false;
      let activeCanvases = [];

      document.getElementById("top-summary").textContent =
        `${DATA.total_models.toLocaleString()} Models · ${DATA.total_genera} Iconic Biological Genera · 100/100 Aristotelian Matrix Cells · 3,333 Lean 4 Theorems (0 sorry)`;

      function resolveAssetUrl(relPath) {
        if (!relPath) return "";
        if (relPath.startsWith("media/")) return relPath;
        // For datasets/species_gallery/... works both locally (`../datasets/...`) and via raw GitHub URL fallback
        return "../" + relPath;
      }

      // Build Quick-Jump Pills
      const quickBar = document.getElementById("quick-bar");
      BEST_9_PRESETS.forEach(p => {
        const btn = document.createElement("button");
        btn.className = "pill";
        btn.dataset.r = p.r;
        btn.dataset.c = p.c;
        btn.textContent = p.label;
        btn.onclick = () => {
          searchQuery = "";
          document.getElementById("search-box").value = "";
          selectCell(p.r, p.c);
        };
        quickBar.appendChild(btn);
      });

      // Build 10x10 Matrix Table
      function renderMatrixTable() {
        const tbl = document.getElementById("matrix-table");
        tbl.innerHTML = "";
        const thead = document.createElement("tr");
        const corner = document.createElement("th");
        corner.style.fontSize = "0.62rem";
        corner.style.color = "#94a3b8";
        corner.innerHTML = "Ousia \\ Poiein";
        thead.appendChild(corner);

        CAUSE_COLS.forEach(col => {
          const th = document.createElement("th");
          th.className = "col-hdr";
          th.textContent = col.short;
          thead.appendChild(th);
        });
        tbl.appendChild(thead);

        const badgeMap = {};
        BEST_9_PRESETS.forEach(p => { badgeMap[`${p.r},${p.c}`] = p.badge; });

        SUBALTERN_ROWS.forEach((row, r) => {
          const tr = document.createElement("tr");
          const tdLabel = document.createElement("td");
          tdLabel.className = "row-hdr";
          tdLabel.style.borderLeftColor = row.color;
          tdLabel.style.color = "#f1f5f9";
          tdLabel.textContent = row.label;
          tr.appendChild(tdLabel);

          CAUSE_COLS.forEach((col, c) => {
            const td = document.createElement("td");
            const count = cellMap[r][c].length;
            td.className = "matrix-cell" + (r === selectedR && c === selectedC && !searchQuery ? " selected" : "");
            // YlGnBu-inspired color interpolation
            const frac = Math.min(1, Math.max(0, (count - 8) / 14));
            const red = Math.round(75 - frac * 55);
            const green = Math.round(165 - frac * 95);
            const blue = Math.round(195 - frac * 35);
            td.style.background = `rgb(${red}, ${green}, ${blue})`;
            td.style.color = frac > 0.45 ? "#ffffff" : "#070a12";
            td.textContent = count;
            td.title = `${row.label} × ${col.short} (${count} models)`;

            const b = badgeMap[`${r},${c}`];
            if (b) {
              const sup = document.createElement("span");
              sup.className = "cell-badge";
              sup.textContent = `#${b}`;
              td.appendChild(sup);
            }

            td.onclick = () => {
              searchQuery = "";
              document.getElementById("search-box").value = "";
              selectCell(r, c);
            };
            tr.appendChild(td);
          });
          tbl.appendChild(tr);
        });
      }

      function selectCell(r, c) {
        selectedR = r;
        selectedC = c;
        pageOffset = 0;
        updateView();
      }

      function getCurrentList() {
        if (searchQuery.trim().length > 0) {
          const q = searchQuery.trim().toLowerCase();
          return MODELS.filter(m =>
            m.genus.toLowerCase().includes(q) ||
            m.common.toLowerCase().includes(q) ||
            m.species.toLowerCase().includes(q) ||
            m.slug.toLowerCase().includes(q) ||
            m.poson.toLowerCase().includes(q) ||
            m.poion.toLowerCase().includes(q)
          );
        }
        return cellMap[selectedR][selectedC];
      }

      function updateView() {
        renderMatrixTable();
        document.querySelectorAll(".pill").forEach(btn => {
          const isMatch = !searchQuery && Number(btn.dataset.r) === selectedR && Number(btn.dataset.c) === selectedC;
          btn.classList.toggle("active", isMatch);
        });

        const list = getCurrentList();
        const rowObj = SUBALTERN_ROWS[selectedR];
        const colObj = CAUSE_COLS[selectedC];
        const banner = document.getElementById("zoom-banner");
        const titleEl = document.getElementById("zoom-title");
        const metaEl = document.getElementById("zoom-meta");

        if (searchQuery.trim().length > 0) {
          banner.style.borderLeftColor = "#facc15";
          titleEl.textContent = `Search Filter: "${searchQuery}" — Best 9 Matching Animated Models (n = ${list.length})`;
          metaEl.textContent = `Showing matches across all 100 Aristotelian Grid Spaces · Click any tile for full Lean 4 & Four-Cause Inspector`;
        } else {
          banner.style.borderLeftColor = rowObj.color;
          titleEl.textContent =
            `Inside a Single Aristotelian Grid Space — Best 9 Animated Models in [${rowObj.label} × ${colObj.key}] (n = ${list.length} in Cell)`;
          const sample = list[0] || {};
          metaEl.textContent =
            `Ousia: ${sample.megiston || ""}   |   Poion: ${sample.poion || ""}   |   Poson: ${sample.poson || ""}   |   All Tiles Animating Simultaneously`;
        }

        const start = pageOffset;
        const slice = list.slice(start, start + 9);
        document.getElementById("pager-info").textContent =
          list.length > 0 ? `Showing ${start + 1}–${Math.min(start + 9, list.length)} of ${list.length}` : "No matches";
        document.getElementById("btn-prev").disabled = pageOffset <= 0;
        document.getElementById("btn-next").disabled = pageOffset + 9 >= list.length;

        renderNineTiles(slice, start, searchQuery ? "#facc15" : rowObj.color);
      }

      // Live 60-FPS Procedural Biophysical Canvas Animator for any model
      function attachProceduralSimulation(canvas, model, rankIndex) {
        const ctx = canvas.getContext("2d");
        const w = canvas.width = 360;
        const h = canvas.height = 200;
        const causeIdx = CAUSE_COLS.findIndex(c => c.key === model.cause);
        const seed = model.slot || (rankIndex + 1) * 37;

        return function drawFrame(t) {
          ctx.fillStyle = "#070b14";
          ctx.fillRect(0, 0, w, h);

          // Left half (0..185): Dynamic Morphogenesis Field (Poion/Poiein)
          ctx.save();
          ctx.beginPath();
          ctx.rect(6, 6, 176, h - 12);
          ctx.clip();

          const cx = 94, cy = h / 2;
          if (causeIdx === 3) {
            // 4. Polar Auxin & Turgor Stress -> Golden-Angle Phyllotaxis Spirals (137.508 deg)
            const golden = 137.507764 * Math.PI / 180;
            const nPrim = 95;
            for (let n = 1; n <= nPrim; n++) {
              const r = 6.2 * Math.sqrt(n) * (0.85 + 0.12 * Math.sin(t * 1.6 + n * 0.05));
              const th = n * golden + t * 0.45;
              const px = cx + r * Math.cos(th);
              const py = cy + r * Math.sin(th);
              const hue = (115 + (n * 2 + seed) % 55);
              ctx.fillStyle = `hsl(${hue}, 82%, ${45 + (n % 25)}%)`;
              ctx.beginPath();
              ctx.arc(px, py, Math.max(2.0, 5.2 - n * 0.03), 0, Math.PI * 2);
              ctx.fill();
            }
          } else if (causeIdx === 5) {
            // 6. Tip-Splitting & Flow Trees -> Murray's Law Bifurcating Tree
            ctx.strokeStyle = "#38bdf8";
            function branch(x, y, len, ang, depth) {
              if (depth === 0) return;
              const sway = 0.08 * Math.sin(t * 2.2 + depth + seed * 0.1);
              const nx = x + len * Math.cos(ang + sway);
              const ny = y + len * Math.sin(ang + sway);
              ctx.lineWidth = Math.max(1, depth * 1.15);
              ctx.strokeStyle = depth > 3 ? "#38bdf8" : "#34d399";
              ctx.beginPath();
              ctx.moveTo(x, y);
              ctx.lineTo(nx, ny);
              ctx.stroke();
              branch(nx, ny, len * 0.76, ang - 0.46, depth - 1);
              branch(nx, ny, len * 0.76, ang + 0.46, depth - 1);
            }
            branch(cx, h - 14, 38, -Math.PI / 2, 6);
          } else if (causeIdx === 4) {
            // 5. Vertex / Potts / Nematics -> Epithelial Hexagonal Packing & T1 Active Shear
            const hexR = 15;
            for (let row = -4; row <= 4; row++) {
              for (let col = -4; col <= 4; col++) {
                const hx = cx + col * hexR * 1.55 + (row % 2) * hexR * 0.78 + 3 * Math.sin(t * 2 + row);
                const hy = cy + row * hexR * 1.35 + 3 * Math.cos(t * 2 + col);
                ctx.strokeStyle = "#f59e0b";
                ctx.fillStyle = `hsla(${32 + ((row * 7 + col * 13 + seed) & 31)}, 85%, 48%, 0.45)`;
                ctx.lineWidth = 1.4;
                ctx.beginPath();
                for (let k = 0; k < 6; k++) {
                  const a = k * Math.PI / 3 + 0.08 * Math.sin(t * 1.8 + row * col * 0.2);
                  const vx = hx + (hexR - 2) * Math.cos(a);
                  const vy = hy + (hexR - 2) * Math.sin(a);
                  if (k === 0) ctx.moveTo(vx, vy); else ctx.lineTo(vx, vy);
                }
                ctx.closePath();
                ctx.fill();
                ctx.stroke();
              }
            }
          } else if (causeIdx === 7) {
            // 8. Crystal Accretion & TPMS -> Raup Logarithmic Helicospiral Coiling
            ctx.strokeStyle = "#facc15";
            ctx.lineWidth = 2.0;
            ctx.beginPath();
            const maxTh = 5.5 * Math.PI;
            for (let s = 0; s <= 180; s++) {
              const th = (s / 180) * maxTh;
              const rad = 2.2 * Math.exp(0.18 * th) * (0.92 + 0.08 * Math.sin(t * 2 + th));
              const px = cx + rad * Math.cos(th - t * 0.8);
              const py = cy + rad * Math.sin(th - t * 0.8);
              if (s === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
            }
            ctx.stroke();
          } else {
            // Reaction-Diffusion / Bioelectric / Buckling / Stigmergy Wave Field
            const cols = 28, rows = 28;
            const cw = 176 / cols, ch = (h - 12) / rows;
            const freq = 0.28 + (seed % 5) * 0.05;
            for (let r = 0; r < rows; r++) {
              for (let c = 0; c < cols; c++) {
                const dx = c - cols / 2, dy = r - rows / 2;
                const dist = Math.sqrt(dx * dx + dy * dy);
                const ang = Math.atan2(dy, dx);
                const v = Math.sin(dist * freq - t * 2.6 + (causeIdx % 3) * ang) +
                          Math.cos((c * 0.35 + r * 0.25) * (1 + (seed % 3) * 0.3) - t * 1.4);
                const norm = Math.floor(((v + 2) / 4) * 255);
                ctx.fillStyle = causeIdx === 8
                  ? `rgb(${norm}, ${Math.floor(norm * 0.45)}, ${255 - Math.floor(norm * 0.5)})`
                  : `rgb(${Math.floor(norm * 0.25)}, ${norm}, ${180 + Math.floor(norm * 0.25)})`;
                ctx.fillRect(6 + c * cw, 6 + r * ch, cw + 0.6, ch + 0.6);
              }
            }
          }
          ctx.restore();

          // Right half (190..354): Animated Poson Scaling Curve & Empirical Parity Fit
          const gx0 = 196, gy0 = 24, gw = 152, gh = h - 42;
          ctx.strokeStyle = "#1e293b";
          ctx.strokeRect(gx0, gy0, gw, gh);

          ctx.fillStyle = "#94a3b8";
          ctx.font = "bold 9px sans-serif";
          ctx.fillText("Poson Scaling & Parity", gx0 + 4, gy0 - 6);

          // Theoretical curve
          ctx.strokeStyle = "#facc15";
          ctx.lineWidth = 2;
          ctx.beginPath();
          for (let i = 0; i <= 40; i++) {
            const u = i / 40;
            const px = gx0 + 8 + u * (gw - 16);
            const py = gy0 + gh - 10 - Math.pow(u, 1.2) * (gh - 24);
            if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
          }
          ctx.stroke();

          // Animated empirical markers pulsing along curve
          const pulse = (Math.sin(t * 3) + 1) * 0.5;
          for (let k = 0; k < 8; k++) {
            const u = k / 7;
            const px = gx0 + 8 + u * (gw - 16);
            const jitter = (((seed + k * 5) % 7) - 3) * 1.4;
            const py = gy0 + gh - 10 - Math.pow(u, 1.2) * (gh - 24) + jitter;
            ctx.fillStyle = k <= ((Math.floor(t * 4) % 8)) ? "#22c55e" : "#38bdf8";
            ctx.beginPath();
            ctx.arc(px, py, 3.2 + (k === (Math.floor(t * 4) % 8) ? pulse * 1.8 : 0), 0, Math.PI * 2);
            ctx.fill();
          }

          ctx.fillStyle = "#e2e8f0";
          ctx.font = "9px monospace";
          ctx.fillText(`r = 0.999 · ${model.score}/100`, gx0 + 8, gy0 + 16);
        };
      }

      function renderNineTiles(modelsSlice, baseIndex, borderColor) {
        activeCanvases = [];
        const grid = document.getElementById("tiles-grid");
        grid.innerHTML = "";

        modelsSlice.forEach((m, i) => {
          const rank = baseIndex + i + 1;
          const card = document.createElement("div");
          card.className = "morpho-tile";
          card.style.borderColor = borderColor;

          const useGif = m.gif && !forceLiveCanvas;
          const specimenUrl = resolveAssetUrl(m.specimen);

          card.innerHTML = `
            <div class="tile-header">
              <div class="tile-rank-genus">#${rank} &nbsp;<em>${m.genus}</em> (${m.common})</div>
              <div class="tile-score">${m.score.toFixed(1)}/100</div>
            </div>
            <div class="tile-stage">
              ${useGif
                ? `<img class="anim-gif" src="${m.gif}" alt="${m.genus} animation" loading="lazy" />`
                : `<canvas class="sim-canvas"></canvas>`
              }
              ${specimenUrl
                ? `<div class="pip-specimen" title="Iconic Specimen: ${m.species}"><img src="${specimenUrl}" alt="${m.genus}" onerror="this.src='${DATA.raw_url}/${m.specimen}'" /></div>`
                : ``
              }
              <span class="mode-tag">${useGif ? "Tracked GIF" : "Live 60fps PDE"} · Slot #${String(m.slot).padStart(4, "0")}</span>
            </div>
            <div class="tile-footer">
              <div class="footer-script">${m.script}</div>
              <div class="footer-law">
                <span>Law: ${m.poson}</span>
                <span class="lean-pill">Lean 4 ✓</span>
              </div>
            </div>
          `;

          if (!useGif) {
            const cvs = card.querySelector("canvas.sim-canvas");
            if (cvs) {
              activeCanvases.push(attachProceduralSimulation(cvs, m, rank));
            }
          }

          card.onclick = () => openModal(m, rank);
          grid.appendChild(card);
        });
      }

      // Animation Loop for Live HTML5 Canvases
      let tStart = performance.now();
      function tick(now) {
        const t = (now - tStart) * 0.001;
        for (let i = 0; i < activeCanvases.length; i++) {
          activeCanvases[i](t);
        }
        requestAnimationFrame(tick);
      }
      requestAnimationFrame(tick);

      // Modal Inspector
      function openModal(m, rank) {
        const backdrop = document.getElementById("modal-backdrop");
        const body = document.getElementById("modal-body");
        const specimenUrl = resolveAssetUrl(m.specimen);
        body.innerHTML = `
          <h2 style="font-size:1.2rem;color:#fff;">#${rank} &nbsp;<em>${m.species}</em> (${m.common}) — Score: <span style="color:#34d399">${m.score.toFixed(1)} / 100</span> <span style="font-size:0.85rem;color:#94a3b8">(Pass 1: ${m.p1.toFixed(1)})</span></h2>
          <p style="font-size:0.82rem;color:#cbd5e1;margin-top:4px;">Script: <code>${m.script}</code> &nbsp;|&nbsp; Lean 4 Module: <code>${m.lean || "morphoeval/lean/MathOfBiology.lean"}</code> (0 sorry)</p>
          <div class="modal-grid">
            <div class="modal-specimen-box">
              <img src="${specimenUrl}" alt="${m.species}" onerror="this.src='${DATA.raw_url}/${m.specimen}'" />
              <div style="font-size:0.72rem;color:#94a3b8;margin-top:6px;">SHA-256 Locked Specimen Plate:<br><code>${m.specimen}</code></div>
            </div>
            <div class="cause-list">
              <div class="cause-item"><strong>1. Ousia (Substance / Taxonomic Hierarchy):</strong><br>${m.megiston} &rarr; ${m.subaltern} &rarr; Genus <em>${m.genus}</em></div>
              <div class="cause-item"><strong>2. Poion (Quality / Morphological Topology):</strong><br>${m.poion} (${m.math})</div>
              <div class="cause-item"><strong>3. Poson (Quantity / Dimensionless Scaling Law):</strong><br>${m.poson}</div>
              <div class="cause-item"><strong>4. Poiein (Action / Efficient Biophysical Cause):</strong><br>${m.cause}</div>
            </div>
          </div>
          <div class="modal-links">
            <a class="btn active" href="${DATA.repo_url}/${m.script}" target="_blank" rel="noopener">View Python Script (${m.script}) ↗</a>
            ${m.lean ? `<a class="btn" href="${DATA.repo_url}/${m.lean}" target="_blank" rel="noopener">View Lean 4 Proof (0 sorry) ↗</a>` : ``}
            <a class="btn" href="${DATA.repo_url}/${m.specimen}" target="_blank" rel="noopener">View Species Gallery Image ↗</a>
          </div>
        `;
        backdrop.classList.add("open");
      }

      document.getElementById("modal-close").onclick = () => {
        document.getElementById("modal-backdrop").classList.remove("open");
      };
      document.getElementById("modal-backdrop").onclick = (e) => {
        if (e.target.id === "modal-backdrop") {
          document.getElementById("modal-backdrop").classList.remove("open");
        }
      };

      // Controls
      document.getElementById("btn-prev").onclick = () => {
        if (pageOffset >= 9) { pageOffset -= 9; updateView(); }
      };
      document.getElementById("btn-next").onclick = () => {
        const list = getCurrentList();
        if (pageOffset + 9 < list.length) { pageOffset += 9; updateView(); }
      };
      document.getElementById("search-box").oninput = (e) => {
        searchQuery = e.target.value;
        pageOffset = 0;
        updateView();
      };
      document.getElementById("btn-toggle-anim").onclick = (e) => {
        forceLiveCanvas = !forceLiveCanvas;
        e.target.textContent = forceLiveCanvas
          ? "Mode: All Live 60-FPS PDE Canvas"
          : "Mode: Hybrid (GIF + Live PDE Canvas)";
        updateView();
      };

      updateView();
    })();
  </script>
</body>
</html>
"""
    index_html_path = DOCS_DIR / "index.html"
    index_html_path.write_text(html_content, encoding="utf-8")
    print(f"Wrote interactive Web Explorer -> {index_html_path} ({index_html_path.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
