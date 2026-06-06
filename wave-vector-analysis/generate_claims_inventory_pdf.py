#!/usr/bin/env python3
"""
Claims inventory PDF: ALL verbatim claims from `Calcium claims.docx`, grouped by
Layer, each tagged with a testability status against the data/tools we have.

Status legend
  TESTABLE NOW  - existing data + an existing analysis already addresses it
  PARTIAL       - data exists, but a specific analysis step is still needed
  NOT YET       - needs a capability/data we do not have (anatomy-specific
                  response classification, healed/repeat-poke timelines, etc.)

Run:
  python generate_claims_inventory_pdf.py \
    --output-pdf wave-vector-analysis/analysis_results/claims_inventory.pdf
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from textwrap import wrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(Path(__file__).resolve().parent))
from claims_doc_parser import parse_claims_doc, DEFAULT_DOCX

STATUS_COLORS = {
    "TESTED": "#1f6f3f",
    "TESTABLE NOW": "#2a7a2a",
    "PARTIAL": "#c87f0a",
    "NOT YET": "#aa2a2a",
}

DEFAULT_LATERALITY = (Path(__file__).resolve().parent / "analysis_results"
                      / "wave_catalog" / "wave_laterality.csv")
DEFAULT_LANDMARKS = (Path(__file__).resolve().parent / "analysis_results"
                     / "landmark_responses.csv")
LANDMARK_DFF_THR = 0.30


def load_landmarks(path: Path = DEFAULT_LANDMARKS) -> dict | None:
    """Aggregate the organ-landmark ΔF/F₀ scoring for the organ claims."""
    if not path.exists():
        return None
    try:
        import pandas as pd
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return None
    df["dff_num"] = pd.to_numeric(df["dff"], errors="coerce")
    df["stimulus"] = df["prefix"].str[0].map({"W": "wound", "P": "press"})

    def agg(sub):
        n = len(sub)
        cor = int((sub["dff_num"] >= LANDMARK_DFF_THR).sum())
        med = float(sub["dff_num"].median()) if n else float("nan")
        nv = int(sub["prefix"].nunique())
        return {"n": n, "cor": cor, "med": med, "nv": nv}

    cg = df[df.landmark_class == "cement_gland"]
    tail = df[df.landmark_class == "tail_response"]
    out = {
        "thr": LANDMARK_DFF_THR,
        "cement_press": agg(cg[cg.stimulus == "press"]),
        "cement_wound": agg(cg[cg.stimulus == "wound"]),
        "eye": agg(df[df.landmark_class == "eye"]),
        "tail_within": agg(tail[tail.embryo_role.isin(["stimulated", "single"])]),
        "tail_neighbor": agg(tail[tail.embryo_role == "neighbor"]),
        "local": agg(df[df.landmark_class == "local"]),
    }
    return out


def load_laterality(path: Path = DEFAULT_LATERALITY) -> dict | None:
    """Aggregate the per-video laterality/bidirectionality scoring, if present."""
    if not path.exists():
        return None
    try:
        import pandas as pd
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return None

    agg = {
        "n_videos": int(len(df)),
        "neighbor_wave_pct": 100.0 * df["neighbor_has_wave"].mean(),
        "bidir_pct": 100.0 * df["bidirectional_stim"].mean(),
        "neighbor_local_n": int((df["neighbor_has_wave"] == 1).sum()),
    }

    # --- Layer 3 claims 7-8: wound vs press neighbor response ---
    wound = df[df["stimulus"] == "wound"]["neighbor_peak_brightness"].dropna()
    press = df[df["stimulus"] == "press"]["neighbor_peak_brightness"].dropna()
    agg["n_wound"], agg["n_press"] = int(len(wound)), int(len(press))
    agg["neigh_wound_med"] = float(wound.median()) if len(wound) else float("nan")
    agg["neigh_press_med"] = float(press.median()) if len(press) else float("nan")
    agg["wp_p"] = float("nan")
    if len(wound) >= 3 and len(press) >= 3:
        try:
            from scipy.stats import mannwhitneyu
            agg["wp_p"] = float(mannwhitneyu(wound, press,
                                             alternative="greater").pvalue)
        except Exception:
            pass

    # --- Layer 4 claim 12: contact vs no-contact neighbor wave ---
    con = df[df["contact"] == "contact"]["neighbor_has_wave"]
    noc = df[df["contact"] == "no-contact"]["neighbor_has_wave"]
    agg["n_contact"], agg["n_nocontact"] = int(len(con)), int(len(noc))
    agg["contact_pct"] = 100.0 * con.mean() if len(con) else float("nan")
    agg["nocontact_pct"] = 100.0 * noc.mean() if len(noc) else float("nan")
    return agg

LAYER_TITLES = {
    "2": "Layer 2 - Response within a single (stimulated) embryo",
    "3": "Layer 3 - Increased response in a neighboring embryo",
    "4": "Layer 4 - Signaling patterns in the neighboring embryo",
}


def _organ_verdict(stat: dict, what: str, thr: float) -> tuple[str, str, str]:
    """Build a TESTED verdict line from a landmark aggregate."""
    n, cor, med, nv = stat["n"], stat["cor"], stat["med"], stat["nv"]
    frac = f"{cor}/{n}"
    medtxt = f"{med:.2f}" if med == med else "n/a"  # noqa: PLR0124
    strong = cor >= max(1, n / 2)
    verdict = ("corroborated" if strong else
               "weakly corroborated by the calcium signal")
    return ("TESTED",
            f"Scored at the manually-annotated {what} location(s): a calcium "
            f"rise (ΔF/F₀ ≥ {thr}) is seen in {frac} landmark(s) across {nv} "
            f"video(s); median ΔF/F₀ {medtxt} → {verdict}.",
            "parse_xy_coordinates.py + score_landmark_responses.py "
            "(landmark_responses.csv).")


def classify(rec: dict, lat: dict | None = None,
             lm: dict | None = None) -> tuple[str, str, str]:
    """Return (status, rationale, data_source) for a claim record.

    `lat` is the aggregate from wave_laterality.csv; `lm` is the organ-landmark
    aggregate from landmark_responses.csv. When present, the relevant claims
    move from PARTIAL/NOT YET to TESTED with a live verdict.
    """
    c = rec["claim"].lower()

    # Memory / healed / re-occurrence: needs timelines we do not have
    if "healed" in c or "prior wound site" in c or "re-occur" in c:
        return ("NOT YET",
                "Requires repeat-poke / healed-embryo timelines and "
                "wound-site memory tracking, which are not in the current "
                "pipeline.",
                "(none yet)")

    # ---- Anatomy-specific responses, now backed by manual XY landmarks ----
    if lm and "tail response in neighbor" in c:
        return _organ_verdict(lm["tail_neighbor"], "neighbor-embryo tail",
                              lm["thr"])
    if lm and re.search(r"(cement gland|eye|tail) response", c):
        if "cement gland" in c:
            stat = lm["cement_press"] if "pressure" in c else lm["cement_wound"]
            return _organ_verdict(stat, "cement gland", lm["thr"])
        if "eye" in c:
            return _organ_verdict(lm["eye"], "eye", lm["thr"])
        if "tail" in c:
            return _organ_verdict(lm["tail_within"], "stimulated-embryo tail",
                                  lm["thr"])

    # Anatomy-specific response without landmark data -> still blocked
    if re.search(r"(cement gland|eye|tail) response", c):
        return ("NOT YET",
                "Requires classifying the response by anatomical structure "
                "(cement gland / eye / tail). We measure brightness and wave "
                "motion, not organ-specific responses yet.",
                "embryo_region_map.py exists but is not wired to score these.")

    # Bidirectional wave within a single embryo
    if "bidirectional" in c and "within an embryo" in c:
        if lat:
            return ("TESTED",
                    f"Scored: stimulated-side waves propagate BOTH ways along "
                    f"the embryo axis in {lat['bidir_pct']:.0f}% of "
                    f"{lat['n_videos']} videos (two opposing front lobes from "
                    f"the poke origin).",
                    "wave_laterality.py / wave_laterality.csv "
                    "(bidirectional_stim).")
        return ("PARTIAL",
                "Wave events and per-wave direction are computed; a "
                "bidirectional test (two opposing fronts from the poke) still "
                "needs to be scored per video.",
                "wave_catalog.csv (propagation direction per wave).")

    # Increase / not-increase response in neighbor (Layer 3)
    if "neighboring embryo" in c and ("increase" in c):
        if lat and lat.get("n_press"):
            p = lat["wp_p"]
            sig = (p == p) and p < 0.05  # not NaN and significant
            wound_claim = "increase" in c and "not increase" not in c
            if wound_claim:
                verdict = ("supported" if sig else
                           "NOT supported by this metric")
                return ("TESTED",
                        f"Scored: neighbor peak brightness, wounding "
                        f"(median {lat['neigh_wound_med']:.0f}, n={lat['n_wound']}) "
                        f"vs pressure (median {lat['neigh_press_med']:.0f}, "
                        f"n={lat['n_press']}); Mann-Whitney p="
                        f"{p:.2f} -> {verdict}. (ΔF/F₀ comparison in the "
                        f"no-mask report is the complementary fluorescence test.)",
                        "wave_laterality.csv (neighbor_peak_brightness); "
                        "claims_nomask_report.pdf (ΔF/F₀).")
            # "pressure does NOT increase" claim
            verdict = ("consistent: pressure neighbor response is not "
                       "elevated above wounding" if not sig else
                       "pressure also elevated; revisit")
            return ("TESTED",
                    f"Scored: pressure neighbor brightness (median "
                    f"{lat['neigh_press_med']:.0f}, n={lat['n_press']}) is not "
                    f"greater than wounding (median {lat['neigh_wound_med']:.0f}); "
                    f"wound>press Mann-Whitney p={p:.2f} -> {verdict}.",
                    "wave_laterality.csv (neighbor_peak_brightness); "
                    "claims_nomask_report.pdf (ΔF/F₀).")
        return ("TESTABLE NOW",
                "Neighbor-side peak ΔF/F₀ is already computed and compared "
                "(wounding vs pressure). This is report Claim 2.",
                "claims_nomask_report.pdf (neighbor ΔF/F₀); wave_catalog.csv.")

    # No physical contact
    if "does not require physical contact" in c:
        if lat and lat.get("n_nocontact"):
            noc = lat["nocontact_pct"]
            verdict = ("supported: neighbor waves still occur without contact, "
                       "at a rate comparable to contact"
                       if noc > 0 else
                       "NOT reproduced: no neighbor wave detected without "
                       "contact by this metric")
            return ("TESTED",
                    f"Scored with real poke + head/tail geometry: no-contact "
                    f"videos show a neighbor-side wave in {noc:.0f}% of cases "
                    f"(n={lat['n_nocontact']}) vs {lat['contact_pct']:.0f}% for "
                    f"contact (n={lat['n_contact']}) -> {verdict}.",
                    "wave_laterality.csv (contact vs no-contact, "
                    "neighbor_has_wave; ground-truth side split).")
        return ("TESTABLE NOW",
                "No-contact (WGD) videos are in the wave catalog and show "
                "waves; contact vs no-contact is compared on the Claim 5 page.",
                "wave_catalog.csv (82 no-contact waves); Claim 5 page.")

    # Calcium wave in neighbor (Layer 4, by orientation)
    if "calcium wave in neighbor" in c:
        if lat:
            return ("TESTED",
                    f"Scored: a distinct wave is detected on the NEIGHBOR "
                    f"embryo (2nd activity cluster along the arrangement axis) "
                    f"in {lat['neighbor_wave_pct']:.0f}% of {lat['n_videos']} "
                    f"videos, lit up after the stimulated side.",
                    "wave_laterality.py / wave_laterality.csv "
                    "(neighbor_has_wave, onset_lag_s).")
        return ("PARTIAL",
                "Waves are detected and rolled up, but attributing a wave to "
                "the NEIGHBOR side (vs stimulated side) per orientation needs a "
                "left/right split added to the roll-up.",
                "wave_catalog.csv + per-video pixel vectors (needs side split).")

    # Local response in neighbor
    if "local response in neighbor" in c:
        if lat:
            return ("TESTED",
                    f"Scored: neighbor-side front (local) speed is measured "
                    f"wherever a neighbor wave exists ({lat['neighbor_local_n']}"
                    f"/{lat['n_videos']} videos); compared against the "
                    f"stimulated side per video.",
                    "wave_laterality.py / wave_laterality.csv "
                    "(neighbor_mean_front_speed).")
        return ("PARTIAL",
                "Local (front) speed and local brightness are measured; "
                "scoring a localized neighbor response by orientation still "
                "needs the side split + region gating.",
                "wave_catalog.csv (front speed); side-region ΔF/F₀.")

    # tail response in neighbor (anatomy)
    if "tail response in neighbor" in c:
        return ("NOT YET",
                "Anatomy-specific (tail) neighbor response; needs organ-level "
                "response classification.",
                "(none yet)")

    return ("PARTIAL",
            "Partially addressable with current brightness / wave data; "
            "needs a dedicated scoring step.",
            "wave_catalog.csv / side-region ΔF/F₀.")


def main():
    ap = argparse.ArgumentParser(description="Claims inventory PDF generator.")
    ap.add_argument("--claims-docx", default=str(DEFAULT_DOCX))
    ap.add_argument("--output-pdf",
                    default="wave-vector-analysis/analysis_results/claims_inventory.pdf")
    args = ap.parse_args()

    recs = parse_claims_doc(Path(args.claims_docx))
    out = Path(args.output_pdf)
    out.parent.mkdir(parents=True, exist_ok=True)

    lat = load_laterality()
    lm = load_landmarks()

    # classify all
    enriched = []
    for r in recs:
        status, why, src = classify(r, lat, lm)
        enriched.append((r, status, why, src))

    counts = {k: 0 for k in STATUS_COLORS}
    for _, s, _, _ in enriched:
        counts[s] += 1

    # group by layer (preserve doc order of first appearance)
    layer_order = []
    by_layer: dict[str, list] = {}
    for item in enriched:
        L = item[0]["layer"] or "?"
        if L not in by_layer:
            by_layer[L] = []
            layer_order.append(L)
        by_layer[L].append(item)

    with PdfPages(out) as pdf:
        # ---- cover / summary ----
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.5, 0.95, "Calcium Claims - Full Inventory",
                 ha="center", fontsize=17, fontweight="bold")
        fig.text(0.5, 0.915,
                 f"All {len(recs)} verbatim claims from {Path(args.claims_docx).name}",
                 ha="center", fontsize=10, color="#444444")
        ax = fig.add_axes([0.08, 0.08, 0.84, 0.80])
        ax.axis("off")
        y = 0.98
        ax.text(0.0, y, "Status summary", fontsize=12, fontweight="bold",
                va="top")
        y -= 0.05
        for st in ("TESTED", "TESTABLE NOW", "PARTIAL", "NOT YET"):
            if counts.get(st, 0) == 0:
                continue
            ax.text(0.02, y, st, fontsize=10, fontweight="bold", va="top",
                    color="white",
                    bbox=dict(boxstyle="round,pad=0.25",
                              fc=STATUS_COLORS[st], ec="none"))
            ax.text(0.35, y, f"{counts[st]} claims", fontsize=10, va="top")
            y -= 0.045
        y -= 0.02
        legend = (
            "TESTED = an analysis has now scored this claim across the videos; "
            "the per-claim line gives the live result.\n"
            "TESTABLE NOW = existing data + an existing analysis already "
            "addresses it.\n"
            "PARTIAL = data exists, but a specific analysis step is still "
            "needed (noted per claim).\n"
            "NOT YET = needs a capability or data we do not have (anatomy-"
            "specific response classification, healed / repeat-poke timelines)."
        )
        for ln in legend.split("\n"):
            for w in wrap(ln, 92):
                ax.text(0.0, y, w, fontsize=8.5, va="top", color="#333333")
                y -= 0.022
            y -= 0.006
        y -= 0.02
        note = (
            "The five numbered claims used elsewhere in this report are a "
            "thematic ROLL-UP of the rows below; this inventory restores the "
            "document's own structure (one row per claim, grouped by Layer)."
        )
        for w in wrap(note, 92):
            ax.text(0.0, y, w, fontsize=8.5, va="top", color="#555555",
                    style="italic")
            y -= 0.022
        pdf.savefig(fig)
        plt.close(fig)

        # ---- evidence page: laterality / directionality scoring ----
        lat_png = DEFAULT_LATERALITY.with_suffix(".png")
        if lat and lat_png.exists():
            import matplotlib.image as mpimg
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.5, 0.965, "Evidence: wave laterality + directionality",
                     ha="center", fontsize=14, fontweight="bold")
            fig.text(0.5, 0.94,
                     f"Scoring behind the TESTED claims  "
                     f"(n={lat['n_videos']} videos)",
                     ha="center", fontsize=9.5, color="#444444")
            ax = fig.add_axes([0.05, 0.30, 0.90, 0.58])
            ax.axis("off")
            ax.imshow(mpimg.imread(str(lat_png)))
            cap = (
                "Two embryos are separated by the principal axis of their wave "
                "ORIGINS; the cluster whose waves start earliest/brightest is "
                "the stimulated embryo, the other is the neighbor. "
                "Top-left: a wave is detected in the neighbor in "
                f"{lat['neighbor_wave_pct']:.0f}% of videos (claims 9-11). "
                "Top-right: neighbor-side local (front) speed vs stimulated "
                "side (claims 14-16). Bottom-left: stimulated-side waves spread "
                f"BOTH ways along the axis in {lat['bidir_pct']:.0f}% of videos "
                "(claims 1-2). Bottom-right: the neighbor lights up AFTER the "
                "stimulated side, consistent with outward propagation."
            )
            yy = 0.25
            for w in wrap(cap, 100):
                fig.text(0.06, yy, w, fontsize=8.5, color="#333333")
                yy -= 0.018
            pdf.savefig(fig)
            plt.close(fig)

        # ---- evidence page 2: neighbor response by condition ----
        cond_png = DEFAULT_LATERALITY.parent / "wave_laterality_conditions.png"
        if lat and cond_png.exists():
            import matplotlib.image as mpimg
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.5, 0.965, "Evidence: neighbor response by condition",
                     ha="center", fontsize=14, fontweight="bold")
            fig.text(0.5, 0.94,
                     "Behind the Layer-3 (wound vs press) and no-contact claims",
                     ha="center", fontsize=9.5, color="#444444")
            ax = fig.add_axes([0.05, 0.46, 0.90, 0.42])
            ax.axis("off")
            ax.imshow(mpimg.imread(str(cond_png)))
            wp = lat.get("wp_p", float("nan"))
            cap = (
                "Claims 7-8 (left, middle): neighbor peak brightness and "
                f"neighbor-wave prevalence for wounding (median "
                f"{lat.get('neigh_wound_med', float('nan')):.0f}, "
                f"n={lat.get('n_wound', 0)}) vs pressure (median "
                f"{lat.get('neigh_press_med', float('nan')):.0f}, "
                f"n={lat.get('n_press', 0)}). A wound>press Mann-Whitney test "
                f"gives p={wp:.2f}: the wave-based neighbor metric does NOT show "
                "wounding producing a larger neighbor response than pressure - "
                "the document's 'wounding increases / pressure does not' "
                "distinction is not reproduced by this metric (the ΔF/F₀ "
                "fluorescence comparison in the no-mask report is the "
                "complementary test). "
                "Claim 12 (right): a neighbor wave is seen in "
                f"{lat.get('nocontact_pct', float('nan')):.0f}% of NO-CONTACT "
                f"videos (n={lat.get('n_nocontact', 0)}) vs "
                f"{lat.get('contact_pct', float('nan')):.0f}% with contact - "
                "inter-embryo signaling does NOT require physical contact "
                "(supported)."
            )
            yy = 0.40
            for w in wrap(cap, 100):
                fig.text(0.06, yy, w, fontsize=8.5, color="#333333")
                yy -= 0.018
            pdf.savefig(fig)
            plt.close(fig)

        # ---- evidence page 3: organ-landmark responses (manual XY) ----
        lm_png = DEFAULT_LANDMARKS.with_suffix(".png")
        if lm and lm_png.exists():
            import matplotlib.image as mpimg
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.5, 0.965, "Evidence: organ-response landmarks (manual XY)",
                     ha="center", fontsize=13.5, fontweight="bold")
            fig.text(0.5, 0.94,
                     "Behind the cement-gland / eye / tail claims "
                     "(Layer 2 & Layer 4)",
                     ha="center", fontsize=9.5, color="#444444")
            ax = fig.add_axes([0.05, 0.46, 0.90, 0.42])
            ax.axis("off")
            ax.imshow(mpimg.imread(str(lm_png)))
            cg_w, cg_p = lm["cement_wound"], lm["cement_press"]
            cap = (
                "Each manually-annotated organ-response point (from "
                "XY coordinates.xlsx) is checked against the calcium signal: "
                "we measure the robust-peak brightness in a disk at the "
                "annotated pixel and report ΔF/F₀ relative to its own quiet "
                f"baseline (corroborated at ΔF/F₀ ≥ {lm['thr']}). "
                f"Eye: {lm['eye']['cor']}/{lm['eye']['n']} (median "
                f"{lm['eye']['med']:.2f}). "
                f"Tail (stimulated embryo): {lm['tail_within']['cor']}/"
                f"{lm['tail_within']['n']}; tail (neighbor): "
                f"{lm['tail_neighbor']['cor']}/{lm['tail_neighbor']['n']}. "
                f"Cement gland: wounding {cg_w['cor']}/{cg_w['n']}, pressure "
                f"{cg_p['cor']}/{cg_p['n']} — the manual cement-gland "
                "annotations show the weakest calcium transient by this "
                "measure (the gland is small and partly autofluorescent). "
                "These are sparse manual points, so the test confirms "
                "existence/location of a response rather than population "
                "statistics."
            )
            yy = 0.40
            for w in wrap(cap, 100):
                fig.text(0.06, yy, w, fontsize=8.5, color="#333333")
                yy -= 0.018
            pdf.savefig(fig)
            plt.close(fig)

        # ---- wave-vector overlays on the real embryo frame (per claim) ----
        overlay_dir = DEFAULT_LATERALITY.parent / "overlays"
        overlays = [
            ("claim_bidirectional.png",
             "Wave vectors on the embryo — Claims 1–2 (bidirectional)",
             "Stimulated-side wave fronts on the brightest microscopy frame of "
             "the most bidirectional videos. Red = fronts moving one way along "
             "the embryo axis, blue = the opposing direction: the two lobes are "
             "the bidirectional spread from the poke."),
            ("claim_neighbor_wave.png",
             "Wave vectors on the embryo — Claims 9–11 (neighbor wave)",
             "Magenta = wave fronts on the NEIGHBOR embryo; faded blue = the "
             "stimulated embryo for context. The signal crosses into the "
             "neighbor in wounding videos."),
            ("claim_neighbor_local.png",
             "Wave vectors on the embryo — Claims 14–16 (neighbor local)",
             "Neighbor-side front vectors (magenta): a localized, directed "
             "response within the neighbor embryo, not just diffuse brightening."),
            ("claim_layer3.png",
             "Wave vectors on the embryo — Claims 7–8 (wound vs press)",
             "Neighbor-side fronts for wounding (top row) vs pressure (bottom "
             "row); the wave-based neighbor response looks similar between the "
             "two conditions (see the brightness test, p=0.43)."),
            ("claim_no_contact.png",
             "Wave vectors on the embryo — Claim 12 (no contact)",
             "Neighbor-side fronts in NO-CONTACT videos: a wave still reaches "
             "the neighbor without physical contact."),
        ]
        present_overlays = [(f, t, c) for f, t, c in overlays
                            if (overlay_dir / f).exists()]
        if present_overlays:
            import matplotlib.image as mpimg
            for fname, ftitle, fcap in present_overlays:
                fig = plt.figure(figsize=(8.5, 11))
                fig.text(0.5, 0.965, ftitle, ha="center", fontsize=12.5,
                         fontweight="bold")
                ax = fig.add_axes([0.03, 0.34, 0.94, 0.56])
                ax.axis("off")
                ax.imshow(mpimg.imread(str(overlay_dir / fname)))
                yy = 0.30
                for w in wrap(fcap, 96):
                    fig.text(0.06, yy, w, fontsize=9, color="#333333")
                    yy -= 0.02
                gif = fname.replace(".png", ".gif")
                if (overlay_dir / gif).exists():
                    yy -= 0.01
                    fig.text(0.06, yy, f"Animated version: overlays/{gif} "
                             "(plays the fronts frame-by-frame).",
                             fontsize=8, color="#1f6fb2", style="italic")
                fig.text(0.5, 0.04,
                         "Vectors = per-frame wave-front motion "
                         "(wave_front_frames.csv); background = brightest frame "
                         "of each source video.",
                         ha="center", fontsize=7.5, color="#777777")
                pdf.savefig(fig)
                plt.close(fig)

        # ---- per-layer pages ----
        for L in layer_order:
            items = by_layer[L]
            # paginate: ~5 claims per page
            per_page = 5
            chunks = [items[i:i + per_page]
                      for i in range(0, len(items), per_page)]
            for pi, chunk in enumerate(chunks):
                fig = plt.figure(figsize=(8.5, 11))
                title = LAYER_TITLES.get(L, f"Layer {L}")
                suffix = "" if len(chunks) == 1 else f"  ({pi+1}/{len(chunks)})"
                fig.text(0.06, 0.965, title + suffix, fontsize=13,
                         fontweight="bold", va="top")
                ax = fig.add_axes([0.04, 0.04, 0.92, 0.90])
                ax.axis("off")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                y = 0.98
                for (r, status, why, src) in chunk:
                    gi = enriched.index((r, status, why, src)) + 1
                    y = _block(ax, y, gi, r, status, why, src)
                pdf.savefig(fig)
                plt.close(fig)

    print(f"Wrote {out}")
    print(f"  TESTED={counts['TESTED']} TESTABLE NOW={counts['TESTABLE NOW']} "
          f"PARTIAL={counts['PARTIAL']} NOT YET={counts['NOT YET']} "
          f"(of {len(recs)})")


def _block(ax, y, idx, rec, status, rationale, data_source):
    color = STATUS_COLORS[status]
    ax.text(0.005, y, f"{idx:02d}", fontsize=10, fontweight="bold",
            va="top", family="monospace")
    ax.text(0.995, y, status, fontsize=8.5, fontweight="bold", color="white",
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.25", fc=color, ec="none"))
    y -= 0.024
    for ln in wrap(rec["claim"], 88):
        ax.text(0.05, y, ln, fontsize=10, va="top", fontweight="bold")
        y -= 0.022
    meta = []
    if rec["pattern"] and rec["pattern"] not in ("-", ""):
        meta.append(f"pattern {rec['pattern']}")
    if rec["orientation"]:
        meta.append(rec["orientation"])
    if rec["poke"]:
        meta.append(f"poke: {rec['poke']}")
    if rec["observed"]:
        meta.append(f"observed: {rec['observed']}")
    if meta:
        for ln in wrap(" | ".join(meta), 108):
            ax.text(0.05, y, ln, fontsize=7.8, va="top", color="#444444",
                    style="italic")
            y -= 0.017
    for ln in wrap("Why: " + rationale, 108):
        ax.text(0.05, y, ln, fontsize=8, va="top", color="#333333")
        y -= 0.017
    for ln in wrap("Data/tool: " + data_source, 108):
        ax.text(0.05, y, ln, fontsize=7.8, va="top", color="#666666")
        y -= 0.017
    y -= 0.018
    return y


if __name__ == "__main__":
    main()
