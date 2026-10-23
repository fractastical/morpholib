#!/usr/bin/env python3
"""
Generate the two canonical Project Delphi / MorphoEval 12-Step Closed-Loop
Architecture diagrams (matching the exact visual language of `draw_pipeline.py`
and `track_loop.py`) for the D'Arcy Thompson (1917) & Project Delphi workflow:

  1. `docs/thompson_12step_closed_loop_dark.png` (+ `.gif`):
     The dark-mode 12-Step Closed-Loop Architecture (Steps 1..5 Tier A, Box 6
     Output & Compare with the actual Thompson 600-pt silhouette & horizon plots,
     Steps 7..12 Tier B, and the right-hand CLOSED LOOP feedback channel from
     Step 12 -> Step 5) annotated with exact file payloads & prompts.
  2. `docs/thompson_system_architecture_and_dataflow.png`:
     Updated to use the clean 12-Step Tier A / Box 6 / Tier B + Closed-Loop
     visual layout with person/robot icons, color-coded model providers
     (Anthropic orange, Google blue, OpenAI green), and explicit file contracts.
"""

from __future__ import annotations

import io
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch
from PIL import Image, ImageSequence

ROOT = Path(__file__).resolve().parents[1]
OUT_DARK_PNG = ROOT / "docs" / "thompson_12step_closed_loop_dark.png"
OUT_DARK_GIF = ROOT / "docs" / "media" / "thompson_12step_closed_loop.gif"
OUT_ARCH_PNG = ROOT / "docs" / "thompson_system_architecture_and_dataflow.png"


def draw_person_icon(ax, x: float, y: float, s: float = 1.0, color: str = "#a855f7"):
    ax.add_patch(Ellipse((x, y + 2.1 * s), 1.4 * s, 1.8 * s, fc=color, ec="none", zorder=6))
    ax.add_patch(
        FancyBboxPatch(
            (x - 1.3 * s, y),
            2.6 * s,
            1.5 * s,
            boxstyle="round,pad=0,rounding_size=0.8",
            fc=color,
            ec="none",
            zorder=6,
        )
    )


def draw_robot_icon(ax, x: float, y: float, s: float = 1.0, color: str = "#d97757"):
    ax.add_patch(
        FancyBboxPatch(
            (x - 0.8 * s, y + 1.4 * s),
            1.6 * s,
            1.4 * s,
            boxstyle="round,pad=0,rounding_size=0.3",
            fc=color,
            ec="none",
            zorder=6,
        )
    )
    ax.plot([x, x], [y + 2.8 * s, y + 3.4 * s], color=color, lw=1.1, zorder=6)
    ax.add_patch(Ellipse((x, y + 3.5 * s), 0.48 * s, 0.4 * s, fc=color, ec="none", zorder=6))
    for dx in (-0.34, 0.34):
        ax.add_patch(Ellipse((x + dx * s, y + 2.15 * s), 0.38 * s, 0.38 * s, fc="white", ec="none", zorder=7))
    ax.add_patch(
        FancyBboxPatch(
            (x - 1.3 * s, y),
            2.6 * s,
            1.2 * s,
            boxstyle="round,pad=0,rounding_size=0.45",
            fc=color,
            ec="none",
            zorder=6,
        )
    )


def render_dark_12step_thompson(out_png: Path, out_gif: Path | None = None):
    BG_MAIN = "#0d1117"
    BG_CARD = "#161b22"
    BG_BADGE = "#1f2937"
    BG_ACTIVE = "#1e293b"
    STROKE = "#30363d"
    STROKE_ACTIVE = "#58a6ff"
    TEXT_PRI = "#f0f6fc"
    TEXT_SEC = "#8b949e"
    TEXT_TER = "#6e7681"
    ACCENT_BLUE = "#58a6ff"
    ACCENT_GREEN = "#3fb950"
    ACCENT_AMBER = "#d29922"
    ACCENT_PURPLE = "#bc8cff"
    ANTHROPIC_COL = "#d97757"
    GOOGLE_COL = "#4285f4"

    # Load actual Thompson visual panels for Box 6
    plates_path = ROOT / "docs" / "thompson_1917_archival_field_plates.png"
    gif_path = ROOT / "docs" / "media" / "thompson_1917_silhouette_paradox_summary.gif"
    stick_vs_sil_path = ROOT / "thompson_stick_vs_silhouette_reality.png"
    horizon_path = ROOT / "docs" / "outreach_fig2_price_wainwright_horizon.png"

    img_6a = Image.open(plates_path).convert("RGB") if plates_path.exists() else None
    img_6b = Image.open(stick_vs_sil_path).convert("RGB") if stick_vs_sil_path.exists() else None
    img_6c = Image.open(horizon_path).convert("RGB") if horizon_path.exists() else None

    gif_frames: list[Image.Image] = []
    if gif_path.exists():
        with Image.open(gif_path) as im:
            gif_frames = [f.convert("RGB") for f in ImageSequence.Iterator(im)]

    TIER_A_STEPS = [
        (1, "1 Scout (P1 only)", "Prompt: 01_scout", "Actor: human + scout-agent", "Thompson 1917 + Dryad fn457gr", ACCENT_BLUE),
        (2, "2 Paper (P1 only)", "Ch. XVII Figs 373-382", "Actor: builder-agent", "4 Coordinate Families + TPS", ACCENT_BLUE),
        (3, "3 Formal source (P1)", "SHA-256: ab5315f3...", "Actor: gates/1_verify_inputs.py", "791 spp CSV + 9 archival plates", ACCENT_BLUE),
        (4, "4 Pin the rules (P1)", "Lean 4: 0 sorry (exit 0)", "Actor: MathOfBiology.lean", "tps_kernel + unit_shear laws", ACCENT_BLUE),
        (5, "5 Script (P1 + P2)", "Prompt: 01_builder_prompt.md", "P2: +600-pt Chamfer & 10,303 spp", "thompson/1917-thompson-*.py", ACCENT_GREEN),
    ]

    TIER_B_STEPS = [
        (7, "7 Dual-agent audit", "Prompt: 02_referee_prompt.md", "P2: 76.6 (Claude) / 90.8 (Gemini)", "claude-fable-5-1 + gemini-4-argon", "P1: 70.2 -> P2 Panel: 83.7"),
        (8, "8 Measure (Sidecar)", "Gate: measure_outputs.py", "Dual-raster IoU Δ <= 0.0001", "Zero-LLM Python verifier", "*.measurements.yaml (23 items)"),
        (9, "9 Write as claims", "Gate: build_morphoeval_kg.py", "npx:supersedes Pass 1", "Project Delphi RDF TriG", "pk:Claim + pk:evidenceAgainst"),
        (10, "10 Sign & publish", "Gate: sign_morphoeval.py", "RSA-2048 + @nanopub/sign", "Test -> Prod Registry", "80 Signed Nanopubs (RAYFKAfy...)"),
        (11, "11 Public record", "SPARQL / Nanopub Query", "Pre-reg lock: RAilN4nb...", "kg/<audit>/REPORT.md", "5,180 Blind FishShapes v1 spp"),
        (12, "12 Remediate (Trigger)", "Prompt: 03_remediation_prompt", "Loops 12 -> 5 -> 11", "Reads pk:evidenceAgainst", "9/17 Head Seesaw -> 600-pt Fix"),
    ]

    def draw_frame(active_step: int | None = None, pass_num: int = 2, g_idx: int = -1) -> Image.Image:
        fig, ax = plt.subplots(figsize=(14.5, 9.2), dpi=140, facecolor=BG_MAIN)
        fig.subplots_adjust(0, 0, 1, 1)
        ax.set_facecolor(BG_MAIN)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.axis("off")

        is_summary = active_step is None

        # Outer frame
        ax.add_patch(
            FancyBboxPatch(
                (1.2, 2.0),
                97.6,
                86.8,
                boxstyle="round,pad=0.3,rounding_size=1.4",
                fc=BG_MAIN,
                ec=STROKE,
                lw=1.2,
                zorder=1,
            )
        )

        # Top header
        ax.text(
            2.0,
            95.8,
            "Project Delphi 12-Step Closed-Loop Architecture — Frontier Model Orchestration, Prompts & Gates",
            color=TEXT_PRI,
            fontsize=13.8,
            fontweight="bold",
            va="center",
        )
        ax.text(
            2.0,
            91.6,
            "Pass 1 (17-Dot Skeleton Fit, Steps 1→11): 70.2/100 (IoU 0.662→0.618)   |   "
            "Step 12 Queries `pk:evidenceAgainst` → Pass 2 (600-Pt Outline + 10,303 Spp + Blind Test): 83.7/100 Panel (+13.5)",
            color=ACCENT_BLUE if pass_num == 1 and not is_summary else ACCENT_GREEN,
            fontsize=8.8,
            fontweight="bold",
            va="center",
        )
        ax.text(
            98.4,
            95.8,
            "Repo: fractastical/project-delphi-loop · fractastical/morpholib",
            color=TEXT_SEC,
            fontsize=8.6,
            ha="right",
            va="center",
        )
        ax.text(
            98.4,
            91.6,
            "Models: claude-fable-5-1 (Anthropic) × gemini-4-argon (Google) · 80 Signed Nanopubs",
            color=ACCENT_GREEN,
            fontsize=8.6,
            fontweight="bold",
            ha="right",
            va="center",
        )

        # Tier A Header
        ax.text(
            2.8,
            85.4,
            "A  Making a reproduction — Steps 1–4 lock SHA-256 data & Lean 4 theorems (`0 sorry`); Step 5 runs Builder Agent in Pass 1 & Pass 2",
            color=TEXT_PRI,
            fontsize=9.5,
            fontweight="bold",
            va="center",
        )

        tier_a_x = [2.8, 21.9, 41.0, 60.1, 79.2]
        box_a_w, box_a_h, box_a_y = 17.4, 16.5, 66.5

        for idx, (snum, stitle, badge1, badge2, actor_lbl, foot_lbl) in enumerate(TIER_A_STEPS):
            x = tier_a_x[idx]
            y = box_a_y
            is_act = active_step == snum
            ec_col = ACCENT_GREEN if (snum == 5 or (is_act and pass_num == 2)) else (STROKE_ACTIVE if is_act else STROKE)
            lw_val = 2.2 if is_act else (1.5 if snum == 5 else 1.0)

            ax.add_patch(
                FancyBboxPatch(
                    (x, y),
                    box_a_w,
                    box_a_h,
                    boxstyle="round,pad=0.2,rounding_size=0.9",
                    fc=BG_ACTIVE if is_act else BG_CARD,
                    ec=ec_col,
                    lw=lw_val,
                    zorder=2,
                )
            )
            ax.text(x + 0.8, y + box_a_h - 2.3, stitle, color=TEXT_PRI, fontsize=8.6, fontweight="bold", va="center", zorder=3)

            if snum == 1:
                draw_person_icon(ax, x + box_a_w - 1.8, y + box_a_h - 3.2, s=0.55, color=ACCENT_PURPLE)
            elif snum == 5:
                draw_robot_icon(ax, x + box_a_w - 1.8, y + box_a_h - 3.2, s=0.55, color=ANTHROPIC_COL)

            if snum < 5:
                ax.add_patch(
                    FancyBboxPatch(
                        (x + 0.7, y + box_a_h - 7.2),
                        box_a_w - 1.4,
                        3.3,
                        boxstyle="round,pad=0.1,rounding_size=0.45",
                        fc=BG_BADGE,
                        ec="none",
                        zorder=3,
                    )
                )
                ax.text(x + 1.1, y + box_a_h - 5.55, badge1, color=ACCENT_BLUE, fontsize=7.6, fontweight="bold", va="center", zorder=4)
                ax.text(x + 0.8, y + 4.8, badge2, color=TEXT_SEC, fontsize=7.3, va="center", zorder=3)
                ax.text(x + 0.8, y + 2.1, actor_lbl, color=TEXT_TER, fontsize=7.0, family="monospace", va="center", zorder=3)
            else:
                ax.add_patch(
                    FancyBboxPatch(
                        (x + 0.7, y + box_a_h - 6.7),
                        box_a_w - 1.4,
                        2.8,
                        boxstyle="round,pad=0.08,rounding_size=0.4",
                        fc=BG_BADGE,
                        ec="none",
                        zorder=3,
                    )
                )
                ax.text(x + 1.0, y + box_a_h - 5.3, badge1, color=ACCENT_BLUE, fontsize=7.2, fontweight="bold", va="center", zorder=4)
                ax.add_patch(
                    FancyBboxPatch(
                        (x + 0.7, y + box_a_h - 10.1),
                        box_a_w - 1.4,
                        2.8,
                        boxstyle="round,pad=0.08,rounding_size=0.4",
                        fc=BG_BADGE,
                        ec="none",
                        zorder=3,
                    )
                )
                ax.text(x + 1.0, y + box_a_h - 8.7, badge2, color=ACCENT_GREEN, fontsize=7.2, fontweight="bold", va="center", zorder=4)
                ax.text(x + 0.8, y + 4.2, actor_lbl, color=TEXT_SEC, fontsize=7.2, family="monospace", va="center", zorder=3)
                ax.text(x + 0.8, y + 1.8, "Pass 1: Claude · Pass 2: Gemini", color=ACCENT_GREEN, fontsize=6.9, fontweight="bold", va="center", zorder=3)

            if idx < 4:
                ax.add_patch(
                    FancyArrowPatch(
                        (x + box_a_w + 0.15, y + box_a_h / 2),
                        (tier_a_x[idx + 1] - 0.15, y + box_a_h / 2),
                        arrowstyle="-|>",
                        mutation_scale=10,
                        lw=1.3,
                        color=TEXT_SEC,
                        zorder=3,
                    )
                )

        # Arrow Step 5 -> Step 6
        s5_cx = tier_a_x[4] + 3.8
        ax.add_patch(
            FancyArrowPatch(
                (s5_cx, box_a_y - 0.2),
                (s5_cx, 60.8),
                arrowstyle="-|>",
                mutation_scale=11,
                lw=1.5,
                color=ACCENT_GREEN,
                zorder=3,
            )
        )
        ax.text(s5_cx + 0.7, 63.5, "Edge E2: writes PNGs + metrics CSV/JSON", color=TEXT_SEC, fontsize=7.4, va="center")

        # Box 6: Output & Compare
        box6_x, box6_y, box6_w, box6_h = 2.8, 31.5, 79.8, 29.0
        ax.add_patch(
            FancyBboxPatch(
                (box6_x, box6_y),
                box6_w,
                box6_h,
                boxstyle="round,pad=0.25,rounding_size=1.0",
                fc=BG_CARD,
                ec=ACCENT_GREEN,
                lw=1.3,
                zorder=2,
            )
        )
        ax.text(
            box6_x + 1.3,
            box6_y + box6_h - 2.6,
            "6  Output & compare — What the Scripts Produce (Scored by Gate 2 Sidecar & Step 7 Dual Referees)",
            color=TEXT_PRI,
            fontsize=9.6,
            fontweight="bold",
            va="center",
            zorder=3,
        )
        ax.add_patch(
            FancyBboxPatch(
                (box6_x + 48.0, box6_y + box6_h - 4.3),
                30.6,
                3.3,
                boxstyle="round,pad=0.1,rounding_size=0.5",
                fc=BG_BADGE,
                ec="none",
                zorder=3,
            )
        )
        ax.text(
            box6_x + 48.6,
            box6_y + box6_h - 2.65,
            "P1 (17-LM): IoU 0.618  |  P2 (600-pt): |AΔB| 56.2%→28.5%",
            color=ACCENT_GREEN,
            fontsize=7.5,
            fontweight="bold",
            va="center",
            zorder=4,
        )

        sub_y0, sub_y1 = box6_y + 4.4, box6_y + box6_h - 5.2
        if img_6a is not None:
            ax.imshow(img_6a, extent=(4.2, 24.2, sub_y0, sub_y1), zorder=3, interpolation="bilinear", aspect="auto")
            ax.add_patch(FancyBboxPatch((4.2, sub_y0), 20.0, sub_y1 - sub_y0, boxstyle="square,pad=0", fc="none", ec=ACCENT_GREEN, lw=1.1, zorder=4))
        ax.text(
            14.2,
            box6_y + 2.2,
            "A. 9 Archival Museum Plates & 600-pt Outlines",
            color=ACCENT_GREEN,
            fontsize=7.1,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=4,
        )

        if gif_frames:
            f_idx = g_idx % len(gif_frames) if g_idx >= 0 else 0
            ax.imshow(gif_frames[f_idx], extent=(25.4, 57.6, sub_y0, sub_y1), zorder=3, interpolation="bilinear", aspect="auto")
            ax.add_patch(FancyBboxPatch((25.4, sub_y0), 32.2, sub_y1 - sub_y0, boxstyle="square,pad=0", fc="none", ec=ACCENT_BLUE, lw=1.1, zorder=4))
        elif img_6b is not None:
            ax.imshow(img_6b, extent=(25.4, 57.6, sub_y0, sub_y1), zorder=3, interpolation="bilinear", aspect="auto")
            ax.add_patch(FancyBboxPatch((25.4, sub_y0), 32.2, sub_y1 - sub_y0, boxstyle="square,pad=0", fc="none", ec=ACCENT_BLUE, lw=1.1, zorder=4))
        ax.text(
            41.5,
            box6_y + 2.2,
            "B. Pass 1 (17-Dot Head Seesaw) vs. Pass 2 (Direct 600-Pt Chamfer Fit)",
            color=TEXT_PRI,
            fontsize=7.2,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=4,
        )

        if img_6c is not None:
            ax.imshow(img_6c, extent=(58.8, 81.4, sub_y0, sub_y1), zorder=3, interpolation="bilinear", aspect="auto")
            ax.add_patch(FancyBboxPatch((58.8, sub_y0), 22.6, sub_y1 - sub_y0, boxstyle="square,pad=0", fc="none", ec=ACCENT_PURPLE, lw=1.0, zorder=4))
        ax.text(
            70.1,
            box6_y + 2.2,
            "C. ~45–50 Myr Horizon (4,334 Pairs) & 5,180 Blind Species",
            color=ACCENT_GREEN,
            fontsize=7.1,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=4,
        )

        # Arrow Step 6 -> Step 7
        s7_cx = 10.0
        ax.add_patch(
            FancyArrowPatch(
                (s7_cx, box6_y - 0.2),
                (s7_cx, 25.0),
                arrowstyle="-|>",
                mutation_scale=11,
                lw=1.5,
                color=ACCENT_GREEN,
                zorder=3,
            )
        )
        ax.text(
            s7_cx + 0.8,
            29.2,
            "Edge E3: Sidecar `measure_outputs.py` hashes PNG/CSV & checks dual-rasterizer IoU before Dual-Agent Audit",
            color=TEXT_SEC,
            fontsize=7.5,
            va="center",
        )

        # Tier B Header
        ax.text(
            2.8,
            26.6,
            "B  Auditing it (The MorphoEval Cycle) — Dual-Provider Blinded Referees → Sidecar → RDF Nanopubs → Step 12 Remediation",
            color=TEXT_PRI,
            fontsize=9.3,
            fontweight="bold",
            va="center",
        )

        tier_b_x = [2.8, 18.9, 35.0, 51.1, 67.2, 83.3]
        box_b_w, box_b_h, box_b_y = 14.5, 16.8, 7.6

        for idx, (snum, stitle, b1, b2, actor_lbl, foot_lbl) in enumerate(TIER_B_STEPS):
            x = tier_b_x[idx]
            y = box_b_y
            ec_col = ACCENT_AMBER if snum == 12 else ACCENT_GREEN
            ax.add_patch(
                FancyBboxPatch(
                    (x, y),
                    box_b_w,
                    box_b_h,
                    boxstyle="round,pad=0.2,rounding_size=0.9",
                    fc=BG_CARD,
                    ec=ec_col,
                    lw=1.4,
                    zorder=2,
                )
            )
            ax.text(x + 0.6, y + box_b_h - 2.3, stitle, color=TEXT_PRI, fontsize=8.1, fontweight="bold", va="center", zorder=3)

            if snum == 7:
                draw_robot_icon(ax, x + box_b_w - 2.8, y + box_b_h - 3.1, s=0.46, color=ANTHROPIC_COL)
                draw_robot_icon(ax, x + box_b_w - 1.2, y + box_b_h - 3.1, s=0.46, color=GOOGLE_COL)
            elif snum == 12:
                draw_person_icon(ax, x + box_b_w - 1.4, y + box_b_h - 3.1, s=0.48, color=ACCENT_PURPLE)

            ax.add_patch(
                FancyBboxPatch(
                    (x + 0.5, y + box_b_h - 6.7),
                    box_b_w - 1.0,
                    2.8,
                    boxstyle="round,pad=0.06,rounding_size=0.35",
                    fc=BG_BADGE,
                    ec="none",
                    zorder=3,
                )
            )
            ax.text(x + 0.75, y + box_b_h - 5.3, b1, color=ACCENT_BLUE, fontsize=6.85, fontweight="bold", va="center", zorder=4)

            ax.add_patch(
                FancyBboxPatch(
                    (x + 0.5, y + box_b_h - 10.0),
                    box_b_w - 1.0,
                    2.8,
                    boxstyle="round,pad=0.06,rounding_size=0.35",
                    fc=BG_BADGE,
                    ec="none",
                    zorder=3,
                )
            )
            ax.text(
                x + 0.75,
                y + box_b_h - 8.6,
                b2,
                color=ACCENT_AMBER if snum == 12 else ACCENT_GREEN,
                fontsize=6.85,
                fontweight="bold",
                va="center",
                zorder=4,
            )

            ax.text(x + 0.6, y + 4.2, actor_lbl, color=TEXT_SEC, fontsize=6.9, va="center", zorder=3)
            ax.text(
                x + 0.6,
                y + 1.8,
                foot_lbl,
                color=ACCENT_AMBER if snum == 12 else ACCENT_GREEN,
                fontsize=6.85,
                fontweight="bold",
                va="center",
                zorder=3,
            )

            if idx < 5:
                ax.add_patch(
                    FancyArrowPatch(
                        (x + box_b_w + 0.12, y + box_b_h / 2),
                        (tier_b_x[idx + 1] - 0.12, y + box_b_h / 2),
                        arrowstyle="-|>",
                        mutation_scale=9,
                        lw=1.2,
                        color=ACCENT_GREEN,
                        zorder=3,
                    )
                )

        # Right-hand CLOSED LOOP feedback channel (Step 12 -> Step 5)
        loop_x = 90.6
        ax.plot([loop_x, loop_x], [box_b_y + box_b_h, 33.4], color=ACCENT_GREEN, lw=2.2, ls="--", zorder=3)
        ax.add_patch(
            FancyArrowPatch(
                (loop_x, 58.6),
                (loop_x, box_a_y - 0.1),
                arrowstyle="-|>",
                mutation_scale=13,
                lw=2.2,
                ls="--",
                color=ACCENT_GREEN,
                zorder=4,
            )
        )
        ax.add_patch(
            FancyBboxPatch(
                (83.5, 33.4),
                14.2,
                25.2,
                boxstyle="round,pad=0.2,rounding_size=0.9",
                fc=BG_CARD,
                ec=ACCENT_GREEN,
                lw=1.9,
                zorder=4,
            )
        )
        ax.text(90.6, 56.2, "CLOSED LOOP", color=ACCENT_GREEN, fontsize=8.2, fontweight="bold", ha="center", va="center", zorder=5)
        ax.text(90.6, 53.5, "Edge E5 Payload:", color=TEXT_SEC, fontsize=6.9, ha="center", va="center", zorder=5)
        ax.text(90.6, 51.1, "pk:evidenceAgainst", color=ACCENT_AMBER, fontsize=7.2, fontweight="bold", family="monospace", ha="center", va="center", zorder=5)
        ax.text(90.6, 48.2, "Pass 1 (17-Dot Fit):", color=TEXT_SEC, fontsize=6.8, ha="center", va="center", zorder=5)
        ax.text(90.6, 45.9, "70.2 → 76.6 (IoU 0.618)", color=ACCENT_BLUE, fontsize=7.0, fontweight="bold", ha="center", va="center", zorder=5)
        ax.plot([84.3, 96.9], [43.5, 43.5], color=STROKE, lw=0.9, zorder=5)
        ax.text(90.6, 41.2, "Pass 2 (600-Pt + 10k):", color=TEXT_PRI, fontsize=6.9, fontweight="bold", ha="center", va="center", zorder=5)
        ax.text(90.6, 38.8, "90.8 Blinded (83.7 Avg)", color=ACCENT_GREEN, fontsize=7.1, fontweight="bold", ha="center", va="center", zorder=5)
        ax.text(90.6, 36.0, "|AΔB| 56.2% → 28.5%", color=ACCENT_GREEN, fontsize=7.0, fontweight="bold", ha="center", va="center", zorder=5)

        # Bottom summary bar
        ax.text(
            2.8,
            4.3,
            "How It Stitches Together: Builder (`claude-fable-5-1`) writes code → Sidecar (`measure_outputs.py`) verifies numbers → Blinded Referee (`gemini-4-argon`) "
            "emits `pk:evidenceAgainst` in RDF → Step 12 feeds back into Step 5.",
            color=TEXT_SEC,
            fontsize=8.0,
            va="center",
        )

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=140, facecolor=BG_MAIN)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    static_im = draw_frame(active_step=None, pass_num=2, g_idx=0)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    static_im.save(out_png)
    static_im.save(OUT_ARCH_PNG)
    print(f"Saved dark-mode 12-step closed-loop diagram to {out_png} and {OUT_ARCH_PNG}")

    if out_gif is not None and gif_frames:
        frames_out = [draw_frame(active_step=None, pass_num=2, g_idx=k) for k in range(min(18, len(gif_frames)))]
        out_gif.parent.mkdir(parents=True, exist_ok=True)
        frames_out[0].save(
            out_gif,
            save_all=True,
            append_images=frames_out[1:],
            duration=180,
            loop=0,
            optimize=True,
        )
        print(f"Saved animated 12-step closed-loop GIF to {out_gif}")


if __name__ == "__main__":
    render_dark_12step_thompson(OUT_DARK_PNG, OUT_DARK_GIF)
