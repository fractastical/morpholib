#!/usr/bin/env python3
"""
Build the complete plug-and-play integration package for bioelectricity.net
(Bioelectricity Nexus) under docs/bioelectricity/:

  1. docs/bioelectricity/index.html
     -> Standalone & iframe-embeddable interactive web application styled with
        bioelectricity.net's exact typography (Space Grotesk, Inter, JetBrains Mono)
        and dark-mode palette. Includes:
        - Live 60 FPS HTML5 <canvas> Interactive D'Arcy Thompson Coordinate Grid
          Sandbox on real 600-point fish silhouettes (Scarus -> Pomacanthus,
          Argyropelecus -> Sternoptyx, Pristigenys -> Scorpaena, Diodon -> Mola mola)
        - One-click preset buttons comparing:
          (a) Unwarped Baseline
          (b) The '17-Dot Head Seesaw Trap' (Worsens Silhouette!)
          (c) Plain Rectangular 'Box Stretch' (Misses Swept Fin Tips!)
          (d) Thompson's Curved & Skewed 600-Point Grid (Recovered!)
        - Full 5-Step Visual Guide for Everyone (large-font separate diagrams)
        - Bioelectric Morphospace Synthesis (linking global V_mem axial/fin
          prepatterns vs. local neural-crest craniofacial ionocyte domains +
          Lean-verified Ermentrout-Oster neural gate <-> Wolfram Rule 30)
  2. docs/bioelectricity/ProjectDelphiWidget.tsx
     -> Drop-in React + TypeScript component for the bioelectricity.net Vite/React codebase.
  3. docs/bioelectricity/bioelectricity_nexus_feed.json
     -> Structured JSON API payload for bioelectricity.net's /publications, /tools,
        /research, and /glossary routes.
  4. docs/bioelectricity/README.md
     -> 30-second integration instructions for bioelectricity.net.
"""

import importlib.util
import json
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
EXACT_JSON = REPO_ROOT / "datasets" / "thompson" / "thompson_1917_exact_species_landmarks.json"
OUT_DIR = REPO_ROOT / "docs" / "bioelectricity"


def load_module(name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def downsample_contour(B: np.ndarray, n: int = 200) -> list:
    idx = np.linspace(0, len(B) - 1, n, endpoint=False).astype(int)
    return np.round(B[idx], 4).tolist()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sil_mod = load_module("sil_mod", "thompson/1917-thompson-silhouette-area-paradox.py")
    at, sp_mod = sil_mod.load_modules()
    names, families, X = at.load_landmarks()
    Y = at.gpa(X)
    M0 = at.centre_scale(Y.mean(0))
    meta = json.loads(EXACT_JSON.read_text(encoding="utf-8"))
    specs = {s: sil_mod.extract_aligned_silhouette_and_lms(s, meta, M0, sp_mod) for s in meta["specimens"]}
    O_sp = sp_mod.outline_analysis(Y, names, allow_download=False)
    specs["Scarus_ghobban"] = (O_sp["photos"]["Scarus_ghobban"]["landmarks"], O_sp["photos"]["Scarus_ghobban"]["boundary"])
    specs["Pomacanthus_semicirculatus"] = (
        O_sp["photos"]["Pomacanthus_semicirculatus"]["landmarks"],
        O_sp["photos"]["Pomacanthus_semicirculatus"]["boundary"],
    )
    idx_dio = names.index("Diodon_hystrix")
    T_mola, _ = specs["Mola_mola"]
    S_dio = at.opa(Y[idx_dio], T_mola)
    BS_dio = sil_mod.build_diodon_silhouette_from_landmarks(S_dio, 600)
    specs["Diodon_hystrix"] = (S_dio, BS_dio)

    pair_defs = [
        ("scarus_pomacanthus", "Scarus_ghobban", "Pomacanthus_semicirculatus", "coaxial",
         "Scarus (Parrotfish) → Pomacanthus (Angelfish)", "II. Coaxial Circles (Figs. 375–376)"),
        ("pristigenys_scorpaena", "Pristigenys_alta", "Scorpaena_scrofa", "fan",
         "Pristigenys (Bigeye) → Scorpaena (Scorpionfish)", "IIIb. Bilinear Fan (Figs. 378–379)"),
        ("argyropelecus_sternoptyx", "Argyropelecus_olfersii", "Sternoptyx_diaphana", "shear",
         "Argyropelecus → Sternoptyx (Hatchetfishes)", "I. 70° Oblique Shear (Figs. 373–374)"),
        ("diodon_mola", "Diodon_hystrix", "Mola_mola", "radial",
         "Diodon (Porcupinefish) → Mola mola (Ocean Sunfish)", "IV. Hyperbolic Radial Flare (Figs. 381–382)"),
    ]

    sandbox_data = {}
    for pid, src, tgt, fam, title, subtitle in pair_defs:
        S, BS = specs[src]
        T, BT = specs[tgt]
        BS_unw = sil_mod.unwarped_similarity_silhouette(S, BS, T, at, sp_mod)
        S_unw = at.fit_similarity(S, T)

        if fam == "shear":
            Ms_lm, _, _ = at.fit_shear(S_unw, T)
            BS_lm = at.apply_affine(Ms_lm, BS_unw)
            S_lm = at.apply_affine(Ms_lm, S_unw)
        else:
            p_lm, _ = at.fit_nonlinear_single(fam, S_unw, T)
            BS_lm = at.FAMILY_FUNCS[fam](p_lm, BS_unw)
            S_lm = at.FAMILY_FUNCS[fam](p_lm, S_unw)

        p_sil, BS_sil, apply_sil = sil_mod.fit_direct_to_silhouette(fam, S, BS, T, BT, at)
        S_sil = apply_sil(p_sil, S)

        # Plain Rectangular Box Stretch (no skew, no curve):
        sx = (BT[:, 0].max() - BT[:, 0].min()) / (BS_unw[:, 0].max() - BS_unw[:, 0].min())
        sy = (BT[:, 1].max() - BT[:, 1].min()) / (BS_unw[:, 1].max() - BS_unw[:, 1].min())
        cx_s, cy_s = BS_unw[:, 0].mean(), BS_unw[:, 1].mean()
        cx_t, cy_t = BT[:, 0].mean(), BT[:, 1].mean()
        BS_box = np.column_stack([(BS_unw[:, 0] - cx_s) * sx + cx_t, (BS_unw[:, 1] - cy_s) * sy + cy_t])
        S_box = np.column_stack([(S_unw[:, 0] - cx_s) * sx + cx_t, (S_unw[:, 1] - cy_s) * sy + cy_t])

        bd_unw = sil_mod.area_diff_breakdown(BS_unw, BT, grid=400)
        bd_lm = sil_mod.area_diff_breakdown(BS_lm, BT, grid=400)
        bd_box = sil_mod.area_diff_breakdown(BS_box, BT, grid=400)
        bd_sil = sil_mod.area_diff_breakdown(BS_sil, BT, grid=400)

        sandbox_data[pid] = {
            "title": title,
            "subtitle": subtitle,
            "family": fam,
            "target_contour": downsample_contour(BT, 220),
            "target_lms": np.round(T, 4).tolist(),
            "states": {
                "unwarped": {
                    "label": "Unwarped 4p Similarity (No Stretch)",
                    "contour": downsample_contour(BS_unw, 220),
                    "lms": np.round(S_unw, 4).tolist(),
                    "area_diff_pct": round(bd_unw["sym_diff_pct_of_target"], 1),
                    "iou": round(bd_unw["iou"], 3),
                    "note": "Starting fish scaled to same size, before any coordinate deformation.",
                },
                "lm17": {
                    "label": "17-Dot Skeleton Fit (9/17 Head Dots Pull Grid Wrong Way!)",
                    "contour": downsample_contour(BS_lm, 220),
                    "lms": np.round(S_lm, 4).tolist(),
                    "area_diff_pct": round(bd_lm["sym_diff_pct_of_target"], 1),
                    "iou": round(bd_lm["iou"], 3),
                    "note": "9 head dots (53% weight) hijack the fit and swing the posterior body off target!",
                },
                "box": {
                    "label": "Plain Rectangular Box Stretch (Zero Skew / Zero Curve)",
                    "contour": downsample_contour(BS_box, 220),
                    "lms": np.round(S_box, 4).tolist(),
                    "area_diff_pct": round(bd_box["sym_diff_pct_of_target"], 1),
                    "iou": round(bd_box["iou"], 3),
                    "note": "Stretches height vs. width with straight lines — misses swept-back fin tips & canted keels!",
                },
                "thompson600": {
                    "label": "Thompson's Curved/Skewed 600-Point Grid (Recovered!)",
                    "contour": downsample_contour(BS_sil, 220),
                    "lms": np.round(S_sil, 4).tolist(),
                    "area_diff_pct": round(bd_sil["sym_diff_pct_of_target"], 1),
                    "iou": round(bd_sil["iou"], 3),
                    "note": "100% diffeomorphic (det J > 0) curved/skewed grid sweeps fins & body into place!",
                },
            },
        }

    # =========================================================================
    # 0B. BUILD INTERACTIVE ARTHROPOD SANDBOX DATA (ANTS, CRABS, COPEPODS, AMPHIPODS)
    # =========================================================================
    ant_mod = load_module("ant_mod", "thompson/run_antscan_caste_allometry_tests.py")
    crab_mod = load_module("crab_mod", "thompson/run_crab_carapace_paradox_audit.py")
    arth_mod = load_module("arth_mod", "thompson/run_all_1917_arthropods_audit.py")

    _, ex_head = ant_mod.run_camponotus_634_analyses()
    S_ant, T_ant = ex_head["S"], ex_head["T"]
    P_box_ant, P_bf_ant = ex_head["P_box"], ex_head["P_bf"]
    P_359_ant, _, _ = ant_mod.fit_thompson_fig359_radial_cordate_8p(S_ant, T_ant)
    ss0_ant = float(np.sum((T_ant - S_ant) ** 2))
    r2_359_ant = (1.0 - float(np.sum((T_ant - P_359_ant) ** 2)) / ss0_ant) * 100.0

    # Pheidole megacephala minor -> major (23 AntWeb landmarks)
    S_pheid = ant_mod.extract_antweb_head_landmarks(ant_mod.CACHE_DIR / "pheidole_megacephala_minor_head.jpg")
    T_pheid = ant_mod.extract_antweb_head_landmarks(ant_mod.CACHE_DIR / "pheidole_megacephala_major_head.jpg")
    P_box_ph = ant_mod.fit_box_4p(S_pheid, T_pheid)
    P_359_ph, _, _ = ant_mod.fit_thompson_fig359_radial_cordate_8p(S_pheid, T_pheid)
    ss0_ph = float(np.sum((T_pheid - S_pheid) ** 2))
    r2_box_ph = (1.0 - float(np.sum((T_pheid - P_box_ph) ** 2)) / ss0_ph) * 100.0
    r2_359_ph = (1.0 - float(np.sum((T_pheid - P_359_ph) ** 2)) / ss0_ph) * 100.0

    # Real 3D Laser-Scanned Crabs (Scholtz et al. 2020 Fig. 2)
    from PIL import Image as PILImage
    arr_crab = np.array(PILImage.open(crab_mod.CRAB_DIR / "scholtz_2020_fig2_real_3d_crabs.png").convert("RGB"))
    real_crabs = crab_mod.extract_crab_dataset(arr_crab, int(arr_crab.shape[0] * 0.24), int(arr_crab.shape[0] * 0.47), use_real_lms=True)
    crab_rochinia = crab_mod.fit_crab_models(real_crabs["a_Geryon"], real_crabs["c_Scyramathia_Rochinia"])
    crab_chorinus = crab_mod.fit_crab_models(real_crabs["a_Geryon"], real_crabs["f_Chorinus_Leptomithrax"])
    crab_lithodes = crab_mod.fit_crab_models(real_crabs["a_Geryon"], real_crabs["d_Paralomis_Lithodes"])

    cop_res = arth_mod.test_copepods_fig367_368()
    amp_res = arth_mod.test_amphipods_fig370()

    arthropod_sandbox_data = {
        "ant_camponotus_soldier": {
            "title": f"Camponotus {ex_head['species']} (Minor → Major Soldier Head, 82 LMs)",
            "subtitle": "Extension of Fig. 359 (p. 732) Ovate → Cordate Radial-Bilobed Law Across 634 Specimens",
            "target_contour": np.round(T_ant, 4).tolist(),
            "target_lms": np.round(T_ant[::4], 4).tolist(),
            "states": {
                "unwarped": {
                    "label": "Unwarped Minor Worker Head (Ovate Baseline)",
                    "contour": np.round(S_ant, 4).tolist(),
                    "lms": np.round(S_ant[::4], 4).tolist(),
                    "r2_pct": 0.0,
                    "bij": "100% (det J = 1.00)",
                    "note": "Minor worker head capsule (ovate contour) before adductor muscle hypertrophy.",
                },
                "trap_or_box": {
                    "label": "4p Rectangular Box Stretch (Misses Occipital Lobes!)",
                    "contour": np.round(P_box_ant, 4).tolist(),
                    "lms": np.round(P_box_ant[::4], 4).tolist(),
                    "r2_pct": round(ex_head["r2_box"], 1),
                    "bij": "100% (det J > 0)",
                    "note": f"Plain Box Stretch explains only {ex_head['r2_box']:.1f}% (13.3% median across 42 spp.) — cannot flare the posterior occipital lobes or indent the vertex notch!",
                },
                "thompson": {
                    "label": "8p Thompson Bilobed-Fan / Fig. 359 Radial-Cordate Grid",
                    "contour": np.round(P_bf_ant, 4).tolist(),
                    "lms": np.round(P_bf_ant[::4], 4).tolist(),
                    "r2_pct": round(max(ex_head["r2_bf"], r2_359_ant), 1),
                    "bij": f"100% (min det J = {ex_head['min_det']:.2f})",
                    "note": f"Thompson's Bilobed-Fan & Fig. 359 Radial-Cordate law jumps to {max(ex_head['r2_bf'], r2_359_ant):.1f}% (76.0% median across 42/42 species, p = 4.55e-13)!",
                },
            },
        },
        "crab_spider_chorinus": {
            "title": "Geryon → Chorinus (1917) / Leptomithrax (3D Spider Crab)",
            "subtitle": "Fig. 369f vs. Scholtz, Knötel & Baum (2020) — f₁ < 0 Sign Inversion + Lateral Coverage",
            "target_contour": downsample_contour(crab_chorinus["BT"], 180),
            "target_lms": np.round(crab_chorinus["T14"], 4).tolist(),
            "states": {
                "unwarped": {
                    "label": f"Scholtz et al. (2020) 8-LM TPS (IoU = {crab_chorinus['silhouette_iou_scholtz_tps_8lm']:.3f}, 0 Lateral LMs)",
                    "contour": downsample_contour(crab_chorinus["B_8lm"], 180),
                    "lms": np.round(crab_chorinus["S14"], 4).tolist(),
                    "r2_pct": round(crab_chorinus["silhouette_iou_scholtz_tps_8lm"] * 100.0, 1),
                    "bij": "0/8 Lateral LMs (Midline Only)",
                    "note": f"Scholtz et al. rightly proved Thompson inverted the spider-crab fan sign (true f₁ = {crab_chorinus['fan_parameter_f1']:+.2f} < 0); in Fig. 9, 8 midline LMs yield IoU = {crab_chorinus['silhouette_iou_scholtz_tps_8lm']:.3f}.",
                },
                "trap_or_box": {
                    "label": f"4p Rectangular Box Stretch (IoU = {crab_chorinus['silhouette_iou_box_4p']:.3f}, Disp. R² = {crab_chorinus['r2_box_4p_pct']:.1f}%)",
                    "contour": downsample_contour(crab_chorinus["B_box"], 180),
                    "lms": np.round(crab_chorinus["S14"], 4).tolist(),
                    "r2_pct": crab_chorinus["r2_box_4p_pct"],
                    "bij": "100% (det J > 0)",
                    "note": f"Plain Box Stretch explains {crab_chorinus['r2_box_4p_pct']:.1f}% of landmark displacement (IoU = {crab_chorinus['silhouette_iou_box_4p']:.3f}) without pyriform anterior convergence.",
                },
                "thompson": {
                    "label": f"14-LM Corrected f₁ < 0 Thompson Grid (IoU = {crab_chorinus['silhouette_iou_thompson_8p_on_14lm']:.3f}, TPS-14 = {crab_chorinus['silhouette_iou_tps_14lm_with_lateral']:.3f})",
                    "contour": downsample_contour(crab_chorinus["B_th"], 180),
                    "lms": np.round(crab_chorinus["T14"], 4).tolist(),
                    "r2_pct": crab_chorinus["r2_thompson_8p_pct"],
                    "bij": f"100% (min det J = {crab_chorinus['min_jacobian_det']:.2f})",
                    "note": f"Adding 6 post-orbital lateral semilandmarks + anteriorly converging f₁ = {crab_chorinus['fan_parameter_f1']:+.2f} lifts silhouette IoU to {crab_chorinus['silhouette_iou_thompson_8p_on_14lm']:.3f} (TPS-14 IoU = {crab_chorinus['silhouette_iou_tps_14lm_with_lateral']:.3f}, R² = {crab_chorinus['r2_thompson_8p_pct']:.1f}%)!",
                },
            },
        },
        "crab_anomura_lithodes": {
            "title": "Geryon → Paralomis (1917) / Lithodes maja (Convergent King Crab)",
            "subtitle": "Fig. 369d — Convergent Carcinization from Hermit Crab Ancestor (Anomura)",
            "target_contour": downsample_contour(crab_lithodes["BT"], 180),
            "target_lms": np.round(crab_lithodes["T14"], 4).tolist(),
            "states": {
                "unwarped": {
                    "label": f"Scholtz et al. (2020) 8-LM TPS (IoU = {crab_lithodes['silhouette_iou_scholtz_tps_8lm']:.3f}, 0 Lateral LMs)",
                    "contour": downsample_contour(crab_lithodes["B_8lm"], 180),
                    "lms": np.round(crab_lithodes["S14"], 4).tolist(),
                    "r2_pct": round(crab_lithodes["silhouette_iou_scholtz_tps_8lm"] * 100.0, 1),
                    "bij": "0/8 Lateral LMs (Midline Only)",
                    "note": f"Scholtz et al. rightly noted Lithodes is a carcinized anomuran; their 8 midline LMs achieve IoU = {crab_lithodes['silhouette_iou_scholtz_tps_8lm']:.3f} (LOO lateral ratio = {crab_lithodes['loocv_lateral_improvement_ratio_x']:.2f}×).",
                },
                "trap_or_box": {
                    "label": f"4p Rectangular Box Stretch (Disp. R² = {crab_lithodes['r2_box_4p_pct']:.1f}%, IoU = {crab_lithodes['silhouette_iou_box_4p']:.3f})",
                    "contour": downsample_contour(crab_lithodes["B_box"], 180),
                    "lms": np.round(crab_lithodes["S14"], 4).tolist(),
                    "r2_pct": crab_lithodes["r2_box_4p_pct"],
                    "bij": "100% (det J > 0)",
                    "note": f"A rectangular box stretch explains only {crab_lithodes['r2_box_4p_pct']:.1f}% of landmark displacement (IoU = {crab_lithodes['silhouette_iou_box_4p']:.3f}) on the triangular lithodid carapace!",
                },
                "thompson": {
                    "label": f"14-LM Thompson 8p Fan (Disp. R² = {crab_lithodes['r2_thompson_8p_pct']:.1f}%, IoU = {crab_lithodes['silhouette_iou_thompson_8p_on_14lm']:.3f}, TPS-14 = {crab_lithodes['silhouette_iou_tps_14lm_with_lateral']:.3f})",
                    "contour": downsample_contour(crab_lithodes["B_th"], 180),
                    "lms": np.round(crab_lithodes["T14"], 4).tolist(),
                    "r2_pct": crab_lithodes["r2_thompson_8p_pct"],
                    "bij": f"100% (min det J = {crab_lithodes['min_jacobian_det']:.2f})",
                    "note": f"Both the anomuran king crab Lithodes (f₁ = {crab_lithodes['fan_parameter_f1']:+.2f}) and brachyuran spider crabs Pisa/Leptomithrax share the same f₁ < 0 anteriorly converging fan!",
                },
            },
        },
        "copepod_oithona_sapphirina": {
            "title": "Oithona nana → Sapphirina (Figs. 367–368, p. 743 Table)",
            "subtitle": "Slender Cyclopoid → Foliaceous Sapphirinid Copepod (77 Grid Nodes)",
            "target_contour": np.round(cop_res["C_sapp"], 4).tolist(),
            "target_lms": np.round(cop_res["T0"][::3], 4).tolist(),
            "states": {
                "unwarped": {
                    "label": "Unwarped Oithona nana (Slender Cyclopoid Baseline)",
                    "contour": np.round(cop_res["C_oith"], 4).tolist(),
                    "lms": np.round(cop_res["S0"][::3], 4).tolist(),
                    "r2_pct": 0.0,
                    "bij": "100% (det J = 1.00)",
                    "note": "Slender Oithona nana inscribed in equidistant rectangular coordinates (Fig. 367, p. 743).",
                },
                "trap_or_box": {
                    "label": "4p Rectangular Box Stretch (20.4% R² — Misses Broad Urosome)",
                    "contour": np.round(cop_res["C_box"], 4).tolist(),
                    "lms": np.round(cop_res["P_box"][::3], 4).tolist(),
                    "r2_pct": cop_res["box_4p_r2_pct"],
                    "bij": "100% (det J > 0)",
                    "note": f"Uniform linear scaling explains only {cop_res['box_4p_r2_pct']:.1f}% of Thompson's p. 743 coordinate lookup table!",
                },
                "thompson": {
                    "label": "8p Thompson Logarithmic-Quadratic Coordinate Grid (+63.8 pts!)",
                    "contour": np.round(cop_res["C_th"], 4).tolist(),
                    "lms": np.round(cop_res["P_th"][::3], 4).tolist(),
                    "r2_pct": cop_res["thompson_8p_log_quad_r2_pct"],
                    "bij": f"100% (min det J = {cop_res['min_jacobian_det']:.2f})",
                    "note": f"Thompson's logarithmic abscissal spacing + quadratic longitudinal gradient explains {cop_res['thompson_8p_log_quad_r2_pct']:.1f}% (+{cop_res['gain_over_box_pct_pts']:.1f} pts over Box)!",
                },
            },
        },
        "amphipod_harpinia_hyperia": {
            "title": "Harpinia plumosa → Hyperia galba (Fig. 370, p. 746)",
            "subtitle": "Benthic Phoxocephalid → Pelagic Hyperiid Amphipod (Radial-Fan Grid)",
            "target_contour": np.round(amp_res["harpinia_to_hyperia"]["B_tgt"], 4).tolist(),
            "target_lms": np.round(amp_res["harpinia_to_hyperia"]["T0"], 4).tolist(),
            "states": {
                "unwarped": {
                    "label": "Unwarped Harpinia plumosa (Rectangular Grid 1)",
                    "contour": np.round(amp_res["harpinia_to_hyperia"]["B_src"], 4).tolist(),
                    "lms": np.round(amp_res["harpinia_to_hyperia"]["S0"], 4).tolist(),
                    "r2_pct": 0.0,
                    "bij": "100% (det J = 1.00)",
                    "note": "Elongate benthic amphipod Harpinia plumosa in rectangular coordinates (Fig. 370, 1).",
                },
                "trap_or_box": {
                    "label": "4p Rectangular Box Stretch (43.8% R² — No Dorsal Arch)",
                    "contour": np.round(amp_res["harpinia_to_hyperia"]["B_box"], 4).tolist(),
                    "lms": np.round(amp_res["harpinia_to_hyperia"]["P_box"], 4).tolist(),
                    "r2_pct": amp_res["harpinia_to_hyperia"]["box_4p_r2_pct"],
                    "bij": "100% (det J > 0)",
                    "note": "Plain Box Stretch cannot bend the dorsal pereon arch or expand the anterior cephalon/eye!",
                },
                "thompson": {
                    "label": "8p Thompson Radial-Arch Coordinate Grid (89.4% R², +45.6 pts!)",
                    "contour": np.round(amp_res["harpinia_to_hyperia"]["B_th"], 4).tolist(),
                    "lms": np.round(amp_res["harpinia_to_hyperia"]["P_th"], 4).tolist(),
                    "r2_pct": amp_res["harpinia_to_hyperia"]["thompson_8p_r2_pct"],
                    "bij": f"100% (min det J = {amp_res['harpinia_to_hyperia']['min_jacobian_det']:.2f})",
                    "note": "Thompson's radial-arch network bends the dorsal tergites and expands the anterior head capsule with zero grid folding!",
                },
            },
        },
    }

    # =========================================================================
    # 1. WRITE STRUCTURED JSON FEED FOR BIOELECTRICITY.NET
    # =========================================================================
    base_url = "https://fractastical.github.io/morpholib"
    feed = {
        "schema_version": "1.0.0",
        "target_site": "https://bioelectricity.net",
        "primary_author": {
            "name": "Joel Dietz",
            "affiliation": "Project Delphi",
            "github": "https://github.com/fractastical/morpholib",
        },
        "embed_urls": {
            "interactive_module_iframe": f"{base_url}/bioelectricity/",
            "full_morphospace_explorer_iframe": f"{base_url}/",
            "visual_guide_pdf": f"{base_url}/thompson_1917_visual_guide_for_everyone.pdf",
            "academic_manuscript_pdf": f"{base_url}/thompson_1917_fish_transformations_manuscript.pdf",
            "rule30_conus_manuscript_pdf": f"{base_url}/conus_textile_morphogenesis_to_rule30_manuscript.pdf",
        },
        "featured_discoveries": [
            {
                "id": "thompson-1917-morphospace-vs-competition",
                "title": "How Nature Builds 10,000 Fishes: Resolving D'Arcy Thompson's 1917 Transformation Paradox Across 10,303 Species",
                "author": "Joel Dietz",
                "date": "2026-10-22",
                "category": "Morphogenesis & Bioelectric Prepatterns",
                "tldr_13yo": "Global growth rules (D'Arcy Thompson's curved rubber sheet) shape the outside of a fish for swimming over 150 million years (94%–99% alignment across 440 families), while competition with neighbors rewires the jaws and snout on the inside (3.3× to 9.3× faster) so sister species can share the same coral reef without fighting over the same food.",
                "bioelectric_connection": "Directly parallels bioelectric morphogenesis in two ways: (1) Global V_mem axial-dorsoventral gradients establish the low-dimensional coordinate field (Thompson's 8-parameter rubber sheet) across the somitic trunk and median fin folds; (2) Local neural-crest craniofacial ionocyte domains ('Picasso frog' / teleost trophic modules) act as decoupled bioelectric subroutines that rapidly rewire jaw lever mechanics under sympatric trophic competition.",
                "hero_images": [
                    f"{base_url}/guide_step1_1917_rubber_sheet.png",
                    f"{base_url}/guide_step2_17dot_head_trap.png",
                    f"{base_url}/guide_step3_box_cheat_vs_curves.png",
                    f"{base_url}/guide_step4_outside_vs_inside_rules.png",
                    f"{base_url}/guide_step5_10000_fishes_scoreboard.png",
                ],
                "animated_gif": f"{base_url}/media/thompson_1917_silhouette_paradox_summary.gif",
                "visual_guide_pdf": f"{base_url}/thompson_1917_visual_guide_for_everyone.pdf",
                "manuscript_pdf": f"{base_url}/thompson_1917_fish_transformations_manuscript.pdf",
            },
            {
                "id": "wolfram-rule30-conus-neural-mantle-isomorphism",
                "title": "From Bioelectric Mantle Excitation to Wolfram's Rule 30: Infinite Period-Doubling, Exact k₅ = 87,867 Horizon & Lean 4 Neural-Gate Isomorphism",
                "author": "Joel Dietz",
                "date": "2026-10-21",
                "category": "Neural / Bioelectric Cellular Automata",
                "tldr_13yo": "Proves in Lean 4 that the Ermentrout–Campbell–Oster (1986) neurosecretory mantle threshold equation is bit-for-bit isomorphic to Wolfram's Rule 30 cellular automaton, proves unconditionally that left-cone period-doubling never cycles (lim P_k = ∞), and resolves the exact unit-seed P = 16 → 32 horizon at diagonal k₅ = 87,867.",
                "animated_gif": f"{base_url}/media/conus_textile_rule30.gif",
                "progress_gif": f"{base_url}/media/rule30_prize_progress.gif",
                "manuscript_pdf": f"{base_url}/conus_textile_morphogenesis_to_rule30_manuscript.pdf",
            },
        ],
    }
    (OUT_DIR / "bioelectricity_nexus_feed.json").write_text(json.dumps(feed, indent=2), encoding="utf-8")

    # =========================================================================
    # 2. WRITE STANDALONE & IFRAME-EMBEDDABLE INTERACTIVE APP (index.html)
    # =========================================================================
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>How Nature Builds 10,000 Fishes — By Joel Dietz | Bioelectricity Nexus & Project Delphi</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg-deep: #070b14;
      --bg-panel: #0f172a;
      --bg-card: #131e36;
      --bg-card-hover: #192747;
      --border: #263859;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --accent-cyan: #38bdf8;
      --accent-emerald: #22c55e;
      --accent-amber: #f59e0b;
      --accent-rose: #f43f5e;
      --accent-purple: #a855f7;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg-deep);
      color: var(--text-main);
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      line-height: 1.6;
      padding-bottom: 60px;
    }}
    h1, h2, h3, .font-display {{
      font-family: 'Space Grotesk', sans-serif;
    }}
    code, pre, .font-mono {{
      font-family: 'JetBrains Mono', monospace;
    }}
    .container {{
      max-width: 1240px;
      margin: 0 auto;
      padding: 0 24px;
    }}
    .hero {{
      background: radial-gradient(circle at 20% 20%, rgba(56, 189, 248, 0.14), transparent 55%),
                  radial-gradient(circle at 80% 30%, rgba(34, 197, 94, 0.12), transparent 55%),
                  linear-gradient(180deg, #0b1326 0%, #070b14 100%);
      border-bottom: 1px solid var(--border);
      padding: 36px 0 28px;
    }}
    .badge-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      margin-bottom: 14px;
    }}
    .badge {{
      background: rgba(56, 189, 248, 0.14);
      border: 1px solid rgba(56, 189, 248, 0.4);
      color: var(--accent-cyan);
      padding: 4px 12px;
      border-radius: 999px;
      font-size: 0.78rem;
      font-weight: 700;
      letter-spacing: 0.3px;
    }}
    .badge.amber {{
      background: rgba(245, 158, 11, 0.15);
      border-color: rgba(245, 158, 11, 0.45);
      color: #fbbf24;
    }}
    .badge.emerald {{
      background: rgba(34, 197, 94, 0.15);
      border-color: rgba(34, 197, 94, 0.45);
      color: #4ade80;
    }}
    .hero h1 {{
      font-size: clamp(1.75rem, 3.2vw, 2.6rem);
      font-weight: 700;
      line-height: 1.2;
      margin-bottom: 12px;
      color: #ffffff;
    }}
    .hero-sub {{
      font-size: 1.06rem;
      color: #cbd5e1;
      max-width: 960px;
      margin-bottom: 20px;
    }}
    .tldr-banner {{
      background: linear-gradient(90deg, rgba(245, 158, 11, 0.14) 0%, rgba(56, 189, 248, 0.12) 100%);
      border: 1px solid rgba(245, 158, 11, 0.45);
      border-left: 5px solid var(--accent-amber);
      border-radius: 12px;
      padding: 16px 20px;
      margin-bottom: 22px;
      font-size: 0.98rem;
      color: #f8fafc;
    }}
    .nav-tabs {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 18px;
    }}
    .tab-btn {{
      background: var(--bg-card);
      border: 1px solid var(--border);
      color: #cbd5e1;
      padding: 10px 18px;
      border-radius: 10px;
      font-family: 'Space Grotesk', sans-serif;
      font-size: 0.92rem;
      font-weight: 700;
      cursor: pointer;
      transition: all 0.16s ease;
    }}
    .tab-btn:hover {{
      background: var(--bg-card-hover);
      color: #fff;
      border-color: var(--accent-cyan);
    }}
    .tab-btn.active {{
      background: var(--accent-cyan);
      color: #06101e;
      border-color: #fff;
      box-shadow: 0 0 20px rgba(56, 189, 248, 0.35);
    }}
    .action-links {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 14px;
    }}
    .btn-link {{
      display: inline-flex;
      align-items: center;
      gap: 7px;
      background: rgba(255, 255, 255, 0.06);
      border: 1px solid var(--border);
      color: #e2e8f0;
      padding: 7px 14px;
      border-radius: 8px;
      font-size: 0.84rem;
      font-weight: 600;
      text-decoration: none;
      transition: all 0.15s ease;
    }}
    .btn-link:hover {{
      border-color: var(--accent-emerald);
      color: #fff;
      background: rgba(34, 197, 94, 0.14);
    }}
    .tab-pane {{
      display: none;
      padding-top: 30px;
    }}
    .tab-pane.active {{
      display: block;
    }}
    .step-card {{
      background: var(--bg-panel);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 26px;
      margin-bottom: 28px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.35);
    }}
    .step-header {{
      display: flex;
      align-items: center;
      gap: 12px;
      margin-bottom: 12px;
    }}
    .step-num {{
      background: var(--accent-cyan);
      color: #04101f;
      font-family: 'Space Grotesk', sans-serif;
      font-weight: 800;
      font-size: 0.9rem;
      padding: 4px 12px;
      border-radius: 999px;
    }}
    .step-card h2 {{
      font-size: 1.4rem;
      color: #fff;
    }}
    .step-card p {{
      color: #cbd5e1;
      font-size: 1.0rem;
      margin-bottom: 14px;
    }}
    .step-img {{
      width: 100%;
      border-radius: 12px;
      border: 1px solid #334155;
      background: #fff;
      margin-top: 8px;
    }}
    .callout {{
      background: rgba(34, 197, 94, 0.1);
      border-left: 4px solid var(--accent-emerald);
      padding: 14px 18px;
      border-radius: 8px;
      margin-top: 14px;
      color: #e2e8f0;
      font-size: 0.95rem;
    }}
    .callout.rose {{
      background: rgba(244, 63, 94, 0.12);
      border-left-color: var(--accent-rose);
    }}
    .callout.amber {{
      background: rgba(245, 158, 11, 0.12);
      border-left-color: var(--accent-amber);
    }}
    /* Interactive Sandbox Layout */
    .sandbox-grid {{
      display: grid;
      grid-template-columns: 360px 1fr;
      gap: 24px;
      align-items: start;
    }}
    @media (max-width: 920px) {{
      .sandbox-grid {{ grid-template-columns: 1fr; }}
    }}
    .controls-card {{
      background: var(--bg-panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 20px;
    }}
    .control-group {{
      margin-bottom: 16px;
    }}
    .control-group label {{
      display: flex;
      justify-content: space-between;
      font-size: 0.84rem;
      font-weight: 600;
      color: #cbd5e1;
      margin-bottom: 6px;
    }}
    .control-group select, .control-group input[type=range] {{
      width: 100%;
    }}
    .control-group select {{
      background: #080d1a;
      color: #fff;
      border: 1px solid var(--border);
      padding: 8px 10px;
      border-radius: 8px;
      font-size: 0.88rem;
    }}
    .preset-btn {{
      display: block;
      width: 100%;
      text-align: left;
      background: var(--bg-card);
      border: 1px solid var(--border);
      color: #e2e8f0;
      padding: 10px 12px;
      border-radius: 8px;
      font-size: 0.84rem;
      font-weight: 600;
      margin-bottom: 8px;
      cursor: pointer;
      transition: all 0.15s ease;
    }}
    .preset-btn:hover {{
      border-color: var(--accent-cyan);
      background: var(--bg-card-hover);
    }}
    .preset-btn.active {{
      border-color: var(--accent-emerald);
      background: rgba(34, 197, 94, 0.16);
      color: #fff;
    }}
    .canvas-card {{
      background: var(--bg-panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 20px;
      text-align: center;
    }}
    canvas#morphCanvas {{
      width: 100%;
      max-width: 760px;
      height: auto;
      background: #ffffff;
      border-radius: 12px;
      border: 1px solid #475569;
    }}
    .metrics-bar {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
      margin-top: 14px;
    }}
    .metric-box {{
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px;
    }}
    .metric-val {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 1.25rem;
      font-weight: 700;
      color: var(--accent-cyan);
    }}
    .metric-lbl {{
      font-size: 0.76rem;
      color: var(--text-muted);
    }}
    pre.code-block {{
      background: #050811;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 16px;
      overflow-x: auto;
      color: #38bdf8;
      font-size: 0.84rem;
      margin-top: 10px;
    }}
  </style>
</head>
<body>

  <section class="hero">
    <div class="container">
      <div class="badge-row">
        <span class="badge">BIOELECTRICITY NEXUS × PROJECT DELPHI</span>
        <span class="badge amber">PRIMARY AUTHOR: JOEL DIETZ</span>
        <span class="badge emerald">10,303 FISH SPECIES · 440 FAMILIES · LEAN 4 VERIFIED</span>
      </div>
      <h1>How Nature Builds 10,000 Different Fishes</h1>
      <p class="hero-sub">
        Resolving the 109-Year Paradox in D'Arcy Thompson's 1917 Coordinate Transformation Grids — and Separating Global Bioelectric/Morphogenetic Growth Highways From Local Trophic Competition.
      </p>

      <div class="tldr-banner">
        <strong>⚡ The 60-Second Story:</strong> Global growth rules (D'Arcy Thompson's curved rubber sheet) shape the <em>outside</em> of a fish for swimming over 150 million years (<strong>94%–99% alignment across 440 families</strong>), while competition with neighbors rewires the <em>jaws and snout</em> on the inside (<strong>3.3× to 9.3× faster</strong>) so sister species can share the same coral reef without fighting over the same food.
      </div>

      <div class="action-links">
        <a class="btn-link" href="../thompson_1917_visual_guide_for_everyone.pdf" target="_blank">📄 Visual Guide PDF (5 Pages, For Everyone)</a>
        <a class="btn-link" href="../thompson_1917_fish_transformations_manuscript.pdf" target="_blank">🔬 Full Academic Paper PDF (20 Pages)</a>
        <a class="btn-link" href="../KEY_DISCOVERIES.md" target="_blank">📘 Key Discoveries Deep-Dive</a>
        <a class="btn-link" href="./bioelectricity_nexus_feed.json" target="_blank">🔌 JSON API Feed for bioelectricity.net</a>
        <a class="btn-link" href="../index.html">🏛️ 116-Model Aristotelian Morphospace Explorer</a>
      </div>

      <div class="nav-tabs">
        <button class="tab-btn active" onclick="switchTab('guide', this)">1. Visual Guide (For Everyone)</button>
        <button class="tab-btn" onclick="switchTab('sandbox', this)">2. Interactive 600-Point Fish Grid Sandbox</button>
        <button class="tab-btn" onclick="switchTab('arthropods', this)">3. Arthropods: Ant Castes &amp; Crab Carapaces</button>
        <button class="tab-btn" onclick="switchTab('bioelectric', this)">4. Bioelectric Morphogenesis Connection</button>
        <button class="tab-btn" onclick="switchTab('embed', this)">5. Embed on bioelectricity.net (iframe / React / JSON)</button>
      </div>
    </div>
  </section>

  <main class="container">

    <!-- ================================================================== -->
    <!-- TAB 1: VISUAL GUIDE FOR EVERYONE (5 LARGE-FONT SEPARATE DIAGRAMS)  -->
    <!-- ================================================================== -->
    <div id="tab-guide" class="tab-pane active">

      <div class="step-card">
        <div class="step-header">
          <span class="step-num">STEP 1</span>
          <h2>D'Arcy Thompson's 1917 "Rubber Graph-Paper" Idea</h2>
        </div>
        <p>
          In 1917, Scottish biologist and mathematician <strong>D'Arcy Wentworth Thompson</strong> asked a simple question: <em>When one fish species evolves into another, does nature have to redesign every single bone from scratch?</em> He drew a parrotfish (<em>Scarus</em>) on flat graph paper, bent the grid lines into curved circles like a sheet of rubber, and showed that the drawing turned into an angelfish (<em>Pomacanthus</em>). He argued that an embryo only needs to turn a few global <strong>"growth knobs"</strong> to transform one body plan into another.
        </p>
        <img class="step-img" src="../guide_step1_1917_rubber_sheet.png" alt="Step 1: D'Arcy Thompson's 1917 Rubber Graph-Paper Idea" />
      </div>

      <div class="step-card">
        <div class="step-header">
          <span class="step-num" style="background: var(--accent-rose); color: #fff;">STEP 2</span>
          <h2>Mistake #1: Why Computers Thought Thompson Was Wrong ("The 17-Dot Head Trap")</h2>
        </div>
        <p>
          For the last 30 years, biologists tested Thompson's idea on computers and concluded it failed across fish families. Why? Because instead of tracing the <strong>whole outline of the fish</strong>, standard fish databases only measure <strong>17 dots on the skeleton</strong>: <strong>9 of those 17 dots (53% of the math weight) are crammed inside the head and jaws</strong>, and <strong>zero dots (0%) are on the actual outer fins or tail!</strong>
        </p>
        <img class="step-img" src="../guide_step2_17dot_head_trap.png" alt="Step 2: The 17-Dot Head Seesaw Trap" />
        <div class="callout rose">
          <strong>Why the 17 Dots Failed:</strong> Imagine trying to tailor a full-body wetsuit by measuring 9 points on someone's face and 0 points on their arms or legs! Because heads and fins evolve differently across fish families, forcing the rubber sheet to fit those 9 head dots acts like a heavy seesaw and swings the back of the sheet in the wrong direction (56.2% mismatch — worse than not stretching the fish at all!).
        </div>
      </div>

      <div class="step-card">
        <div class="step-header">
          <span class="step-num" style="background: var(--accent-emerald);">STEP 3</span>
          <h2>Mistake #2: Why "Just Stretching a Box" Fails True Anatomy — And Why Thompson's Curves Win</h2>
        </div>
        <p>
          When we switch from 17 skeleton dots to a <strong>600-point outline</strong> of the whole fish, how do we know Thompson's slanted and curved grids are really needed, instead of just stretching a plain rectangle taller or wider ("Plain Box Stretch")?
        </p>
        <img class="step-img" src="../guide_step3_box_cheat_vs_curves.png" alt="Step 3: Box Stretch vs Thompson's Curved Grids" />
        <div class="callout">
          <strong>The "Shadow Cheat" vs. True Anatomy:</strong> A plain rectangular Box Stretch lifts the middle of the back straight up (red <strong>X</strong> in Panel A), missing the swept-back fin tips of the angelfish (only <strong>2.4% match</strong>!). Once you require the grid to map the true <strong>Dorsal Fin Tip</strong>, <strong>Anal Fin Tip</strong>, and fin bases along the body (without the 9 internal skull dots), a plain Box Stretch fails and <strong>Thompson's curved and slanted grids jump by +13% to +48% accuracy</strong>!
        </div>
      </div>

      <div class="step-card">
        <div class="step-header">
          <span class="step-num" style="background: var(--accent-amber);">STEP 4</span>
          <h2>The Big Picture Across 10,303 Species: Two Simple Rules of Fish Evolution</h2>
        </div>
        <p>
          Testing <strong>three independent datasets totaling 10,303 fish species across 987 genera and 440 families</strong> revealed that fish evolution obeys two simple rules working at the same time:
        </p>
        <img class="step-img" src="../guide_step4_outside_vs_inside_rules.png" alt="Step 4: Two Simple Rules of Fish Evolution" />
        <div class="callout amber">
          <strong>Outside = Swimming Wing (Rule 1) · Inside = Food Tool (Rule 2):</strong> Water flows around the whole outer body and fins like an airplane wing, so for 150 million years across 440 families, the outside silhouette follows D'Arcy Thompson's 8-knob rubber sheet (94%–99% match). Meanwhile, sister species living side-by-side on the same coral reef rewire their <strong>jaws, snouts, and eyes 3.3× to 9.3× faster</strong> than their bodies so they can eat different foods (coral vs. plankton vs. algae) without fighting!
        </div>
      </div>

      <div class="step-card">
        <div class="step-header">
          <span class="step-num">STEP 5</span>
          <h2>The Final Scoreboard in Four Simple Numbers</h2>
        </div>
        <img class="step-img" src="../guide_step5_10000_fishes_scoreboard.png" alt="Step 5: Four-Number Scoreboard" />
      </div>

    </div>

    <!-- ================================================================== -->
    <!-- TAB 2: INTERACTIVE 600-POINT THOMPSON COORDINATE GRID SANDBOX      -->
    <!-- ================================================================== -->
    <div id="tab-sandbox" class="tab-pane">
      <div class="sandbox-grid">
        <div class="controls-card">
          <h3 style="margin-bottom: 12px; color: #fff;">Interactive Coordinate Transformer</h3>
          <div class="control-group">
            <label><span>1. Select 1917 Fish Pair:</span></label>
            <select id="pairSelect" onchange="changePair(this.value)">
              <option value="scarus_pomacanthus">Scarus (Parrotfish) → Pomacanthus (Angelfish)</option>
              <option value="pristigenys_scorpaena">Pristigenys (Bigeye) → Scorpaena (Scorpionfish)</option>
              <option value="argyropelecus_sternoptyx">Argyropelecus → Sternoptyx (Hatchetfishes)</option>
              <option value="diodon_mola">Diodon (Porcupinefish) → Mola mola (Sunfish)</option>
            </select>
          </div>

          <div class="control-group">
            <label><span>2. Compare Transformation Method:</span></label>
            <button class="preset-btn" id="btn-unwarped" onclick="setMode('unwarped')">A. Unwarped Baseline (No Stretch)</button>
            <button class="preset-btn" id="btn-lm17" onclick="setMode('lm17')">B. 17-Dot Skeleton Fit (9 Head Dots Trap!)</button>
            <button class="preset-btn" id="btn-box" onclick="setMode('box')">C. Plain Box Stretch (No Curve — Misses Fin Tips)</button>
            <button class="preset-btn active" id="btn-thompson600" onclick="setMode('thompson600')">D. Thompson's Curved/Skewed 600-pt Fit (Recovered!)</button>
          </div>

          <div class="control-group">
            <label>
              <span>3. Warp Progression (α):</span>
              <span id="alphaVal" class="font-mono">100%</span>
            </label>
            <input type="range" id="alphaSlider" min="0" max="100" value="100" oninput="updateAlpha(this.value)" />
          </div>

          <div class="control-group">
            <label style="cursor: pointer;">
              <span>Show 17 Skeleton Dots (9 Gold in Head):</span>
              <input type="checkbox" id="showLmsToggle" checked onchange="drawSandbox()" />
            </label>
          </div>

          <div class="callout" id="sandboxNote" style="margin-top: 10px; font-size: 0.86rem;">
            100% diffeomorphic (det J &gt; 0) curved/skewed grid sweeps fins &amp; body into place!
          </div>
        </div>

        <div class="canvas-card">
          <h3 id="canvasTitle" style="color: #fff; margin-bottom: 4px;">Scarus (Parrotfish) → Pomacanthus (Angelfish)</h3>
          <p id="canvasSub" style="color: var(--accent-cyan); font-size: 0.88rem; margin-bottom: 12px;">II. Coaxial Circles (Figs. 375–376)</p>
          <canvas id="morphCanvas" width="760" height="460"></canvas>
          <div class="metrics-bar">
            <div class="metric-box">
              <div class="metric-val" id="metricErr">28.5%</div>
              <div class="metric-lbl">Non-Overlapping Area |A Δ B| (Lower = Better)</div>
            </div>
            <div class="metric-box">
              <div class="metric-val" id="metricIou">0.749</div>
              <div class="metric-lbl">Whole-Body Silhouette Overlap (IoU)</div>
            </div>
            <div class="metric-box">
              <div class="metric-val" id="metricBij" style="color: var(--accent-emerald);">100% (det J &gt; 0)</div>
              <div class="metric-lbl">2D Tissue Bijectivity (Zero Grid Folding)</div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- ================================================================== -->
    <!-- TAB 3: ARTHROPODS: ANT CASTES, CRAB CARAPACES, COPEPODS & AMPHIPODS -->
    <!-- ================================================================== -->
    <div id="tab-arthropods" class="tab-pane">
      <div class="sandbox-grid">
        <div class="controls-card">
          <h3 style="margin-bottom: 12px; color: #fff;">Interactive Arthropod Grid Lab</h3>
          <div class="control-group">
            <label><span>1. Select Arthropod Clade (Figs. 359, 367–370):</span></label>
            <select id="arthPairSelect" onchange="changeArthPair(this.value)">
              <option value="ant_camponotus_soldier">Ant Soldier Caste: Minor → Major Head (Fig. 359 Cordate)</option>
              <option value="crab_spider_chorinus">Spider Crab: Geryon → Chorinus / Leptomithrax (Fig. 369f)</option>
              <option value="crab_anomura_lithodes">Convergent King Crab: Geryon → Paralomis / Lithodes (Fig. 369d)</option>
              <option value="copepod_oithona_sapphirina">Copepods: Oithona nana → Sapphirina (Figs. 367–368)</option>
              <option value="amphipod_harpinia_hyperia">Amphipods: Harpinia → Hyperia galba (Fig. 370)</option>
            </select>
          </div>

          <div class="control-group">
            <label><span>2. Compare Transformation Method:</span></label>
            <button class="preset-btn" id="btn-arth-unwarped" onclick="setArthMode('unwarped')">A. Baseline / 8-LM Midline Trap</button>
            <button class="preset-btn" id="btn-arth-trap_or_box" onclick="setArthMode('trap_or_box')">B. 4p Rectangular Box Stretch (No Curve/Fan)</button>
            <button class="preset-btn active" id="btn-arth-thompson" onclick="setArthMode('thompson')">C. 8p Thompson Coordinate Grid (Recovered!)</button>
          </div>

          <div class="control-group">
            <label>
              <span>3. Warp Progression (α):</span>
              <span id="arthAlphaVal" class="font-mono">100%</span>
            </label>
            <input type="range" id="arthAlphaSlider" min="0" max="100" value="100" oninput="updateArthAlpha(this.value)" />
          </div>

          <div class="callout" id="arthSandboxNote" style="margin-top: 10px; font-size: 0.86rem;">
            Thompson's Bilobed-Fan &amp; Fig. 359 Radial-Cordate law jumps to 79.6% (76.0% median across 42/42 species, p = 4.55e-13)!
          </div>
        </div>

        <div class="canvas-card">
          <h3 id="arthCanvasTitle" style="color: #fff; margin-bottom: 4px;">Camponotus Soldier Head (Minor → Major, 82 LMs)</h3>
          <p id="arthCanvasSub" style="color: var(--accent-cyan); font-size: 0.88rem; margin-bottom: 12px;">Extension of Fig. 359 (p. 732) Ovate → Cordate Radial-Bilobed Law</p>
          <canvas id="arthCanvas" width="760" height="460"></canvas>
          <div class="metrics-bar">
            <div class="metric-box">
              <div class="metric-val" id="arthMetricR2">79.6%</div>
              <div class="metric-lbl">Shape Variance Explained R² (Higher = Better)</div>
            </div>
            <div class="metric-box">
              <div class="metric-val" id="arthMetricWin" style="color: var(--accent-amber);">42 / 42 (p = 4.5e-13)</div>
              <div class="metric-lbl">Cross-Species Statistical Replication</div>
            </div>
            <div class="metric-box">
              <div class="metric-val" id="arthMetricBij" style="color: var(--accent-emerald);">100% (det J &gt; 0)</div>
              <div class="metric-lbl">2D Cuticle Bijectivity (Zero Grid Folding)</div>
            </div>
          </div>
        </div>
      </div>

      <div class="step-card">
        <h2>Resolving 100% of D'Arcy Thompson's (1917) Arthropods + Ant Caste Allometry</h2>
        <p style="margin-top: 8px;">
          In Chapter XVII of <em>On Growth and Form</em> (1917), Thompson tested three crustacean groups (<strong>Figs. 367–368 Copepods</strong>, <strong>Fig. 369 Crab Carapaces</strong>, and <strong>Fig. 370 Amphipods</strong>) and formulated his <strong>Fig. 359 Radial-Cordate Law</strong> (p. 732) for how an ovate structure transforms into a heart-shaped (cordate) bilobed structure around an arrest node. Testing all four across modern 3D laser scans, Antscan (2026) synchrotron micro-CT meshes, and 634 multi-caste landmark configurations reveals the exact same law we found in fishes:
        </p>
        <img class="step-img" src="../outreach_fig8_antscan_caste_allometry_hypotheses.png" alt="Antscan 2026 & Ant Caste Allometry Thompson Tests" />
        <img class="step-img" src="../outreach_fig9_crab_carapace_paradox_resolution.png" alt="Resolving the Scholtz et al. 2020 Crab Carapace Paradox" style="margin-top: 18px;" />
        <img class="step-img" src="../outreach_fig10_complete_arthropod_atlas.png" alt="Complete 1917 Arthropod Atlas: Copepods, Crabs, Amphipods & Ants" style="margin-top: 18px;" />
      </div>
    </div>

    <!-- ================================================================== -->
    <!-- TAB 4: BIOELECTRIC MORPHOGENESIS CONNECTION                        -->
    <!-- ================================================================== -->
    <div id="tab-bioelectric" class="tab-pane">
      <div class="step-card">
        <h2>Why This Matters for Bioelectricity &amp; Anatomical Compilers</h2>
        <p style="margin-top: 10px;">
          In bioelectric morphogenesis (Levin 2014, 2021; Cervera, Levin &amp; Mafé 2021), resting membrane voltage gradients (<strong>V<sub>mem</sub> prepatterns</strong>) across gap-junction-coupled cell sheets act as low-dimensional target-morphology coordinates that instruct downstream gene expression and tissue growth. Our cross-dataset discovery across <strong>10,303 teleost species</strong> provides a direct macroevolutionary map of how those bioelectric and morphogenetic coordinates are organized:
        </p>
        <div class="callout" style="margin-bottom: 16px;">
          <strong>1. Global Bioelectric Axial/Fin Prepatterns = D'Arcy Thompson's 8-Parameter Coordinate Highway:</strong><br/>
          Across 440 fish families and 150 million years of radiation, whole-body shape evolves strictly along the developmental "lines of least resistance" captured by Thompson's 8-parameter coordinate fields (<strong>|cos θ| = 0.935 to 0.989</strong>). Rather than micromanaging individual mesenchymal cells, bioelectric and morphogen gradients modulate global axial elongation, dorsoventral scaling, and anterior-to-posterior fin-fold fanning.
        </div>
        <div class="callout amber" style="margin-bottom: 16px;">
          <strong>2. Local Craniofacial Bioelectric Subroutines ("Picasso Frogs" &amp; Reef Fish Jaws):</strong><br/>
          Why do the 9 cranial neural-crest landmarks decouple from the body trunk and evolve <strong>3.3× to 9.3× faster</strong> among sympatric sister species? In vertebrate embryogenesis (Vandenberg, Morrie &amp; Adams 2011; Bischof et al. 2020; Larouche, Zelditch &amp; Cloutier 2018), the craniofacial ectoderm and neural crest form localized <strong>V<sub>mem</sub> "electric face" compartments</strong> (H<sup>+</sup>-V-ATPase / K<sup>+</sup> ionocyte domains) that can remodel jaw lever mechanics, eye diameter, and snout length independently of the somitic trunk!
        </div>
        <div class="callout" style="border-left-color: var(--accent-purple); background: rgba(168, 85, 247, 0.12);">
          <strong>3. Machine-Checked in Lean 4: Neurosecretory Mantle Bioelectricity = Wolfram's Rule 30:</strong><br/>
          In <code>morphoeval/lean/MathOfBiology.lean</code> (theorem <code>ermentrout_oster_neural_gate_eq_rule30</code>), we proved with <code>0 sorry</code> that the Ermentrout–Campbell–Oster (1986) neurosecretory mantle excitation-inhibition equation <em>E(l, c, r) = 2l + (c ∨ r) ∈ [1, 2]</em> is <strong>bit-for-bit isomorphic to Wolfram's Rule 30</strong> cellular automaton, while proving unconditionally that Rule 30's left-cone period-doubling cascade never cycles (<em>lim P<sub>k</sub> = ∞</em>) and resolving the exact unit-seed <em>P = 16 → 32</em> transition at diagonal <strong>k<sub>5</sub> = 87,867</strong>.
        </div>
        <img class="step-img" src="../outreach_fig6_three_dataset_validation.png" alt="3-Dataset Validation Across 10,303 Species" style="margin-top: 18px;" />
      </div>
    </div>

    <!-- ================================================================== -->
    <!-- TAB 4: EMBED ON BIOELECTRICITY.NET (IFRAME / REACT / JSON)         -->
    <!-- ================================================================== -->
    <div id="tab-embed" class="tab-pane">
      <div class="step-card">
        <h2>30-Second Integration into <code>bioelectricity.net</code></h2>
        <p style="margin-top: 8px;">
          You can embed this interactive module directly into any page on <strong>bioelectricity.net</strong> (e.g., <code>/research</code>, <code>/tools</code>, or <code>/latest-papers</code>) using any of the three methods below:
        </p>

        <h3 style="margin-top: 18px; color: var(--accent-cyan);">Option A — Responsive <code>&lt;iframe&gt;</code> Embed (Zero Dependencies)</h3>
        <pre class="code-block">&lt;iframe
  src="https://fractastical.github.io/morpholib/bioelectricity/"
  title="How Nature Builds 10,000 Fishes — Interactive D'Arcy Thompson &amp; Bioelectric Morphospace Guide by Joel Dietz"
  style="width: 100%; height: 920px; border: 1px solid #263859; border-radius: 16px; background: #070b14;"
  loading="lazy"
  allowfullscreen
&gt;&lt;/iframe&gt;</pre>

        <h3 style="margin-top: 22px; color: var(--accent-emerald);">Option B — Drop-In React / TypeScript Component (<code>ProjectDelphiWidget.tsx</code>)</h3>
        <p>Copy <a href="./ProjectDelphiWidget.tsx" style="color: var(--accent-cyan);"><code>docs/bioelectricity/ProjectDelphiWidget.tsx</code></a> directly into your <code>bioelectricity.net</code> React codebase and render <code>&lt;ProjectDelphiWidget /&gt;</code>.</p>

        <h3 style="margin-top: 22px; color: var(--accent-amber);">Option C — Structured JSON Feed (<code>bioelectricity_nexus_feed.json</code>)</h3>
        <p>Fetch <a href="./bioelectricity_nexus_feed.json" style="color: var(--accent-cyan);"><code>https://fractastical.github.io/morpholib/bioelectricity/bioelectricity_nexus_feed.json</code></a> to populate publication cards, tool cards, and downloadable PDFs automatically.</p>
      </div>
    </div>

  </main>

  <script>
    const SANDBOX_DATA = {json.dumps(sandbox_data)};
    const ARTHROPOD_DATA = {json.dumps(arthropod_sandbox_data)};
    let currentPair = 'scarus_pomacanthus';
    let currentMode = 'thompson600';
    let currentAlpha = 1.0;

    let currentArthPair = 'ant_camponotus_soldier';
    let currentArthMode = 'thompson';
    let currentArthAlpha = 1.0;

    function switchTab(tabId, btnEl) {{
      document.querySelectorAll('.tab-pane').forEach(el => el.classList.remove('active'));
      document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
      document.getElementById('tab-' + tabId).classList.add('active');
      btnEl.classList.add('active');
      if (tabId === 'sandbox') drawSandbox();
      if (tabId === 'arthropods') drawArthSandbox();
    }}

    function changeArthPair(pid) {{
      currentArthPair = pid;
      drawArthSandbox();
    }}

    function setArthMode(mode) {{
      currentArthMode = mode;
      ['unwarped', 'trap_or_box', 'thompson'].forEach(m => {{
        document.getElementById('btn-arth-' + m).classList.toggle('active', m === mode);
      }});
      drawArthSandbox();
    }}

    function updateArthAlpha(val) {{
      currentArthAlpha = parseFloat(val) / 100.0;
      document.getElementById('arthAlphaVal').textContent = val + '%';
      drawArthSandbox();
    }}

    function drawArthSandbox() {{
      const canvas = document.getElementById('arthCanvas');
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      const W = canvas.width, H = canvas.height;
      ctx.clearRect(0, 0, W, H);

      const pdata = ARTHROPOD_DATA[currentArthPair];
      const stUnw = pdata.states.unwarped;
      const stCur = pdata.states[currentArthMode];
      const tgt = pdata.target_contour;

      document.getElementById('arthCanvasTitle').textContent = pdata.title;
      document.getElementById('arthCanvasSub').textContent = pdata.subtitle + ' — ' + stCur.label;
      document.getElementById('arthSandboxNote').textContent = stCur.note;

      const r2Val = (1 - currentArthAlpha) * stUnw.r2_pct + currentArthAlpha * stCur.r2_pct;
      document.getElementById('arthMetricR2').textContent = r2Val.toFixed(1) + '%';
      document.getElementById('arthMetricR2').style.color = currentArthMode === 'thompson' ? '#22c55e' : '#f59e0b';
      document.getElementById('arthMetricBij').textContent = stCur.bij;

      const src = stUnw.contour.map((pt, i) => [
        (1 - currentArthAlpha) * pt[0] + currentArthAlpha * stCur.contour[i][0],
        (1 - currentArthAlpha) * pt[1] + currentArthAlpha * stCur.contour[i][1]
      ]);

      const allPts = tgt.concat(src);
      let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
      for (const [x, y] of allPts) {{
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }}
      const pad = (maxX - minX) * 0.14;
      minX -= pad; maxX += pad; minY -= pad; maxY += pad;
      const scale = Math.min((W - 60) / (maxX - minX), (H - 60) / (maxY - minY));
      const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
      const toX = x => W / 2 + (x - cx) * scale;
      const toY = y => H / 2 - (y - cy) * scale;

      ctx.strokeStyle = '#e2e8f0';
      ctx.lineWidth = 1;
      const stepG = (maxX - minX) / 8.0;
      for (let gx = minX; gx <= maxX; gx += stepG) {{
        ctx.beginPath();
        ctx.moveTo(toX(gx), 20);
        ctx.lineTo(toX(gx), H - 20);
        ctx.stroke();
      }}
      for (let gy = minY; gy <= maxY; gy += stepG) {{
        ctx.beginPath();
        ctx.moveTo(20, toY(gy));
        ctx.lineTo(W - 20, toY(gy));
        ctx.stroke();
      }}

      ctx.beginPath();
      tgt.forEach(([x, y], idx) => idx === 0 ? ctx.moveTo(toX(x), toY(y)) : ctx.lineTo(toX(x), toY(y)));
      ctx.closePath();
      ctx.fillStyle = 'rgba(244, 63, 94, 0.22)';
      ctx.fill();
      ctx.strokeStyle = '#dc2626';
      ctx.lineWidth = 2.4;
      ctx.setLineDash([7, 5]);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.beginPath();
      src.forEach(([x, y], idx) => idx === 0 ? ctx.moveTo(toX(x), toY(y)) : ctx.lineTo(toX(x), toY(y)));
      ctx.closePath();
      const isGood = currentArthMode === 'thompson';
      ctx.fillStyle = isGood ? 'rgba(34, 197, 94, 0.42)' : 'rgba(249, 115, 22, 0.35)';
      ctx.fill();
      ctx.strokeStyle = isGood ? '#15803d' : '#ea580c';
      ctx.lineWidth = 2.6;
      ctx.stroke();

      if (pdata.target_lms) {{
        pdata.target_lms.forEach(([x, y], i) => {{
          ctx.beginPath();
          ctx.arc(toX(x), toY(y), 5.0, 0, Math.PI * 2);
          ctx.fillStyle = i < 8 ? '#eab308' : '#10b981';
          ctx.fill();
          ctx.strokeStyle = '#0f172a';
          ctx.lineWidth = 1.4;
          ctx.stroke();
        }});
      }}
    }}

    function changePair(pid) {{
      currentPair = pid;
      drawSandbox();
    }}

    function setMode(mode) {{
      currentMode = mode;
      ['unwarped', 'lm17', 'box', 'thompson600'].forEach(m => {{
        document.getElementById('btn-' + m).classList.toggle('active', m === mode);
      }});
      drawSandbox();
    }}

    function updateAlpha(val) {{
      currentAlpha = parseFloat(val) / 100.0;
      document.getElementById('alphaVal').textContent = val + '%';
      drawSandbox();
    }}

    function drawSandbox() {{
      const canvas = document.getElementById('morphCanvas');
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      const W = canvas.width, H = canvas.height;
      ctx.clearRect(0, 0, W, H);

      const pdata = SANDBOX_DATA[currentPair];
      const stUnw = pdata.states.unwarped;
      const stCur = pdata.states[currentMode];
      const tgt = pdata.target_contour;
      const tgtLms = pdata.target_lms;

      document.getElementById('canvasTitle').textContent = pdata.title;
      document.getElementById('canvasSub').textContent = pdata.subtitle + ' — ' + stCur.label;
      document.getElementById('sandboxNote').textContent = stCur.note;

      const errVal = (1 - currentAlpha) * stUnw.area_diff_pct + currentAlpha * stCur.area_diff_pct;
      const iouVal = (1 - currentAlpha) * stUnw.iou + currentAlpha * stCur.iou;
      document.getElementById('metricErr').textContent = errVal.toFixed(1) + '%';
      document.getElementById('metricErr').style.color = (currentMode === 'lm17' && stCur.area_diff_pct > stUnw.area_diff_pct) ? '#f43f5e' : '#38bdf8';
      document.getElementById('metricIou').textContent = iouVal.toFixed(3);

      // Blend source contour and landmarks by currentAlpha:
      const src = stUnw.contour.map((pt, i) => [
        (1 - currentAlpha) * pt[0] + currentAlpha * stCur.contour[i][0],
        (1 - currentAlpha) * pt[1] + currentAlpha * stCur.contour[i][1]
      ]);
      const srcLms = stUnw.lms.map((pt, i) => [
        (1 - currentAlpha) * pt[0] + currentAlpha * stCur.lms[i][0],
        (1 - currentAlpha) * pt[1] + currentAlpha * stCur.lms[i][1]
      ]);

      // Compute bounding box
      const allPts = tgt.concat(src);
      let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
      for (const [x, y] of allPts) {{
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }}
      const pad = 0.08;
      minX -= pad; maxX += pad; minY -= pad; maxY += pad;
      const scale = Math.min((W - 60) / (maxX - minX), (H - 60) / (maxY - minY));
      const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
      const toX = x => W / 2 + (x - cx) * scale;
      const toY = y => H / 2 - (y - cy) * scale;

      // Draw subtle background grid
      ctx.strokeStyle = '#e2e8f0';
      ctx.lineWidth = 1;
      for (let gx = -0.45; gx <= 0.45; gx += 0.1) {{
        ctx.beginPath();
        ctx.moveTo(toX(gx), 20);
        ctx.lineTo(toX(gx), H - 20);
        ctx.stroke();
      }}
      for (let gy = -0.35; gy <= 0.35; gy += 0.1) {{
        ctx.beginPath();
        ctx.moveTo(20, toY(gy));
        ctx.lineTo(W - 20, toY(gy));
        ctx.stroke();
      }}

      // Draw Target Contour (Red Dashed Fill)
      ctx.beginPath();
      tgt.forEach(([x, y], idx) => idx === 0 ? ctx.moveTo(toX(x), toY(y)) : ctx.lineTo(toX(x), toY(y)));
      ctx.closePath();
      ctx.fillStyle = 'rgba(244, 63, 94, 0.22)';
      ctx.fill();
      ctx.strokeStyle = '#dc2626';
      ctx.lineWidth = 2.4;
      ctx.setLineDash([7, 5]);
      ctx.stroke();
      ctx.setLineDash([]);

      // Draw Warped Source Contour (Emerald or Blue Fill)
      ctx.beginPath();
      src.forEach(([x, y], idx) => idx === 0 ? ctx.moveTo(toX(x), toY(y)) : ctx.lineTo(toX(x), toY(y)));
      ctx.closePath();
      const isGood = currentMode === 'thompson600';
      ctx.fillStyle = isGood ? 'rgba(34, 197, 94, 0.42)' : 'rgba(59, 130, 246, 0.38)';
      ctx.fill();
      ctx.strokeStyle = isGood ? '#15803d' : '#1d4ed8';
      ctx.lineWidth = 2.6;
      ctx.stroke();

      // Draw 17 Skeleton Landmarks if enabled
      if (document.getElementById('showLmsToggle').checked) {{
        const headIdx = new Set([0, 1, 10, 11, 12, 13, 14, 15, 16]);
        srcLms.forEach(([x, y], i) => {{
          ctx.beginPath();
          ctx.arc(toX(x), toY(y), headIdx.has(i) ? 6.5 : 5.0, 0, Math.PI * 2);
          ctx.fillStyle = headIdx.has(i) ? '#f59e0b' : '#10b981';
          ctx.fill();
          ctx.strokeStyle = '#0f172a';
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }});
      }}
    }}

    window.addEventListener('DOMContentLoaded', drawSandbox);
  </script>
</body>
</html>
"""
    (OUT_DIR / "index.html").write_text(html_content, encoding="utf-8")

    # =========================================================================
    # 3. WRITE DROP-IN REACT COMPONENT (ProjectDelphiWidget.tsx)
    # =========================================================================
    tsx_content = """import React, { useState } from 'react';

const BASE_URL = 'https://fractastical.github.io/morpholib';

export interface ProjectDelphiWidgetProps {
  defaultTab?: 'guide' | 'sandbox' | 'bioelectric';
  height?: number;
}

/**
 * Drop-in React component for bioelectricity.net (Bioelectricity Nexus).
 * Primary Author: Joel Dietz (Project Delphi)
 */
export const ProjectDelphiWidget: React.FC<ProjectDelphiWidgetProps> = ({
  height = 900,
}) => {
  const [view, setView] = useState<'cards' | 'interactive'>('cards');

  return (
    <section
      style={{
        background: '#070b14',
        color: '#f8fafc',
        borderRadius: 16,
        border: '1px solid #263859',
        padding: 28,
        fontFamily: "'Inter', sans-serif",
      }}
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between', gap: 12, marginBottom: 16 }}>
        <div>
          <span
            style={{
              background: 'rgba(56, 189, 248, 0.14)',
              border: '1px solid rgba(56, 189, 248, 0.4)',
              color: '#38bdf8',
              padding: '4px 12px',
              borderRadius: 999,
              fontSize: 12,
              fontWeight: 700,
            }}
          >
            RESEARCH MODULE · BY JOEL DIETZ
          </span>
          <h2 style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: 26, fontWeight: 700, marginTop: 10 }}>
            How Nature Builds 10,000 Different Fishes: Bioelectric Morphospace vs. Trophic Competition
          </h2>
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <button
            onClick={() => setView('cards')}
            style={{
              background: view === 'cards' ? '#38bdf8' : '#131e36',
              color: view === 'cards' ? '#06101e' : '#e2e8f0',
              border: '1px solid #263859',
              borderRadius: 8,
              padding: '8px 14px',
              fontWeight: 700,
              cursor: 'pointer',
            }}
          >
            Visual Guide
          </button>
          <button
            onClick={() => setView('interactive')}
            style={{
              background: view === 'interactive' ? '#38bdf8' : '#131e36',
              color: view === 'interactive' ? '#06101e' : '#e2e8f0',
              border: '1px solid #263859',
              borderRadius: 8,
              padding: '8px 14px',
              fontWeight: 700,
              cursor: 'pointer',
            }}
          >
            Interactive 600-pt Grid Lab
          </button>
          <a
            href={`${BASE_URL}/thompson_1917_visual_guide_for_everyone.pdf`}
            target="_blank"
            rel="noreferrer"
            style={{
              background: 'rgba(34, 197, 94, 0.15)',
              border: '1px solid rgba(34, 197, 94, 0.45)',
              color: '#4ade80',
              borderRadius: 8,
              padding: '8px 14px',
              fontWeight: 700,
              textDecoration: 'none',
              fontSize: 14,
            }}
          >
            Download 5-Page Visual Guide PDF
          </a>
        </div>
      </div>

      {view === 'interactive' ? (
        <iframe
          src={`${BASE_URL}/bioelectricity/`}
          title="Interactive D'Arcy Thompson & Bioelectric Morphospace Lab by Joel Dietz"
          style={{ width: '100%', height, border: '1px solid #263859', borderRadius: 12, background: '#070b14' }}
          loading="lazy"
        />
      ) : (
        <div style={{ display: 'grid', gap: 20 }}>
          <img
            src={`${BASE_URL}/guide_step1_1917_rubber_sheet.png`}
            alt="Step 1: D'Arcy Thompson's 1917 Rubber Graph-Paper Idea"
            style={{ width: '100%', borderRadius: 12, background: '#fff' }}
          />
          <img
            src={`${BASE_URL}/guide_step2_17dot_head_trap.png`}
            alt="Step 2: The 17-Dot Head Seesaw Trap"
            style={{ width: '100%', borderRadius: 12, background: '#fff' }}
          />
          <img
            src={`${BASE_URL}/guide_step3_box_cheat_vs_curves.png`}
            alt="Step 3: Plain Box Stretch vs Thompson's Curved Grids"
            style={{ width: '100%', borderRadius: 12, background: '#fff' }}
          />
          <img
            src={`${BASE_URL}/guide_step4_outside_vs_inside_rules.png`}
            alt="Step 4: Two Simple Rules of Fish Evolution Across 10,303 Species"
            style={{ width: '100%', borderRadius: 12, background: '#fff' }}
          />
          <img
            src={`${BASE_URL}/guide_step5_10000_fishes_scoreboard.png`}
            alt="Step 5: The Four-Number Scoreboard Across 10,303 Species"
            style={{ width: '100%', borderRadius: 12, background: '#fff' }}
          />
        </div>
      )}
    </section>
  );
};

export default ProjectDelphiWidget;
"""
    (OUT_DIR / "ProjectDelphiWidget.tsx").write_text(tsx_content, encoding="utf-8")

    # =========================================================================
    # 4. WRITE INTEGRATION README (docs/bioelectricity/README.md)
    # =========================================================================
    readme_md = """# `bioelectricity.net` Integration Package (By Joel Dietz)

This folder (`docs/bioelectricity/`) provides a turnkey integration bundle so our D'Arcy Thompson 1917 / 10,303-species Morphospace & Bioelectric Prepattern discoveries (and Lean-4-verified *Conus textile* Rule 30 neural-gate proofs) can be dropped straight into **[bioelectricity.net](https://bioelectricity.net)** (*Bioelectricity Nexus*).

## Live Hosted Endpoints (via GitHub Pages)

- **Interactive Web App & `<iframe />` Widget:**  
  `https://fractastical.github.io/morpholib/bioelectricity/`
- **Structured JSON API Feed:**  
  `https://fractastical.github.io/morpholib/bioelectricity/bioelectricity_nexus_feed.json`
- **Drop-In React / TypeScript Component (`ProjectDelphiWidget.tsx`):**  
  [`docs/bioelectricity/ProjectDelphiWidget.tsx`](./ProjectDelphiWidget.tsx)
- **5-Page Visual Guide PDF (13-Year-Old Level, Large-Font Separate Diagrams):**  
  `https://fractastical.github.io/morpholib/thompson_1917_visual_guide_for_everyone.pdf`
- **20-Page Full Academic Manuscript PDF:**  
  `https://fractastical.github.io/morpholib/thompson_1917_fish_transformations_manuscript.pdf`

---

## 3 Ways to Integrate Into `bioelectricity.net`

### 1. Zero-Dependency `<iframe />` Embed (30 Seconds)
Paste this into any page on `bioelectricity.net`:

```html
<iframe
  src="https://fractastical.github.io/morpholib/bioelectricity/"
  title="How Nature Builds 10,000 Fishes — Interactive D'Arcy Thompson & Bioelectric Morphospace Guide by Joel Dietz"
  style="width: 100%; height: 920px; border: 1px solid #263859; border-radius: 16px; background: #070b14;"
  loading="lazy"
  allowfullscreen
></iframe>
```

### 2. Native React / Vite Component (`ProjectDelphiWidget.tsx`)
Copy [`ProjectDelphiWidget.tsx`](./ProjectDelphiWidget.tsx) into your `bioelectricity.net` `src/components/` folder and render:

```tsx
import { ProjectDelphiWidget } from './components/ProjectDelphiWidget';

export function MorphospacePage() {
  return <ProjectDelphiWidget />;
}
```

### 3. JSON Feed Ingestion (`bioelectricity_nexus_feed.json`)
Fetch `https://fractastical.github.io/morpholib/bioelectricity/bioelectricity_nexus_feed.json` to automatically populate cards in `/publications`, `/tools`, and `/research`.
"""
    (OUT_DIR / "README.md").write_text(readme_md, encoding="utf-8")
    print(f"Built complete bioelectricity.net integration bundle in {OUT_DIR}")


if __name__ == "__main__":
    main()
