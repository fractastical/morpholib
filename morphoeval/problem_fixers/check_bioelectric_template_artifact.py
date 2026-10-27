#!/usr/bin/env python3
"""
BIO-TPL-DQ-01 — detect slot-seeded alpha/beta template curves masquerading as
empirical bioelectric_polarity observations in core catalog reference JSON.

Exit 0 when all watched datasets match the registered falsifier (confirmed
template artifact). Exit 1 when any dataset fails the inequality checks or
uses a non-template poson_law string (refuted / needs human review).

Usage:
  python3 morphoeval/problem_fixers/check_bioelectric_template_artifact.py
  python3 morphoeval/problem_fixers/check_bioelectric_template_artifact.py --json
  python3 morphoeval/problem_fixers/check_bioelectric_template_artifact.py --self-test
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

WATCHED = (
    ROOT / "datasets/danio_rerio_bioelectric_polarity/danio_rerio_bioelectric_polarity_reference.json",
    ROOT
    / "datasets/drosophila_melanogaster_bioelectric_polarity/drosophila_melanogaster_bioelectric_polarity_reference.json",
    ROOT
    / "datasets/xenopus_tropicalis_bioelectric_polarity/xenopus_tropicalis_bioelectric_polarity_reference.json",
)

EXPECTED_POSON_LAW = "Bistable V_mem / G_ij Threshold"
MIN_R_LINEAR_ALPHA = 0.99
MIN_R_EMP_SIM = 0.99
GENERATOR_SCRIPT = "morphoeval/run_overnight_1000_loops.py"
GENERATOR_FORMULA = "alpha_scale * x_control**1.25 + beta_offset + slot_seeded_gaussian_noise"


def pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n != len(y) or n < 2:
        return float("nan")
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    den = math.sqrt(sum((xi - mx) ** 2 for xi in x) * sum((yi - my) ** 2 for yi in y))
    return num / den if den else float("nan")


def slot_template_params(slot: int) -> tuple[float, float]:
    alpha = round(1.15 + 0.08 * (slot % 9), 4)
    beta = round(0.25 + 0.03 * (slot % 7), 4)
    return alpha, beta


def evaluate_reference(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    es = data["empirical_series"]
    params = data["parameters"]
    slot = int(data["slot"])
    alpha = float(params["alpha_scale"])
    beta = float(params["beta_offset"])
    x = [float(v) for v in es["x_control"]]
    y_emp = [float(v) for v in es["y_empirical"]]
    y_sim = [float(v) for v in es["y_simulated"]]
    poson = data["aristotelian_categories"]["poson_law"]

    linear_template = [alpha * xi for xi in x]
    slot_formula = [alpha * (xi**1.25) + beta for xi in x]
    exp_alpha, exp_beta = slot_template_params(slot)

    r_emp_linear = pearson(y_emp, linear_template)
    r_emp_sim = pearson(y_emp, y_sim)
    r_emp_slot_formula = pearson(y_emp, slot_formula)

    poson_ok = poson == EXPECTED_POSON_LAW
    linear_ok = r_emp_linear >= MIN_R_LINEAR_ALPHA
    emp_sim_ok = r_emp_sim >= MIN_R_EMP_SIM
    slot_params_match = alpha == exp_alpha and beta == exp_beta

    confirmed_template = poson_ok and linear_ok and emp_sim_ok and slot_params_match

    return {
        "path": str(path.relative_to(ROOT)),
        "sim_id": data.get("sim_id"),
        "slot": slot,
        "poson_law": poson,
        "alpha_scale": alpha,
        "beta_offset": beta,
        "expected_slot_alpha": exp_alpha,
        "expected_slot_beta": exp_beta,
        "pearson_r_y_empirical_vs_alpha_x": round(r_emp_linear, 6),
        "pearson_r_y_empirical_vs_y_simulated": round(r_emp_sim, 6),
        "pearson_r_y_empirical_vs_slot_formula": round(r_emp_slot_formula, 6),
        "stored_pearson_r": es.get("pearson_r"),
        "poson_law_ok": poson_ok,
        "linear_alpha_template_ok": linear_ok,
        "empirical_vs_simulated_ok": emp_sim_ok,
        "slot_parameter_derivation_ok": slot_params_match,
        "bio_tpl_dq_01_confirmed_template_artifact": confirmed_template,
    }


def run_check() -> tuple[int, dict]:
    rows = [evaluate_reference(p) for p in WATCHED]
    all_confirmed = all(r["bio_tpl_dq_01_confirmed_template_artifact"] for r in rows)
    payload = {
        "check_id": "BIO-TPL-DQ-01",
        "generator_script": GENERATOR_SCRIPT,
        "generator_formula": GENERATOR_FORMULA,
        "thresholds": {
            "min_r_y_empirical_vs_alpha_x": MIN_R_LINEAR_ALPHA,
            "min_r_y_empirical_vs_y_simulated": MIN_R_EMP_SIM,
            "expected_poson_law": EXPECTED_POSON_LAW,
        },
        "datasets": rows,
        "all_confirmed_template_artifacts": all_confirmed,
    }
    return (0 if all_confirmed else 1, payload)


def self_test() -> int:
    code, payload = run_check()
    if code != 0:
        print("self-test failed: not all datasets confirmed as template artifacts", file=sys.stderr)
        print(json.dumps(payload, indent=2), file=sys.stderr)
        return 1
    for row in payload["datasets"]:
        if row["pearson_r_y_empirical_vs_alpha_x"] < MIN_R_LINEAR_ALPHA:
            print("self-test failed: linear alpha correlation below threshold", file=sys.stderr)
            return 1
    print("check_bioelectric_template_artifact self-test OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="Print full JSON report on stdout")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()

    code, payload = run_check()
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        for row in payload["datasets"]:
            status = "CONFIRMED_TEMPLATE" if row["bio_tpl_dq_01_confirmed_template_artifact"] else "REFUTED"
            print(
                f"{row['path']}: {status} "
                f"r(emp,alpha*x)={row['pearson_r_y_empirical_vs_alpha_x']} "
                f"r(emp,sim)={row['pearson_r_y_empirical_vs_y_simulated']}"
            )
        print("overall:", "all confirmed template artifacts" if code == 0 else "check failed")
    return code


if __name__ == "__main__":
    sys.exit(main())
