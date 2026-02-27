# High-Priority Feature Requests

This document tracks high-priority improvements identified in the codebase review, formatted as actionable feature requests.

## FR-001: Modularize Core Pipeline Scripts

**Priority:** High  
**Area:** Architecture / Maintainability

### Problem
The core analysis scripts are monolithic and difficult to evolve safely:
- `wave-vector-tiff-parser.py` mixes IO, detection, tracking, geometry, annotation, and CSV writing.
- `generate_detection_summary.py` combines detection, matching, validation, plotting, PDF generation, and reporting.

This increases regression risk and slows development.

### Feature Request
Refactor core functionality into importable modules with clear boundaries (IO, detection/tracking, geometry, units, reporting, CLI orchestration).

### Acceptance Criteria
- Core scripts are split into logical modules.
- Public APIs are documented for major components.
- Existing CLI behavior remains backward compatible.
- Regression tests validate parity for key outputs (`spark_tracks.csv`, summary artifacts).

---

## FR-002: Replace Broad Exception Handling with Structured Error Modes

**Priority:** High  
**Area:** Reliability / Debuggability

### Problem
There are many broad `except Exception` handlers in critical paths. This can hide failures and produce partial or degraded outputs without clear fail conditions.

### Feature Request
Introduce typed exceptions, structured warning/error reporting, and strict/lenient execution modes.

### Acceptance Criteria
- Replace broad catches with targeted exception types where possible.
- Add `--strict` mode that fails on critical data/processing errors.
- Keep lenient mode available but with explicit warning logs and counters.
- Output includes a machine-readable error summary for each run.

---

## FR-003: Establish and Enforce Canonical Data Schema + Units

**Priority:** High  
**Area:** Reproducibility / Data Integrity

### Problem
Documentation and outputs are drifting (for example, new unit-related fields can appear in code before downstream docs and checks are synchronized).

### Feature Request
Define a canonical schema for `spark_tracks.csv`, `vector_clusters.csv`, and comparison outputs, including required columns and units.

### Acceptance Criteria
- Single schema source-of-truth file (or module) exists.
- Docs (`wave-vector.md`, pipeline READMEs) are synchronized with schema.
- Outputs include explicit unit metadata (px, em/ecm/emm, μm where applicable).
- Validation step checks schema compliance before analysis completion.

---

## FR-004: Standardize Per-Run Literature Comparison Artifacts (Zoom-Aware)

**Priority:** High  
**Area:** Scientific Comparison / Cross-run Consistency

### Problem
Comparisons to literature are sensitive to zoom/scale differentials. Without per-run scale metadata, comparisons can be invalid.

### Feature Request
Require a dedicated per-run comparison artifact that records zoom and scale assumptions, and compares pipeline speeds against benchmark ranges.

### Acceptance Criteria
- Each run can generate a dedicated comparison file (JSON/CSV) with:
  - `run_id`
  - `zoom`
  - `um_per_px` or derivation fields (`embryo_length_px`, `embryo_length_um`)
  - computed speed summaries
  - benchmark range comparisons
- Batch summary table can combine multiple runs while preserving per-run zoom/scale columns.
- Comparison docs clearly state assumptions used for each run.

---

## FR-005: Add Automated Test and CI Baseline

**Priority:** High  
**Area:** Quality Assurance

### Problem
There is minimal formal test coverage for core pipeline behavior and no CI guardrail to prevent regressions.

### Feature Request
Create a baseline automated test suite and CI workflow focused on data integrity and unit/scale correctness.

### Acceptance Criteria
- Add automated tests for:
  - speed/unit conversion helpers,
  - metadata extraction/parsing (including zoom formats),
  - schema validation for key outputs.
- Add at least one integration smoke test on a small fixture dataset.
- Add CI workflow to run tests on pull requests.

---

## Suggested Implementation Order
1. FR-003 (Schema + units)
2. FR-004 (Zoom-aware comparisons)
3. FR-002 (Error handling modes)
4. FR-001 (Modularization)
5. FR-005 (Testing + CI, expanded continuously during 1-4)
