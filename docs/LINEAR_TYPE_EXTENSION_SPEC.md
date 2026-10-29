# Linear-resource proofs for Project Delphi: extension spec

Status: draft for review (issue "Review linear type theory spec" on
[fractastical/project-delphi-ontology](https://github.com/fractastical/project-delphi-ontology)).
Prototype: `morphoeval/lean/LinearResource.lean` (core Lean 4, kernel-checked, 0 `sorry`).
Pre-registered predictions: `predictions/registrations/LINEAR-TYPES-01.json`.

## 1. Summary

Project Delphi 1.3 added four prover-specific terms (`delphi:lean4`, `delphi:leanDeclaration`,
`delphi:leanTheorem`, `delphi:leanVerified`) so that a `delphi:Concept` standing for a mathematical
law can carry its Lean 4 formalisation and whether the kernel accepted it. MorphoEval is the
originating consumer: `morphoeval/math_kg.py` emits those terms for every law in
`morphoeval/lean/MathOfBiology.lean` and the 136 loop modules in `morphoeval/lean/loops/`.

Every one of those theorems is *equational*: a free-standing arithmetic identity over `Int`/`Nat`
stated and proved per law. For a large class of morphogenesis laws (auxin canalization, cell sorting,
somite budgets, vascular and Physarum transport, gap-junction charge) the biological content is a
*conservation or partition* statement: a quantity moves between compartments and is neither created
nor destroyed. That content is better carried by a type discipline than by a per-law identity.

This spec proposes:

1. **Lean side.** A small linear-resource DSL embedded in Lean (no change to Lean itself): a ledger
   of a conserved quantity, programs whose only constructors are `skip`, `transfer`, `seq`, and a
   soundness theorem proved once (`MathOfBiology.Linear.conserves`) that every well-scoped program
   preserves the ledger total. Each biological law becomes an *instantiation*: it builds a program,
   and its conservation theorem is `conserves` applied, not a new identity.
2. **Ontology side.** Four additive terms so a Delphi reader can tell *what kind* of proof backs a
   node and *which toolchain* checked it: `delphi:proofKind`, `delphi:formalSystem`,
   `delphi:toolchainVersion`, `delphi:conservedQuantity`, plus one optional link
   `delphi:soundnessTheorem`. No existing term changes meaning. Minor version (2.1).
3. **Emitter side.** `math_kg.py` and the `math_kg_loops/*.yaml` schema gain the matching fields.

Ten predictions about what this lets us formalise, with priors, are pre-registered and published
as `delphi:TestableHypothesis` nanopublications (section 6).

## 2. How it works now

### 2.1 Ontology terms (Delphi 2.0, unchanged since 1.3)

| Term | Kind | Meaning |
| --- | --- | --- |
| `delphi:lean4` | datatype | Lexical form is Lean 4 source; used on `skos:definition`. |
| `delphi:leanDeclaration` | data property, domain `delphi:Concept` | Fully qualified Lean identifier of the `def` that formalises the law. |
| `delphi:leanTheorem` | data property, repeatable | Fully qualified identifier of a theorem proved about it. |
| `delphi:leanVerified` | `xsd:boolean` | Kernel accepted the module, no `sorry`, no extra axioms, at the commit in `rdfs:seeAlso`. Explicitly *not* a probability and *not* a faithfulness claim. |

The `skos:editorialNote` on `delphi:leanVerified` says verification is a property of a
(file, commit, toolchain) triple and that consumers needing the toolchain must read it from the
owning claim's `delphi:context` or the repository. In practice no node records it.

### 2.2 MorphoEval emission

`morphoeval/math_kg.py` reads each `math_kg_loops/<sim>.yaml`, which per equation carries
`formula` (human), `lean_code`, `lean_decl`, `lean_theorem`, `lean_locus`, and writes:

```
eq  skos:notation        "<formula>" ;
    skos:definition      "<lean_code>"^^pk:lean4 ;
    pk:leanDeclaration   "<lean_decl>" ;
    pk:leanTheorem       "<lean_theorem>" ;
    pk:leanVerified      true/false .
```

`lean_verified` comes from `verify_lean_module()` (`math_kg.py` ~L798): a `sorry` grep over every
module plus a kernel run of `MathOfBiology.lean` and a sample of loop files. The Lean version string
is captured in the run report but not emitted into the graph. The emitter still uses the `pk:`
(ProbKnow) prefix; Delphi 2.0 provides `owl:equivalentProperty` crosswalks.

### 2.3 The seven problem classes as currently formalised

| Problem | Loops | Current theorem (locus) | What it proves | What it does not |
| --- | --- | --- | --- | --- |
| Auxin canalization / PIN1 allocation | 032, 034, 035, 038, 0401 | `jonsson_2006_intercellular_flux_conservation` (Loop_032 L64); `feugier_2005_polar_flux_conservation` (Loop_034 L98); `jonsson_2006_up_gradient_pin1_allocation` (L37); `mitchison_1980_canalization_winner_take_all` (L186) | Per-edge flux antisymmetry \(J_{ij}+J_{ji}=0\); a 3-face PIN1 partition sums to the pool | That a *tissue-level stepper* built from these conserves total auxin; nothing about the reflux fountain's open boundary |
| Cell sorting (DAH / Potts / DITH) | 041, 042, 044, 014, zhang | `ggh_differential_adhesion_sorting_condition` (Loop_014 L153); `cpm_equal_adhesion_null_tension`, `cpm_volume_elasticity_restoring_identity` (Loop_042) | Sign conditions on \(J\) and restoring-force identities | That any sequence of swap moves preserves the cell-type multiset and target volumes; no link to a trajectory |
| Fate switching | 007, 004 | `iridophore_switch_exclusive_and_conservative` (Loop_007 L222); `kondo_asai_1995_pigment_agent_fate_partition_and_bounded_feedback` (Loop_004 L204) | Exclusivity and conservation as an `Int` identity on indicator sums | Exclusivity as a sum type: a cell is in exactly one state by construction |
| Somitogenesis budget | 021, 024, 026, 028, 0301 | `dubrulle_2001_fgf8_bead_somite_pair_conservation` (Loop_024 L83); `meinhardt_1986_half_somite_cleft_count` (Loop_021 L154); `flank_influx_volume_budget` (Loop_017 L204) | Separate counting identities | That both counts are corollaries of one PSM-consumption law |
| Transport networks / Murray | 052, 057, MathOfBiology L782–816 | `murray_1926_tree_conservation` (VeinTree, by induction); `tero_kirchhoff_antisymmetry_and_conductivity_equilibrium` (Loop_057 L70); `peristaltic_closed_source_zero_sum` (L179) | Root \(r^\gamma\) equals leaf sum; per-edge Kirchhoff antisymmetry | Flow conservation at every node of an arbitrary graph, including sources/sinks |
| Rule 30 reversibility | MathOfBiology L20–80, Ruliobiology | `jen_1990_left_bijective` (L48); `quadrant_pde_bwd_*` (L72–78) | Left-inverse identities by `cases … rfl` | That the backward map is the inverse *of a typed bijection*; engine in-place updates rely on Lean's refcount optimiser |
| Bioelectric charge | 081, 082, 084, 0901 | `gj_voltage_gate_symmetry_and_isolation` (Loop_081 L61); `beane_bistable_vmem_equilibria` (L33) | Gate symmetry and equilibria on \(V_\text{mem}\) | Charge conservation across the gap-junction network |

The pattern is the same in every row: the per-edge or per-node identity is proved, the global
conservation that is the actual biological claim is not, and the KG cannot distinguish the two.

## 3. The extension

### 3.1 Lean side: the linear-resource DSL

`morphoeval/lean/LinearResource.lean` (prototype, compiles with Lean 4.35.0, core only):

```lean
namespace MathOfBiology.Linear
abbrev Ledger := List Int                      -- one Int per compartment

inductive Prog where                            -- the ONLY ways to act on a ledger
  | skip : Prog
  | transfer (i j : Nat) (q : Int) : Prog       -- move q from i to j
  | seq : Prog → Prog → Prog

inductive WellScoped (n : Nat) : Prog → Prop    -- typing judgment: indices in range
def exec : Prog → Ledger → Ledger

theorem conserves (n : Nat) (p : Prog) (hp : WellScoped n p) (L : Ledger) (hL : L.length = n) :
    (exec p L).sum = L.sum                      -- soundness, proved once
```

There is no constructor that creates or destroys resource; that absence *is* the linear discipline.
Instantiations then look like:

```lean
def auxin_edge_step (T D P_ij P_ji : Int) (i j : Nat) (L : Ledger) : Prog :=
  .transfer j i (intercellular_auxin_flux T D P_ij P_ji (L.getD i 0) (L.getD j 0))

theorem auxin_edge_step_conserves … : (exec (auxin_edge_step …) L).sum = L.sum :=
  conserves L.length _ (.transfer hj hi) L rfl      -- no new arithmetic

theorem auxin_edge_step_reverse … :
  auxin_edge_step T D P_ji P_ij j i L = .transfer i j (-(intercellular_auxin_flux …))
  -- Jönsson antisymmetry is now a statement that the reverse edge is the reverse program
```

and `murray_bifurcation` as `seq (transfer parent left q_l) (transfer parent right q_r)` with
`murray_bifurcation_conserves := conserves …`.

Planned, not yet in the prototype:

- `split i [(j₁,q₁),…]` as sugar for a sequence of transfers (PIN1 partition, Murray fan-out).
- A typed `swap i j` for Potts moves over per-label ledgers.
- A graded variant (multiplicity on binders) if the somite clock partition needs a *period* rather
  than a *quantity* ledger (prediction LT-04 is the test of whether that is necessary).
- A second ledger carrier (`Rat` or fixed-point) for bioelectric models (LT-06).

What the DSL does **not** do: dispersion relations, buckling thresholds, Turing wavelengths,
scaling exponents. Those stay equational (prediction LT-09 registers this as a negative control).

### 3.2 Ontology side: proposed additive terms (Delphi 2.1)

Proposed Turtle, to sit next to the v1.3 formal-verification block in `delphi.ttl`:

```turtle
delphi:proofKind a owl:DatatypeProperty ;
    rdfs:label "proof kind"@en ;
    rdfs:domain delphi:Concept ;
    rdfs:range xsd:string ;
    rdfs:comment "What kind of formal artefact backs this law. Controlled values: 'equational' (a free-standing identity proved per law), 'conservation-by-typing' (the law is an instantiation of a resource discipline whose soundness theorem is named in delphi:soundnessTheorem; the conservation property is inherited, not restated), 'executable' (the declaration is run to produce data, e.g. a CA engine), 'placeholder' (compiles but formalises nothing about the named law). Orthogonal to delphi:leanVerified: a placeholder can be kernel-verified."@en .

delphi:soundnessTheorem a owl:DatatypeProperty ;
    rdfs:label "soundness theorem"@en ;
    rdfs:domain delphi:Concept ;
    rdfs:range xsd:string ;
    rdfs:comment "Fully qualified identifier of the generic theorem a 'conservation-by-typing' node depends on, e.g. 'MathOfBiology.Linear.conserves'. Lets a reader (or a SPARQL query) find every law that inherits one proof."@en .

delphi:conservedQuantity a owl:DatatypeProperty ;
    rdfs:label "conserved quantity"@en ;
    rdfs:domain delphi:Concept ;
    rdfs:range xsd:string ;
    rdfs:comment "The biological quantity the type discipline conserves, in the source model's own terms: 'total auxin (arbitrary units)', 'cell count per type', 'volumetric flow', 'net charge'. Human-readable; pair with a QUDT or UO IRI via rdfs:seeAlso when one exists."@en .

delphi:formalSystem a owl:DatatypeProperty ;
    rdfs:label "formal system"@en ;
    rdfs:domain delphi:Concept ;
    rdfs:range xsd:string ;
    rdfs:comment "Name of the proof assistant or type system that checked the artefact ('Lean 4', 'Idris 2', 'Granule', 'Coq'). The delphi:lean* terms remain the Lean-specific carrier; this term exists so a second system can be recorded without minting a parallel term set first."@en .

delphi:toolchainVersion a owl:DatatypeProperty ;
    rdfs:label "toolchain version"@en ;
    rdfs:domain delphi:Concept ;
    rdfs:range xsd:string ;
    rdfs:comment "Exact version string of the checker, e.g. 'Lean (version 4.35.0, arm64-apple-darwin24.6.0, commit eb6cafb7…)'. Completes the (file, commit, toolchain) triple that the editorial note on delphi:leanVerified says consumers need."@en .
```

`delphi:leanVerified`'s `skos:editorialNote` would be updated to point at `delphi:toolchainVersion`
instead of telling consumers to infer it. No term is deprecated; no range or domain changes.

### 3.3 Spec difference on a node

Before (today's emission for Jönsson flux conservation, Loop_032 law 2):

```turtle
<…/math-law/auxin-pin1-phyllotaxis-2> a delphi:Concept ;
    skos:notation       "J_ij = T (P_ji a_j − P_ij a_i) + D (a_j − a_i); J_ij + J_ji = 0" ;
    skos:definition     "def intercellular_auxin_flux … theorem jonsson_2006_intercellular_flux_conservation …"^^delphi:lean4 ;
    delphi:leanDeclaration "MathOfBiology.Loop_032.intercellular_auxin_flux" ;
    delphi:leanTheorem     "MathOfBiology.Loop_032.jonsson_2006_intercellular_flux_conservation" ;
    delphi:leanVerified    true .
```

After:

```turtle
<…/math-law/auxin-pin1-phyllotaxis-2> a delphi:Concept ;
    skos:notation       "J_ij = T (P_ji a_j − P_ij a_i) + D (a_j − a_i); Σ_i a_i conserved under any edge schedule" ;
    skos:definition     "def auxin_edge_step … theorem auxin_edge_step_conserves … := conserves …"^^delphi:lean4 ;
    delphi:leanDeclaration  "MathOfBiology.Linear.auxin_edge_step" ;
    delphi:leanTheorem      "MathOfBiology.Linear.auxin_edge_step_conserves" ,
                            "MathOfBiology.Linear.auxin_edge_step_reverse" ;
    delphi:leanVerified     true ;
    delphi:proofKind        "conservation-by-typing" ;
    delphi:soundnessTheorem "MathOfBiology.Linear.conserves" ;
    delphi:conservedQuantity "total auxin across the cell graph (model units)" ;
    delphi:formalSystem     "Lean 4" ;
    delphi:toolchainVersion "Lean (version 4.35.0, …)" .
```

A reader can now ask "which laws in the graph inherit their conservation from one soundness
theorem" (`?law delphi:soundnessTheorem "MathOfBiology.Linear.conserves"`), and "which verified
nodes are placeholders", neither of which is expressible today.

### 3.4 Emitter and YAML schema diff

`math_kg_loops/<sim>.yaml`, per equation, adds optional keys:

```yaml
    proof_kind: conservation-by-typing      # default: equational
    soundness_theorem: MathOfBiology.Linear.conserves
    conserved_quantity: total auxin across the cell graph (model units)
```

`math_kg.py`:

- emit the three keys when present; default `proof_kind` to `equational`, or to `placeholder` when
  `is_template_lean_equation` fires (today that filters the node out; it would instead be emitted
  and labelled);
- emit `delphi:formalSystem "Lean 4"` and `delphi:toolchainVersion` from the
  `verify_lean_module()` version string on every node;
- add `morphoeval/lean/LinearResource.lean` to the kernel-check set (it is currently outside
  `LEAN_PATH` + `loops/*`);
- migrate the prefix from `pk:` to `delphi:` in the same change.

`check_lean_formula_alignment.py` gains one rule: a `conservation-by-typing` equation's
`lean_code` must reference the named `soundness_theorem`.

## 4. Per-problem: now, with the extension, and the spec difference

| Problem | Now | With the DSL | Node-level difference |
| --- | --- | --- | --- |
| Auxin canalization | Edge antisymmetry per loop | One `auxin_edge_step`; tissue conservation = `conserves`; reverse edge = reverse program | `proofKind` + `soundnessTheorem` + `conservedQuantity "total auxin"` |
| Cell sorting | Sign conditions on \(J\) | `swap` over per-label ledgers; type-multiset invariance for any move sequence; cross-check vs Potts Python | `conservedQuantity "cell count per type; target volume"` |
| Fate switching | Indicator-sum identity | Sum type; exclusivity by construction | `proofKind "conservation-by-typing"` on the state count |
| Somite budget | Two counting identities | One consumption program; both counts as corollaries | Both nodes share one `soundnessTheorem` |
| Transport / Murray | Tree induction; per-edge Kirchhoff | `murray_bifurcation` as split; node conservation on arbitrary graphs including boundary | `conservedQuantity "volumetric flow"` |
| Rule 30 | `cases … rfl` inverses | Typed bijection; inverses are projections; uniqueness-typed engine | `proofKind "executable"` on the engine, `"equational"` on the identities |
| Bioelectric | Gate symmetry on \(V_\text{mem}\) | Gap-junction transfer on a charge ledger (`Rat` carrier) | `conservedQuantity "net charge"` |

## 5. What this does not change

- `delphi:leanVerified` still means only "the kernel accepted the statement as written." A typed
  conservation law can be a faithful or unfaithful rendering of Jönsson or Tero exactly as an
  equational one can. Faithfulness stays on `delphi:Assessment` and on
  `check_lean_formula_alignment.py`.
- Nothing here requires linear types *in Lean itself*. The DSL is an object language; Lean checks
  it. If Lean grows graded binders, or if an Idris 2/Granule module is added, `delphi:formalSystem`
  and `delphi:toolchainVersion` already have room for it.
- The 3,000 template stubs recorded in `morphoeval/problem_scouts/lean_stub_census.json` (issue
  #268) have been deleted; the 136 remaining loop files are mechanism-specific. `proofKind
  "placeholder"` is proposed so that, if generators ever emit stubs again, the graph says so rather
  than filtering them silently.

## 6. Pre-registered predictions

Registration: `predictions/registrations/LINEAR-TYPES-01.json` (priors fixed before any work beyond
the prototype). Each is published as a `delphi:TestableHypothesis` nanopub by
`nanopubs/linear_types_predictions_to_nanopubs.py`; trusty URIs are recorded in
`nanopubs/linear_types_predictions/published_manifest_prod.txt`. Priors are credences that the
*formal-verification outcome* is achieved by the deadline, not probabilities about biology.

| ID | Prediction | Prior | Resolvability | Deadline |
| --- | --- | --- | --- | --- |
| LT-01 | DSL soundness theorem merged, 0 `sorry`, no extra axioms | 0.97 | 1.00 | 2026-11-30 |
| LT-02 | All five auxin loops carry a DSL instantiation and pass the alignment checker | 0.75 | 0.95 | 2027-01-31 |
| LT-03 | Potts swap stepper conserves type multiset by typing; 1,000-MCS cross-check identical | 0.60 | 0.90 | 2027-01-31 |
| LT-04 | Dubrulle pair-conservation and Meinhardt cleft-count derived from one PSM consumption law | 0.50 | 0.85 | 2027-02-28 |
| LT-05 | At least one current Python reference implementation (Physarum, canalization, Potts) violates conservation > 1e-9 relative | 0.55 | 0.95 | 2026-12-31 |
| LT-06 | Gap-junction charge transfer formalised; cross-check vs Beane planarian model | 0.45 | 0.70 | 2027-02-28 |
| LT-07 | Delphi 2.x ships a proof-kind + toolchain term and ≥1 published nanopub uses it | 0.55 | 1.00 | 2027-03-31 |
| LT-08 | ≥40 of 136 loops DSL-expressible; ≥20 emitted as `conservation-by-typing` | 0.40 | 0.90 | 2027-03-31 |
| LT-09 | Negative control: elasticity / Turing-wavelength loops remain `equational` | 0.85 | 0.90 | 2027-03-31 |
| LT-10 | Rule 30 left-bijectivity as a typed bijection; 1,024-step bit-identical cross-check | 0.50 | 0.80 | 2027-02-28 |

Mean prior 0.612; expected confirmations 6.1 of 10; scored by Brier at 2027-03-31.

## 7. Questions for the ontology review

1. Is `delphi:proofKind` as a string with a controlled vocabulary acceptable, or should the values
   be `skos:Concept`s under `delphir:proof-kind/…` so they can carry definitions and be extended
   without a release?
2. Should `delphi:soundnessTheorem` be an object property to a `delphi:Concept` node for the
   soundness theorem itself (so the generic theorem is a first-class law in the graph), rather than
   a string identifier?
3. `delphi:conservedQuantity` is deliberately a string. Is there appetite for an object property to
   QUDT/UO now, or defer until a second consumer needs it?
4. `delphi:formalSystem` / `delphi:toolchainVersion` are on `delphi:Concept`. Should they instead
   hang off the `prov:Activity` in the provenance graph, since they describe the check rather than
   the law?
5. Does the ontology want to say anything about *cross-checks* (DSL stepper vs reference Python,
   LT-03/05/10), or is that purely `delphi:EvidenceItem` territory?
