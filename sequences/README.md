# Linking known DNA/RNA sequences to the morphogenesis simulations

**Yes — and it is now wired up.** Every model in `models/` currently runs on
anonymous morphogens (`U`, `V`, `X`, `Y` with hand-picked `D`, `F`, `k`). This
folder replaces the anonymity with named, accession-pinned molecules: the real
*bmp2b* gene from zebrafish, real *dpp* from *Drosophila*, real *KIT* from cow,
and 14 more — resolved live from public databases, cached as FASTA, and
SHA-256 pinned like the specimen photographs in `datasets/turing/`.

## Quick start

```bash
python3 sequences/fetch_sequences.py              # resolve all 17 genes, write sequence_links.json
python3 sequences/fetch_sequences.py --only bmp2b,dpp
python3 sequences/fetch_sequences.py --offline    # what's already cached?
python3 sequences/fetch_sequences.py --self-test  # no network
```

No new dependencies: Python stdlib `urllib` only.

## Where the sequences come from

| Layer | Source | Stable link pattern | Gives you |
|---|---|---|---|
| Gene + transcript + CDS/RNA + protein | **Ensembl REST** | `rest.ensembl.org/lookup/symbol/{species}/{gene}`, `rest.ensembl.org/sequence/id/{transcript}?type=cds\|protein` (genomic / transcript / translated variants too) | Canonical accession IDs, assembly, isoform, actual sequence |
| Protein cross-reference | **UniProt REST** | `rest.uniprot.org/uniprotkb/search?query=gene:{g}` ([www.uniprot.org](https://www.uniprot.org/uniprotkb?query=gene:bmp2b)) | Reviewed protein record, length, function |
| Gene cross-reference | **NCBI Gene / E-utilities / Datasets** | [ncbi.nlm.nih.gov/gene](https://www.ncbi.nlm.nih.gov/gene/?term=bmp2b), `eutils.ncbi.nlm.nih.gov/entrez/eutils/` | RefSeq `NM_`/`NP_` transcripts, Gene ID |
| ncRNA / RNA families | **RNAcentral**, **Rfam** | [rnacentral.org](https://rnacentral.org/), [rfam.org](https://rfam.org/) | miRNAs (e.g. miR-125 in coelomocyte / appendage patterning), lncRNAs |
| Where/when it's expressed | **ZFIN** (zebrafish), **Xenbase** (*Xenopus*), **FlyBase** (fly), **WormBase** (worm), **MGI** (mouse) | Linked per-gene in `catalog.json` → `expression_atlas` | The spatial/temporal source term a simulation actually needs |

Per-gene records in the generated `sequence_links.json` carry direct
clickable links in exactly these forms, plus `cds_sha256` / `protein_sha256`
for the cached FASTA files in `sequences/cache/`.

## The curated panel (`catalog.json`)

Seventeen genes chosen because a model in this repo can already use them:

- **Turing / reaction–diffusion pairs** (`models/1952-turing-morpho.py`,
  `models/1981-murray-bos-taurus-turing-wavelength.py`): zebrafish `bmp2b`
  (activator-side gradient) + `chd`/chordin (its secreted inhibitor) — a real
  activator–inhibitor pair instead of anonymous *U/V*; mouse `Bmp4`/`Shh` and
  *Xenopus* `bmp4`/`nog` (noggin) give a functionally analogous pair in another
  species, so diffusion priors can be compared across species. (An earlier
  wording here called them "orthologs of the same pair" — that claim was
  logged as a test and REFUTED on 2026-10-26; see below.)
- **Clock-and-wavefront**: zebrafish `her1` (somitogenesis clock oscillator).
- **Wing venation** (`models/2018-hoffmann-wing-venation.py`): *Drosophila*
  `dpp` (long-range BMP) vs `wg`/wingless (short-range lipid-tethered Wnt) —
  the textbook case where two real sequences predict *different* gradient
  lengths in the same tissue.
- **Worm lineage** (`models/2015-santella-wormguides-celegans.py`):
  `lin-44` (Wnt) and `mab-5` (Hox readout).
- **Bioelectric coupling** (`models/2021-Cervera–Levin–Mafe.py`,
  `models/2017-pietak-bioelectric-morphogen-coupling.py`): *Xenopus* `bmp4`,
  `nog`, `shh`.
- **The negative claim, anchored** (`datasets/bos_taurus_turing_wavelength/`):
  cow `KIT` + `MITF` — the melanoblast-migration genes that dataset's README
  cites *against* a Turing account of Holstein spotting. Linking their real
  sequences ties that counter-claim to molecules, not just citations.

## Using it in a model (wired up: `models/1952-turing-morpho.py`)

```bash
./venv/bin/python3 models/1952-turing-morpho.py --named-morphogens --no-show --skip-gif
# other pair from the catalog:
./venv/bin/python3 models/1952-turing-morpho.py --named-morphogens \
    --morphogen-pair xenopus_tropicalis:bmp4,xenopus_tropicalis:nog --no-show --skip-gif
```

With `--named-morphogens` the script loads the pair from `sequence_links.json`,
**verifies both cached protein FASTAs against their pinned SHA-256** (and CDS,
when present) before simulating, then labels the Gray–Scott fields with the
named pair in the figure title/axis and appends 11 provenance columns to the
CSV (`activator_gene`, `activator_accession`, `activator_protein_aa`,
`inhibitor_*`, `diffusion_prior_ratio_inhibitor_over_activator`,
`size_prior_supports_faster_inhibitor`, `protein_sha256_verified`).

Default pair: zebrafish **bmp2b** (411 aa, `ENSDARG00000041430`) as the
activator arm (`V`), **chordin/chd** (940 aa, UniProt `O57472`) as the
inhibitory arm. Their size-only diffusion-prior ratio is
\(D_{chd}/D_{bmp2b} = (411/940)^{1/3} \approx 0.76 < 1\) — i.e. sequence size
alone does *not* give the faster-inhibitor ordering classical Turing needs
(in vivo this pair also works by shuttling/ECM binding). The script prints
that as an explicit note and **does not** rescale \(D_u/D_v\): named and
anonymous runs are numerically identical on every shared CSV column
(verified). The `--self-test` suite gates the loader, the SHA verification,
the 0.759 ratio, and the named CSV columns.

## Logged test: pair-equivalence claim — REFUTED (2026-10-26)

Claim logged for testing: *Xenopus* `bmp4`/`nog` are "orthologs of the same
pair" as zebrafish `bmp2b`/`chd`, interchangeable as a Turing
activator–inhibitor pair. Checker: `check_pair_equivalence.py` (offline =
diffusion half; `--online` adds Ensembl Compara orthology); evidence log:
`pair_equivalence_log.json`; hermetic wrapper:
`tests/test_sequence_pair_equivalence.py` (4 tests, all passing — the harness
passes, the *claim* does not).

- **Orthology (Compara, live):** `bmp2b` → *Xenopus* `bmp2`
  (`ENSXETG00000005519`, ortholog_one2many), not the catalog `bmp4`
  (`ENSXETG00000008620`) — paralog substitution. `chd` → *Xenopus* `chrd`
  (`ENSXETG00000017290`, ortholog_one2one), not `nog`
  (`ENSXETG00000043161`) — Chordin and Noggin are non-homologous BMP
  antagonists.
- **Diffusion equivalence (pinned sequences, all 17 SHA-256 verified):**
  zebrafish \(D_{chd}/D_{bmp2b} = (411/940)^{1/3} \approx 0.759 < 1\)
  (inhibitor predicted *slower* — fails Turing ordering); *Xenopus*
  \(D_{nog}/D_{bmp4} = (469/222)^{1/3} \approx 1.283 > 1\) (passes). Opposite
  orderings: the pairs are not interchangeable under the repo's own prior.

Corrected statement: functionally analogous activator–antagonist pairs —
paralogous ligand, non-homologous antagonist — of which only BMP/Noggin
satisfies the size-prior Turing ordering. BMP/Chordin works by
shuttling/ECM binding in vivo and should not be cast as anonymous *U/V*
without kinetic evidence. Rerun:
`python3 sequences/check_pair_equivalence.py --online` (exit 1 = REFUTED).

## Pair screen (`pair_screen.py`)

```bash
python3 sequences/pair_screen.py               # prints both tables, writes pair_screen.csv
python3 sequences/pair_screen.py --self-test
```

SHA-verifies all 17 proteins, then ranks (a) same-species, pathway-matched
activator/inhibitor pairs by \(D_{inh}/D_{act}=(L_{act}/L_{inh})^{1/3}\), and
(b) same-pathway ortholog pairs by departure from size conservation.
Non-diffusible roles (`her1` clock, `mab-5`/`KIT`/`MITF` competence) are
excluded, not forced into morphogen pairs. Results (2026-10-26 run,
`pair_screen.csv`):

- *Xenopus* **bmp4/nog**: \(D_{inh}/D_{act}=1.283\) — passes the size-only
  faster-inhibitor Turing ordering.
- Zebrafish **bmp2b/chd**: \(0.759\) — fails it; flags a shuttling/ECM
  mechanism rather than free-diffusion Turing (the natural next model).
- Hedgehog orthologs are almost perfectly size-conserved across zebrafish,
  *Xenopus*, mouse and fly (416–471 aa, all ratios 0.96–1.03), and zebrafish
  `bmp2b` vs mouse `Bmp4` is 411 vs 408 aa (ratio 1.002) — fly `dpp` (588 aa)
  is the size-diverged BMP outlier.

## How a sequence feeds a simulation — and where it stops

```
sequence ──► identity / isoform / length / GC / protein size ──► EMULATION
             (this folder: verified, hashed, reproducible)       │
                                                                  ▼
literature ─► D, production, decay, receptor kinetics,      model parameters
expression    expression place & time (atlas links above)   (stays in models/*.py)
```

What the sequence legitimately contributes to `models/1952-turing-morpho.py`
and friends:

1. **Identity**: the activator *is* BMP2b (411 aa in zebrafish), the inhibitor
   *is* chordin — not free parameters.
2. **A relative diffusion prior**: `diffusion_prior_vs_smallest` in
   `sequence_links.json` computes \(D \propto M_W^{-1/3}\) (Stokes–Einstein) —
   bigger secreted ligand, slower prior. Good enough to *rank* BMP vs Wnt vs
   Shh and to sanity-check the \(D_V < D_U\) ordering Turing requires, e.g.
   lipid/cholesterol-modified Shh and Wingless should sit at the slow end.
3. **Composition checks**: CDS length, GC%, signal-peptide-scale size — catches
   wrong-isoform or wrong-species mistakes before you simulate.

What it does **not** contribute — be explicit in any paper text, as this repo
already is in its `claim_boundary` blocks: absolute \(D\) in µm²/s,
production/degradation rates, binding affinities, enhancer-driven expression
timing. Those come from FRAP/FCS measurements and expression atlases, and
belong in the model script's parameter table with a DOI, not in the sequence
record. Anyone claiming "we derived the Turing parameters from the genome" is
over-claiming; "we emulated named morphogens X/Y, with size-ranked diffusion
priors from their sequences and kinetics from [paper]" is defensible.

## Adding a gene

Add one block to `catalog.json` (species + symbol + which model uses it +
*why*), re-run the fetcher. Accessions are never hand-typed: if Ensembl can't
resolve the symbol, the run reports `[FAIL]` rather than inventing an ID
(planarian *Schmidtea* genes, for instance, are not in Ensembl — resolve those
via NCBI Gene / PlanMine and add them as NCBI-only entries).

## Files

- `catalog.json` — hand-curated panel + claim boundary (the only hand-edited file)
- `fetch_sequences.py` — stdlib-only resolver, `--self-test` passes offline
- `check_pair_equivalence.py` — logged pair-equivalence test (`--online` for Compara)
- `pair_equivalence_log.json` — generated evidence log, 2026-10-26 run: REFUTED
- `sequence_links.json` — generated, accession + hash record (regenerate, don't edit)
- `cache/*.fasta` — generated CDS + protein FASTA, SHA-256 pinned in `sequence_links.json`
