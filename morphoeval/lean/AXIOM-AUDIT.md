# Axiom audit

`axiom-audit.sh` answers one question about the Lean files in this directory: **what does each proof
actually rest on?**

## Why a separate script

"0 sorry" is a claim about a file, and a reader cannot check it from the file. Three different things
can be true of a Lean file that compiles:

- the proofs are complete, and nothing in the file shows it;
- a gap is in there, because `sorry` is a *warning*, not an error, so the file still compiles;
- a result was simply assumed with a hand-written `axiom`, which does not even warn, and which a
  search for the word "sorry" will never find.

Lean can settle it. `#print axioms Foo` prints the axioms `Foo` depends on. If that set is within
`propext`, `Classical.choice` and `Quot.sound` — the three the standard library itself rests on — the
proof is complete and uses nothing of the author's own. `sorry` shows up in that output as `sorryAx`,
so it cannot hide.

The script does not modify anything. It copies each file, appends one `#print axioms` line per
theorem to the copy, runs Lean over the copy, and reports what Lean printed.

## Running it

```sh
./axiom-audit.sh MathOfBiology.lean LinearResource.lean Ruliobiology.lean
```

Exit status is 0 only when every theorem in every file was accepted within those three axioms. These
files import nothing, so no Lake project and no Mathlib are needed — only a Lean on `PATH`. If `lean`
is not on `PATH`, point the script at one:

```sh
LEAN="lake env lean" ./axiom-audit.sh MathOfBiology.lean       # inside a Lake project
ELAN_TOOLCHAIN=leanprover/lean4:v4.35.0 ./axiom-audit.sh MathOfBiology.lean
```

## What it said on 2026-11-19

Run against `main` at `e356466`, with Lean 4.35.0 (`x86_64-w64-windows-gnu`, commit `eb6cafb7`):

| file | sha256 | theorems | audited | verdict | axioms |
|---|---|---|---|---|---|
| `MathOfBiology.lean` | `ac8ca6c4…38dda38b0` | 76 | 76 | CHECKED | `Classical.choice`, `propext`, `Quot.sound` |
| `Ruliobiology.lean` | `b5df9acd…bb2f4739b` | 31 | 31 | CHECKED | `Classical.choice`, `propext`, `Quot.sound` |
| `LinearResource.lean` | `957bf486…add2dd1d2` | 8 | 8 | CHECKED | `propext`, `Quot.sound` |

115 theorems, no errors, no `sorry`, no hand-written axiom, nothing outside the standard three. The
proofs hold. Before this script there was no way to see that from the published files.

## What this does not say

It does not say a theorem is the *right* theorem. A proof whose hypotheses contradict each other is
accepted by Lean and says nothing at all; so is a theorem that is true but about a different object
than its name suggests. No script catches either. That needs a reader.

It is also a statement about the bytes in the table above, not about the repository forever. Re-run it
when the files change — which is the point of it being a script.
