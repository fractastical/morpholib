#!/usr/bin/env bash
# Report which axioms every theorem in a Lean file actually rests on.
#
#   ./axiom-audit.sh MathOfBiology.lean
#   ./axiom-audit.sh *.lean loops/*.lean
#
# Why: "0 sorry" is a claim about a file, and a reader cannot check it from the file. `sorry` is a
# warning, not an error, so a file containing one still compiles; and a hand-written `axiom` does not
# even warn. What Lean can be asked for is the axiom set each proof depends on. If that set is within
# propext, Classical.choice and Quot.sound — the three the standard library itself rests on — then the
# proof is complete and uses nothing of the author's own.
#
# This script does not modify the file. It copies it, appends one `#print axioms` line per theorem to
# the copy, runs Lean over the copy, and reports what Lean printed. Exit status is 0 only when every
# theorem was accepted within those three axioms.
#
# Needs `lean` on PATH (elan install leanprover/lean4:stable). Files here import nothing, so no Lake
# project and no Mathlib are required.
set -uo pipefail

LEAN_CMD="${LEAN:-lean}"
ALLOWED="propext Classical.choice Quot.sound"
fail=0

audit_one() {
  local f="$1"
  [ -f "$f" ] || { printf '%-28s MISSING\n' "$(basename "$f")"; fail=1; return; }

  local tmp base ns names count out
  tmp="$(mktemp -d)"
  base="$(basename "$f")"
  cp "$f" "$tmp/$base"

  # the namespace a declaration ends up in, so the appended lines name it correctly
  ns="$(grep -E '^namespace[[:space:]]+' "$f" | head -1 | awk '{print $2}')"
  names="$(grep -oE '^[[:space:]]*(private |protected )?(theorem|lemma)[[:space:]]+[A-Za-z_][A-Za-z0-9_.'"'"'!?]*' "$f" | awk '{print $NF}')"

  count=0
  printf '\n' >> "$tmp/$base"
  for n in $names; do
    if [ -n "$ns" ]; then printf '#print axioms %s.%s\n' "$ns" "$n" >> "$tmp/$base"
    else printf '#print axioms %s\n' "$n" >> "$tmp/$base"; fi
    count=$((count + 1))
  done

  out="$($LEAN_CMD "$tmp/$base" 2>&1)"
  rm -rf "$tmp"

  local errors sorries audited axioms extra verdict
  errors="$(printf '%s\n' "$out" | grep -c 'error:')"
  sorries="$(printf '%s\n' "$out" | grep -c 'declaration uses .sorry')"
  audited="$(printf '%s\n' "$out" | grep -cE "depends on axioms|does not depend on any axioms")"
  axioms="$(printf '%s\n' "$out" | grep -oE 'depends on axioms: \[[^]]*\]' | sed 's/.*\[//;s/\]//' | tr ',' '\n' | tr -d ' ' | sort -u | tr '\n' ' ')"

  extra=""
  for a in $axioms; do
    case " $ALLOWED " in *" $a "*) ;; *) extra="$extra $a" ;; esac
  done

  if [ "$errors" -gt 0 ]; then verdict="ERRORS ($errors)"; fail=1
  elif [ "$sorries" -gt 0 ]; then verdict="SORRY ($sorries)"; fail=1
  elif [ -n "$extra" ]; then verdict="EXTRA AXIOMS:$extra"; fail=1
  elif [ "$audited" -lt "$count" ]; then verdict="ONLY $audited OF $count AUDITED"; fail=1
  else verdict="CHECKED"; fi

  printf '%-28s %3s theorem(s), %3s audited  %-10s  axioms: %s\n' \
    "$base" "$count" "$audited" "$verdict" "${axioms:-none}"
}

[ $# -gt 0 ] || { echo "usage: $0 <file.lean> [more.lean ...]"; exit 2; }
:
echo "lean: $($LEAN_CMD --version 2>&1 | head -1)"
for f in "$@"; do audit_one "$f"; done
exit $fail
