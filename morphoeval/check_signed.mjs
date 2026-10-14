#!/usr/bin/env node
// check_signed.mjs — independent verification of signed nanopubs (trusty URI + RSA signature)
// with the Rust/WASM nanopub toolkit, @nanopub/sign (vemonet/nanopub-rs).
//
//   npm install @nanopub/sign            # once, anywhere (a node_modules dir is searched upward)
//   node morphoeval/check_signed.mjs morphoeval/kg/<audit-id>/signed/*.trig
//
// Why: nanopub-py 2.1.0's own verifier is broken on Python 3.14 (it rejects nanopub-java-signed,
// published nanopubs), so we never trust a signer to grade its own homework. @nanopub/sign 0.2.0
// ships its node build as CommonJS inside a "type": "module" package and loads its .wasm relative
// to the cwd; this script works around both without modifying anything in node_modules.
import { readFileSync, existsSync, copyFileSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join, resolve } from "node:path";
import { tmpdir } from "node:os";

function findPkg(start) {
  for (let d = resolve(start); ; d = dirname(d)) {
    const p = join(d, "node_modules", "@nanopub", "sign");
    if (existsSync(join(p, "node.js"))) return p;
    if (dirname(d) === d) return null;
  }
}
const pkg = (process.env.NANOPUB_SIGN_DIR && findPkg(process.env.NANOPUB_SIGN_DIR))
  || findPkg(process.cwd()) || findPkg(dirname(new URL(import.meta.url).pathname)) || findPkg(tmpdir());
if (!pkg) { console.error("@nanopub/sign not found: run `npm install @nanopub/sign` (or set NANOPUB_SIGN_DIR to the dir containing node_modules)"); process.exit(2); }

// CommonJS shim: copy node.js -> node.cjs with an absolute wasm path (node_modules untouched if read-only)
let shim = join(pkg, "node.cjs");
const src = readFileSync(join(pkg, "node.js"), "utf8")
  .replace(/['"]\.\/node_bg\.wasm['"]/, JSON.stringify(join(pkg, "node_bg.wasm")));
try { writeFileSync(shim, src); } catch { shim = join(tmpdir(), "nanopub-sign-node.cjs"); writeFileSync(shim, src); }
const { Nanopub } = createRequire(import.meta.url)(shim);

let ok = 0, fail = 0;
for (const f of process.argv.slice(2)) {
  try {
    const np = new Nanopub(readFileSync(f, "utf8")).check();   // throws on bad trusty hash / signature
    const uri = np.info().uri; const u = typeof uri === "string" ? uri : uri?.value;
    console.log("OK   ", f.split("/").pop(), u); ok++;
  } catch (e) { console.log("FAIL ", f.split("/").pop(), String(e).slice(0, 160)); fail++; }
}
console.log(`${ok} ok, ${fail} failed`);
process.exit(fail ? 1 : 0);
