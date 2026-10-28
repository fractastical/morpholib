#!/usr/bin/env python3
"""
nanopubs/galvani_twitch_estimates_to_nanopubs.py
================================================
Generate, cryptographically sign, and (optionally) publish nanopublications for
quantitative muscle-twitch estimates across Galvani's 1780–1794 frog experiments,
Matteucci's 1838–1842 Galvanoscopic frog & series frog battery, Pflüger's 1859
high-current anodal block, Hoyle's 1955 insect SETi/FETi axons, and human
percutaneous electroacupuncture (ST36/LI4) — plus any custom user-entered
muscle-twitch estimate passed on the CLI or recorded from the interactive Canvas/HTML lab.

Usage:
  # 1. Generate all canonical Galvani + Insect + Human needle twitch nanopubs (unsigned + RSA-signed):
  python3 nanopubs/galvani_twitch_estimates_to_nanopubs.py --sign

  # 2. Record a custom muscle-twitch estimate as its own signed nanopublication:
  python3 nanopubs/galvani_twitch_estimates_to_nanopubs.py --sign \
    --custom-id "human-st36-deqi-2ma" \
    --species "Homo sapiens (tibialis anterior at ST36 Zusanli)" \
    --current-uA 2000.0 \
    --voltage-mV 4500.0 \
    --duration-ms 0.20 \
    --freq-hz 2.0 \
    --polarity "cathodal_descending" \
    --make-twitch-pct 12.5 \
    --break-twitch-pct 0.0 \
    --force-newton 3.40 \
    --kick-angle-deg 6.5 \
    --notes "Visible rhythmic De Qi ankle dorsiflexion twitch at 2.0 mA, 2 Hz percutaneous needle stimulation"
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from rdflib import Dataset, Graph, Literal, Namespace, URIRef
from rdflib.namespace import DCTERMS, PROV, RDF, RDFS, XSD

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "morphoeval"))
REF_JSON = REPO_ROOT / "datasets" / "galvani" / "galvani_1791_reference.json"
OUT_DIR = REPO_ROOT / "nanopubs" / "galvani_twitch_predictions"
PROFILE_PATH = Path.home() / ".nanopub" / "profile.yml"

TEMP_BASE = "http://purl.org/nanopub/temp/"
MLG = Namespace("https://w3id.org/morpholib/galvani/")
NP = Namespace("http://www.nanopub.org/nschema#")
NPX = Namespace("http://purl.org/nanopub/x/")
NTEMPLATE = Namespace("https://w3id.org/np/o/ntemplate/")
PK = Namespace("https://w3id.org/probknow/ontology/")
PKR = Namespace("https://w3id.org/probknow/resource/")
SCHEMA = Namespace("https://schema.org/")
ORCID = Namespace("https://orcid.org/")

SCRIPT_URI = URIRef("https://github.com/fractastical/morpholib/blob/main/models/1791-galvani-bioelectricity.py")
LEAN_URI = URIRef("https://github.com/fractastical/morpholib/blob/main/morphoeval/lean/loops/Loop_101_galvani_1791_neuromuscular_bioelectricity.lean")
PICCOLINO_DOI = URIRef("https://doi.org/10.1016/S0361-9230(98)00026-4")
FH_DOI = URIRef("https://doi.org/10.1113/jphysiol.1964.sp007378")
VOLTA_DOI = URIRef("https://doi.org/10.1098/rstl.1800.0018")


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def slugify(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-").lower()


def load_profile_dict() -> dict[str, str]:
    if not PROFILE_PATH.exists():
        return {}
    out: dict[str, str] = {}
    for line in PROFILE_PATH.read_text(encoding="utf-8").splitlines():
        m = re.match(r"(\w+):\s*(.*)", line)
        if m and not line.startswith("#"):
            k, v = m.groups()
            if v.strip():
                out[k.strip()] = v.strip()
    return out


from build_morphoeval_kg import plain_doubles


def build_twitch_estimate_nanopub(
    rec: dict[str, Any],
    commit: str,
    created_iso: str,
    orcid_url: str,
    creator_name: str,
    ref_sha256: str,
) -> tuple[str, Dataset]:
    """Construct an rdflib Dataset for a single quantitative muscle-twitch estimate nanopublication."""
    exp_id = slugify(str(rec["experiment_id"]))
    np_slug = f"galvani-twitch-{exp_id}"
    base = URIRef(f"{TEMP_BASE}{np_slug}/")
    this_uri = URIRef(f"{TEMP_BASE}{np_slug}")

    ds = Dataset()
    for pfx, ns in [
        ("this", Namespace(str(this_uri))),
        ("sub", Namespace(str(base))),
        ("np", NP),
        ("npx", NPX),
        ("ntemplate", NTEMPLATE),
        ("dcterms", DCTERMS),
        ("prov", PROV),
        ("rdfs", RDFS),
        ("xsd", XSD),
        ("mlg", MLG),
        ("pk", PK),
        ("pkr", PKR),
        ("schema", SCHEMA),
        ("orcid", ORCID),
    ]:
        ds.bind(pfx, ns)

    head = ds.graph(URIRef(f"{base}Head"))
    assertion = ds.graph(URIRef(f"{base}assertion"))
    provenance = ds.graph(URIRef(f"{base}provenance"))
    pubinfo = ds.graph(URIRef(f"{base}pubinfo"))

    head.add((this_uri, RDF.type, NP.Nanopublication))
    head.add((this_uri, NP.hasAssertion, assertion.identifier))
    head.add((this_uri, NP.hasProvenance, provenance.identifier))
    head.add((this_uri, NP.hasPublicationInfo, pubinfo.identifier))

    claim_uri = URIRef(f"https://w3id.org/morpholib/galvani/claim/{np_slug}")
    species = rec.get("species", "Pelophylax esculentus (= Rana esculenta)")
    curr_ua = float(rec["current_uA"])
    volt_mv = float(rec.get("driving_voltage_mV", curr_ua * 12.5))
    dur_ms = float(rec.get("duration_ms", 2.5))
    freq_hz = float(rec.get("freq_hz", 1.0))
    polarity = str(rec.get("polarity", "cathodal_descending"))
    make_pct = float(rec["make_twitch_pct"])
    break_pct = float(rec.get("break_twitch_pct", 0.0))
    force_n = float(rec["single_twitch_force_N"])
    kick_deg = float(rec["leg_kick_angle_deg"])
    regime = str(rec.get("regime", "Neuromuscular current-to-twitch estimate"))
    apparatus = str(rec.get("apparatus", "Bimetallic or percutaneous electrode"))
    investigator = str(rec.get("investigator", "Galvani Bioelectricity Lab"))
    year_val = int(rec.get("year", 2026))

    label_str = (
        f"Muscle-twitch estimate ({exp_id}): {curr_ua:.1f} uA ({volt_mv:.1f} mV, {freq_hz:.0f} Hz, {polarity}) "
        f"in {species} -> {make_pct:.1f}% Make twitch, {break_pct:.1f}% Break twitch, "
        f"{force_n:.2f} N force, {kick_deg:.1f} deg kick angle"
    )
    desc_str = (
        f"Quantitative neuromuscular excitation & muscle-twitch estimate for {species} ({investigator}, {year_val}; "
        f"apparatus: {apparatus}). Applied stimulus current I = {curr_ua:.2f} uA (driving EMF = {volt_mv:.1f} mV, "
        f"contact duration d = {dur_ms:.2f} ms, frequency f = {freq_hz:.1f} Hz, polarity = {polarity}) produces "
        f"{make_pct:.1f}% circuit-closing ('Make') contraction and {break_pct:.1f}% circuit-opening ('Break') contraction, "
        f"yielding peak isometric force F = {force_n:.2f} N and limb deflection angle theta = {kick_deg:.1f} deg ({regime})."
    )

    assertion.add((claim_uri, RDF.type, PK.Claim))
    assertion.add((claim_uri, RDF.type, MLG.NeuromuscularTwitchEstimate))
    assertion.add((claim_uri, RDFS.label, Literal(label_str)))
    assertion.add((claim_uri, DCTERMS.description, Literal(desc_str)))
    assertion.add((claim_uri, MLG.organism, Literal(species)))
    assertion.add((claim_uri, MLG.investigator, Literal(investigator)))
    assertion.add((claim_uri, MLG.experimentYear, Literal(year_val, datatype=XSD.integer)))
    assertion.add((claim_uri, MLG.apparatus, Literal(apparatus)))
    assertion.add((claim_uri, MLG.electrodePolarity, Literal(polarity)))
    assertion.add((claim_uri, MLG.pflugerRegime, Literal(regime)))

    # Quantitative PropertyValue measurements attached to the claim
    metrics = [
        ("stimulus_current_uA", "Stimulus current through nerve/tissue", curr_ua, "uA"),
        ("driving_voltage_mV", "Driving electromotive force (EMF)", volt_mv, "mV"),
        ("pulse_duration_ms", "Stimulus contact duration", dur_ms, "ms"),
        ("stimulus_frequency_Hz", "Stimulus pulse train frequency", freq_hz, "Hz"),
        ("make_twitch_pct", "Circuit-closing ('Make') contraction amplitude", make_pct, "% of max single twitch"),
        ("break_twitch_pct", "Circuit-opening ('Break') contraction amplitude", break_pct, "% of max single twitch"),
        ("peak_twitch_force_N", "Peak isometric muscle contraction force", force_n, "N"),
        ("limb_kick_angle_deg", "Peak kinematic limb deflection / kick angle", kick_deg, "deg"),
    ]
    for m_key, m_lbl, m_val, m_unit in metrics:
        m_node = URIRef(f"{claim_uri}/{m_key}")
        assertion.add((claim_uri, MLG.hasQuantitativeEstimate, m_node))
        assertion.add((m_node, RDF.type, SCHEMA.PropertyValue))
        assertion.add((m_node, RDFS.label, Literal(m_lbl)))
        assertion.add((m_node, SCHEMA.propertyID, Literal(m_key)))
        assertion.add((m_node, SCHEMA.value, Literal(round(m_val, 4), datatype=XSD.double)))
        assertion.add((m_node, SCHEMA.unitText, Literal(m_unit)))

    # Provenance graph
    creator_uri = URIRef(orcid_url)
    provenance.add((assertion.identifier, PROV.wasAttributedTo, creator_uri))
    provenance.add((assertion.identifier, PROV.wasGeneratedBy, SCRIPT_URI))
    provenance.add((assertion.identifier, PROV.wasDerivedFrom, LEAN_URI))
    provenance.add((assertion.identifier, PROV.wasDerivedFrom, PICCOLINO_DOI))
    provenance.add((assertion.identifier, PROV.wasDerivedFrom, FH_DOI))
    provenance.add((assertion.identifier, PROV.wasDerivedFrom, VOLTA_DOI))
    provenance.add((SCRIPT_URI, RDF.type, PROV.SoftwareAgent))
    provenance.add((SCRIPT_URI, RDFS.label, Literal(f"models/1791-galvani-bioelectricity.py at commit {commit[:10]} (dataset SHA-256 {ref_sha256[:16]})")))

    # Pubinfo graph
    pubinfo.add((this_uri, RDFS.label, Literal(f"Nanopub: {label_str[:110]}")))
    pubinfo.add((this_uri, DCTERMS.created, Literal(created_iso, datatype=XSD.dateTime)))
    pubinfo.add((this_uri, DCTERMS.creator, creator_uri))
    pubinfo.add((creator_uri, RDFS.label, Literal(creator_name)))
    pubinfo.add((this_uri, DCTERMS.license, URIRef("https://creativecommons.org/licenses/by/4.0/")))
    pubinfo.add((this_uri, NPX.hasNanopubType, PK.Claim))
    pubinfo.add((this_uri, NPX.hasNanopubType, MLG.NeuromuscularTwitchEstimate))
    pubinfo.add((this_uri, NPX.introduces, claim_uri))
    pubinfo.add((this_uri, PK.domain, URIRef("https://w3id.org/probknow/resource/domain/morphogenesis")))

    return np_slug, ds


def build_all_canonical_records(ref_data: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for exp in ref_data["current_to_twitch_experiments"]:
        r = dict(exp)
        r["species"] = ref_data["taxon"]["scientific_name"]
        r["duration_ms"] = 2.5 if exp["current_uA"] < 50 else 2.0
        r["freq_hz"] = 40.0 if "tetanic" in exp["experiment_id"] else 1.0
        r["polarity"] = "anodal_ascending" if "ascending" in exp["experiment_id"] else "cathodal_descending"
        records.append(r)

    # Add the 2 canonical Insect (SETi / FETi) and 2 Human Electroacupuncture (ST36 De Qi / Tibial M-max) estimates
    records.extend([
        {
            "experiment_id": "9_insect_seti_slow_axon_25hz",
            "year": 1955,
            "investigator": "Hoyle (1955) & Burrows (1996) Locust/Cockroach Tibial Extensor",
            "species": "Schistocerca gregaria / Periplaneta americana",
            "apparatus": "Fine silver wire electrodes on metathoracic nerve 5 (SETi slow glutamatergic axon at 25 Hz)",
            "driving_voltage_mV": 300.0,
            "current_uA": 1.2,
            "duration_ms": 2.0,
            "freq_hz": 25.0,
            "polarity": "cathodal_descending",
            "make_twitch_pct": 43.2,
            "break_twitch_pct": 0.0,
            "single_twitch_force_N": 0.52,
            "leg_kick_angle_deg": 32.4,
            "regime": "Polyneuronal Slow Axon (SETi) Frequency-Graded Extension",
        },
        {
            "experiment_id": "10_insect_feti_fast_axon_jump",
            "year": 1955,
            "investigator": "Hoyle (1955) & Usherwood (1962) Locust Fast Extensor Tibiae",
            "species": "Schistocerca gregaria (Desert Locust)",
            "apparatus": "Suprathreshold pulse recruiting Fast Extensor Tibiae (FETi) glutamatergic axon",
            "driving_voltage_mV": 800.0,
            "current_uA": 3.8,
            "duration_ms": 2.0,
            "freq_hz": 1.0,
            "polarity": "cathodal_descending",
            "make_twitch_pct": 100.0,
            "break_twitch_pct": 0.0,
            "single_twitch_force_N": 2.50,
            "leg_kick_angle_deg": 75.0,
            "regime": "All-or-None Fast Axon (FETi) Ballistic Jump Twitch (~15 N under co-contraction)",
        },
        {
            "experiment_id": "11_human_electroacupuncture_st36_deqi",
            "year": 2022,
            "investigator": "Peng et al. (2022) & Lu et al. (2019) Human Electroacupuncture",
            "species": "Homo sapiens (tibialis anterior / mylohyoid / FDI at ST36, LI4, CV23)",
            "apparatus": "0.30 mm x 25 mm stainless-steel acupuncture needles (HANS-200A, 2 Hz motor threshold)",
            "driving_voltage_mV": 3800.0,
            "current_uA": 1950.0,
            "duration_ms": 0.20,
            "freq_hz": 2.0,
            "polarity": "cathodal_descending",
            "make_twitch_pct": 12.0,
            "break_twitch_pct": 0.0,
            "single_twitch_force_N": 3.20,
            "leg_kick_angle_deg": 6.0,
            "regime": "Human Percutaneous De Qi Visible Rhythmic Motor Twitch (0.9-3.0 mA clinical window)",
        },
        {
            "experiment_id": "12_human_percutaneous_tibial_mmax",
            "year": 2023,
            "investigator": "Grospretre & Martin (2015) / Nakagawa et al. (2023) Percutaneous Motor Point",
            "species": "Homo sapiens (posterior tibial nerve / soleus & gastrocnemius)",
            "apparatus": "Percutaneous needle / motor-point stimulation at supramaximal M-max intensity",
            "driving_voltage_mV": 85000.0,
            "current_uA": 60000.0,
            "duration_ms": 1.00,
            "freq_hz": 1.0,
            "polarity": "cathodal_descending",
            "make_twitch_pct": 100.0,
            "break_twitch_pct": 0.0,
            "single_twitch_force_N": 22.0,
            "leg_kick_angle_deg": 18.0,
            "regime": "Human Supramaximal M-Wave (M-max, 40-100 mA) 100% Alpha-Motor Recruitment",
        },
    ])
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sign", action="store_true", help="Cryptographically sign every generated .trig file with ~/.nanopub/profile.yml")
    ap.add_argument("--custom-id", type=str, default=None, help="Optional custom estimate ID to mint as a nanopublication")
    ap.add_argument("--species", type=str, default="Pelophylax esculentus (= Rana esculenta)")
    ap.add_argument("--investigator", type=str, default="Joel Dietz (Galvani Interactive Lab)")
    ap.add_argument("--apparatus", type=str, default="Interactive Current-to-Twitch Bench")
    ap.add_argument("--current-uA", type=float, default=None)
    ap.add_argument("--voltage-mV", type=float, default=None)
    ap.add_argument("--duration-ms", type=float, default=2.5)
    ap.add_argument("--freq-hz", type=float, default=1.0)
    ap.add_argument("--polarity", type=str, default="cathodal_descending")
    ap.add_argument("--make-twitch-pct", type=float, default=None)
    ap.add_argument("--break-twitch-pct", type=float, default=0.0)
    ap.add_argument("--force-newton", type=float, default=None)
    ap.add_argument("--kick-angle-deg", type=float, default=None)
    ap.add_argument("--notes", type=str, default="User-registered quantitative muscle-twitch estimate")
    args = ap.parse_args()

    ref_bytes = REF_JSON.read_bytes()
    ref_sha256 = hashlib.sha256(ref_bytes).hexdigest()
    ref_data = json.loads(ref_bytes.decode("utf-8"))

    prof = load_profile_dict()
    orcid_url = prof.get("orcid_id", "https://orcid.org/0009-0007-4948-9192")
    creator_name = prof.get("name", "Joel Dietz")
    commit = git_commit()
    created_iso = "2026-10-28T15:30:00Z"

    records = build_all_canonical_records(ref_data)
    if args.custom_id and args.current_uA is not None:
        curr = float(args.current_uA)
        volt = float(args.voltage_mV) if args.voltage_mV is not None else curr * 12.5
        make_p = float(args.make_twitch_pct) if args.make_twitch_pct is not None else 100.0
        force_n = float(args.force_newton) if args.force_newton is not None else round((make_p / 100.0) * 1.20, 2)
        kick_d = float(args.kick_angle_deg) if args.kick_angle_deg is not None else round(0.65 * min(100.0, make_p), 1)
        custom_rec = {
            "experiment_id": args.custom_id,
            "year": 2026,
            "investigator": args.investigator,
            "species": args.species,
            "apparatus": args.apparatus,
            "driving_voltage_mV": volt,
            "current_uA": curr,
            "duration_ms": float(args.duration_ms),
            "freq_hz": float(args.freq_hz),
            "polarity": args.polarity,
            "make_twitch_pct": make_p,
            "break_twitch_pct": float(args.break_twitch_pct),
            "single_twitch_force_N": force_n,
            "leg_kick_angle_deg": kick_d,
            "regime": args.notes,
        }
        records.append(custom_rec)
        # Also append to a local custom estimates registry JSON
        custom_reg_path = OUT_DIR / "custom_twitch_estimates.json"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        existing = json.loads(custom_reg_path.read_text(encoding="utf-8")) if custom_reg_path.exists() else []
        existing = [e for e in existing if e.get("experiment_id") != args.custom_id] + [custom_rec]
        custom_reg_path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    signed_dir = OUT_DIR / "signed"
    signed_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = OUT_DIR / "published.json"
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    conf = None
    if args.sign and prof.get("private_key"):
        from nanopub import NanopubConf, Profile
        p_obj = Profile(
            orcid_id=orcid_url,
            name=creator_name,
            private_key=Path(prof["private_key"]),
            public_key=Path(prof["public_key"]),
        )
        conf = NanopubConf(profile=p_obj, use_test_server=True)

    for rec in records:
        np_slug, ds = build_twitch_estimate_nanopub(rec, commit, created_iso, orcid_url, creator_name, ref_sha256)
        unsigned_path = OUT_DIR / f"{np_slug}.trig"
        trig_str = plain_doubles(ds.serialize(format="trig"))
        unsigned_path.write_text(trig_str, encoding="utf-8")

        if conf is not None:
            from nanopub import Nanopub
            ds_sign = Dataset()
            ds_sign.parse(data=trig_str, format="trig")
            np_obj = Nanopub(rdf=ds_sign, conf=conf)
            np_obj.sign()
            trusty_uri = str(np_obj.source_uri)
            signed_trig = plain_doubles(np_obj.rdf.serialize(format="trig"))
            signed_path = signed_dir / f"{np_slug}.trig"
            signed_path.write_text(signed_trig, encoding="utf-8")
            manifest[f"{np_slug}.trig"] = {
                "nanopub": trusty_uri,
                "experiment_id": rec["experiment_id"],
                "species": rec.get("species"),
                "current_uA": rec["current_uA"],
                "make_twitch_pct": rec["make_twitch_pct"],
                "break_twitch_pct": rec.get("break_twitch_pct", 0.0),
                "single_twitch_force_N": rec["single_twitch_force_N"],
                "leg_kick_angle_deg": rec["leg_kick_angle_deg"],
                "published": None,
                "date": dt.date.today().isoformat(),
            }
            print(f"Signed {np_slug}.trig -> {trusty_uri}")
        else:
            print(f"Wrote unsigned {unsigned_path.name}")

    if conf is not None:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Saved trusty URI manifest ({len(manifest)} nanopubs) -> {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
