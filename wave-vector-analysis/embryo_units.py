"""
Embryo units: standardize measurements relative to one embryo length.

Convention:
  - 1 em (embryo meter) = one reference embryo length (head–tail).
  - 1 ecm (embryo centimeter) = 0.01 em = 1/100 embryo length.
  - 1 emm (embryo millimeter) = 0.001 em = 1/1000 embryo length.

Pick a reference embryo length (e.g. from detection or a canonical value in pixels),
call that 1 em, then express all distances in em, ecm, or emm for comparison
across embryos and experiments.

Usage:
  from embryo_units import px_to_emm, px_to_ecm, px_to_em

  reference_length_px = 1200   # e.g. head–tail distance in pixels for this video
  dist_px = 60
  dist_emm = px_to_emm(dist_px, reference_length_px)   # 50 emm
  dist_ecm = px_to_ecm(dist_px, reference_length_px)    # 5 ecm
  dist_em = px_to_em(dist_px, reference_length_px)      # 0.05 em
"""

# Scale factors relative to 1 em (one embryo length)
EMM_PER_EM = 1000.0   # millimeters per embryo meter
ECM_PER_EM = 100.0    # centimeters per embryo meter


def px_to_em(px, reference_length_px):
    """
    Convert pixels to embryo meters (em).
    1 em = reference embryo length (e.g. head–tail in pixels).
    """
    if reference_length_px is None or reference_length_px <= 0:
        return None
    return float(px) / float(reference_length_px)


def px_to_ecm(px, reference_length_px):
    """
    Convert pixels to embryo centimeters (ecm).
    1 ecm = 0.01 em = 1/100 of reference embryo length.
    """
    em = px_to_em(px, reference_length_px)
    if em is None:
        return None
    return em * ECM_PER_EM


def px_to_emm(px, reference_length_px):
    """
    Convert pixels to embryo millimeters (emm).
    1 emm = 0.001 em = 1/1000 of reference embryo length.
    """
    em = px_to_em(px, reference_length_px)
    if em is None:
        return None
    return em * EMM_PER_EM


def em_to_px(em, reference_length_px):
    """Convert embryo meters (em) to pixels."""
    if reference_length_px is None or reference_length_px <= 0:
        return None
    return float(em) * float(reference_length_px)


def ecm_to_px(ecm, reference_length_px):
    """Convert embryo centimeters (ecm) to pixels."""
    if reference_length_px is None or reference_length_px <= 0:
        return None
    em = float(ecm) / ECM_PER_EM
    return em * float(reference_length_px)


def emm_to_px(emm, reference_length_px):
    """Convert embryo millimeters (emm) to pixels."""
    if reference_length_px is None or reference_length_px <= 0:
        return None
    em = float(emm) / EMM_PER_EM
    return em * float(reference_length_px)


def add_embryo_unit_columns(df, reference_length_px=None, length_column="embryo_length_px"):
    """
    Add embryo-unit columns to a DataFrame that has pixel distances.

    Either pass a scalar reference_length_px (same for all rows) or ensure the
    DataFrame has a column length_column (e.g. embryo_length_px) with per-row
    reference length. If both are None/missing, no columns are added.

    Adds (when source columns exist):
      - dist_from_poke_emm, dist_from_poke_ecm from dist_from_poke_px
      - dv_emm, dv_ecm from dv_px
    """
    use_per_row = length_column in df.columns and reference_length_px is None
    out = df.copy()
    if use_per_row:
        ref = df[length_column]
    else:
        ref = reference_length_px
    if ref is None:
        return out
    if "dist_from_poke_px" in df.columns:
        if use_per_row:
            out["dist_from_poke_emm"] = [px_to_emm(px, r) for px, r in zip(df["dist_from_poke_px"], ref)]
            out["dist_from_poke_ecm"] = [px_to_ecm(px, r) for px, r in zip(df["dist_from_poke_px"], ref)]
        else:
            out["dist_from_poke_emm"] = df["dist_from_poke_px"].apply(lambda px: px_to_emm(px, ref))
            out["dist_from_poke_ecm"] = df["dist_from_poke_px"].apply(lambda px: px_to_ecm(px, ref))
    if "dv_px" in df.columns:
        if use_per_row:
            out["dv_emm"] = [px_to_emm(px, r) for px, r in zip(df["dv_px"], ref)]
            out["dv_ecm"] = [px_to_ecm(px, r) for px, r in zip(df["dv_px"], ref)]
        else:
            out["dv_emm"] = df["dv_px"].apply(lambda px: px_to_emm(px, ref))
            out["dv_ecm"] = df["dv_px"].apply(lambda px: px_to_ecm(px, ref))
    return out
