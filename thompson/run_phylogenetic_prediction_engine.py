#!/usr/bin/env python3
"""
Phylogenetic prediction engine for D'Arcy Thompson's (1917) fish body-plan coordinate law.

Turns the Thompson coordinate law (validated pairwise in 1917-thompson-four-families-atlas.py)
into a *time-calibrated, uncertainty-quantified* predictor of body planform on the Rabosky et
al. (2018) megaphylogeny, and tests it blind.

  1. Trait space: "Thompson modes".  The three non-linear 1917 families (coaxial Moebius
     inversion, bilinear fan, parabolic/exponential radial) and the affine block are
     linearised at the Procrustes consensus M0.  Their deformation fields, evaluated at the
     17 consensus landmarks and made orthogonal to the rigid modes (translation, rotation,
     scale), span a 7-D linear subspace of shape space:
         aspect (u,-v), shear (v,u), fan (uv,0), fan (0,uv), radial bow (v^2,0),
         coaxial Re (u^2-v^2, 2uv), coaxial Im (2uv, -(u^2-v^2)).
     A species' Thompson coordinates z (7-D) are the least-squares projection of its
     Procrustes residual Y_i - M0 onto that subspace.  Because the map z -> landmarks is
     linear, Gaussian conditioning on the tree is exact and a predicted z is always a
     legitimate shape.  The craniofacial/jaw module is what is left after removing the
     Thompson field (LM 1-2, 11-17), reduced to PCs.

  2. Process models.  Multivariate Brownian motion (BM), single-optimum Ornstein-Uhlenbeck
     (OU), Early Burst (EB), Pagel's lambda and a star/white-noise null, by phylogenetic GLS
     (closed-form root + rate matrix; alpha / r / lambda profiled), compared by AICc.  Fitted
     separately to the Thompson modes, the jaw residual, and the full 30-D shape space.

  3. Prediction = Gaussian conditioning.  For any target node (held-out tip, internal node,
     or a *virtual* node at any age along any lineage) the conditional mean/covariance of z
     given the observed tips follow from the model covariance.  Predicted landmarks are
     M0 + B z.

  4. Validation.  (a) Leave-one-species / genus / family-out: error vs. horizon tau (age of
     the MRCA with the nearest remaining tip), 95% ellipsoid coverage, and comparison with
     (i) full 30-D shape-space phylogenetic prediction, (ii) an equal-dimension PCA basis,
     (iii) copying the nearest sampled relative, (iv) the consensus.  (b) Blind prediction of
     the seven museum specimens of Thompson's own 1917 species, which are on the tree but not
     in the training data.  (c) Monte Bolca (~50 Ma) fossils, if digitised in
     datasets/thompson/monte_bolca_fossil_landmarks.json.

  5. Registered predictions: crown-node ancestral body plans for every sampled family and
     the state of each Bolca-recorded family's lineage at 50 Ma, each with a 95% region.

Outputs
  thompson_phylogenetic_prediction_engine.json
  docs/outreach_fig9_phylogenetic_prediction_engine.png
  docs/THOMPSON_PHYLOGENETIC_PREDICTIONS.md
"""
from __future__ import annotations

import importlib.util
import json
import lzma
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar

REPO_ROOT = Path(__file__).resolve().parents[1]
BOLCA_JSON = REPO_ROOT / "datasets" / "thompson" / "monte_bolca_fossil_landmarks.json"
BOLCA_RATIOS_JSON = REPO_ROOT / "datasets" / "thompson" / "monte_bolca_fossil_ratios.json"
OUT_JSON = REPO_ROOT / "thompson_phylogenetic_prediction_engine.json"
OUT_FIG = REPO_ROOT / "docs" / "outreach_fig9_phylogenetic_prediction_engine.png"
OUT_MD = REPO_ROOT / "docs" / "THOMPSON_PHYLOGENETIC_PREDICTIONS.md"

BOLCA_AGE_MA = 50.0
# Families with a Monte Bolca (Ypresian) fossil record that are also sampled in the Claverie &
# Wainwright (2014) landmark set.  Bellwood (1996) Coral Reefs 15:11; Carnevale, Bannikov,
# Marrama, Tyler & Zorzin (2014) "The Bolca Fossil-Lagerstaetten"; Friedman et al. (2016).
BOLCA_FAMILIES = {
    "Acanthuridae": "Eonaso, Padovathurus, Sorbinithurus, Tylerichthys",
    "Apogonidae": "Eoapogon",
    "Carangidae": "Eastmanalepes, Ceratoichthys, Pseudovomer",
    "Ephippidae": "Eoplatax papilio, Exellia velifera",
    "Holocentridae": "Eoholocentrum macrocephalum",
    "Labridae": "Eocoris bloti, Phyllopharyngodon longipinnis, Bellwoodilabrus landinii",
    "Leiognathidae": "Eoleiognathus dorsalis (ex 'Pygaeus dorsalis')",
    "Lutjanidae": "Ottaviania leptacanthus, Veranichthys ventralis",
    "Pomacentridae": "Palaeopomacentrus orphae, Lorenzichthys olihan",
    "Priacanthidae": "Pristigenys substriata (type species of Pristigenys)",
    "Serranidae": "Eoserranus",
    "Siganidae": "Ruffoichthys spinosus, Ruffoichthys bannikovi",
    "Sparidae": "Sparnodus vulgaris, Sparnodus elongatus, Goujetia crassispina, Abromasta",
    "Sphyraenidae": "Sphyraena bolcensis",
    "Zanclidae": "Eozanclus brevirostris",
    "Scombridae": "Eocoelopoma, Scombrinus",
}

HEAD_LMS = [0, 1, 10, 11, 12, 13, 14, 15, 16]
FIN_LMS = [2, 3, 4, 5, 6, 7, 8, 9]
MODE_NAMES = ["aspect (u,-v)", "shear (v,u)", "fan (uv,0)", "fan (0,uv)", "radial bow (v^2,0)",
              "coaxial Re (u^2-v^2, 2uv)", "coaxial Im (2uv, -(u^2-v^2))"]


def load_module(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


at = load_module("at", "thompson/1917-thompson-four-families-atlas.py")


# ----------------------------------------------------------------------------- Thompson modes
def thompson_mode_basis(M0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (B, Rg): B is 34 x 7 (Thompson tangent fields at the consensus, orthogonalised
    against the rigid modes), Rg is 34 x 4 (translation x/y, rotation, scale)."""
    u, v = M0[:, 0], M0[:, 1]
    z = np.zeros_like(u)
    one = np.ones_like(u)
    fields = [
        (u, -v), (v, u), (u * v, z), (z, u * v), (v ** 2, z),
        (u ** 2 - v ** 2, 2 * u * v), (2 * u * v, -(u ** 2 - v ** 2)),
    ]
    rigid = [(one, z), (z, one), (-v, u), (u, v)]
    Rg = np.column_stack([np.column_stack([fx, fy]).ravel() for fx, fy in rigid])
    B = np.column_stack([np.column_stack([fx, fy]).ravel() for fx, fy in fields])
    Qr, _ = np.linalg.qr(Rg)
    B = B - Qr @ (Qr.T @ B)
    B = B / np.linalg.norm(B, axis=0, keepdims=True)
    return B, Rg


def project(B: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Least-squares coefficients of rows of D (n x 34) on basis B (34 x m)."""
    return np.linalg.lstsq(B, D.T, rcond=None)[0].T


def shape_rms(pred: np.ndarray, target: np.ndarray) -> float:
    return at.rms(at.opa(at.centre_scale(pred), target), target)


# ----------------------------------------------------------------------------- tree
class Tree:
    def __init__(self, parent_all: List[int], depth_all: List[float], keep: Dict[str, int]):
        keep_nodes = set()
        for u in keep.values():
            c = u
            while c != -1 and c not in keep_nodes:
                keep_nodes.add(c)
                c = parent_all[c]
        ch: Dict[int, List[int]] = defaultdict(list)
        root = None
        for u in keep_nodes:
            p = parent_all[u]
            if p == -1 or p not in keep_nodes:
                root = u
            else:
                ch[p].append(u)

        def collapse(u: int) -> int:
            while len(ch[u]) == 1:
                u = ch[u][0]
            return u

        new_root = collapse(root)
        order, new_parent = [], {}
        stack = [new_root]
        while stack:
            u = stack.pop()
            order.append(u)
            for c in ch[u]:
                c2 = collapse(c)
                new_parent[c2] = u
                stack.append(c2)
        idx = {u: i for i, u in enumerate(order)}
        self.N = len(order)
        self.parent = np.array([idx[new_parent[u]] if u in new_parent else -1 for u in order])
        h = np.array([depth_all[u] for u in order], dtype=float)
        self.h = h - h[idx[new_root]]
        self.T = float(np.max(self.h))
        id_to_name = {u: n for n, u in keep.items()}
        self.names = [id_to_name.get(u, "") for u in order]
        self.name_to_node = {n: i for i, n in enumerate(self.names) if n}
        self.children: List[List[int]] = [[] for _ in range(self.N)]
        for v in range(self.N):
            if self.parent[v] >= 0:
                self.children[self.parent[v]].append(v)
        self.desc: List[np.ndarray] = [None] * self.N
        for u in reversed(range(self.N)):
            d = [u]
            for c in self.children[u]:
                d.extend(self.desc[c].tolist())
            self.desc[u] = np.array(d)
        C = np.zeros((self.N, self.N))
        for u in range(self.N):
            d = self.desc[u]
            C[u, d] = self.h[u]
            C[d, u] = self.h[u]
            cs = self.children[u]
            for i in range(len(cs)):
                for j in range(i + 1, len(cs)):
                    C[np.ix_(self.desc[cs[i]], self.desc[cs[j]])] = self.h[u]
                    C[np.ix_(self.desc[cs[j]], self.desc[cs[i]])] = self.h[u]
        self.C = C

    def mrca(self, nodes: Sequence[int]) -> int:
        anc_sets = []
        for u in nodes:
            s, c = [], u
            while c != -1:
                s.append(c)
                c = self.parent[c]
            anc_sets.append(s)
        common = set(anc_sets[0])
        for s in anc_sets[1:]:
            common &= set(s)
        return max(common, key=lambda x: self.h[x])

    def virtual_node(self, tip: int, age_ma: float) -> Tuple[np.ndarray, float]:
        hv = self.T - age_ma
        if hv < 0:
            raise ValueError("age exceeds root age")
        c = tip
        while self.parent[c] != -1 and self.h[self.parent[c]] >= hv:
            c = self.parent[c]
        return np.minimum(hv, self.C[c]), hv


# ----------------------------------------------------------------------------- process models
class Model:
    T: float = 0.0

    def __init__(self, kind: str, theta: Optional[float] = None):
        self.kind, self.theta = kind, theta
        self.n_extra = 0 if kind in ("BM", "WN") else 1

    def cov(self, t, hi, hj):
        k = self.kind
        if k == "BM":
            return t
        if k == "OU":
            a = self.theta
            d = hi + hj - 2.0 * t
            return np.exp(-a * d) * (1.0 - np.exp(-2.0 * a * t)) / (2.0 * a)
        if k == "EB":
            r = self.theta
            return (np.exp(r * t) - 1.0) / r
        if k == "lambda":
            lam = self.theta
            same = np.isclose(t, hi) & np.isclose(t, hj)
            is_present_tip = same & np.isclose(hi, self.T)
            return np.where(is_present_tip, t, lam * t)
        if k == "WN":
            same = np.isclose(t, hi) & np.isclose(t, hj)
            return np.where(same, 1.0, 0.0)
        raise ValueError(k)

    def cov_matrix(self, C, h, rows, cols):
        return self.cov(C[np.ix_(rows, cols)], h[rows][:, None], h[cols][None, :])

    def label(self) -> str:
        if self.kind == "OU":
            return f"OU (alpha={self.theta:.4f}/Myr, t1/2={np.log(2) / self.theta:.0f} Myr)"
        if self.kind == "EB":
            return f"EB (r={self.theta:.4f}/Myr)"
        if self.kind == "lambda":
            return f"Pagel lambda={self.theta:.3f}"
        return {"BM": "Brownian motion", "WN": "White noise (star)"}[self.kind]


def gls(V: np.ndarray, X: np.ndarray):
    n, p = X.shape
    L = np.linalg.cholesky(V + 1e-10 * np.eye(n))
    one = np.ones((n, 1))
    Vi1 = np.linalg.solve(L.T, np.linalg.solve(L, one))
    ViX = np.linalg.solve(L.T, np.linalg.solve(L, X))
    a = (one.T @ ViX) / float((one.T @ Vi1)[0, 0])
    E = X - a
    ViE = ViX - Vi1 @ a
    R = (E.T @ ViE) / n
    logdetV = 2.0 * float(np.sum(np.log(np.diag(L))))
    _, logdetR = np.linalg.slogdet(R)
    ll = -0.5 * (n * p * np.log(2 * np.pi) + p * logdetV + n * logdetR + n * p)
    return a.ravel(), R, float(ll), L


def fit_models(tree: Tree, tips: np.ndarray, X: np.ndarray) -> Dict[str, dict]:
    n, p = X.shape
    k_base = p + p * (p + 1) // 2

    def neg_ll(kind, theta):
        m = Model(kind, theta)
        try:
            return -gls(m.cov_matrix(tree.C, tree.h, tips, tips), X)[2]
        except np.linalg.LinAlgError:
            return 1e30

    out = {}
    for kind in ["WN", "BM", "lambda", "OU", "EB"]:
        if kind == "OU":
            res = minimize_scalar(lambda la: neg_ll("OU", np.exp(la)), bounds=(np.log(1e-4), np.log(1.0)), method="bounded")
            m = Model("OU", float(np.exp(res.x)))
        elif kind == "EB":
            res = minimize_scalar(lambda lr: neg_ll("EB", -np.exp(lr)), bounds=(np.log(1e-5), np.log(0.2)), method="bounded")
            m = Model("EB", float(-np.exp(res.x)))
        elif kind == "lambda":
            res = minimize_scalar(lambda la: neg_ll("lambda", la), bounds=(0.0, 1.0), method="bounded")
            m = Model("lambda", float(res.x))
        else:
            m = Model(kind)
        a, R, ll, _ = gls(m.cov_matrix(tree.C, tree.h, tips, tips), X)
        k = k_base + m.n_extra
        aicc = 2 * k - 2 * ll + 2 * k * (k + 1) / max(n - k - 1, 1)
        out[kind] = {"model": m, "root": a, "R": R, "logL": ll, "k": k, "AICc": aicc}
    best = min(d["AICc"] for d in out.values())
    for d in out.values():
        d["dAICc"] = d["AICc"] - best
    return out


def predict(model: Model, tree: Tree, obs: np.ndarray, X_obs: np.ndarray, tgt_h: np.ndarray, tgt_C: np.ndarray):
    """Conditional Gaussian prediction. Returns mean (m x p), scalar variance factor (m,), root, R."""
    Voo = model.cov_matrix(tree.C, tree.h, obs, obs)
    a, R, _, L = gls(Voo, X_obs)
    Vto = model.cov(tgt_C[:, obs], tgt_h[:, None], tree.h[obs][None, :])
    Vtt = model.cov(tgt_h, tgt_h, tgt_h)
    K = np.linalg.solve(L.T, np.linalg.solve(L, Vto.T))
    mean = a[None, :] + K.T @ (X_obs - a)
    s2 = np.maximum(Vtt - np.einsum("mn,nm->m", Vto, K), 1e-12)
    return mean, s2, a, R


def mahal2(z, mu, S) -> float:
    d = z - mu
    return float(d @ np.linalg.solve(S, d))


# ----------------------------------------------------------------------------- main
def main(argv: Optional[Sequence[str]] = None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("[1] landmarks, GPA, consensus, Thompson modes")
    names, families, X = at.load_landmarks()
    Y = at.gpa(X)
    M0 = at.centre_scale(Y.mean(0))
    Y_opa = np.array([at.opa(y, M0) for y in Y])
    fam_of = dict(zip(names, families))
    B, Rg = thompson_mode_basis(M0)
    m_dim = B.shape[1]
    D_all = (Y_opa - M0).reshape(len(names), -1)
    Z_all = project(B, D_all)
    recon = M0.reshape(-1) + Z_all @ B.T
    repr_rms_all = np.sqrt(((D_all - Z_all @ B.T) ** 2).reshape(len(names), 17, 2).sum(-1).mean(-1))
    E_all = D_all - Z_all @ B.T  # residual after removing the Thompson field

    print("[2] Rabosky et al. (2018) tree -> pruned to sampled species + held-out 1917 species")
    tree_path = at.fetch_phylogeny(allow_download=False)
    with lzma.open(tree_path, "rt") as fh:
        parent_all, depth_all, tip_to_id = at.parse_newick_depths(fh.read())
    exact = at.load_exact_1917_species(M0, allow_download=False)
    exact_meta = json.loads(at.EXACT_SPECIES_JSON.read_text(encoding="utf-8"))["specimens"]
    for s_, cfg_ in exact_meta.items():
        fam_of.setdefault(s_, cfg_.get("family", ""))
    keep = {n: tip_to_id[n] for n in names if n in tip_to_id}
    for n in exact:
        if n in tip_to_id:
            keep[n] = tip_to_id[n]
    tree = Tree(parent_all, depth_all, keep)
    Model.T = tree.T
    train_names = [n for n in names if n in tree.name_to_node]
    tips = np.array([tree.name_to_node[n] for n in train_names])
    name_idx = {n: i for i, n in enumerate(names)}
    tr = np.array([name_idx[n] for n in train_names])
    Zt = Z_all[tr]
    Dt = D_all[tr]
    n = len(train_names)
    print(f"    {n} training tips, crown age of sampled tree {tree.T:.1f} Ma, {tree.N} nodes")

    # Full shape space (30 non-rigid dims) via exact PCA rotation of the Procrustes residuals
    Dc = Dt.mean(0)
    U_, S_, Vt_ = np.linalg.svd(Dt - Dc, full_matrices=False)
    n_full = 30
    P_full = Vt_[:n_full].T                                   # 34 x 30
    F_t = (Dt - Dc) @ P_full                                  # full-space coordinates
    P_pca7 = Vt_[:m_dim].T                                    # equal-dimension PCA baseline
    Q_t = (Dt - Dc) @ P_pca7
    repr_rms_pca7 = np.sqrt((((Dt - Dc) - Q_t @ P_pca7.T) ** 2).reshape(n, 17, 2).sum(-1).mean(-1))
    var_thompson = float(np.sum((Zt @ B.T - (Zt @ B.T).mean(0)) ** 2) / np.sum((Dt - Dc) ** 2))
    var_pca7 = float(np.sum(S_[:m_dim] ** 2) / np.sum(S_ ** 2))

    # Jaw module: residual after Thompson field, craniofacial landmarks, top PCs
    Eh = E_all[tr].reshape(n, 17, 2)[:, HEAD_LMS, :].reshape(n, -1)
    Eh_c = Eh - Eh.mean(0)
    Uj, Sj, Vtj = np.linalg.svd(Eh_c, full_matrices=False)
    n_jaw = 4
    jaw = Uj[:, :n_jaw] * Sj[:n_jaw]
    jaw_var_expl = float(np.sum(Sj[:n_jaw] ** 2) / np.sum(Sj ** 2))

    print("[3] process-model fits: Thompson modes / jaw residual / full 30-D shape")
    fits_body = fit_models(tree, tips, Zt)
    fits_jaw = fit_models(tree, tips, jaw)
    fits_full = fit_models(tree, tips, F_t)
    fits_pca7 = fit_models(tree, tips, Q_t)
    for lab, fits in [("THOMPSON MODES (7-D)", fits_body), ("JAW residual (4 PCs)", fits_jaw), ("FULL shape (30-D)", fits_full), ("PCA-7 baseline", fits_pca7)]:
        print(f"    {lab}")
        for k, d in sorted(fits.items(), key=lambda kv: kv[1]["AICc"]):
            print(f"      {d['model'].label():44s} logL={d['logL']:10.1f}  k={d['k']:4d}  dAICc={d['dAICc']:9.1f}")
    best_body = min(fits_body, key=lambda k: fits_body[k]["AICc"])
    best_jaw = min(fits_jaw, key=lambda k: fits_jaw[k]["AICc"])
    best_full = min(fits_full, key=lambda k: fits_full[k]["AICc"])
    best_pca7 = min(fits_pca7, key=lambda k: fits_pca7[k]["AICc"])
    model = fits_body[best_body]["model"]
    model_full = fits_full[best_full]["model"]
    model_pca7 = fits_pca7[best_pca7]["model"]
    cond_R = float(np.linalg.cond(fits_body[best_body]["R"]))
    cond_full = float(np.linalg.cond(fits_full[best_full]["R"]))
    CHI2_95 = float(stats.chi2.ppf(0.95, m_dim)); CHI2_50 = float(stats.chi2.ppf(0.50, m_dim))

    # Per-mode phylogenetic signal (univariate lambda) for interpretability
    mode_lambda = []
    for j in range(m_dim):
        f1 = fit_models(tree, tips, Zt[:, [j]])
        mode_lambda.append({"mode": MODE_NAMES[j], "lambda": f1["lambda"]["model"].theta, "best": min(f1, key=lambda k: f1[k]["AICc"]),
                            "sd_across_species": float(np.std(Zt[:, j]))})

    # Floors
    gen_members = defaultdict(list)
    for n_ in train_names:
        gen_members[n_.split("_")[0]].append(name_idx[n_])
    sq = []
    for g, idxs in gen_members.items():
        if len(idxs) >= 3:
            m = Y_opa[idxs].mean(0)
            sq.extend([at.rms(Y_opa[i], m) ** 2 for i in idxs])
    sigma_genus = float(np.sqrt(np.mean(sq)))
    repr_floor = float(np.sqrt(np.mean(repr_rms_all[tr] ** 2)))
    repr_floor_pca7 = float(np.sqrt(np.mean(repr_rms_pca7 ** 2)))
    print(f"    sigma_genus = {sigma_genus:.4f}; Thompson-7 representation floor = {repr_floor:.4f} ({100 * var_thompson:.1f}% of shape variance); PCA-7 floor = {repr_floor_pca7:.4f} ({100 * var_pca7:.1f}%)")

    # ------------------------------------------------------------------ [4] held-out validation
    print("[4] leave-one-out validation (species / genus / family)")

    def to_lm(vec34):
        return (M0.reshape(-1) + vec34).reshape(17, 2)

    def holdout(groups: Dict[str, List[int]], label: str) -> List[dict]:
        recs = []
        for g, members in groups.items():
            held = np.array([tree.name_to_node[train_names[i]] for i in members])
            obs_mask = np.ones(n, bool)
            obs_mask[members] = False
            obs = tips[obs_mask]
            mean, s2, a, R = predict(model, tree, obs, Zt[obs_mask], tree.h[held], tree.C[held])
            mean_f, _, _, _ = predict(model_full, tree, obs, F_t[obs_mask], tree.h[held], tree.C[held])
            mean_q, _, _, _ = predict(model_pca7, tree, obs, Q_t[obs_mask], tree.h[held], tree.C[held])
            for j, i in enumerate(members):
                tau = tree.T - float(np.max(tree.C[held[j], obs]))
                yi = Y_opa[tr[i]]
                nn = obs[int(np.argmax(tree.C[held[j], obs]))]
                m2 = mahal2(Zt[i], mean[j], s2[j] * R)
                recs.append({
                    "scheme": label, "group": g, "species": train_names[i], "tau_myr": tau,
                    "rms_thompson": shape_rms(to_lm(mean[j] @ B.T), yi),
                    "rms_full30": shape_rms(to_lm(Dc + mean_f[j] @ P_full.T), yi),
                    "rms_pca7": shape_rms(to_lm(Dc + mean_q[j] @ P_pca7.T), yi),
                    "rms_consensus": shape_rms(M0, yi),
                    "rms_nearest_tip": shape_rms(Y_opa[name_idx[tree.names[nn]]], yi),
                    "rms_thompson_oracle": float(repr_rms_all[tr[i]]),
                    "mahal2": m2, "in95": bool(m2 <= CHI2_95), "in50": bool(m2 <= CHI2_50),
                })
        return recs

    species_groups = {train_names[i]: [i] for i in range(n)}
    genus_groups, family_groups = defaultdict(list), defaultdict(list)
    for i, n_ in enumerate(train_names):
        genus_groups[n_.split("_")[0]].append(i)
        family_groups[fam_of[n_]].append(i)
    val = holdout(species_groups, "species") + holdout(dict(genus_groups), "genus") + holdout(dict(family_groups), "family")

    bins = [(0, 5), (5, 10), (10, 25), (25, 50), (50, 80), (80, 120), (120, 400)]
    keys = ["rms_thompson", "rms_full30", "rms_pca7", "rms_nearest_tip", "rms_consensus", "rms_thompson_oracle"]
    horizon_table = []
    for lo, hi in bins:
        sub = [r for r in val if lo <= r["tau_myr"] < hi]
        if not sub:
            continue
        row = {"tau_bin_myr": [lo, hi], "n": len(sub)}
        for k in keys:
            row["median_" + k] = float(np.median([r[k] for r in sub]))
        row["pct_within_sigma_genus_thompson"] = 100.0 * float(np.mean([r["rms_thompson"] <= sigma_genus for r in sub]))
        row["pct_within_sigma_genus_full30"] = 100.0 * float(np.mean([r["rms_full30"] <= sigma_genus for r in sub]))
        row["pct_thompson_beats_consensus"] = 100.0 * float(np.mean([r["rms_thompson"] < r["rms_consensus"] for r in sub]))
        row["pct_full30_beats_nearest_tip"] = 100.0 * float(np.mean([r["rms_full30"] < r["rms_nearest_tip"] for r in sub]))
        row["coverage95_pct"] = 100.0 * float(np.mean([r["in95"] for r in sub]))
        row["coverage50_pct"] = 100.0 * float(np.mean([r["in50"] for r in sub]))
        horizon_table.append(row)
        print(f"    tau {lo:3d}-{hi:3d}  n={len(sub):4d}  Thompson {row['median_rms_thompson']:.4f} | full30 {row['median_rms_full30']:.4f} | pca7 {row['median_rms_pca7']:.4f} | nearest {row['median_rms_nearest_tip']:.4f} | consensus {row['median_rms_consensus']:.4f} | oracle {row['median_rms_thompson_oracle']:.4f}  cov95 {row['coverage95_pct']:5.1f}%")
    m2_all = np.array([r["mahal2"] for r in val])

    # Prediction skill: how much of the full-space phylogenetically predictable improvement over the
    # consensus is captured inside the 7 Thompson modes?
    def skill(key):
        g = np.array([r["rms_consensus"] ** 2 - r[key] ** 2 for r in val])
        return g
    gain_full = skill("rms_full30"); gain_th = skill("rms_thompson")
    skill_share = float(np.sum(gain_th) / np.sum(gain_full)) if np.sum(gain_full) > 0 else float("nan")

    # ------------------------------------------------------------------ [5] blind 1917 species
    print("[5] blind prediction of Thompson's own 1917 species (on tree, not in training data)")
    held_names = [s for s in exact if s in tree.name_to_node]
    held = np.array([tree.name_to_node[s] for s in held_names])
    mean, s2, a, R = predict(model, tree, tips, Zt, tree.h[held], tree.C[held])
    mean_f, _, _, _ = predict(model_full, tree, tips, F_t, tree.h[held], tree.C[held])
    exact_recs = []
    for j, s in enumerate(held_names):
        yi = exact[s]
        d_i = (yi - M0).reshape(-1)
        z_own = project(B, d_i[None, :])[0]
        tau = tree.T - float(np.max(tree.C[held[j], tips]))
        nn = tips[int(np.argmax(tree.C[held[j], tips]))]
        pred_lm = to_lm(mean[j] @ B.T)
        rec = {
            "species": s, "family": fam_of.get(s, ""), "tau_myr": tau, "nearest_training_tip": tree.names[nn],
            "rms_thompson": shape_rms(pred_lm, yi), "rms_full30": shape_rms(to_lm(Dc + mean_f[j] @ P_full.T), yi),
            "rms_consensus": shape_rms(M0, yi), "rms_nearest_tip": shape_rms(Y_opa[name_idx[tree.names[nn]]], yi),
            "rms_thompson_oracle": shape_rms(to_lm(z_own @ B.T), yi),
            "mahal2": mahal2(z_own, mean[j], s2[j] * R), "z_pred": mean[j].tolist(), "z_obs": z_own.tolist(),
            "z_sd": np.sqrt(np.diag(s2[j] * R)).tolist(),
            "pred_landmarks": at.opa(at.centre_scale(pred_lm), yi).tolist(), "obs_landmarks": yi.tolist(),
        }
        rec["in95"] = bool(rec["mahal2"] <= CHI2_95)
        exact_recs.append(rec)
        print(f"    {s:24s} tau={tau:6.1f}  nearest={tree.names[nn]:26s} Thompson {rec['rms_thompson']:.4f} | full30 {rec['rms_full30']:.4f} | consensus {rec['rms_consensus']:.4f} | nearest {rec['rms_nearest_tip']:.4f} | oracle {rec['rms_thompson_oracle']:.4f}  M2={rec['mahal2']:.1f} in95={rec['in95']}")

    # ------------------------------------------------------------------ [6] registered predictions
    print("[6] registered ancestral predictions: family crown nodes, 1917-pair MRCAs, Bolca-age lineage states")
    fam_tips = defaultdict(list)
    for i, n_ in enumerate(train_names):
        fam_tips[fam_of[n_]].append(tips[i])
    rows_C, rows_h, labels = [], [], []
    for f, tl in sorted(fam_tips.items()):
        if len(tl) < 3:
            continue
        node = tree.mrca(tl)
        rows_C.append(tree.C[node]); rows_h.append(tree.h[node]); labels.append(("crown", f, len(tl), None, None))
    specials = [("Scarus_ghobban", "Pomacanthus_semicirculatus"), ("Diodon_hystrix", "Mola_mola"),
                ("Pristigenys_alta", "Scorpaena_scrofa"), ("Scorpaena_scrofa", "Antigonia_capros"),
                ("Argyropelecus_olfersii", "Sternoptyx_diaphana"), ("Polyprion_americanus", "Pristigenys_alta")]
    for s1, s2_ in specials:
        if s1 in tree.name_to_node and s2_ in tree.name_to_node:
            node = tree.mrca([tree.name_to_node[s1], tree.name_to_node[s2_]])
            rows_C.append(tree.C[node]); rows_h.append(tree.h[node]); labels.append(("mrca", f"{s1} + {s2_}", 2, None, None))
    for f in sorted(BOLCA_FAMILIES):
        tl = fam_tips.get(f, [])
        if not tl:
            continue
        crown = tree.mrca(tl) if len(tl) > 1 else tl[0]
        crown_age = tree.T - tree.h[crown]
        # lineage: if the sampled crown is younger than 50 Ma use the stem (any tip gives the same
        # branch); otherwise use the lineage to the tip closest to the family's mean shape.
        if crown_age < BOLCA_AGE_MA:
            rep = tl[0]
        else:
            fm = Y_opa[[name_idx[tree.names[t]] for t in tl]].mean(0)
            rep = min(tl, key=lambda t: at.rms(Y_opa[name_idx[tree.names[t]]], fm))
        row, hv = tree.virtual_node(rep, BOLCA_AGE_MA)
        rows_C.append(row); rows_h.append(hv); labels.append(("bolca50", f, len(tl), float(crown_age), tree.names[rep]))
    rows_C = np.array(rows_C); rows_h = np.array(rows_h)
    mean, s2, a, R = predict(model, tree, tips, Zt, rows_h, rows_C)
    evals, evecs = np.linalg.eigh(R)
    ax1_dir = evecs[:, -1]

    def body_depth(L):  # dorsal-fin origin (LM3) to anal-fin origin (LM8)
        return float(np.linalg.norm(L[2] - L[7]))

    def std_len(L):  # snout (mean LM1,2) to caudal-fin insertion midpoint (LM5,6)
        return float(np.linalg.norm(0.5 * (L[0] + L[1]) - 0.5 * (L[4] + L[5])))

    def head_len(L):  # snout to opercle (LM11)
        return float(np.linalg.norm(0.5 * (L[0] + L[1]) - L[10]))

    def predorsal(L):  # snout to dorsal-fin origin, straight-line (the published convention)
        return float(np.linalg.norm(L[2] - 0.5 * (L[0] + L[1])))

    def bdmax(L):  # maximum body depth excluding fin webs = vertical extent of the 17 body landmarks
        return float(L[:, 1].max() - L[:, 1].min())

    def ratios(L):
        sl = std_len(L)
        return {"BD_over_SL": body_depth(L) / sl, "BDmax_over_SL": bdmax(L) / sl, "HL_over_SL": head_len(L) / sl, "PD_over_SL": predorsal(L) / sl}

    rng = np.random.default_rng(1917)

    def mc_ratio_interval(mu, S, to_landmarks, n_mc=1500):
        zs = rng.multivariate_normal(mu, S, size=n_mc)
        vals = defaultdict(list)
        for zz in zs:
            for k_, v_ in ratios(at.centre_scale(to_landmarks(zz))).items():
                vals[k_].append(v_)
        return {k_: [float(np.percentile(v_, 2.5)), float(np.percentile(v_, 97.5))] for k_, v_ in vals.items()}

    def lm_th(zz):
        return to_lm(zz @ B.T)

    def lm_full(ff):
        return to_lm(Dc + ff @ P_full.T)

    mean_f30, s2_f30, a_f30, R_f30 = predict(model_full, tree, tips, F_t, rows_h, rows_C)
    taus = np.array([r["tau_myr"] for r in val]); errs = np.array([r["rms_thompson"] for r in val])
    registered = []
    for j, lab in enumerate(labels):
        S = s2[j] * R
        S_f = s2_f30[j] * R_f30
        lm = at.centre_scale(lm_th(mean[j]))
        lmf = at.centre_scale(lm_full(mean_f30[j]))
        age = float(tree.T - rows_h[j])
        sel = np.abs(taus - age) <= max(5.0, 0.2 * age)
        sel_f = sel
        rec = {
            "kind": lab[0], "target": lab[1], "n_sampled_tips": lab[2], "age_ma": age,
            "thompson7": {"z_mean": mean[j].tolist(), "z_sd": np.sqrt(np.diag(S)).tolist(), "pred_var_factor": float(s2[j]),
                          "pred_landmarks": lm.tolist(), "ratios": ratios(lm), "ratios_95": mc_ratio_interval(mean[j], S, lm_th)},
            "full30": {"pred_landmarks": lmf.tolist(), "pred_var_factor": float(s2_f30[j]), "ratios": ratios(lmf), "ratios_95": mc_ratio_interval(mean_f30[j], S_f, lm_full)},
            "expected_landmark_rms_at_this_horizon_thompson7": float(np.median(errs[sel])) if sel.sum() >= 10 else None,
            "expected_landmark_rms_at_this_horizon_full30": float(np.median(np.array([r["rms_full30"] for r in val])[sel_f])) if sel_f.sum() >= 10 else None,
        }
        if lab[0] == "bolca50":
            rec["family_crown_age_ma"] = lab[3]; rec["lineage_tip"] = lab[4]; rec["bolca_taxa"] = BOLCA_FAMILIES[lab[1]]
            fm = Y_opa[[name_idx[tree.names[t]] for t in fam_tips[lab[1]]]].mean(0)
            rec["living_family_mean_ratios"] = ratios(fm)
            fam_ratio_sd = {k_: float(np.std([ratios(Y_opa[name_idx[tree.names[t]]])[k_] for t in fam_tips[lab[1]]])) for k_ in ratios(fm)}
            rec["living_family_ratio_sd"] = fam_ratio_sd
        registered.append(rec)

    # ------------------------------------------------------------------ [7] Monte Bolca fossils (optional)
    bolca_recs = []
    if BOLCA_JSON.exists():
        print("[7] Monte Bolca fossil test")
        meta = json.loads(BOLCA_JSON.read_text())
        for taxon, cfg in meta["specimens"].items():
            w, h_ = float(cfg["width"]), float(cfg["height"])
            pts = np.array([[x * w, (1.0 - y) * h_] for x, y in cfg["landmarks_fractional_top_left"]], dtype=float)
            if cfg.get("flip_x"):
                pts[:, 0] = -pts[:, 0]
            yi = at.opa(at.centre_scale(pts), M0)
            f = cfg["family"]
            tl = fam_tips.get(f, [])
            if not tl:
                print(f"    {taxon}: family {f} not sampled; skipped")
                continue
            age = float(cfg.get("age_ma", BOLCA_AGE_MA))
            crown = tree.mrca(tl) if len(tl) > 1 else tl[0]
            crown_age = tree.T - tree.h[crown]
            if crown_age < age:
                rep = tl[0]
            else:
                fm_ = Y_opa[[name_idx[tree.names[t]] for t in tl]].mean(0)
                rep = min(tl, key=lambda t: at.rms(Y_opa[name_idx[tree.names[t]]], fm_))
            row, hv = tree.virtual_node(rep, age)
            mean_b, s2_b, a_b, R_b = predict(model, tree, tips, Zt, np.array([hv]), row[None, :])
            mean_fb, _, _, _ = predict(model_full, tree, tips, F_t, np.array([hv]), row[None, :])
            pred_lm = to_lm(mean_b[0] @ B.T)
            z_own = project(B, (yi - M0).reshape(1, -1))[0]
            fam_mean = Y_opa[[name_idx[tree.names[t]] for t in tl]].mean(0)
            rec = {
                "taxon": taxon, "family": f, "age_ma": age, "sampled_crown_age_ma": float(crown_age), "lineage_tip": tree.names[rep],
                "placement": cfg.get("placement", ""), "source": cfg.get("source_citation", ""), "url": cfg.get("url", ""),
                "rms_thompson": shape_rms(pred_lm, yi), "rms_full30": shape_rms(to_lm(Dc + mean_fb[0] @ P_full.T), yi),
                "rms_living_family_mean": shape_rms(fam_mean, yi), "rms_consensus": shape_rms(M0, yi),
                "rms_thompson_oracle": shape_rms(to_lm(z_own @ B.T), yi),
                "mahal2": mahal2(z_own, mean_b[0], s2_b[0] * R_b), "z_pred": mean_b[0].tolist(), "z_obs": z_own.tolist(),
                "ratios_pred": ratios(at.centre_scale(pred_lm)), "ratios_obs": ratios(yi), "ratios_living_family_mean": ratios(fam_mean),
                "pred_landmarks": at.opa(at.centre_scale(pred_lm), yi).tolist(), "obs_landmarks": yi.tolist(),
                "digitisation_note": cfg.get("note", ""),
            }
            rec["in95"] = bool(rec["mahal2"] <= CHI2_95)
            bolca_recs.append(rec)
            print(f"    {taxon:28s} {f:14s} Thompson {rec['rms_thompson']:.4f} | full30 {rec['rms_full30']:.4f} | living-fam-mean {rec['rms_living_family_mean']:.4f} | consensus {rec['rms_consensus']:.4f} | oracle {rec['rms_thompson_oracle']:.4f}  BDmax/SL pred {rec['ratios_pred']['BDmax_over_SL']:.3f} obs {rec['ratios_obs']['BDmax_over_SL']:.3f} living {rec['ratios_living_family_mean']['BDmax_over_SL']:.3f}  M2={rec['mahal2']:.1f}")
    else:
        print("[7] no Monte Bolca landmark file; skipping landmark-level fossil test")

    # Published-measurement fossil test: compare predicted BDmax/SL (and other ratios) of the family
    # lineage at the fossil's age with ratios reported in the fossil's description.
    bolca_ratio_recs = []
    if BOLCA_RATIOS_JSON.exists():
        print("[7b] Monte Bolca published-ratio test")
        meta = json.loads(BOLCA_RATIOS_JSON.read_text())
        pred_by_fam = {r["target"]: r for r in registered if r["kind"] == "bolca50"}
        for taxon, cfg in meta["fossils"].items():
            f = cfg["family"]
            if f not in pred_by_fam:
                print(f"    {taxon}: family {f} not sampled; skipped")
                continue
            pr = pred_by_fam[f]
            rec = {"taxon": taxon, "family": f, "age_ma": cfg.get("age_ma", BOLCA_AGE_MA), "placement": cfg.get("placement", ""),
                   "source": cfg.get("source", ""), "note": cfg.get("note", ""), "measurements": {}}
            for k_, obs in cfg.get("ratios", {}).items():
                lo_obs, hi_obs = (obs if isinstance(obs, list) else [obs, obs])
                mid = 0.5 * (lo_obs + hi_obs)
                p_full = pr["full30"]["ratios"][k_]; ci_full = pr["full30"]["ratios_95"][k_]
                p_th = pr["thompson7"]["ratios"][k_]; ci_th = pr["thompson7"]["ratios_95"][k_]
                liv = pr["living_family_mean_ratios"][k_]
                rec["measurements"][k_] = {
                    "fossil": [lo_obs, hi_obs], "pred_full30": p_full, "ci95_full30": ci_full, "pred_thompson7": p_th, "ci95_thompson7": ci_th,
                    "living_family_mean": liv,
                    "fossil_inside_ci95_full30": bool(ci_full[0] <= hi_obs and lo_obs <= ci_full[1]),
                    "fossil_inside_ci95_thompson7": bool(ci_th[0] <= hi_obs and lo_obs <= ci_th[1]),
                    "prediction_closer_than_living_mean_full30": bool(abs(p_full - mid) < abs(liv - mid)),
                    "direction_of_change_correct_full30": bool(np.sign(p_full - liv) == np.sign(mid - liv)) if abs(p_full - liv) > 1e-6 else None,
                }
                m_ = rec["measurements"][k_]
                print(f"    {taxon:26s} {f:13s} {k_:13s} fossil {lo_obs:.2f}-{hi_obs:.2f} | pred full30 {p_full:.3f} [{ci_full[0]:.2f},{ci_full[1]:.2f}] | Thompson7 {p_th:.3f} [{ci_th[0]:.2f},{ci_th[1]:.2f}] | living {liv:.3f}  in95={m_['fossil_inside_ci95_full30']} closer={m_['prediction_closer_than_living_mean_full30']} dir={m_['direction_of_change_correct_full30']}")
            bolca_ratio_recs.append(rec)
    else:
        print("[7b] no Monte Bolca published-ratio file; skipping")

    # ------------------------------------------------------------------ [8] lineage through time
    lineage_demo = []
    for tip_name in ["Pomacanthus_semicirculatus", "Naso_brevirostris", "Platax_orbicularis", "Scarus_ghobban"]:
        if tip_name not in tree.name_to_node:
            continue
        tnode = tree.name_to_node[tip_name]
        ages = [0.0, 10.0, 25.0, 50.0, 75.0, 100.0]
        rows, hs = [], []
        for ag in ages:
            if ag == 0.0:
                rows.append(tree.C[tnode]); hs.append(tree.h[tnode])
            else:
                r_, hv = tree.virtual_node(tnode, ag); rows.append(r_); hs.append(hv)
        obs_mask = np.array([t != tnode for t in tips])
        mean_l, s2_l, _, R_l = predict(model, tree, tips[obs_mask], Zt[obs_mask], np.array(hs), np.array(rows))
        mean_lf, _, _, _ = predict(model_full, tree, tips[obs_mask], F_t[obs_mask], np.array(hs), np.array(rows))
        lineage_demo.append({"lineage": tip_name, "ages_ma": ages, "z_mean": mean_l.tolist(), "pred_var_factor": s2_l.tolist(),
                             "pred_landmarks": [at.centre_scale(lm_th(m)).tolist() for m in mean_l],
                             "pred_landmarks_full30": [at.centre_scale(lm_full(m)).tolist() for m in mean_lf],
                             "BDmax_over_SL": [ratios(at.centre_scale(lm_th(m)))["BDmax_over_SL"] for m in mean_l],
                             "BDmax_over_SL_full30": [ratios(at.centre_scale(lm_full(m)))["BDmax_over_SL"] for m in mean_lf],
                             "observed_tip_BDmax_over_SL": ratios(Y_opa[name_idx[tip_name]])["BDmax_over_SL"]})

    # ------------------------------------------------------------------ [9] figure
    print("[9] figure")
    fig = plt.figure(figsize=(19, 13))
    gs = fig.add_gridspec(3, 4, hspace=0.5, wspace=0.32, height_ratios=[1.2, 1.0, 1.0])

    axA = fig.add_subplot(gs[0, 0])
    kinds = ["WN", "BM", "lambda", "OU", "EB"]
    xb = np.arange(len(kinds))
    axA.bar(xb - 0.27, [fits_body[k]["dAICc"] for k in kinds], 0.27, color="#1f77b4", label="Thompson modes (7-D body field)")
    axA.bar(xb, [fits_jaw[k]["dAICc"] for k in kinds], 0.27, color="#d62728", label="Jaw residual (4 PCs)")
    axA.bar(xb + 0.27, [fits_full[k]["dAICc"] for k in kinds], 0.27, color="#7f7f7f", label="Full 30-D shape")
    axA.set_xticks(xb); axA.set_xticklabels(["White\nnoise", "BM", "Pagel\nlambda", "OU", "Early\nBurst"], fontsize=8)
    axA.set_ylabel("dAICc (0 = best)", fontsize=8.5); axA.set_yscale("symlog", linthresh=10)
    for off, fits_ in [(-0.27, fits_body), (0.0, fits_jaw), (0.27, fits_full)]:
        for i_, k in enumerate(kinds):
            if fits_[k]["dAICc"] == 0.0:
                axA.text(i_ + off, 1.5, "best", ha="center", va="bottom", fontsize=6.5, rotation=90)
    axA.set_title(f"A. Process-model selection (n={n} species)\nbody: {fits_body[best_body]['model'].label()}\njaw: {fits_jaw[best_jaw]['model'].label()}", fontsize=8.3)
    axA.legend(fontsize=6.8, loc="upper right"); axA.grid(alpha=0.25, axis="y")

    axB = fig.add_subplot(gs[0, 1:3])
    xs = [0.5 * (r["tau_bin_myr"][0] + min(r["tau_bin_myr"][1], 160)) for r in horizon_table]
    axB.plot(xs, [r["median_rms_thompson"] for r in horizon_table], "o-", color="#1f77b4", lw=2.2, label="Thompson-mode prediction (7-D, this engine)")
    axB.plot(xs, [r["median_rms_full30"] for r in horizon_table], "D-", color="#2ca02c", lw=1.6, label="Full 30-D shape-space prediction")
    axB.plot(xs, [r["median_rms_pca7"] for r in horizon_table], "x--", color="#9467bd", lw=1.2, label="PCA-7 basis (equal dimension)")
    axB.plot(xs, [r["median_rms_nearest_tip"] for r in horizon_table], "s--", color="#ff7f0e", label="Copy nearest sampled relative")
    axB.plot(xs, [r["median_rms_consensus"] for r in horizon_table], "^:", color="#7f7f7f", label="Grand consensus (no phylogeny)")
    axB.plot(xs, [r["median_rms_thompson_oracle"] for r in horizon_table], "-", color="#1f77b4", lw=0.9, alpha=0.5, label="Thompson-7 oracle (representation floor)")
    axB.axhline(sigma_genus, color="k", ls="--", lw=1.0, label=f"sigma_genus = {sigma_genus:.4f}")
    axB.axvline(50, color="k", lw=0.8, alpha=0.4); axB.text(51, 0.03, "Monte Bolca\n(~50 Ma)", fontsize=7.5, va="bottom", transform=axB.get_xaxis_transform())
    ymax_b = max(max(r["median_rms_nearest_tip"], r["median_rms_thompson"], r["median_rms_consensus"]) for r in horizon_table)
    axB.set_ylim(0.0, ymax_b * 1.45)
    for r, x in zip(horizon_table, xs):
        axB.annotate(f"n={r['n']}\n{r['coverage95_pct']:.0f}% in 95%", (x, r["median_rms_thompson"]), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=6.3, color="#1f77b4")
    axB.set_xlabel("Prediction horizon tau (Myr to nearest sampled relative)", fontsize=8.5)
    axB.set_ylabel("Median 17-landmark Procrustes RMS vs. held-out species", fontsize=8.5)
    axB.set_title(f"B. Held-out prediction error vs. time horizon ({len(val)} leave-one-species/genus/family-out predictions)", fontsize=8.5)
    axB.legend(fontsize=6.6, loc="upper left", ncol=3); axB.grid(alpha=0.25)

    axC = fig.add_subplot(gs[0, 3])
    qs = np.linspace(0.02, 0.98, 49)
    emp = [float(np.mean(m2_all <= stats.chi2.ppf(q, m_dim))) for q in qs]
    axC.plot(qs, emp, "-", color="#1f77b4", lw=2, label="Empirical coverage")
    axC.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    axC.set_xlabel("Nominal credible level", fontsize=8.5); axC.set_ylabel("Fraction of held-out species inside", fontsize=8.5)
    axC.set_title(f"C. Calibration of 7-D predictive ellipsoids\n95% nominal -> {100 * np.mean(m2_all <= CHI2_95):.1f}%; 50% -> {100 * np.mean(m2_all <= CHI2_50):.1f}%", fontsize=8.5)
    axC.legend(fontsize=7); axC.grid(alpha=0.25)

    for j, rec in enumerate(exact_recs[:7]):
        ax = fig.add_subplot(gs[1 + j // 4, j % 4])
        obs = np.array(rec["obs_landmarks"]); prd = np.array(rec["pred_landmarks"])
        at.draw_fish(ax, obs, "#333333", lw=1.6)
        at.draw_fish(ax, prd, "#1f77b4", lw=1.6)
        at.finish(ax, f"D{j + 1}. {rec['species'].replace('_', ' ')} (blind)\ntau={rec['tau_myr']:.0f} Myr from {rec['nearest_training_tip'].replace('_', ' ')}\nRMS: Thompson {rec['rms_thompson']:.3f} | consensus {rec['rms_consensus']:.3f} | nearest {rec['rms_nearest_tip']:.3f} | oracle {rec['rms_thompson_oracle']:.3f}", lim=0.5)
        ax.title.set_fontsize(7.4)
    axF = fig.add_subplot(gs[2, 3])
    if lineage_demo:
        ld = lineage_demo[0]
        cols = plt.cm.viridis(np.linspace(0, 0.9, len(ld["ages_ma"])))
        for k_, (ag, L) in enumerate(zip(ld["ages_ma"], ld["pred_landmarks"])):
            at.draw_fish(axF, np.array(L), cols[k_], lw=1.2, dots=False)
        at.finish(axF, f"F. Predicted body plan along the {ld['lineage'].split('_')[0]} lineage\n(0, 10, 25, 50, 75, 100 Ma; dark=present, light=100 Ma)\nBDmax/SL: " + ", ".join(f"{b:.2f}" for b in ld["BDmax_over_SL"]), lim=0.5)
        axF.title.set_fontsize(7.4)
    fig.suptitle("A phylogenetic prediction engine for Thompson's 1917 coordinate law: process models, horizon-calibrated error, blind tests on Thompson's own species, time-sliced ancestral body plans",
                 fontsize=11, fontweight="bold", y=0.995)
    fig.text(0.5, 0.004, "Black = observed 17 landmarks; blue = blind prediction from the 7 linearised Thompson transformation modes conditioned on the sampled relatives (Rabosky et al. 2018 tree). Joel Dietz | Project Delphi.", ha="center", fontsize=7.5)
    fig.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
    print(f"    wrote {OUT_FIG}")

    # ------------------------------------------------------------------ [10] JSON + markdown
    def fit_summary(fits):
        return {k: {"label": d["model"].label(), "theta": d["model"].theta, "logL": d["logL"], "k": d["k"], "AICc": d["AICc"], "dAICc": d["dAICc"]} for k, d in fits.items()}

    out = {
        "trait_space": {"basis": "7 linearised Thompson transformation modes at the consensus, orthogonal to rigid modes", "modes": MODE_NAMES,
                        "n_training_species": n, "tree": "Rabosky et al. 2018 actinopt_12k_treePL (pruned)", "sampled_crown_age_ma": tree.T,
                        "sigma_genus_procrustes_rms": sigma_genus, "representation_floor_rms_thompson7": repr_floor, "representation_floor_rms_pca7": repr_floor_pca7,
                        "shape_variance_captured_thompson7": var_thompson, "shape_variance_captured_pca7": var_pca7,
                        "rate_matrix_condition_number_thompson7": cond_R, "rate_matrix_condition_number_full30": cond_full,
                        "jaw_module_var_explained_4pc": jaw_var_expl, "per_mode_phylogenetic_signal": mode_lambda,
                        "basis_matrix_34x7": B.tolist(), "consensus_M0": M0.tolist()},
        "model_selection": {"thompson_modes": fit_summary(fits_body), "jaw_residual": fit_summary(fits_jaw), "full_shape_30d": fit_summary(fits_full), "pca7": fit_summary(fits_pca7),
                            "best": {"thompson_modes": best_body, "jaw_residual": best_jaw, "full_shape_30d": best_full, "pca7": best_pca7},
                            "thompson_rate_matrix": fits_body[best_body]["R"].tolist(), "thompson_root_state": fits_body[best_body]["root"].tolist()},
        "holdout_validation": {"n_predictions": len(val), "horizon_table": horizon_table,
                               "coverage95_overall_pct": 100.0 * float(np.mean(m2_all <= CHI2_95)), "coverage50_overall_pct": 100.0 * float(np.mean(m2_all <= CHI2_50)),
                               "share_of_full_space_prediction_gain_captured_by_thompson7": skill_share, "records": val},
        "blind_1917_species": exact_recs,
        "registered_predictions": registered,
        "monte_bolca_fossil_test": bolca_recs,
        "monte_bolca_published_ratio_test": bolca_ratio_recs,
        "lineage_through_time": lineage_demo,
    }
    OUT_JSON.write_text(json.dumps(out, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.bool_)) else o))
    print(f"    wrote {OUT_JSON}")
    write_markdown(out)
    print(f"    wrote {OUT_MD}")
    return 0


def eoplatax_sentence(out: dict) -> str:
    rec = next((r for r in out["monte_bolca_published_ratio_test"] if r["taxon"] == "Eoplatax_papilio"), None)
    if rec is None or "BDmax_over_SL" not in rec["measurements"]:
        return "\n"
    m = rec["measurements"]["BDmax_over_SL"]
    lo, hi = m["fossil"]
    verdict = ("consistent with the stem-state prediction" if m["fossil_inside_ci95_full30"]
               else "outside the stem-state prediction's 95% interval: the stem-state prediction is falsified as stated")
    return (f"The sharpest single outcome is *Eoplatax papilio*: the engine predicts the ephippid lineage at 50 Ma to have BDmax/SL {m['pred_full30']:.2f} "
            f"[{m['ci95_full30'][0]:.2f}–{m['ci95_full30'][1]:.2f}], far shallower than living *Platax* ({m['living_family_mean']:.2f}); the fossil measures about {lo:.2f}–{hi:.2f}, {verdict}. "
            "Either the extreme disc body of *Platax* was already in place 50 Myr ago (deep-time stasis that the fitted Pagel-lambda process, which regresses stem states toward the clade ancestor, does not anticipate), "
            "or *Eoplatax* sits on a side branch closer to crown *Platax* than a stem placement assumes. Both readings are informative.\n")


def write_markdown(out: dict) -> None:
    ts = out["trait_space"]; ms = out["model_selection"]; hv = out["holdout_validation"]
    L = []
    L.append("# Registered Phylogenetic Predictions from Thompson's 1917 Coordinate Law\n")
    L.append(f"Generated by [`thompson/run_phylogenetic_prediction_engine.py`](../thompson/run_phylogenetic_prediction_engine.py); all numbers in "
             f"[`thompson_phylogenetic_prediction_engine.json`](../thompson_phylogenetic_prediction_engine.json); figure "
             f"[`outreach_fig9_phylogenetic_prediction_engine.png`](./outreach_fig9_phylogenetic_prediction_engine.png). "
             f"Every registered prediction and fossil-test outcome below is also recorded as a signed nanopublication in "
             f"[`nanopubs/thompson_predictions/`](../nanopubs/thompson_predictions/) (generated by `nanopubs/thompson_predictions_to_nanopubs.py`).\n")
    L.append("## 1. What is predicted, and how\n")
    L.append(f"Each species' whole-body planform is encoded by **7 linearised Thompson transformation modes** evaluated at the Procrustes consensus of "
             f"{ts['n_training_species']} Indo-Pacific reef-fish species (Claverie & Wainwright 2014) that sit on the Rabosky et al. (2018) time tree: "
             + "; ".join(ts["modes"]) + ". "
             f"These 7 modes capture {100 * ts['shape_variance_captured_thompson7']:.1f}% of all 17-landmark shape variance (an equal-dimension PCA captures {100 * ts['shape_variance_captured_pca7']:.1f}%). "
             f"A multivariate evolutionary process is fitted to the 7 mode coefficients on the tree; the state of any lineage at any age is the Gaussian conditional expectation given the sampled living species, with a 7-D predictive ellipsoid. "
             f"Reference scales: intrageneric disparity sigma_genus = {ts['sigma_genus_procrustes_rms']:.4f} Procrustes RMS; Thompson-7 representation floor = {ts['representation_floor_rms_thompson7']:.4f}; PCA-7 floor = {ts['representation_floor_rms_pca7']:.4f}.\n")
    L.append("## 2. Which evolutionary process the body plan follows (tested, not assumed)\n")
    L.append("| Model | Thompson modes (7-D) | dAICc | Jaw residual (4 PCs) | dAICc | Full shape (30-D) | dAICc |\n|---|---|---|---|---|---|---|")
    for k in ["WN", "BM", "lambda", "OU", "EB"]:
        L.append(f"| {k} | {ms['thompson_modes'][k]['label']} | {ms['thompson_modes'][k]['dAICc']:.1f} | {ms['jaw_residual'][k]['label']} | {ms['jaw_residual'][k]['dAICc']:.1f} | {ms['full_shape_30d'][k]['label']} | {ms['full_shape_30d'][k]['dAICc']:.1f} |")
    L.append(f"\nRate-matrix condition number: {ts['rate_matrix_condition_number_thompson7']:.1f} (Thompson-7) vs {ts['rate_matrix_condition_number_full30']:.0f} (full 30-D).\n")
    L.append("Per-mode phylogenetic signal (univariate Pagel lambda):\n")
    L.append("| Mode | lambda | best model | SD across species |\n|---|---|---|---|")
    for m in ts["per_mode_phylogenetic_signal"]:
        L.append(f"| {m['mode']} | {m['lambda']:.3f} | {m['best']} | {m['sd_across_species']:.4f} |")
    L.append("\n## 3. How far the engine can see (held-out calibration)\n")
    L.append("Median Procrustes RMS between the prediction and the held-out species' real 17 landmarks, by time horizon tau (age of the MRCA with the nearest species left in the training set).\n")
    L.append("| tau (Myr) | n | Thompson-7 prediction | Full 30-D prediction | PCA-7 prediction | Copy nearest relative | Consensus | Thompson-7 oracle floor | % Thompson within sigma_genus | % Full-30 within sigma_genus | 95% ellipsoid coverage |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for r in hv["horizon_table"]:
        L.append(f"| {r['tau_bin_myr'][0]}–{r['tau_bin_myr'][1]} | {r['n']} | **{r['median_rms_thompson']:.4f}** | {r['median_rms_full30']:.4f} | {r['median_rms_pca7']:.4f} | {r['median_rms_nearest_tip']:.4f} | {r['median_rms_consensus']:.4f} | {r['median_rms_thompson_oracle']:.4f} | {r['pct_within_sigma_genus_thompson']:.1f}% | {r['pct_within_sigma_genus_full30']:.1f}% | {r['coverage95_pct']:.1f}% |")
    L.append(f"\nOverall: 95% ellipsoids contain {hv['coverage95_overall_pct']:.1f}% of held-out species; 50% ellipsoids contain {hv['coverage50_overall_pct']:.1f}%. "
             f"The 7 Thompson modes capture {100 * hv['share_of_full_space_prediction_gain_captured_by_thompson7']:.1f}% of the squared-error reduction (relative to the consensus) that full 30-D shape-space phylogenetic prediction achieves.\n")
    first, last_in = hv["horizon_table"][0], [r for r in hv["horizon_table"] if r["tau_bin_myr"][1] <= 50][-1]
    L.append(f"Reading the table honestly: within 50 Myr the full 30-D model is the most accurate predictor at every horizon (it beats copying the nearest relative, {first['median_rms_full30']:.4f} vs {first['median_rms_nearest_tip']:.4f} at tau<5 Myr and {last_in['median_rms_full30']:.4f} vs {last_in['median_rms_nearest_tip']:.4f} at 25–50 Myr), "
             f"the data-driven PCA-7 basis beats the a-priori Thompson-7 basis, and the Thompson-7 error is bounded below by its representation floor (~0.017) rather than by the phylogenetic signal. Beyond ~50 Myr every phylogenetic predictor collapses to the consensus level: "
             f"this is the same ~50 Myr horizon found independently in the Thompson-pair work, recovered here from the held-out error curve alone. The Thompson modes' value is interpretability (7 named deformations, condition number {ts['rate_matrix_condition_number_thompson7']:.0f} vs {ts['rate_matrix_condition_number_full30']:.0f}), not accuracy.\n")
    L.append("## 4. Blind test on D'Arcy Thompson's own 1917 species (on the tree, never in the training data)\n")
    L.append("| Species | Horizon (Myr) | Nearest sampled relative | Thompson-7 prediction | Full 30-D | Consensus | Copy nearest | Thompson-7 oracle | Mahalanobis² (chi²₇ 95% = 14.1) |\n|---|---|---|---|---|---|---|---|---|")
    for r in out["blind_1917_species"]:
        L.append(f"| *{r['species'].replace('_', ' ')}* ({r['family']}) | {r['tau_myr']:.0f} | *{r['nearest_training_tip'].replace('_', ' ')}* | **{r['rms_thompson']:.4f}** | {r['rms_full30']:.4f} | {r['rms_consensus']:.4f} | {r['rms_nearest_tip']:.4f} | {r['rms_thompson_oracle']:.4f} | {r['mahal2']:.1f} {'✓' if r['in95'] else '✗'} |")
    L.append("\n## 5. Registered predictions (falsifiable against future fossils and unsampled living species)\n")
    L.append("Scalar summaries of the predicted 17-landmark configuration. **BDmax/SL** = maximum body depth excluding fin webs over standard length (the ratio reported in species descriptions); "
             "BD/SL = dorsal-fin origin to anal-fin origin over SL; HL/SL = head length; PD/SL = predorsal length. Intervals are 95% Monte Carlo intervals of the predictive distribution. "
             "'Full' = full 30-D shape-space model (most accurate); 'T7' = 7 Thompson modes. The expected landmark RMS is the empirical median held-out error at this horizon (Section 3).\n")
    L.append("### 5a. Crown-node ancestral body plans of sampled families\n")
    L.append("| Family (n sampled) | Crown age (Ma) | BDmax/SL Full [95%] | BDmax/SL T7 [95%] | HL/SL Full | PD/SL Full | Expected RMS Full / T7 |\n|---|---|---|---|---|---|---|")
    for r in out["registered_predictions"]:
        if r["kind"] == "crown":
            fu, th = r["full30"], r["thompson7"]
            ef, et = r["expected_landmark_rms_at_this_horizon_full30"], r["expected_landmark_rms_at_this_horizon_thompson7"]
            L.append(f"| {r['target']} ({r['n_sampled_tips']}) | {r['age_ma']:.1f} | **{fu['ratios']['BDmax_over_SL']:.3f}** [{fu['ratios_95']['BDmax_over_SL'][0]:.3f}–{fu['ratios_95']['BDmax_over_SL'][1]:.3f}] | {th['ratios']['BDmax_over_SL']:.3f} [{th['ratios_95']['BDmax_over_SL'][0]:.3f}–{th['ratios_95']['BDmax_over_SL'][1]:.3f}] | {fu['ratios']['HL_over_SL']:.3f} | {fu['ratios']['PD_over_SL']:.3f} | {f'{ef:.4f}' if ef else '–'} / {f'{et:.4f}' if et else '–'} |")
    L.append("\n### 5b. Common ancestors of Thompson's 1917 pairs\n")
    L.append("| Pair | MRCA age (Ma) | BDmax/SL Full [95%] | BDmax/SL T7 [95%] | HL/SL Full | PD/SL Full |\n|---|---|---|---|---|---|")
    for r in out["registered_predictions"]:
        if r["kind"] == "mrca":
            fu, th = r["full30"], r["thompson7"]
            L.append(f"| {r['target'].replace('_', ' ')} | {r['age_ma']:.1f} | **{fu['ratios']['BDmax_over_SL']:.3f}** [{fu['ratios_95']['BDmax_over_SL'][0]:.3f}–{fu['ratios_95']['BDmax_over_SL'][1]:.3f}] | {th['ratios']['BDmax_over_SL']:.3f} [{th['ratios_95']['BDmax_over_SL'][0]:.3f}–{th['ratios_95']['BDmax_over_SL'][1]:.3f}] | {fu['ratios']['HL_over_SL']:.3f} | {fu['ratios']['PD_over_SL']:.3f} |")
    L.append(f"\n### 5c. State of each Bolca-recorded family's lineage at {BOLCA_AGE_MA:.0f} Ma (Monte Bolca, Ypresian)\n")
    L.append("A Monte Bolca fossil of the family should match these if it sits on or near the lineage leading to the sampled living species. 'Living' is the present-day family mean (± SD across sampled species), so the predicted direction of change over 50 Myr is explicit.\n")
    L.append("| Family | Bolca taxa | Sampled crown age (Ma) | Lineage | BDmax/SL at 50 Ma, Full [95%] | T7 [95%] | Living BDmax/SL (± SD) | HL/SL Full (living) | PD/SL Full (living) |\n|---|---|---|---|---|---|---|---|---|")
    for r in out["registered_predictions"]:
        if r["kind"] == "bolca50":
            fu, th, lv, sd = r["full30"], r["thompson7"], r["living_family_mean_ratios"], r["living_family_ratio_sd"]
            L.append(f"| {r['target']} | {r['bolca_taxa']} | {r['family_crown_age_ma']:.1f} | *{r['lineage_tip'].replace('_', ' ')}* | **{fu['ratios']['BDmax_over_SL']:.3f}** [{fu['ratios_95']['BDmax_over_SL'][0]:.3f}–{fu['ratios_95']['BDmax_over_SL'][1]:.3f}] | {th['ratios']['BDmax_over_SL']:.3f} [{th['ratios_95']['BDmax_over_SL'][0]:.3f}–{th['ratios_95']['BDmax_over_SL'][1]:.3f}] | {lv['BDmax_over_SL']:.3f} (± {sd['BDmax_over_SL']:.3f}) | {fu['ratios']['HL_over_SL']:.3f} ({lv['HL_over_SL']:.3f}) | {fu['ratios']['PD_over_SL']:.3f} ({lv['PD_over_SL']:.3f}) |")
    L.append("\n### 5d. Lineages through time (BDmax/SL at 0 / 10 / 25 / 50 / 75 / 100 Ma; tip itself held out)\n")
    L.append("| Lineage | Observed today | Full 30-D | Thompson-7 |\n|---|---|---|---|")
    for ld in out["lineage_through_time"]:
        L.append(f"| *{ld['lineage'].replace('_', ' ')}* | {ld['observed_tip_BDmax_over_SL']:.3f} | " + " / ".join(f"{b:.3f}" for b in ld["BDmax_over_SL_full30"]) + " | " + " / ".join(f"{b:.3f}" for b in ld["BDmax_over_SL"]) + " |")
    if out["monte_bolca_published_ratio_test"]:
        L.append("\n## 6. Monte Bolca fossil test against published measurements\n")
        L.append("For each fossil, the ratio reported in its description is compared with the predicted state of its family's lineage at the fossil's age (Section 5c). "
                 "'Closer' = the time-sliced prediction is closer to the fossil than the living family mean is; 'direction' = the predicted sign of change from the living mean matches the fossil.\n")
        def short_src(s: str) -> str:
            mm = re.match(r"^(.*?\(\d{4}\))", s)
            return mm.group(1) if mm else (s[:40] + "…")
        L.append("| Fossil | Family | Ratio | Fossil value | Prediction Full [95%] | Prediction T7 [95%] | Living mean | Fossil in 95% (Full / T7) | Closer than living mean | Direction correct | Source |\n|---|---|---|---|---|---|---|---|---|---|---|")
        for r in out["monte_bolca_published_ratio_test"]:
            for k_, m in r["measurements"].items():
                fv = f"{m['fossil'][0]:.2f}" if m['fossil'][0] == m['fossil'][1] else f"{m['fossil'][0]:.2f}–{m['fossil'][1]:.2f}"
                dirn = "–" if m["direction_of_change_correct_full30"] is None else ("✓" if m["direction_of_change_correct_full30"] else "✗")
                L.append(f"| *{r['taxon'].replace('_', ' ')}* | {r['family']} | {k_} | {fv} | **{m['pred_full30']:.3f}** [{m['ci95_full30'][0]:.2f}–{m['ci95_full30'][1]:.2f}] | {m['pred_thompson7']:.3f} [{m['ci95_thompson7'][0]:.2f}–{m['ci95_thompson7'][1]:.2f}] | {m['living_family_mean']:.3f} | {'✓' if m['fossil_inside_ci95_full30'] else '✗'} / {'✓' if m['fossil_inside_ci95_thompson7'] else '✗'} | {'✓' if m['prediction_closer_than_living_mean_full30'] else '✗'} | {dirn} | {short_src(r['source'])} |")
        L.append("\nMeasurement sources and placements:\n")
        for r in out["monte_bolca_published_ratio_test"]:
            L.append(f"- *{r['taxon'].replace('_', ' ')}* ({r['family']}; {r['placement']}): {r['source']}" + (f" Note: {r['note']}." if r.get("note") else ""))
        n_meas = sum(len(r["measurements"]) for r in out["monte_bolca_published_ratio_test"])
        n_in = sum(m["fossil_inside_ci95_full30"] for r in out["monte_bolca_published_ratio_test"] for m in r["measurements"].values())
        n_closer = sum(m["prediction_closer_than_living_mean_full30"] for r in out["monte_bolca_published_ratio_test"] for m in r["measurements"].values())
        n_dir = [m["direction_of_change_correct_full30"] for r in out["monte_bolca_published_ratio_test"] for m in r["measurements"].values() if m["direction_of_change_correct_full30"] is not None]
        n_bd = [m for r in out["monte_bolca_published_ratio_test"] for k_, m in r["measurements"].items() if k_ == "BDmax_over_SL"]
        n_bd_in = sum(m["fossil_inside_ci95_full30"] for m in n_bd); n_bd_closer = sum(m["prediction_closer_than_living_mean_full30"] for m in n_bd)
        n_bd_dir = [m["direction_of_change_correct_full30"] for m in n_bd if m["direction_of_change_correct_full30"] is not None]
        L.append(f"\nSummary (all ratios): fossil inside the 95% interval in {n_in}/{n_meas} measurements; time-sliced prediction closer than the living family mean in {n_closer}/{n_meas}; direction of change correct in {sum(n_dir)}/{len(n_dir)}. "
                 f"BDmax/SL only (the ratio least affected by measurement convention): inside 95% in {n_bd_in}/{len(n_bd)}, closer than living mean in {n_bd_closer}/{len(n_bd)}, direction correct in {sum(n_bd_dir)}/{len(n_bd_dir)}.\n")
        L.append("How to read this. The competing null is phylogenetic stasis (fossil = living family mean). Under that null 'closer' and 'direction' should each succeed in about half the measurements, so the tallies above are the test statistic; with this many measurements neither outcome is far from the null, and the test is reported as such. "
                 "Three limits are structural, not tunable: (i) measurement convention — the landmark head length ends at the opercular landmark rather than the opercular-membrane margin and the landmark body depth is the vertical extent of the 17 points, so landmark-based HL/SL runs a few percent lower than a calliper HL/SL and fossil HL values above every living mean partly reflect that; (ii) placement — several Bolca taxa (*Bellwoodilabrus*, *Ruffoichthys*, *Eoleiognathus*) are stem or incertae-sedis forms that need not sit on the lineage to the sampled living species; (iii) the sampled crown of most of these families is younger than 50 Ma, so the 50 Ma prediction is a stem-lineage state that regresses toward the clade ancestor rather than a within-family interpolation. "
                 + eoplatax_sentence(out))
    if out["monte_bolca_fossil_test"]:
        L.append("\n### 6b. Landmark-level fossil test\n")
        L.append("| Fossil | Family | Age (Ma) | Thompson-7 RMS | Full 30-D | Living family mean | Consensus | T7 oracle | BDmax/SL pred / obs / living | Mahalanobis² | Source |\n|---|---|---|---|---|---|---|---|---|---|---|")
        for r in out["monte_bolca_fossil_test"]:
            L.append(f"| *{r['taxon'].replace('_', ' ')}* | {r['family']} | {r['age_ma']:.0f} | **{r['rms_thompson']:.4f}** | {r['rms_full30']:.4f} | {r['rms_living_family_mean']:.4f} | {r['rms_consensus']:.4f} | {r['rms_thompson_oracle']:.4f} | {r['ratios_pred']['BDmax_over_SL']:.3f} / {r['ratios_obs']['BDmax_over_SL']:.3f} / {r['ratios_living_family_mean']['BDmax_over_SL']:.3f} | {r['mahal2']:.1f} {'✓' if r['in95'] else '✗'} | {r['source']} |")
    L.append("\n## 7. Scope and caveats\n")
    L.append("- Predictions concern the global body planform (7 Thompson modes). The craniofacial/jaw residual is modelled separately and is not part of these predictions.\n"
             "- A lineage's state at an age is the conditional expectation under the fitted process given the sampled living species: a best estimate with a stated ellipsoid, not a claim about any particular fossil's phylogenetic position.\n"
             "- Fossil comparisons assume the fossil lies on or near the stem/crown lineage of the named family; fossil landmark digitisation error adds to the reported RMS.\n"
             "- Section 3 is the error model. Read any prediction's expected landmark error from its horizon there.\n")
    OUT_MD.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
