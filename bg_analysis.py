# -*- coding: utf-8 -*-
"""
bg_analysis.py - image-cryptography metrics (pure NumPy, no plotting, no cryptography).

Everything here is a read-only analysis of arrays the pipeline already produced. Nothing in this
module touches keys, random-number generators or the encryption/decryption code.

Conventions
    * Images are HxWx3 uint8 arrays (RGB, intensity range 0..255, L = 255).
    * Correlations are EXACT Pearson coefficients over ALL neighbouring pairs of each channel
      (computed in bounded-memory chunks). Sampling is used only for scatter plots.
    * A value that cannot be computed is returned as None together with a reason - never as 0.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

CHANNELS = ("R", "G", "B")
DIRECTIONS = ("horizontal", "vertical", "diagonal")
STAGE_ORDER = ("input", "scrambled", "encrypted", "decrypted", "descrambled")
CORR_STAGES = ("input", "scrambled", "encrypted", "decrypted")
RECON_PAIRS = (
    ("input_vs_scrambled", "input", "scrambled"),
    ("input_vs_encrypted", "input", "encrypted"),
    ("input_vs_decrypted", "input", "decrypted"),
    ("scrambled_vs_decrypted", "scrambled", "decrypted"),
    ("input_vs_descrambled", "input", "descrambled"),
)
FINAL_PAIR = "input_vs_descrambled"
SSIM_WINDOW = 7
NPCR_IDEAL = 100.0 * (1.0 - 2.0 ** -8)                              # 99.609375 %
UACI_IDEAL = 100.0 * ((256 ** 2 - 1) / (3.0 * 256)) / 255.0         # 33.4635 %


# ----------------------------------------------------------------------------
# hashing / array summaries
# ----------------------------------------------------------------------------
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            blk = f.read(chunk)
            if not blk:
                break
            h.update(blk)
    return h.hexdigest()


def array_summary(a: np.ndarray, full_limit: int = 64) -> dict:
    """Shape/dtype/memory/min/max/SHA-256 of an array. The hash is a checksum, NOT the array."""
    d = dict(count=int(a.size), shape=list(a.shape), dtype=str(a.dtype), nbytes=int(a.nbytes),
             min=int(a.min()) if a.size else None, max=int(a.max()) if a.size else None,
             sha256=sha256_bytes(np.ascontiguousarray(a).tobytes()))
    if a.size <= full_limit:
        d["full"] = [int(v) for v in a.ravel()]
    if a.ndim == 1 and a.size and np.issubdtype(a.dtype, np.integer):
        d["is_permutation_of_0_to_N_minus_1"] = bool(np.all(np.bincount(a, minlength=a.size) == 1)
                                                      and a.min() == 0 and a.max() == a.size - 1)
    return d


# ----------------------------------------------------------------------------
# histograms and entropy
# ----------------------------------------------------------------------------
def histograms(arr: np.ndarray) -> np.ndarray:
    """(3, 256) int64 counts, intensity values 0..255, one row per channel (R, G, B)."""
    return np.stack([np.bincount(arr[:, :, c].ravel(), minlength=256).astype(np.int64) for c in range(3)])


def entropy_counts(counts: np.ndarray):
    total = int(counts.sum())
    if total == 0:
        return None
    p = counts[counts > 0].astype(np.float64) / total
    return float(-(p * np.log2(p)).sum())


def entropy_report(hist: np.ndarray) -> dict:
    """Shannon entropy in bits per 8-bit value (maximum 8). 'pooled' uses all channels together."""
    out = {c: entropy_counts(hist[i]) for i, c in enumerate(CHANNELS)}
    out["pooled"] = entropy_counts(hist.sum(axis=0))
    return out


# ----------------------------------------------------------------------------
# adjacent-pixel correlation (exact)
# ----------------------------------------------------------------------------
def neighbor_views(ch: np.ndarray, direction: str):
    if direction == "horizontal":
        return ch[:, :-1], ch[:, 1:]
    if direction == "vertical":
        return ch[:-1, :], ch[1:, :]
    if direction == "diagonal":                      # (x, y) with (x+1, y+1)
        return ch[:-1, :-1], ch[1:, 1:]
    raise ValueError("unknown direction %r" % direction)


def pearson_2d(a: np.ndarray, b: np.ndarray, chunk_elems: int = 1_000_000):
    """Exact Pearson r over all paired elements. Returns (r | None, pair_count, reason | None)."""
    n = int(a.size)
    if n < 2:
        return None, n, "fewer than 2 pixel pairs"
    h, w = a.shape
    rows = max(1, chunk_elems // max(1, w))
    sa = sb = 0.0
    for r0 in range(0, h, rows):
        sa += float(a[r0:r0 + rows].sum(dtype=np.float64))
        sb += float(b[r0:r0 + rows].sum(dtype=np.float64))
    ma, mb = sa / n, sb / n
    saa = sbb = sab = 0.0
    for r0 in range(0, h, rows):
        x = a[r0:r0 + rows].astype(np.float64) - ma
        y = b[r0:r0 + rows].astype(np.float64) - mb
        saa += float((x * x).sum())
        sbb += float((y * y).sum())
        sab += float((x * y).sum())
    if saa == 0.0 or sbb == 0.0:
        return None, n, "zero variance (constant channel): correlation undefined"
    return max(-1.0, min(1.0, sab / math.sqrt(saa * sbb))), n, None


def correlations(arr: np.ndarray) -> dict:
    """Per direction: per-channel r, mean over the DEFINED channels, pair counts, reasons."""
    out = {}
    for d in DIRECTIONS:
        per, why, pairs = {}, {}, 0
        for ci, cname in enumerate(CHANNELS):
            a, b = neighbor_views(arr[:, :, ci], d)
            r, n, reason = pearson_2d(a, b)
            per[cname], pairs = r, n
            if reason:
                why[cname] = reason
        defined = [v for v in per.values() if v is not None]
        out[d] = dict(per_channel=per, mean=(sum(defined) / len(defined) if defined else None),
                      channels_defined=len(defined), pairs_per_channel=pairs, undefined_reasons=why)
    return out


def sample_pairs(arr: np.ndarray, direction: str, channel: int, k: int, seed: int):
    """Reproducible sample of neighbouring pairs for PLOTTING ONLY. Returns (x, y, total_pairs)."""
    a, b = neighbor_views(arr[:, :, channel], direction)
    total = int(a.size)
    if total == 0:
        return np.empty(0, np.uint8), np.empty(0, np.uint8), 0
    if total <= k:
        return a.ravel().copy(), b.ravel().copy(), total
    idx = np.random.default_rng(seed).choice(total, size=k, replace=False)
    r, c = np.unravel_index(idx, a.shape)
    return a[r, c], b[r, c], total


# ----------------------------------------------------------------------------
# MSE / PSNR / SSIM
# ----------------------------------------------------------------------------
def _sum_sq_diff(a: np.ndarray, b: np.ndarray) -> int:
    fa, fb = np.ascontiguousarray(a).reshape(-1), np.ascontiguousarray(b).reshape(-1)
    ss, step = 0, 1 << 22
    for i in range(0, fa.size, step):
        d = fa[i:i + step].astype(np.int64) - fb[i:i + step].astype(np.int64)
        ss += int(np.dot(d, d))
    return ss


def mse_psnr(a: np.ndarray, b: np.ndarray) -> dict:
    """Pooled and per-channel MSE; PSNR = 10 log10(255^2 / MSE) (inf when MSE == 0)."""
    if a.shape != b.shape:
        raise ValueError("shape mismatch %s vs %s" % (a.shape, b.shape))
    per_ss = [_sum_sq_diff(a[:, :, c], b[:, :, c]) for c in range(3)]
    per_n = a.shape[0] * a.shape[1]
    mse_c = {CHANNELS[c]: per_ss[c] / per_n for c in range(3)}
    mse = sum(per_ss) / (3 * per_n)
    psnr = lambda m: math.inf if m == 0 else 10.0 * math.log10(255.0 ** 2 / m)
    return dict(mse=mse, psnr_db=psnr(mse), mse_per_channel=mse_c,
                psnr_per_channel={k: psnr(v) for k, v in mse_c.items()},
                identical=(sum(per_ss) == 0), differing_values=int(np.count_nonzero(a != b)))


def _box_mean(x: np.ndarray, k: int) -> np.ndarray:
    c = np.zeros((x.shape[0] + 1, x.shape[1] + 1), np.float64)
    np.cumsum(np.cumsum(x, axis=0), axis=1, out=c[1:, 1:])
    s = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    return s / float(k * k)


def ssim_channel(x: np.ndarray, y: np.ndarray, win: int = SSIM_WINDOW):
    """Mean SSIM, uniform win x win window, sample covariance, K1=0.01, K2=0.03, L=255
    (same convention as scikit-image's default structural_similarity on a 2-D image)."""
    h, w = x.shape
    if h < win or w < win:
        return None
    x, y = x.astype(np.float64), y.astype(np.float64)
    npx = win * win
    cov_norm = npx / (npx - 1.0)
    ux, uy = _box_mean(x, win), _box_mean(y, win)
    vx = cov_norm * (_box_mean(x * x, win) - ux * ux)
    vy = cov_norm * (_box_mean(y * y, win) - uy * uy)
    vxy = cov_norm * (_box_mean(x * y, win) - ux * uy)
    c1, c2 = (0.01 * 255.0) ** 2, (0.03 * 255.0) ** 2
    s = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux * ux + uy * uy + c1) * (vx + vy + c2))
    return float(s.mean())


def ssim_report(a: np.ndarray, b: np.ndarray, identical: bool) -> dict:
    if identical:     # SSIM(x, x) = 1 exactly
        return dict(R=1.0, G=1.0, B=1.0, mean=1.0, note="identical images: SSIM = 1 by definition")
    per = {CHANNELS[c]: ssim_channel(a[:, :, c], b[:, :, c]) for c in range(3)}
    if any(v is None for v in per.values()):
        return dict(per, mean=None, note="undefined: image smaller than %dx%d SSIM window" % (SSIM_WINDOW, SSIM_WINDOW))
    per["mean"] = sum(per.values()) / 3.0
    per["note"] = "uniform %dx%d window, sample covariance, mean over R,G,B" % (SSIM_WINDOW, SSIM_WINDOW)
    return per


# ----------------------------------------------------------------------------
# chi-square uniformity test (own incomplete-gamma implementation, no SciPy needed)
# ----------------------------------------------------------------------------
def _gammaincc(a: float, x: float) -> float:
    """Regularised upper incomplete gamma Q(a, x)."""
    if x <= 0:
        return 1.0
    lg = math.lgamma(a)
    if x < a + 1.0:                                     # series for P, return 1 - P
        ap, s = a, 1.0 / a
        d = s
        for _ in range(100000):
            ap += 1.0
            d *= x / ap
            s += d
            if abs(d) < abs(s) * 1e-16:
                break
        return max(0.0, 1.0 - s * math.exp(-x + a * math.log(x) - lg))
    tiny = 1e-300                                       # continued fraction (modified Lentz)
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 100000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return math.exp(-x + a * math.log(x) - lg) * h


def chi2_sf(chi2: float, dof: int) -> float:
    return _gammaincc(dof / 2.0, chi2 / 2.0)


def chi_square_uniform(counts: np.ndarray) -> dict:
    """H0: all 256 intensity values are equally likely. Valid when expected count per bin >= 5."""
    total = int(counts.sum())
    k = int(counts.size)
    if total == 0:
        return dict(chi2=None, dof=k - 1, p_value=None, expected_per_bin=None, valid=False,
                    note="no samples")
    exp = total / k
    chi2 = float(((counts.astype(np.float64) - exp) ** 2 / exp).sum())
    valid = exp >= 5.0
    p = chi2_sf(chi2, k - 1) if valid else None
    return dict(chi2=chi2, dof=k - 1, p_value=p, expected_per_bin=exp, samples=total, valid=valid,
                verdict=(None if p is None else
                         ("uniformity NOT rejected at alpha=0.05" if p >= 0.05 else "uniformity rejected at alpha=0.05")),
                note=("" if valid else "p-value undefined: expected count per bin %.3f < 5 (chi-square approximation invalid)" % exp))


def chi_square_report(hist: np.ndarray) -> dict:
    out = {"pooled": chi_square_uniform(hist.sum(axis=0))}
    for i, c in enumerate(CHANNELS):
        out[c] = chi_square_uniform(hist[i])
    return out


# ----------------------------------------------------------------------------
# NPCR / UACI  (only for two ciphertexts of two controlled plaintexts)
# ----------------------------------------------------------------------------
def npcr_uaci(c1: np.ndarray, c2: np.ndarray) -> dict:
    """NPCR = 100 * mean(c1 != c2); UACI = 100 * mean(|c1 - c2|) / 255; per channel, then mean."""
    if c1.shape != c2.shape:
        raise ValueError("ciphertext shapes differ: %s vs %s" % (c1.shape, c2.shape))
    n = c1.shape[0] * c1.shape[1]
    npcr, uaci, changed = {}, {}, {}
    for i, c in enumerate(CHANNELS):
        a, b = c1[:, :, i], c2[:, :, i]
        k = int(np.count_nonzero(a != b))
        changed[c] = k
        npcr[c] = 100.0 * k / n
        uaci[c] = 100.0 * float(np.abs(a.astype(np.int16) - b.astype(np.int16)).sum(dtype=np.int64)) / (255.0 * n)
    npcr["mean"] = sum(npcr[c] for c in CHANNELS) / 3.0
    uaci["mean"] = sum(uaci[c] for c in CHANNELS) / 3.0
    return dict(npcr=npcr, uaci=uaci, changed_values=changed, changed_values_total=sum(changed.values()),
                values_per_channel=n, ideal_npcr=NPCR_IDEAL, ideal_uaci=UACI_IDEAL)


# ----------------------------------------------------------------------------
# whole-image analysis
# ----------------------------------------------------------------------------
def analyze_image(A: dict, rec: dict | None = None):
    """A maps stage name -> HxWx3 uint8 array for every stage that was produced.
    Returns (analysis dict, histogram dict {stage: (3,256) counts})."""
    out = dict(stages={}, reconstruction={}, chi_square=None, warnings=[], validation={})
    W = out["warnings"]
    hist = {}
    for s in STAGE_ORDER:
        arr = A.get(s)
        if arr is None:
            continue
        hist[s] = histograms(arr)
        st = dict(shape=list(arr.shape), dtype=str(arr.dtype), nbytes=int(arr.nbytes),
                  pixels=int(arr.shape[0] * arr.shape[1]), values=int(arr.size),
                  entropy=entropy_report(hist[s]))
        if s in CORR_STAGES:
            st["correlation"] = correlations(arr)
            for d, e in st["correlation"].items():
                for ch, why in e["undefined_reasons"].items():
                    W.append("%s %s correlation, channel %s: %s" % (s, d, ch, why))
        out["stages"][s] = st
    for key, a, b in RECON_PAIRS:
        if a in A and b in A:
            m = mse_psnr(A[a], A[b])
            m["ssim"] = ssim_report(A[a], A[b], m["identical"])
            if m["ssim"]["mean"] is None:
                W.append("%s: %s" % (key, m["ssim"]["note"]))
            out["reconstruction"][key] = m
    if "encrypted" in hist:
        out["chi_square"] = chi_square_report(hist["encrypted"])
        if not out["chi_square"]["pooled"]["valid"]:
            W.append("chi-square: %s" % out["chi_square"]["pooled"]["note"])

    # ---- metric validation (mathematical sanity checks on the numbers themselves) ----
    V = out["validation"]
    ref = next(iter(out["stages"].values()), None)
    V["all_stage_shapes_equal"] = len({tuple(v["shape"]) for v in out["stages"].values()}) <= 1
    V["histogram_counts_sum_to_pixels"] = all(int(h.sum(axis=1)[0]) == out["stages"][s]["pixels"] and
                                              len(set(h.sum(axis=1).tolist())) == 1 for s, h in hist.items())
    V["entropy_within_0_8_bits"] = all(0.0 <= (e or 0.0) <= 8.0 + 1e-9
                                       for v in out["stages"].values() for e in v["entropy"].values())
    corr_vals = [x for v in out["stages"].values() if "correlation" in v
                 for e in v["correlation"].values() for x in e["per_channel"].values() if x is not None]
    V["correlations_within_minus1_1"] = all(-1.0 <= x <= 1.0 for x in corr_vals)
    V["mse_non_negative_and_psnr_consistent"] = all(
        m["mse"] >= 0 and (m["psnr_db"] == math.inf) == (m["mse"] == 0) and
        (m["mse"] == 0 or abs(m["psnr_db"] - 10 * math.log10(255.0 ** 2 / m["mse"])) < 1e-9)
        for m in out["reconstruction"].values())
    V["ssim_within_minus1_1"] = all(m["ssim"]["mean"] is None or -1.0 - 1e-9 <= m["ssim"]["mean"] <= 1.0 + 1e-9
                                    for m in out["reconstruction"].values())
    if out["chi_square"] and out["chi_square"]["pooled"]["chi2"] is not None:
        V["chi_square_sample_count_equals_values"] = (out["chi_square"]["pooled"]["samples"] ==
                                                      out["stages"]["encrypted"]["values"])
        p = out["chi_square"]["pooled"]["p_value"]
        V["chi_square_p_value_in_0_1"] = p is None or 0.0 <= p <= 1.0
    if rec and FINAL_PAIR in out["reconstruction"] and rec.get("mse") is not None:
        V["final_mse_matches_pipeline_mse"] = abs(out["reconstruction"][FINAL_PAIR]["mse"] - rec["mse"]) < 1e-9
    return out, hist
