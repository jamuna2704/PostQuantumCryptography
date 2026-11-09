# -*- coding: utf-8 -*-
"""
bg_visuals.py - high-resolution PNG plots for the Lorenz + Blum-Goldwasser image pipeline.

Plots only READ the arrays handed to them. Sampling for scatter plots uses a local, seeded NumPy
generator, so it is reproducible and never touches the encryption random-number source.

Output layout (one sub-folder per processed image, never mixed with the pipeline stage folders):
    visualizations/<image>/<image>__hist_<stage>.png
    visualizations/<image>/<image>__corr_scatter_<stage>.png
    visualizations/<image>/<image>__comparison_input_vs_encrypted.png
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from bg_analysis import CHANNELS, DIRECTIONS, CORR_STAGES, STAGE_ORDER, sample_pairs

HAVE_MPL, MPL_VERSION, MPL_ERROR = False, None, ""
try:
    import matplotlib
    matplotlib.use("Agg")                      # file output only; never needs a display
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D
    HAVE_MPL, MPL_VERSION = True, matplotlib.__version__
except Exception as exc:                        # pragma: no cover
    MPL_ERROR = "%s: %s" % (type(exc).__name__, exc)

CH_COLORS = ("#d62728", "#2ca02c", "#1f77b4")
STAGE_TITLES = dict(input="input (original RGB)", scrambled="scrambled (Lorenz)", encrypted="encrypted (BG)",
                    decrypted="decrypted (= scrambled)", descrambled="descrambled (final reconstruction)")


def _save(fig, path: Path, dpi: int) -> str:
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return str(path)


def _thumb(arr: np.ndarray, target: int = 600) -> np.ndarray:
    step = max(1, math.ceil(max(arr.shape[0], arr.shape[1]) / target))
    return arr[::step, ::step]


def _fr(v, p=4):
    return "undefined" if v is None else "%.*f" % (p, v)


def plot_stage_histogram(path, base, stage, hist, entropy, pixels, dpi):
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8.5))
    x = np.arange(256)
    for c, ax in enumerate(axes):
        ax.bar(x, hist[c], width=1.0, color=CH_COLORS[c], alpha=0.85,
               label="%s channel   Shannon entropy = %s bits (max 8)" % (CHANNELS[c], _fr(entropy[CHANNELS[c]])))
        if stage == "encrypted":
            ax.axhline(pixels / 256.0, color="black", ls="--", lw=1.0,
                       label="uniform expectation = %.2f per value" % (pixels / 256.0))
        ax.set_ylabel("Pixel count")
        ax.set_xlim(-0.5, 255.5)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Intensity value (0-255, 8-bit)")
    fig.suptitle("%s - %s\nPixel-intensity histogram per channel (pooled entropy %s bits)"
                 % (base, STAGE_TITLES.get(stage, stage), _fr(entropy["pooled"])), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, path, dpi)


def plot_stage_scatter(path, base, stage, arr, corr, k, seed, dpi):
    fig, axes = plt.subplots(3, 3, figsize=(12.5, 12))
    for ci, cname in enumerate(CHANNELS):
        for di, d in enumerate(DIRECTIONS):
            ax = axes[ci][di]
            xs, ys, total = sample_pairs(arr, d, ci, k, seed + 1000 * ci + 100 * di)
            r = corr[d]["per_channel"][cname]
            ax.scatter(xs, ys, s=2, alpha=0.3, color=CH_COLORS[ci], linewidths=0, rasterized=True)
            ax.set_xlim(-3, 258)
            ax.set_ylim(-3, 258)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title("%s channel - %s" % (cname, d), fontsize=9)
            ax.set_xlabel("Pixel value p(x, y)", fontsize=8)
            ax.set_ylabel("Neighbour value", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.2)
            lab = "r = %s | %s of %s pairs plotted" % (_fr(r, 6), format(len(xs), ","), format(total, ","))
            ax.legend(handles=[Line2D([], [], ls="", marker="o", ms=4, color=CH_COLORS[ci], label=lab)],
                      loc="lower right", fontsize=6.5)
    fig.suptitle("%s - %s\nAdjacent-pixel pairs (r = exact Pearson over ALL pairs; points = seeded sample)"
                 % (base, STAGE_TITLES.get(stage, stage)), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, path, dpi)


def plot_comparison(path, base, A, an, hist, k, seed, dpi):
    inp, enc = A["input"], A["encrypted"]
    si, se = an["stages"]["input"], an["stages"]["encrypted"]
    fig = plt.figure(figsize=(16, 13))
    gs = GridSpec(3, 3, figure=fig)
    ax = fig.add_subplot(gs[0, 0]); ax.imshow(_thumb(inp)); ax.set_title("Input (original RGB)"); ax.axis("off")
    ax = fig.add_subplot(gs[0, 1]); ax.imshow(_thumb(enc)); ax.set_title("Encrypted (ciphertext bytes as RGB)"); ax.axis("off")

    # correlation bars (mean over defined channels)
    ax = fig.add_subplot(gs[0, 2])
    xs = np.arange(3)
    for off, (lab, st, col) in enumerate((("input", si, "#4c72b0"), ("encrypted", se, "#dd8452"))):
        vals = [st["correlation"][d]["mean"] for d in DIRECTIONS]
        ax.bar(xs + (off - 0.5) * 0.38, [0 if v is None else v for v in vals], 0.38, label=lab, color=col)
        for i, v in enumerate(vals):
            ax.text(xs[i] + (off - 0.5) * 0.38, 0.02 if v is None else v, "undef." if v is None else "%.4f" % v,
                    ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(xs); ax.set_xticklabels(DIRECTIONS)
    ax.set_ylim(-0.1, 1.25); ax.set_ylabel("Pearson r (mean of R,G,B)"); ax.legend(fontsize=8); ax.grid(alpha=0.25)
    ax.set_title("Adjacent-pixel correlation")

    x = np.arange(256)
    for col, (stage, st) in enumerate((("input", si), ("encrypted", se))):
        ax = fig.add_subplot(gs[1, col])
        for c in range(3):
            ax.plot(x, hist[stage][c], color=CH_COLORS[c], lw=0.8, label="%s  H=%s" % (CHANNELS[c], _fr(st["entropy"][CHANNELS[c]])))
        if stage == "encrypted":
            ax.axhline(st["pixels"] / 256.0, color="black", ls="--", lw=1, label="uniform expectation")
        ax.set_xlim(0, 255); ax.set_xlabel("Intensity value (0-255)"); ax.set_ylabel("Pixel count")
        ax.set_title("%s histogram (pooled H = %s bits)" % (stage, _fr(st["entropy"]["pooled"])))
        ax.legend(fontsize=7); ax.grid(alpha=0.25)
    ax = fig.add_subplot(gs[1, 2])
    labels = list(CHANNELS) + ["pooled"]
    ax.bar(np.arange(4) - 0.19, [si["entropy"][c] or 0 for c in labels], 0.38, label="input", color="#4c72b0")
    ax.bar(np.arange(4) + 0.19, [se["entropy"][c] or 0 for c in labels], 0.38, label="encrypted", color="#dd8452")
    ax.axhline(8, color="black", ls="--", lw=1, label="maximum (8 bits)")
    ax.set_xticks(np.arange(4)); ax.set_xticklabels(labels); ax.set_ylim(0, 8.6)
    ax.set_ylabel("Shannon entropy (bits per value)"); ax.set_title("Entropy"); ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.25)

    for col, (stage, arr, st) in enumerate((("input", inp, si), ("encrypted", enc, se))):
        ax = fig.add_subplot(gs[2, col])
        xs_, ys_, total = sample_pairs(arr, "horizontal", 1, k, seed + 7)
        ax.scatter(xs_, ys_, s=2, alpha=0.3, color=CH_COLORS[1], linewidths=0, rasterized=True)
        ax.set_xlim(-3, 258); ax.set_ylim(-3, 258); ax.set_aspect("equal", adjustable="box")
        r = st["correlation"]["horizontal"]["per_channel"]["G"]
        ax.set_xlabel("Pixel value p(x, y)"); ax.set_ylabel("Right neighbour p(x+1, y)")
        ax.set_title("%s: horizontal pairs, G channel" % stage)
        ax.legend(handles=[Line2D([], [], ls="", marker="o", ms=4, color=CH_COLORS[1],
                                  label="r = %s | %s of %s pairs" % (_fr(r, 6), format(len(xs_), ","), format(total, ",")))],
                  loc="lower right", fontsize=7)
        ax.grid(alpha=0.2)
    ax = fig.add_subplot(gs[2, 2]); ax.axis("off")
    cs = an.get("chi_square") or {}
    pooled = cs.get("pooled") or {}
    lines = ["Encrypted image summary", "-----------------------",
            "pixels            : %s" % format(se["pixels"], ","),
            "entropy (pooled)  : %s bits" % _fr(se["entropy"]["pooled"], 6),
            "chi-square        : %s" % _fr(pooled.get("chi2"), 3),
            "degrees of freedom: %s" % pooled.get("dof", "n/a"),
            "p-value           : %s" % ("n/a" if pooled.get("p_value") is None else "%.6g" % pooled["p_value"]),
            "corr H / V / D    : %s / %s / %s" % tuple(_fr(se["correlation"][d]["mean"], 4) for d in DIRECTIONS)]
    ax.text(0.0, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=9.5)
    fig.suptitle("%s - original vs encrypted" % base, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, path, dpi)


def make_plots(base, A, analysis, hist, out_root, dpi=200, scatter_points=8000, seed=20240601) -> dict:
    """Never raises: every figure is attempted independently and failures become warnings."""
    res = dict(files=[], warnings=[], folder=None)
    if not HAVE_MPL:
        res["warnings"].append("matplotlib unavailable (%s): no plots. Install with: py -m pip install matplotlib" % MPL_ERROR)
        return res
    folder = Path(out_root) / base
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        res["warnings"].append("cannot create %s: %s" % (folder, e))
        return res
    res["folder"] = str(folder)

    def attempt(name, fn):
        try:
            res["files"].append(fn())
        except Exception as e:
            plt.close("all")
            res["warnings"].append("plot '%s' failed: %s: %s" % (name, type(e).__name__, e))

    for s in STAGE_ORDER:
        if s in hist and s in analysis["stages"]:
            st = analysis["stages"][s]
            attempt("hist_" + s, lambda s=s, st=st: plot_stage_histogram(
                folder / ("%s__hist_%s.png" % (base, s)), base, s, hist[s], st["entropy"], st["pixels"], dpi))
    for s in CORR_STAGES:
        if s in A and s in analysis["stages"] and "correlation" in analysis["stages"][s]:
            attempt("corr_" + s, lambda s=s: plot_stage_scatter(
                folder / ("%s__corr_scatter_%s.png" % (base, s)), base, s, A[s],
                analysis["stages"][s]["correlation"], scatter_points, seed, dpi))
    if "input" in A and "encrypted" in A and "input" in hist and "encrypted" in hist:
        attempt("comparison", lambda: plot_comparison(
            folder / ("%s__comparison_input_vs_encrypted.png" % base), base, A, analysis, hist,
            scatter_points, seed, dpi))
    return res
