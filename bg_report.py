# -*- coding: utf-8 -*-
"""
bg_report.py - structured terminal / text / JSON / CSV reporting for the Lorenz + Blum-Goldwasser pipeline.

This module only FORMATS values that the pipeline already computed. It never generates keys, states
or random numbers.

Modes
    research (default) : local debugging; prints p, q, a, b, x0, d_p, d_q, u, v in full. Do NOT share.
    safe (--hide-secrets / --safe): secrets are neither printed nor written to any report file.
Terminal verbosity
    full    : everything is echoed (default)
    compact : large integers' digits/bit-lengths and summaries only; the saved .txt still has everything
    pager   : nothing echoed live; the complete report is shown in a pager at the end
"""
from __future__ import annotations

import csv
import json
import math
import shutil
import sys
import textwrap
from pathlib import Path

from bg_analysis import CHANNELS, DIRECTIONS, CORR_STAGES, FINAL_PAIR, NPCR_IDEAL, UACI_IDEAL

if hasattr(sys, "set_int_max_str_digits"):      # Python >= 3.11 limits int->str to 4300 digits by default
    sys.set_int_max_str_digits(0)

SECRET_FIELDS = ("x0", "dp", "dq", "u", "v", "x0_rec")
MODE_CHANNELS = {"1": 1, "L": 1, "P": 1, "I": 1, "F": 1, "I;16": 1, "LA": 2, "PA": 2, "RGB": 3,
                 "YCbCr": 3, "LAB": 3, "HSV": 3, "RGBA": 4, "CMYK": 4}
RECON_LABELS = {
    "input_vs_scrambled": "input vs scrambled",
    "input_vs_encrypted": "input vs encrypted",
    "input_vs_decrypted": "input vs decrypted (decrypted == scrambled)",
    "scrambled_vs_decrypted": "scrambled vs decrypted (BG round trip)",
    "input_vs_descrambled": "input vs descrambled (FINAL reconstruction)",
}
SENS_LABELS = (
    ("same_key_same_state", "A: same key, same x0 (1-pixel change only)"),
    ("same_key_fresh_state", "B: same key, fresh x0, 1-pixel change"),
    ("control_same_plaintext_fresh_state", "Control: SAME plaintext, fresh x0"),
)


def int_info(x: int, secret: bool, safe: bool) -> dict:
    return dict(value=None if (secret and safe) else x, digits=len(str(abs(x))), bits=abs(x).bit_length(),
                secret=secret, negative=x < 0)


def fnum(v, p=6, inf="INF") -> str:
    if v is None:
        return "undefined"
    if isinstance(v, float) and math.isinf(v):
        return inf
    return "%.*f" % (p, v)


def pval(v) -> str:
    return "n/a" if v is None else ("%.6g" % v)


def _ok(b) -> str:
    return "PASS" if b else "FAIL"


class Reporter:
    def __init__(self, safe=False, unicode_ok=False, terminal="full", width=None):
        self.safe, self.uni, self.terminal = safe, unicode_ok, terminal
        cols = width or shutil.get_terminal_size((100, 24)).columns
        self.W = max(72, min(cols - 1, 110))
        self.R1 = ("═" if unicode_ok else "=") * self.W
        self.R2 = ("─" if unicode_ok else "-") * self.W
        self.dash = "–" if unicode_ok else "-"
        self.bg = "Blum%sGoldwasser" % self.dash
        self.lines = []                  # list of (text, detail_only)
        self._slot = None
        self.warnings_all = []

    # ------------------------------------------------------------------ low level
    def line(self, text="", detail=False):
        self.lines.append((text, detail))
        if self.terminal == "pager" or (detail and self.terminal == "compact"):
            return
        try:
            print(text)
        except UnicodeEncodeError:
            print(text.encode("ascii", "replace").decode("ascii"))

    def text(self) -> str:
        return "\n".join(t for t, _ in self.lines) + "\n"

    def section(self, num, title):
        self.line()
        self.line("[%s] %s" % (num, title))
        self.line(self.R2)

    def sub(self, title):
        self.line()
        self.line("  " + title)

    def _kv_lines(self, rows, indent=2, sep=" : "):
        if not rows:
            return []
        lw = max(len(l) for l, _ in rows)
        avail = max(24, self.W - indent - lw - len(sep))
        out = []
        for l, v in rows:
            parts = textwrap.wrap(str(v), avail, break_long_words=True, break_on_hyphens=False,
                                  replace_whitespace=False, drop_whitespace=False) or [""]
            out.append("%s%-*s%s%s" % (" " * indent, lw, l, sep, parts[0]))
            out += [" " * (indent + lw + len(sep)) + p for p in parts[1:]]
        return out

    def kv(self, rows, indent=2, detail=False, wrap=True):
        if not wrap:                       # paths: keep each value on one unbroken line
            lw = max(len(l) for l, _ in rows)
            for l, v in rows:
                self.line("%s%-*s : %s" % (" " * indent, lw, l, v), detail)
            return
        for s in self._kv_lines(rows, indent):
            self.line(s, detail)

    def table(self, headers, rows, indent=4, aligns=None, detail=False):
        rows = [[str(c) for c in r] for r in rows]
        widths = [max([len(h)] + [len(r[i]) for r in rows]) for i, h in enumerate(headers)]
        aligns = aligns or ["<"] + [">"] * (len(headers) - 1)
        fmt = lambda cells: " " * indent + "  ".join(("%-*s" if a == "<" else "%*s") % (w, c)
                                                    for a, w, c in zip(aligns, widths, cells)).rstrip()
        self.line(fmt(headers), detail)
        self.line(" " * indent + "  ".join("-" * w for w in widths), detail)
        for r in rows:
            self.line(fmt(r), detail)

    def big_int(self, label, info, note="", indent=4):
        pad = " " * indent
        self.line("%s%s%s" % (" " * (indent - 2), label, ("   " + note) if note else ""))
        self.line("%sdecimal digits : %d" % (pad, info["digits"]))
        self.line("%sbit length     : %d" % (pad, info["bits"]))
        if info["value"] is None:
            self.line("%svalue          : <hidden - safe mode>" % pad)
            self.line()
            return
        x = info["value"]
        if x < 0:
            self.line("%ssign           : negative (magnitude shown)" % pad)
        dec, hx = str(abs(x)), format(abs(x), "x")
        hx = hx.zfill(len(hx) + (len(hx) % 2))
        cw = max(32, self.W - indent - 4)
        self.line("%sdecimal value  (complete; line breaks inserted every %d digits for display only):" % (pad, cw), True)
        for i in range(0, len(dec), cw):
            self.line("%s  %s" % (pad, dec[i:i + cw]), True)
        hw = max(16, min(64, cw))
        self.line("%shexadecimal    (complete, %d hex digits):" % (pad, len(hx)), True)
        for i in range(0, len(hx), hw):
            self.line("%s  %s" % (pad, hx[i:i + hw]), True)
        self.line()

    # ------------------------------------------------------------------ [01] / [02] global
    def _summary_rows(self, run, status, total, n_done):
        return [("Run ID", run["run_id"]), ("Date and time", run["started"]),
                ("Algorithm", "Lorenz 2-D chaotic-map pixel scrambling + %s (HAC Alg. 8.56)" % self.bg),
                ("Report mode", "SAFE (secrets hidden, shareable)" if self.safe else
                 "RESEARCH (full secret disclosure - local use only, do not share)"),
                ("Images found / processed", "%d / %s" % (run["n_images"], n_done)),
                ("Total execution time", "n/a (running)" if total is None else "%.2f s" % total),
                ("Overall status", status)]

    def header(self, run):
        self.line(self.R1)
        self.line("IMAGE CRYPTOGRAPHY REPORT")
        self.line("Lorenz chaotic-map scrambling + %s encryption" % self.bg)
        self.line()
        self.line("[01] EXECUTION SUMMARY")
        self.line(self.R2)
        start = len(self.lines)
        self.kv(self._summary_rows(run, "RUNNING", None, "..."))
        self._slot = (start, len(self.lines))
        if not self.safe:
            self.line()
            self.line("  WARNING: research mode prints and saves secret keys. Re-run with --hide-secrets before sharing.")
        if self.terminal == "compact":
            self.line("  (compact terminal mode: complete integers and detail tables are in the saved report file)")

        K, bg = run["keys"], run["bg"]
        self.section("02", "CRYPTOGRAPHIC PARAMETERS")
        self.sub("%s parameters (key material generated in this run)" % self.bg)
        self.kv([("reference", "Handbook of Applied Cryptography, Algorithm 8.56 (stores x_(t+1); exponent t+1)"),
                 ("prime size", "%d bits each (requested)" % bg["prime_bits"]),
                 ("modulus size", "%d bits" % bg["modulus_bits"]),
                 ("block size h", "%d bits = %d bytes" % (bg["block_bits"], bg["block_bytes"])),
                 ("textbook bound", "floor(lg lg n) = %d bits (provable-security regime; see warnings)" % bg["textbook_bound"]),
                 ("byte order", bg["byte_order"]),
                 ("key generation time", "%.3f s" % run["keygen_seconds"])], indent=4)
        self.line()
        self.big_int("p  (secret prime, p = 3 mod 4)", K["p"])
        self.big_int("q  (secret prime, q = 3 mod 4)", K["q"])
        self.big_int("n = p*q  (public modulus)", K["n"])
        self.big_int("a  (Bezout coefficient, a*p + b*q = 1, secret)", K["a"])
        self.big_int("b  (Bezout coefficient, a*p + b*q = 1, secret)", K["b"])
        self.sub("Key validation (performed on the actual key of this run)")
        self.table(["result", "check"], [[_ok(ok), name] for name, ok in run["key_checks"]], indent=4, aligns=["<", "<"])
        self.sub("Self-tests executed before processing (textbook HAC example, round trips, failure detection, BMP, Lorenz)")
        for l in run["unit_tests"]["pass_lines"]:
            self.line("    [PASS] " + l)
        self.line("    Per-image state (x0, x_(t+1), t, d_p, d_q, u, v) is printed in each image's [02] block.")

    # ------------------------------------------------------------------ per image
    def image_report(self, idx, total, img):
        rec, an, V = img["rec"], img.get("analysis") or {}, img["values"]
        T, C = rec.get("timings_s", {}), rec.get("checks", {})
        self.line()
        self.line(self.R1)
        self.line(" IMAGE %d / %d : %s" % (idx, total, rec["source_file"]))
        self.line(self.R1)
        warns = list(img.get("warnings", []))
        if "input_size" not in rec:
            self.section("05", "PIPELINE STATUS")
            self.kv([("Result", "FAILED at stage '%s'" % rec.get("failed_stage")), ("Reason", rec.get("error", "unknown"))])
            self.section("07", "VALIDATION")
            self.kv([("Errors or warnings", "image could not be loaded; no metrics available")])
            self.line()
            self.line("  STATUS : FAIL")
            return
        w, h = rec["input_size"]
        sw, sh = rec["source_size"]
        N, L, B = rec["pixels"], rec["rgb_bytes"], rec["bg_block_bytes"]
        src = img["src"]

        # [02] per-image state
        self.section("02", "CRYPTOGRAPHIC PARAMETERS - per-image %s state" % self.bg)
        self.kv([("plaintext bit length", "%s bits (%s bytes of scrambled RGB data)" % (format(8 * L, ","), format(L, ","))),
                 ("ciphertext bit length", "%s bits (stream cipher: equal to plaintext length)" % format(8 * L, ",")),
                 ("number of encrypted bits", format(8 * L, ",")),
                 ("block size h", "%d bits (%d bytes)" % (8 * B, B)),
                 ("blocks t", "%s (last block %d bytes)" % (format(rec["bg_blocks"], ","), rec["bg_final_block_bytes"])),
                 ("decryption exponent", "t+1 = %s" % format(rec["bg_blocks"] + 1, ",")),
                 ("random seed", "x0 = r^2 mod n, r drawn from random.SystemRandom (OS entropy; not seedable/reproducible)")])
        self.line()
        sec = lambda key, label, note="": self.big_int(
            label, int_info(V[key], True, self.safe) if key in SECRET_FIELDS else int_info(V[key], False, self.safe), note) \
            if key in V else None
        sec("x0", "x0  (initial state = r^2 mod n)", "[secret]")
        sec("xfinal", "x_(t+1)  (final state; stored in metadata.json)", "[public]")
        sec("dp", "d_p = ((p+1)/4)^(t+1) mod (p-1)", "[secret]")
        sec("dq", "d_q = ((q+1)/4)^(t+1) mod (q-1)", "[secret]")
        sec("u", "u = x_(t+1)^d_p mod p", "[secret]")
        sec("v", "v = x_(t+1)^d_q mod q", "[secret]")
        sec("x0_rec", "x0 recovered = v*a*p + u*b*q mod n", "[secret]")

        # [03] Lorenz
        lz = rec.get("lorenz")
        self.section("03", "LORENZ SCRAMBLING PARAMETERS")
        if lz:
            ps = img.get("perm") or {}
            self.kv([("map", "x1(n+1) = (1 + A*h)*x1(n) - h*x1(n)*x2(n) ; x2(n+1) = (1 - h)*x1(n) + h*x1(n)^2"),
                     ("initial conditions", "X1_0 = %r, X2_0 = %r" % (lz["x1_initial"], lz["x2_initial"])),
                     ("system constants", "A = %r, h = %r   (Lorenz-map h; unrelated to the BG block size)" % (lz["A"], lz["h_lorenz"])),
                     ("transient iterations", "%d (discarded)" % lz["transient"]),
                     ("sequence length", "%s values (one per pixel)" % format(lz["N"], ",")),
                     ("total iterations", "%s = %d transient + %s used" % (format(lz["transient"] + lz["N"], ","), lz["transient"], format(lz["N"], ","))),
                     ("ordering key / method", "s = x1 + x2 ; permutation = stable argsort(s)  (ties broken by index)"),
                     ("sequence precision", "float64 (IEEE-754 binary64, ~15-17 significant digits), stored via array('d')"),
                     ("sequence minimum", "%.17g" % lz["key_min"]), ("sequence maximum", "%.17g" % lz["key_max"]),
                     ("tied key values", "%d" % lz["ties"]),
                     ("permutation size", "%s pixel positions (each RGB triplet moves as one unit)" % format(lz["N"], ","))])
            if ps:
                self.kv([("permutation array", "shape %s, dtype %s, memory %s bytes, min %s, max %s" % (
                    tuple(ps["shape"]), ps["dtype"], format(ps["nbytes"], ","), ps["min"], ps["max"])),
                         ("is a permutation of 0..N-1", _ok(ps.get("is_permutation_of_0_to_N_minus_1"))),
                         ("SHA-256 checksum", "%s  (a checksum of the array, NOT the array itself)" % ps["sha256"])])
                if "full" in ps:
                    self.line("    permutation array (complete, N <= 64): %s" % ps["full"])
                else:
                    self.line("    (array has %s elements: too large to print; saved scrambling is reproducible from the constants above)" % format(ps["count"], ","))
        else:
            self.line("  not reached")

        # [04] image details
        self.section("04", "IMAGE DETAILS")
        nch = MODE_CHANNELS.get(rec["source_mode"], "?")
        arr_nbytes = rec["rgb_bytes"]
        self.kv([("Filename", rec["source_file"]), ("Path", "<hidden in safe mode>" if self.safe else src["path"]),
                 ("Format / source mode", "%s / %s" % (rec["source_format"], rec["source_mode"])),
                 ("Dimensions (source)", "%d x %d (width x height)" % (sw, sh)),
                 ("Dimensions (pipeline)", "%d x %d%s" % (w, h, "  [native, no resizing]" if (w, h) == (sw, sh) else "  [RESIZED]")),
                 ("Channels", "source %s -> pipeline 3 (RGB)" % nch),
                 ("Pixel count", format(N, ",")), ("Scalar channel values", format(3 * N, ",")),
                 ("Bits per channel", "8"), ("Raw data bit count", format(8 * L, ",")),
                 ("File size (source)", "%s bytes" % format(src["bytes"], ",")),
                 ("Array (pipeline input)", "shape (%d, %d, 3), dtype uint8, memory %s bytes" % (h, w, format(arr_nbytes, ","))),
                 ("SHA-256 (source file)", src["sha256"]),
                 ("SHA-256 (RGB input array)", rec.get("hashes_sha256", {}).get("input", "n/a"))])
        if "sizes" in rec:
            self.sub("Sizes at every stage")
            self.table(["stage", "w x h", "pixels", "RGB bytes", "file bytes", "format"],
                       [[s["stage"], "%d x %d" % (s["width"], s["height"]), format(s["pixels"], ","),
                         format(s["rgb_bytes"], ","), format(s["file_bytes"], ","), s["kind"]] for s in rec["sizes"]],
                       aligns=["<", "<", ">", ">", ">", "<"])
            bl = rec["bmp_layout"]
            self.line("    BMP: %d-byte header + %d rows x %d-byte stride (%d padding bytes/row) = %s bytes" % (
                bl["header_bytes"], bl["rows"], bl["row_stride"], bl["padding_per_row"], format(rec["bmp_file_size"], ",")))

        # [05] pipeline
        self.section("05", "PIPELINE STATUS")
        have = img["have"]
        order = [("Original image (-> RGB input)", "input"), ("Scrambling (Lorenz)", "scrambled"),
                 ("Encryption (BG)", "encrypted"), ("Decryption (BG)", "decrypted"), ("Descrambling (inverse Lorenz)", "descrambled")]
        rows, failed_seen = [], False
        for label, key in order:
            if key in have:
                st = "OK"
                if key == "decrypted" and C.get("decrypted_equals_scrambled") is False:
                    st = "MISMATCH"
                if key == "descrambled" and C.get("descrambled_equals_input") is False:
                    st = "MISMATCH"
            elif not failed_seen:
                st, failed_seen = "FAILED (%s)" % rec.get("failed_stage"), True
            else:
                st = "NOT RUN"
            bs = rec.get("bmp_stages", {}).get(key)
            rows.append([label, st, "%s/%s.bmp" % (key, img["base"]), format(bs["file_size"], ",") if bs else "n/a"])
        fin = C.get("descrambled_equals_input")
        rows.append(["Final reconstruction", "OK (bit-exact)" if fin else ("FAILED" if fin is False else "NOT RUN"), "", ""])
        self.table(["stage", "status", "file", "bytes"], rows, aligns=["<", "<", "<", ">"])
        H = rec.get("hashes_sha256", {})
        for k in ("input", "scrambled", "encrypted", "decrypted", "descrambled"):
            if k in H:
                self.line("    SHA-256 %-11s %s" % (k, H[k]), True)

        # [06] metrics
        self.section("06", "QUANTITATIVE METRICS")
        if not an or "stages" not in an:
            self.line("  analysis unavailable: %s" % (an.get("error", "no stages completed") if an else "no stages completed"))
        else:
            self._metrics(img, an, T)

        # [07] validation
        self.section("07", "VALIDATION")
        self._validation(img, an, C, rec, warns)
        self.line()
        self.line("  STATUS : %s%s" % (rec["status"], "" if rec["status"] == "PASS" else
                                      "   (stage: %s %s)" % (rec["failed_stage"], rec.get("error", ""))))

    def _metrics(self, img, an, T):
        rc, st, cs = an["reconstruction"], an["stages"], an.get("chi_square")
        fin = rc.get(FINAL_PAIR)
        sens = img.get("sensitivity") or {}
        enc = st.get("encrypted")
        sA = sens.get("same_key_same_state")
        sB = sens.get("same_key_fresh_state")
        self.sub("Headline values")
        pool = (cs or {}).get("pooled") or {}
        rows = [
            ("MSE (input vs descrambled)", fnum(fin["mse"]) if fin else "n/a"),
            ("PSNR (input vs descrambled)", ("INF dB (identical images, MSE = 0)" if fin and fin["psnr_db"] == math.inf else
                                              "%s dB" % fnum(fin["psnr_db"]) if fin else "n/a")),
            ("SSIM (input vs descrambled)", fnum(fin["ssim"]["mean"]) if fin else "n/a"),
            ("NPCR  [A: same key+x0]", "%s %%" % fnum(sA["npcr"]["mean"], 6) if sA else "n/a (not run)"),
            ("UACI  [A: same key+x0]", "%s %%" % fnum(sA["uaci"]["mean"], 6) if sA else "n/a (not run)"),
            ("NPCR  [B: same key, fresh x0]", "%s %%" % fnum(sB["npcr"]["mean"], 6) if sB else "n/a (not run)"),
            ("UACI  [B: same key, fresh x0]", "%s %%" % fnum(sB["uaci"]["mean"], 6) if sB else "n/a (not run)"),
            ("Chi-square (encrypted, pooled)", fnum(pool.get("chi2"), 4) if pool else "n/a"),
            ("Degrees of freedom", str(pool.get("dof", "n/a"))),
            ("p-value", pval(pool.get("p_value")) if pool.get("valid") else "n/a (%s)" % pool.get("note", "unavailable")),
            ("Shannon entropy (encrypted)", "%s bits per value (max 8)" % fnum(enc["entropy"]["pooled"], 6) if enc else "n/a"),
        ]
        for d in DIRECTIONS:
            c = enc["correlation"][d]["mean"] if enc else None
            rows.append(("%s correlation (encrypted)" % d.capitalize(), fnum(c, 6) if c is not None else "undefined"))
        T = img["rec"].get("timings_s", {})
        rows += [("Encryption time (BG only)", "%.4f s" % T["bg_encryption"] if "bg_encryption" in T else "n/a"),
                 ("Decryption time (BG only)", "%.4f s" % T["bg_decryption"] if "bg_decryption" in T else "n/a")]
        self.kv(rows, indent=4)
        self.line("    NPCR/UACI intensity range 0..255 (L = 255); ideal random 8-bit reference: NPCR %.4f %%, UACI %.4f %%." % (NPCR_IDEAL, UACI_IDEAL))

        self.sub("Reconstruction metrics (matched image pairs; MSE/PSNR pooled over R,G,B; peak = 255)")
        r = []
        for key, lab in RECON_LABELS.items():
            m = rc.get(key)
            if m:
                r.append([lab, fnum(m["mse"], 4), "INF" if m["psnr_db"] == math.inf else fnum(m["psnr_db"], 3),
                          fnum(m["ssim"]["mean"], 6)])
        self.table(["pair", "MSE", "PSNR (dB)", "SSIM"], r, aligns=["<", ">", ">", ">"])
        self.line("    'decrypted' is the scrambled image recovered by BG; the original is recovered only after descrambling.")

        self.sub("Shannon entropy (bits per 8-bit value, maximum 8)")
        self.table(["stage", "R", "G", "B", "pooled"],
                   [[s] + [fnum(st[s]["entropy"][c], 6) for c in CHANNELS + ("pooled",)] for s in st])

        if cs:
            self.sub("Chi-square uniformity test of encrypted pixel intensities (H0: all 256 values equally likely)")
            rows = []
            for k in ("pooled",) + CHANNELS:
                c = cs[k]
                rows.append([k, fnum(c["chi2"], 4), str(c["dof"]), fnum(c["expected_per_bin"], 2),
                             pval(c["p_value"]), c.get("verdict") or "n/a (invalid)"])
            self.table(["scope", "chi-square", "dof", "expected/bin", "p-value", "verdict"], rows,
                       aligns=["<", ">", ">", ">", ">", "<"])
            self.line("    Valid when expected count per bin >= 5. A single p-value is evidence, not proof, of randomness.")

        self.sub("Adjacent-pixel Pearson correlation (exact, all pairs; mean = average of defined channels)")
        rows = []
        for s in CORR_STAGES:
            if s in st and "correlation" in st[s]:
                for d in DIRECTIONS:
                    e = st[s]["correlation"][d]
                    rows.append([s, d] + [fnum(e["per_channel"][c], 6) for c in CHANNELS] +
                                [fnum(e["mean"], 6), format(e["pairs_per_channel"], ",")])
        self.table(["stage", "direction", "R", "G", "B", "mean", "pairs/channel"], rows,
                   aligns=["<", "<", ">", ">", ">", ">", ">"])
        self.line("    metadata.json keeps the legacy 30,000-pair sample estimate; this table uses every pair.")

        if sens:
            m = sens["modification"]
            self.sub("Plaintext sensitivity: P1 vs P2 differing by one bit")
            self.line("    P2 = P1 with the LSB of the R value at pixel (x=%d, y=%d) flipped (%d -> %d); same key, same Lorenz permutation." % (
                m["pixel_x"], m["pixel_y"], m["old"], m["new"]))
            rows = []
            for k, lab in SENS_LABELS:
                s = sens.get(k)
                if s:
                    rows.append([lab] + [fnum(s["npcr"][c], 4) for c in CHANNELS + ("mean",)] +
                                [fnum(s["uaci"][c], 4) for c in CHANNELS + ("mean",)] + [format(s["changed_values_total"], ",")])
            self.table(["case", "NPCR R", "G", "B", "mean %", "UACI R", "G", "B", "mean %", "changed values"], rows,
                       aligns=["<"] + [">"] * 9)
            self.line("    Case A isolates the cipher's plaintext dependence; case B/Control show how much comes from changing x0 alone.")

        self.sub("Timings (seconds, in-memory; excludes file I/O except where noted)")
        lab = [("resize", "load + RGB convert"), ("lorenz_generation", "Lorenz generation"), ("scrambling", "scrambling"),
               ("bg_encryption", "BG encryption"), ("bg_decryption", "BG decryption"),
               ("lorenz_regeneration", "Lorenz regeneration"), ("descrambling", "descrambling")]
        rows = [[l, "%.4f" % T[k]] for k, l in lab if k in T]
        enc_t = sum(T.get(k, 0) for k in ("lorenz_generation", "scrambling", "bg_encryption"))
        dec_t = sum(T.get(k, 0) for k in ("bg_decryption", "lorenz_regeneration", "descrambling"))
        rows += [["encryption pipeline (Lorenz + BG)", "%.4f" % enc_t], ["decryption pipeline (BG + Lorenz + inverse)", "%.4f" % dec_t],
                 ["analysis (metrics)", "%.4f" % img["timings"].get("analysis", 0)], ["plots", "%.4f" % img["timings"].get("plots", 0)],
                 ["pipeline total", "%.4f" % T.get("total", 0)]]
        self.table(["step", "seconds"], rows, aligns=["<", ">"])

        if img.get("plots") and (img["plots"].get("files") or img["plots"].get("warnings")):
            self.sub("Generated plots")
            for f in img["plots"]["files"]:
                self.line("    " + f)

    def _validation(self, img, an, C, rec, warns):
        rows = [[_ok(C.get(k)), d] for k, d in (
            ("x0_quadratic_residue", "x0 is a quadratic residue mod p and mod q"),
            ("crt_residues", "u = x0 mod p and v = x0 mod q"),
            ("bg_state_recovery", "recovered x0 == generated x0"))]
        self.sub("Mathematical parameter checks (this image)")
        self.table(["result", "check"], rows, aligns=["<", "<"])
        self.sub("Image dimension checks")
        self.table(["result", "check"], [[_ok(C.get(k)), d] for k, d in (
            ("source_dims_preserved", "native dimensions preserved (no resizing)"),
            ("resize_target_met", "resize target met"),
            ("input_bmp_readback", "input BMP reads back identical"),
            ("dimension_checks", "5 stage BMPs: identical width, height, 24-bit"),
            ("rgb_bytes_equal", "5 stage BMPs: identical RGB byte count"),
            ("bmp_size_equality", "5 stage BMPs: identical file size")) if k in C], aligns=["<", "<"])
        self.sub("Decryption correctness")
        self.table(["result", "check"], [
            [_ok(C.get("lorenz_inverse")), "inverse Lorenz reproduces input"],
            [_ok(C.get("decrypted_equals_scrambled")), "decrypted == scrambled (differing bits: %s)" % rec.get("decrypted_vs_scrambled_bit_diff", "n/a")],
            [_ok(C.get("descrambled_equals_input")), "descrambled == input (differing bits: %s, differing values: %s)" % (
                rec.get("descrambled_vs_input_bit_diff", "n/a"), rec.get("different_values", "n/a"))],
            [_ok(C.get("hash_chain")), "SHA-256: decrypted = scrambled and descrambled = input"]], aligns=["<", "<"])
        self.sub("Metric validation")
        names = {"all_stage_shapes_equal": "all stage arrays have identical shape",
                 "histogram_counts_sum_to_pixels": "histogram counts sum to the pixel count (every stage and channel)",
                 "entropy_within_0_8_bits": "entropy within [0, 8] bits",
                 "correlations_within_minus1_1": "correlations within [-1, 1]",
                 "mse_non_negative_and_psnr_consistent": "MSE >= 0 and PSNR consistent with MSE",
                 "ssim_within_minus1_1": "SSIM within [-1, 1]",
                 "chi_square_sample_count_equals_values": "chi-square sample count == number of encrypted values",
                 "chi_square_p_value_in_0_1": "chi-square p-value in [0, 1]",
                 "final_mse_matches_pipeline_mse": "report MSE == pipeline MSE (independent computations)"}
        self.table(["result", "check"], [[_ok(v), names.get(k, k)] for k, v in (an.get("validation") or {}).items()],
                   aligns=["<", "<"])
        sens = img.get("sensitivity")
        if sens:
            self.line("    %s  NPCR/UACI computed only between ciphertexts of two controlled plaintexts (1-bit difference), same key" % _ok(True))
        self.sub("Errors or warnings")
        if warns:
            for w in warns:
                for i, p in enumerate(textwrap.wrap(w, self.W - 8) or [""]):
                    self.line("    %s %s" % ("-" if i == 0 else " ", p))
        else:
            self.line("    none")
        self.warnings_all.extend("%s: %s" % (img["rec"]["source_file"], w) for w in warns)

    # ------------------------------------------------------------------ final
    def finish(self, run, imgs, status, total_s, files):
        n = len(imgs)
        self.line()
        self.line(self.R1)
        self.section("08", "FINAL SUMMARY ACROSS ALL IMAGES")
        rows = []
        for im in imgs:
            rec, an = im["rec"], im.get("analysis") or {}
            st = (an.get("stages") or {}).get("encrypted")
            sA = (im.get("sensitivity") or {}).get("same_key_same_state")
            fin = (an.get("reconstruction") or {}).get(FINAL_PAIR)
            T = rec.get("timings_s", {})
            rows.append([rec["source_file"], "%dx%d" % tuple(rec["input_size"]) if "input_size" in rec else "n/a", rec["status"],
                         "%.3f" % T["bg_encryption"] if "bg_encryption" in T else "n/a",
                         "%.3f" % T["bg_decryption"] if "bg_decryption" in T else "n/a",
                         fnum(st["entropy"]["pooled"], 4) if st else "n/a",
                         "/".join(fnum(st["correlation"][d]["mean"], 3) if st else "n/a" for d in DIRECTIONS),
                         fnum(sA["npcr"]["mean"], 3) if sA else "n/a",
                         "INF" if fin and fin["psnr_db"] == math.inf else (fnum(fin["psnr_db"], 2) if fin else "n/a")])
        self.table(["image", "size", "status", "enc s", "dec s", "entropy", "corr H/V/D (enc)", "NPCR A %", "PSNR dB"], rows,
                   indent=2, aligns=["<", "<", "<", ">", ">", ">", ">", ">", ">"])
        npass = sum(i["rec"]["status"] == "PASS" for i in imgs)
        self.line()
        self.kv([("Images processed", n), ("PASS", npass), ("FAIL", n - npass),
                 ("Unsupported files skipped", run["unsupported"]), ("Total execution time", "%.2f s" % total_s),
                 ("Overall status", status)])
        if self.warnings_all:
            self.sub("All warnings (%d)" % len(self.warnings_all))
            for w in self.warnings_all:
                for i, p in enumerate(textwrap.wrap(w, self.W - 8) or [""]):
                    self.line("    %s %s" % ("-" if i == 0 else " ", p))
        if any((im.get("sensitivity") or {}).get("same_key_same_state", {}).get("npcr", {}).get("mean", 100) < 99.0 for im in imgs):
            self.sub("Interpretation note: NPCR/UACI")
            for p in textwrap.wrap(
                    "Case A keeps the key AND the initial state x0 fixed. The BG keystream depends only on (n, x0), never on the "
                    "plaintext, so flipping one plaintext bit flips exactly one ciphertext bit and NPCR/UACI are near zero. This is a "
                    "property of the stream-cipher construction (and of the Lorenz stage being a pure permutation), not a reporting error. "
                    "Cases B and Control reach roughly the ideal values only because a fresh x0 changes the whole keystream; the Control "
                    "row (identical plaintext) shows the plaintext plays no role in that number. Do not present case B as plaintext "
                    "sensitivity in a paper; report case A and discuss the limitation, or bind x0 to the plaintext hash if a "
                    "plaintext-dependent design is required.", self.W - 6):
                self.line("    " + p)
        self.sub("Files generated")
        self.kv([(k, v) for k, v in files.items()], indent=4, wrap=False)
        self.line()
        self.line(self.R1)
        self.line("END OF REPORT")
        self.line(self.R1)
        # rewrite the [01] block (shown as RUNNING at the start) with the final values in the saved file
        a, b = self._slot
        new = [(s, False) for s in self._kv_lines(self._summary_rows(run, status, total_s, n))]
        self.lines[a:b] = new

    # ------------------------------------------------------------------ files
    @staticmethod
    def paths(reports_dir, run_id, safe):
        tag = "SAFE" if safe else "RESEARCH"
        d = Path(reports_dir)
        return dict(txt=d / ("report_%s_%s.txt" % (run_id, tag)), json=d / ("results_%s_%s.json" % (run_id, tag)),
                    csv_summary=d / ("summary_%s_%s.csv" % (run_id, tag)), csv_metrics=d / ("metrics_long_%s_%s.csv" % (run_id, tag)))

    def write_all(self, paths, run, imgs):
        paths["txt"].write_text(self.text(), encoding="utf-8")
        write_json(paths["json"], run, imgs, self.safe)
        write_csv(paths["csv_summary"], paths["csv_metrics"], run, imgs)


# ----------------------------------------------------------------------------
# JSON / CSV
# ----------------------------------------------------------------------------
def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_clean(v) for v in o]
    if isinstance(o, bool) or o is None or isinstance(o, str):
        return o
    if isinstance(o, int):
        return o if abs(o) < 2 ** 53 else str(o)
    if isinstance(o, float):
        return o if math.isfinite(o) else ("INF" if o > 0 else ("-INF" if o < 0 else "NaN"))
    if hasattr(o, "item"):
        return _clean(o.item())
    return str(o)


def _bigjson(x):
    return dict(decimal=str(x), hex=hex(x), decimal_digits=len(str(abs(x))), bit_length=abs(x).bit_length())


def write_json(path, run, imgs, safe):
    keys = {"n": _bigjson(run["keys"]["n"]["value"])}
    if not safe:
        for k in ("p", "q", "a", "b"):
            keys[k] = _bigjson(run["keys"][k]["value"])
    else:
        for k in ("p", "q", "a", "b"):
            keys[k] = dict(hidden=True, decimal_digits=run["keys"][k]["digits"], bit_length=run["keys"][k]["bits"])
    out = dict(report_version=1, mode="safe" if safe else "research", run={k: run[k] for k in (
        "run_id", "started", "software", "args", "bg", "lorenz", "keygen_seconds", "key_checks", "n_images")},
        unit_tests=dict(ok=run["unit_tests"]["ok"], passed=run["unit_tests"]["pass_lines"]), key_material=keys, images=[])
    for im in imgs:
        V = im["values"]
        bgv = {}
        if "xfinal" in V:
            bgv["x_final"] = _bigjson(V["xfinal"])
            bgv["t"] = V.get("t")
        for k in SECRET_FIELDS:
            if k in V:
                bgv[k] = dict(hidden=True) if safe else _bigjson(V[k])
        out["images"].append(dict(file=im["rec"]["source_file"], base=im["base"], status=im["rec"]["status"],
                                  source=im["src"] if not safe else {k: v for k, v in im["src"].items() if k != "path"},
                                  record=im["rec"], bg_values=bgv, permutation=im.get("perm"), analysis=im.get("analysis"),
                                  sensitivity=im.get("sensitivity"), histograms_counts_RGB_0_255=im.get("histograms"),
                                  plots=(im.get("plots") or {}).get("files"), warnings=im.get("warnings")))
    Path(path).write_text(json.dumps(_clean(out), indent=2), encoding="utf-8")


def _flat(prefix, o, out):
    if isinstance(o, dict):
        for k, v in o.items():
            _flat("%s.%s" % (prefix, k) if prefix else str(k), v, out)
    elif isinstance(o, (list, tuple)):
        if len(o) <= 16 and all(not isinstance(x, (dict, list, tuple)) for x in o):
            out.append((prefix, json.dumps(_clean(list(o)))))
    else:
        out.append((prefix, _clean(o)))


def write_csv(path_summary, path_long, run, imgs):
    cols = ["run_id", "image", "status", "width", "height", "pixels", "rgb_bytes", "bg_blocks", "enc_time_s", "dec_time_s",
            "mse_final", "psnr_final_db", "ssim_final", "entropy_encrypted", "chi2", "chi2_dof", "chi2_p",
            "corr_enc_h", "corr_enc_v", "corr_enc_d", "corr_in_h", "corr_in_v", "corr_in_d",
            "npcr_A_mean", "uaci_A_mean", "npcr_B_mean", "uaci_B_mean"]
    with open(path_summary, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(cols)
        for im in imgs:
            rec, an = im["rec"], im.get("analysis") or {}
            st, T = an.get("stages") or {}, rec.get("timings_s", {})
            fin = (an.get("reconstruction") or {}).get(FINAL_PAIR) or {}
            cs = ((an.get("chi_square") or {}).get("pooled")) or {}
            sens = im.get("sensitivity") or {}
            g = lambda s, d: ((st.get(s) or {}).get("correlation") or {}).get(d, {}).get("mean")
            row = [run["run_id"], rec["source_file"], rec["status"], *(rec.get("input_size") or ["", ""]), rec.get("pixels"),
                   rec.get("rgb_bytes"), rec.get("bg_blocks"), T.get("bg_encryption"), T.get("bg_decryption"),
                   fin.get("mse"), fin.get("psnr_db"), (fin.get("ssim") or {}).get("mean"),
                   ((st.get("encrypted") or {}).get("entropy") or {}).get("pooled"), cs.get("chi2"), cs.get("dof"), cs.get("p_value"),
                   g("encrypted", "horizontal"), g("encrypted", "vertical"), g("encrypted", "diagonal"),
                   g("input", "horizontal"), g("input", "vertical"), g("input", "diagonal"),
                   ((sens.get("same_key_same_state") or {}).get("npcr") or {}).get("mean"),
                   ((sens.get("same_key_same_state") or {}).get("uaci") or {}).get("mean"),
                   ((sens.get("same_key_fresh_state") or {}).get("npcr") or {}).get("mean"),
                   ((sens.get("same_key_fresh_state") or {}).get("uaci") or {}).get("mean")]
            wr.writerow(["" if v is None else _clean(v) for v in row])
    with open(path_long, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["run_id", "image", "metric", "value"])
        for im in imgs:
            pairs = []
            _flat("analysis", im.get("analysis") or {}, pairs)
            _flat("sensitivity", im.get("sensitivity") or {}, pairs)
            _flat("timings_s", im["rec"].get("timings_s", {}), pairs)
            for k, v in pairs:
                wr.writerow([run["run_id"], im["rec"]["source_file"], k, v])
