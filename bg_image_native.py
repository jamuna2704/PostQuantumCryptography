#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bg_image_native.py - NATIVE-SIZE version (no resizing). Lorenz scrambling + standard Blum-Goldwasser (BG) image encryption.

FOLDERS (created automatically next to this script; source images are never modified)
    original/       your source pictures (PNG, JPG, JPEG, BMP, TIFF, WEBP)
    input/          every picture converted to RGB at its ORIGINAL width x height (24-bit BMP) = pipeline input
    scrambled/      input after 2D Lorenz pixel permutation
    encrypted/      scrambled after BG encryption (ciphertext bytes shown as a same-size BMP)
    decrypted/      encrypted after BG decryption     -> must equal scrambled bit-for-bit
    descrambled/    decrypted after inverse Lorenz    -> must equal input bit-for-bit
    metadata.json   BG final states x_(t+1), timings, hashes, statistics
    reports/        (new) full text report, JSON results, summary CSV, long-format metrics CSV for every run
    visualizations/ (new) histogram, correlation-scatter and comparison PNGs, one sub-folder per image

PIPELINE
    encrypt: original -> RGB (no resize) -> input -> Lorenz -> scrambled -> BG -> encrypted
    decrypt: encrypted (+ x_(t+1) from metadata.json) -> BG -> decrypted -> inverse Lorenz -> descrambled

INSTALL (Windows PowerShell):   py -m pip install --upgrade pillow numpy matplotlib
RUN:                            py bg_image_native.py
OPTIONS:                        --hide-secrets / --safe  shareable report: p, q, a, b, x0, d_p, d_q, u, v hidden everywhere
                                --test-only      run only the unit tests
                                --block-bytes N  BG block size in bytes (default 64 = 512 bits)
                                --prime-bits N   size of p and q (default 1024)
                                --terminal {full,compact,pager}   terminal verbosity (file always complete)
                                --unicode        box-drawing rules and en dashes (default: ASCII only)
                                --no-plots, --dpi N, --scatter-points N, --skip-sensitivity
                                --reports-dir DIR, --viz-dir DIR

BG REFERENCE AND STATE CONVENTION (Handbook of Applied Cryptography, Algorithm 8.56)
    Key:     p, q distinct primes, p = q = 3 (mod 4); n = p*q; a*p + b*q = 1.
    Encrypt: x0 = r^2 mod n.  For i = 1..t:  x_i = x_(i-1)^2 mod n;  p_i = h low bits of x_i;
             c_i = m_i XOR p_i.   Then x_(t+1) = x_t^2 mod n  (the stored final state).
    Decrypt: d_p = ((p+1)/4)^(t+1) mod (p-1), d_q likewise with q;  u = x_(t+1)^d_p mod p,
             v = x_(t+1)^d_q mod q;  x0 = v*a*p + u*b*q mod n; regenerate keystream; XOR.
    The exponent is t+1 because the stored state is x0^(2^(t+1)).

EXPERIMENTAL CHOICES THAT ARE NOT TEXTBOOK (disclose in the paper)
    * Block size h = 512 bits by default; BG's provable security only covers h = O(log log n)
      (about 11 bits for a 2048-bit n). --block-bytes 1 (8 bits) is inside that bound.
    * A final partial block (not occurring at 512x512, but supported) uses the lowest 8*r bits of x_t.
    * h-bit keystream values are serialised big-endian and XORed with the RGB bytes in image order.
    * Lorenz scrambling only permutes pixels; it is not the source of BG security.
    * No resizing. RGB conversion (alpha composited on white) is a preprocessing step; "exact
      recovery" means equality with the RGB input array, not the original PNG/JPG file bytes.
      EXIF orientation is ignored so width and height are never changed.
    * A final partial block occurs whenever width*height*3 is not a multiple of the block size.

REPORTING LAYER (new): bg_analysis.py (metrics), bg_visuals.py (plots), bg_report.py (formatting/files).
    The cryptographic functions below are unchanged. process_image only hands its in-memory arrays to the
    report (no reloading). The sensitivity test and the report use their OWN random generators, so the
    main encryption never consumes extra randomness.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import platform
import random
import secrets
import struct
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

try:
    import numpy as np
    from PIL import Image
    import PIL
except ImportError as exc:  # pragma: no cover
    print("Missing dependency: %s\nInstall with:  py -m pip install pillow numpy matplotlib" % exc)
    sys.exit(3)

import bg_analysis
import bg_report
import bg_visuals

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
Image.MAX_IMAGE_PIXELS = None   # allow very large legitimate images
SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
STAGE_DIRS = ["original", "input", "scrambled", "encrypted", "decrypted", "descrambled"]
METADATA_FILE = "metadata.json"
RESIZE_TO = None   # native version: NO resizing
BG_ALGORITHM = "Blum-Goldwasser, HAC Alg. 8.56 (stores x_(t+1); decrypt exponent ((p+1)/4)^(t+1))"

LORENZ_A = 0.95
LORENZ_H = 1.0          # Lorenz-map parameter "h"  (NOT the BG block size)
LORENZ_X1 = 0.11
LORENZ_X2 = 0.12
LORENZ_TRANSIENT = 1000
MAX_TIE_FRACTION = 1e-3  # Lorenz key sequence considered degenerate above this
SENS_MAX_BYTES = 64 * 1024 * 1024   # skip the extra sensitivity encryptions above this payload size


class StageError(Exception):
    """Raised with the pipeline/BG stage that failed."""

    def __init__(self, stage: str, message: str):
        super().__init__("[%s] %s" % (stage, message))
        self.stage = stage
        self.message = message


RULE = "=" * 72
SUBRULE = "-" * 72


# ----------------------------------------------------------------------------
# Number theory / key generation
# ----------------------------------------------------------------------------
_SMALL_PRIMES = [p for p in range(3, 2000) if all(p % d for d in range(2, int(p ** 0.5) + 1))]


def is_probable_prime(n: int, rng, rounds: int = 40) -> bool:
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    for sp in _SMALL_PRIMES:
        if n == sp:
            return True
        if n % sp == 0:
            return False
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for _ in range(rounds):
        a = rng.randrange(2, n - 1)
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def gen_blum_prime(bits: int, rng) -> int:
    """Random prime of exactly `bits` bits with p = 3 (mod 4)."""
    while True:
        cand = rng.getrandbits(bits) | (1 << (bits - 1)) | (1 << (bits - 2)) | 3
        if is_probable_prime(cand, rng):
            return cand


def egcd(a: int, b: int):
    """Return (g, x, y) with a*x + b*y = g."""
    x0, y0, x1, y1 = 1, 0, 0, 1
    while b:
        qq, a, b = a // b, b, a % b
        x0, x1 = x1, x0 - qq * x1
        y0, y1 = y1, y0 - qq * y1
    return a, x0, y0


@dataclass(frozen=True)
class PublicKey:
    n: int


@dataclass(frozen=True)
class PrivateKey:
    p: int
    q: int
    a: int   # Bezout: a*p + b*q = 1
    b: int

    def __repr__(self) -> str:  # never leak secrets in logs
        return "PrivateKey(<redacted>)"

    __str__ = __repr__


def make_private_key(p: int, q: int) -> tuple[PublicKey, PrivateKey]:
    g, a, b = egcd(p, q)
    if g != 1:
        raise StageError("key generation", "gcd(p,q) != 1")
    return PublicKey(p * q), PrivateKey(p, q, a, b)


def generate_keypair(prime_bits: int, rng) -> tuple[PublicKey, PrivateKey]:
    p = gen_blum_prime(prime_bits, rng)
    q = gen_blum_prime(prime_bits, rng)
    while q == p:
        q = gen_blum_prime(prime_bits, rng)
    pub, priv = make_private_key(p, q)
    validate_keypair(pub, priv)
    return pub, priv


def validate_keypair(pub: PublicKey, priv: PrivateKey) -> None:
    p, q = priv.p, priv.q
    if p == q:
        raise StageError("key generation", "p == q")
    if p % 4 != 3 or q % 4 != 3:
        raise StageError("key generation", "prime is not 3 mod 4")
    if pub.n != p * q:
        raise StageError("key generation", "n != p*q")
    if priv.a * p + priv.b * q != 1:
        raise StageError("key generation", "Bezout identity a*p+b*q=1 failed")


def generate_x0(pub: PublicKey, rng) -> int:
    """x0 = r^2 mod n with gcd(r, n) = 1: a quadratic residue mod p and mod q."""
    n = pub.n
    while True:
        r = rng.randrange(2, n - 1)
        if math.gcd(r, n) != 1:
            continue
        x0 = r * r % n
        if x0 > 1:
            return x0


def validate_x0(x0: int, pub: PublicKey, priv: PrivateKey) -> None:
    if not (1 < x0 < pub.n) or math.gcd(x0, pub.n) != 1:
        raise StageError("state generation", "x0 out of range or not coprime to n")
    # Euler criterion: x0 is a QR mod p and mod q
    if pow(x0, (priv.p - 1) // 2, priv.p) != 1 or pow(x0, (priv.q - 1) // 2, priv.q) != 1:
        raise StageError("state generation", "x0 is not a quadratic residue")


# ----------------------------------------------------------------------------
# BG core (bytes layer; block size B bytes, h = 8*B bits)
# ----------------------------------------------------------------------------
def block_layout(length: int, block_bytes: int) -> tuple[int, int]:
    """(t, final_block_length). t = 0 and final length 0 for empty input."""
    if length == 0:
        return 0, 0
    t = -(-length // block_bytes)
    return t, length - (t - 1) * block_bytes


def _check_block_size(n: int, block_bytes: int) -> None:
    if block_bytes < 1 or 8 * block_bytes >= n.bit_length():
        raise StageError("keystream extraction",
                         "block size %d bits must be < modulus bits (%d)" % (8 * block_bytes, n.bit_length()))


def bg_keystream(x0: int, n: int, length: int, block_bytes: int) -> tuple[bytes, int, int]:
    """Return (keystream of `length` bytes, final state x_(t+1), t)."""
    _check_block_size(n, block_bytes)
    full, rem = divmod(length, block_bytes)
    mask = (1 << (8 * block_bytes)) - 1
    out = bytearray()
    x = x0
    for _ in range(full):
        x = x * x % n                                    # x_i
        out += (x & mask).to_bytes(block_bytes, "big")  # lowest h bits of x_i
    if rem:
        x = x * x % n                                    # x_t (partial block)
        out += (x & ((1 << (8 * rem)) - 1)).to_bytes(rem, "big")
    x = x * x % n                                        # x_(t+1)
    t = full + (1 if rem else 0)
    return bytes(out), x, t


def _xor(a: bytes, b: bytes) -> bytes:
    if len(a) != len(b):
        raise StageError("byte packing/unpacking", "XOR length mismatch %d vs %d" % (len(a), len(b)))
    return np.bitwise_xor(np.frombuffer(a, np.uint8), np.frombuffer(b, np.uint8)).tobytes()


def bg_encrypt(data: bytes, pub: PublicKey, x0: int, block_bytes: int) -> tuple[bytes, int, int]:
    ks, xfinal, t = bg_keystream(x0, pub.n, len(data), block_bytes)
    return _xor(data, ks), xfinal, t


def recover_x0_detail(xfinal: int, t: int, pub: PublicKey, priv: PrivateKey) -> dict:
    p, q = priv.p, priv.q
    dp = pow((p + 1) // 4, t + 1, p - 1)
    dq = pow((q + 1) // 4, t + 1, q - 1)
    u = pow(xfinal % p, dp, p)
    v = pow(xfinal % q, dq, q)
    return dict(dp=dp, dq=dq, u=u, v=v, x0=(v * priv.a * p + u * priv.b * q) % pub.n)


def recover_x0(xfinal: int, t: int, pub: PublicKey, priv: PrivateKey) -> int:
    return recover_x0_detail(xfinal, t, pub, priv)["x0"]


def bg_decrypt(ct: bytes, xfinal: int, pub: PublicKey, priv: PrivateKey,
               block_bytes: int) -> tuple[bytes, int]:
    t, _ = block_layout(len(ct), block_bytes)
    x0 = recover_x0(xfinal, t, pub, priv)
    ks, xfinal2, t2 = bg_keystream(x0, pub.n, len(ct), block_bytes)
    if t2 != t or xfinal2 != xfinal:
        raise StageError("final-state handling",
                         "regenerated final state differs from stored x_(t+1) (wrong key/state/t)")
    return _xor(ct, ks), x0


# ---- textbook (integer-block) implementation used for Test A -----------------
def textbook_encrypt(blocks, h, n, x0):
    states, ks, cs = [], [], []
    x = x0
    for m in blocks:
        x = x * x % n
        states.append(x)
        pi = x & ((1 << h) - 1)
        ks.append(pi)
        cs.append(m ^ pi)
    x = x * x % n
    states.append(x)             # x_(t+1)
    return cs, states, ks


def textbook_decrypt(cs, xfinal, h, p, q, a, b):
    t = len(cs)
    n = p * q
    dp = pow((p + 1) // 4, t + 1, p - 1)
    dq = pow((q + 1) // 4, t + 1, q - 1)
    u = pow(xfinal % p, dp, p)
    v = pow(xfinal % q, dq, q)
    x0 = (v * a * p + u * b * q) % n
    ms, x = [], x0
    for c in cs:
        x = x * x % n
        ms.append(c ^ (x & ((1 << h) - 1)))
    return ms, x0


def naive_reference_encrypt(data: bytes, n: int, x0: int, block_bytes: int):
    """Independent bit-string implementation used only to cross-check bg_keystream."""
    out = bytearray()
    x = x0
    L = len(data)
    t = -(-L // block_bytes) if L else 0
    for i in range(t):
        x = pow(x, 2, n)
        blen = min(block_bytes, L - i * block_bytes)
        bits = format(x, "b").zfill(8 * blen)[-8 * blen:]
        ks = int(bits, 2).to_bytes(blen, "big")
        blk = data[i * block_bytes:i * block_bytes + blen]
        out += bytes(c ^ k for c, k in zip(blk, ks))
    x = pow(x, 2, n)
    return bytes(out), x


# ----------------------------------------------------------------------------
# BG unit tests
# ----------------------------------------------------------------------------
def test_textbook(log) -> None:
    p, q, n, a, b, x0, t, h = 499, 547, 272953, -57, 52, 159201, 5, 4
    if p * q != n or a * p + b * q != 1:
        raise StageError("key generation", "textbook p,q,n or Bezout coefficients inconsistent")
    if x0 != 399 * 399 % n:
        raise StageError("state generation", "textbook x0 != 399^2 mod n")
    if int(math.log2(math.log2(n))) != h:
        raise StageError("keystream extraction", "floor(lg lg n) != 4")
    m = [0b1001, 0b1100, 0b0001, 0b0000, 0b1100]
    exp_states = [180539, 193932, 245613, 130286, 40632, 139680]
    exp_p = [0b1011, 0b1100, 0b1101, 0b1110, 0b1000]
    exp_c = [0b0010, 0b0000, 0b1100, 0b1110, 0b0100]
    cs, states, ks = textbook_encrypt(m, h, n, x0)
    if states != exp_states:
        raise StageError("state generation", "states %s != reference %s" % (states, exp_states))
    if ks != exp_p:
        raise StageError("keystream extraction", "p_i %s != reference %s" % (ks, exp_p))
    if cs != exp_c:
        raise StageError("keystream extraction", "c_i %s != reference %s" % (cs, exp_c))
    if states[-1] != 139680:
        raise StageError("final-state handling", "x6 != 139680")
    ms, x0r = textbook_decrypt(cs, states[-1], h, p, q, a, b)
    if x0r != x0:
        raise StageError("CRT reconstruction", "recovered x0 %d != %d" % (x0r, x0))
    if ms != m:
        raise StageError("partial-block handling", "recovered plaintext blocks differ")
    log("Test A   textbook HAC example (p=499, q=547, n=272953, t=5, h=4): states, p_i, c_i, x6, x0, m_i all match")
    print()
    print("          i     x_i      p_i    m_i    c_i = m_i XOR p_i")
    for i in range(t):
        print("          %d  %7d     %s   %s   %s" % (i + 1, states[i], format(ks[i], "04b"),
                                                     format(m[i], "04b"), format(cs[i], "04b")))
    print("          final state x6 = %d | recovered x0 = %d (original x0 = %d)" % (states[-1], x0r, x0))
    print()

    # Same recurrence through the BYTE layer with 1-byte blocks (h=8) on the textbook key.
    pub, priv = make_private_key(p, q)
    data = bytes([0x9C, 0x1C, 0xC3, 0x00, 0x7F, 0xFF, 0x01])
    ct, xf, t2 = bg_encrypt(data, pub, x0, 1)
    cs8, st8, _ = textbook_encrypt(list(data), 8, n, x0)
    if list(ct) != cs8 or xf != st8[-1] or t2 != len(data):
        raise StageError("byte packing/unpacking", "byte layer (B=1) disagrees with integer reference")
    pt, x0b = bg_decrypt(ct, xf, pub, priv, 1)
    if pt != data or x0b != x0:
        raise StageError("CRT reconstruction", "byte-layer round trip failed on textbook key")
    log("Test A2  byte layer (B=1) on textbook key == integer reference; round trip OK")


def _diagnose(label, data, pub, priv, x0, B, ct, xf, t):
    ref_ct, ref_xf = naive_reference_encrypt(data, pub.n, x0, B)
    exp_t, _ = block_layout(len(data), B)
    if t != exp_t:
        raise StageError("partial-block handling", "%s: block count %d != %d" % (label, t, exp_t))
    if len(ct) != len(data):
        raise StageError("byte packing/unpacking", "%s: ciphertext length %d != %d" % (label, len(ct), len(data)))
    if ct != ref_ct:
        raise StageError("keystream extraction", "%s: ciphertext differs from independent reference" % label)
    if xf != ref_xf:
        raise StageError("final-state handling", "%s: final state differs from independent reference" % label)
    if not (0 < xf < pub.n):
        raise StageError("final-state handling", "%s: final state out of range" % label)
    if int(hex(xf), 16) != xf:
        raise StageError("final-state handling", "%s: state serialisation round trip failed" % label)
    p, q = priv.p, priv.q
    dp = pow((p + 1) // 4, t + 1, p - 1)
    dq = pow((q + 1) // 4, t + 1, q - 1)
    u, v = pow(xf % p, dp, p), pow(xf % q, dq, q)
    if u != x0 % p or v != x0 % q:
        raise StageError("decryption exponent calculation",
                         "%s: u/v residues do not equal x0 mod p / mod q (exponent t+1 inconsistent)" % label)
    x0r = recover_x0(xf, t, pub, priv)
    if x0r != x0:
        raise StageError("CRT reconstruction", "%s: residues right but CRT result wrong" % label)
    pt, x0b = bg_decrypt(ct, xf, pub, priv, B)
    if x0b != x0:
        raise StageError("CRT reconstruction", "%s: recovered state mismatch" % label)
    if pt != data:
        stage = "partial-block handling" if len(data) % B else "byte packing/unpacking"
        raise StageError(stage, "%s: recovered plaintext differs" % label)


def test_roundtrips(log, seed: int) -> None:
    rng = random.Random(seed)
    lengths = [0, 1, 2, 63, 64, 65, 127, 128, 129, 191, 192, 193, 640, 1000, 4097]
    lengths += [rng.randrange(0, 3000) for _ in range(10)]
    cases = 0
    for prime_bits, blocks in ((128, (1, 8, 16)), (256, (1, 8, 32)), (512, (1, 16, 64))):
        for _ in range(2):  # two independent keys per size
            pub, priv = generate_keypair(prime_bits, rng)
            for B in blocks:
                if 8 * B >= pub.n.bit_length():
                    continue
                for L in lengths:
                    x0 = generate_x0(pub, rng)
                    validate_x0(x0, pub, priv)
                    data = bytes(rng.getrandbits(8) for _ in range(L))
                    ct, xf, t = bg_encrypt(data, pub, x0, B)
                    _diagnose("bits=%d B=%d L=%d" % (prime_bits, B, L), data, pub, priv, x0, B, ct, xf, t)
                    cases += 1
    log("Test B   %d randomized round trips (empty/1/partial/64/65/multi-block, 6 keys, B in {1,8,16,32,64}) "
        "match independent bit-string reference; x0 and plaintext recovered exactly" % cases)


def test_failure_detection(log, seed: int) -> None:
    rng = random.Random(seed + 1)
    pub, priv = generate_keypair(256, rng)
    x0 = generate_x0(pub, rng)
    data = bytes(rng.getrandbits(8) for _ in range(200))
    ct, xf, t = bg_encrypt(data, pub, x0, 16)
    try:
        bg_decrypt(ct, (xf * 3 + 1) % pub.n, pub, priv, 16)
    except StageError:
        pass
    else:
        raise StageError("final-state handling", "corrupted final state was NOT detected")
    # wrong exponent (t instead of t+1) must NOT work: guards against silent exponent changes
    p, q = priv.p, priv.q
    bad = (pow(pow(xf % q, pow((q + 1) // 4, t, q - 1), q) * priv.a * p
               + pow(xf % p, pow((p + 1) // 4, t, p - 1), p) * priv.b * q, 1, pub.n))
    if bad == x0:
        raise StageError("decryption exponent calculation", "exponent t unexpectedly works; convention test is vacuous")
    log("Test C   corrupted final state detected; exponent t (instead of t+1) provably fails")


def test_bmp_and_lorenz(log) -> None:
    # BMP writer/reader
    import tempfile
    rng = np.random.default_rng(7)
    with tempfile.TemporaryDirectory() as td:
        for (w, h) in ((1, 1), (2, 3), (5, 4), (7, 7), (13, 2), (64, 33)):
            arr = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
            path = Path(td) / ("t_%dx%d.bmp" % (w, h))
            write_bmp24(path, arr)
            back, info = read_bmp24(path)
            if not np.array_equal(arr, back):
                raise StageError("BMP", "write/read mismatch for %dx%d" % (w, h))
            if path.stat().st_size != bmp_expected_size(w, h) or info["bpp"] != 24:
                raise StageError("BMP", "size/bit depth wrong for %dx%d" % (w, h))
            with Image.open(path) as im:   # independent decoder
                if im.mode != "RGB" or im.size != (w, h) or not np.array_equal(np.asarray(im), arr):
                    raise StageError("BMP", "Pillow decodes %dx%d differently" % (w, h))
    log("Test D   24-bit BMP writer/reader (row padding, header, Pillow cross-decode) OK")
    # Lorenz
    dims = [(1, 1), (1, 9), (9, 1), (3, 2), (7, 5), (100, 37), (257, 255), (640, 480)]
    dims.append((512, 512))
    if RESIZE_TO is None:
        dims.append((3000, 2000))
    for (w, h) in dims:
        arr = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        perm, info = lorenz_permutation(w * h)
        sc = arr.reshape(-1, 3)[perm].reshape(h, w, 3)
        back = inverse_scramble(sc, perm)
        if not np.array_equal(back, arr):
            raise StageError("Lorenz", "inverse failed for %dx%d" % (w, h))
        if w * h > 1 and np.array_equal(sc, arr) and w * h > 4:
            raise StageError("Lorenz", "permutation is identity for %dx%d" % (w, h))
    log("Test E   Lorenz permutation + exact inverse for %d sizes (incl. %s)" %
        (len(dims), "512x512" if RESIZE_TO is not None else "3000x2000"))


def run_unit_tests(seed: int) -> bool:
    lines = []

    def log(s):
        lines.append(s)
        print("  [PASS] " + s)
        sys.stdout.flush()

    print(RULE)
    print(" UNIT TESTS   (reproducible seed = %d)" % seed)
    print(RULE)
    print()
    t0 = time.perf_counter()
    try:
        test_textbook(log)
        test_roundtrips(log, seed)
        test_failure_detection(log, seed)
        test_bmp_and_lorenz(log)
    except StageError as e:
        print("  [FAIL] stage = %s\n         %s" % (e.stage, e.message))
        print("\n  UNIT TESTS FAILED - image processing will not run.")
        return False
    print("  RESULT: ALL UNIT TESTS PASS  (%.2f s)" % (time.perf_counter() - t0))
    return True


# ----------------------------------------------------------------------------
# BMP (24-bit uncompressed) writer / reader
# ----------------------------------------------------------------------------
def bmp_stride(w: int) -> int:
    return ((w * 3 + 3) // 4) * 4


def bmp_expected_size(w: int, h: int) -> int:
    return 14 + 40 + bmp_stride(w) * h


def write_bmp24(path: Path, arr: np.ndarray) -> None:
    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
        raise StageError("BMP", "array must be HxWx3 uint8")
    h, w, _ = arr.shape
    stride = bmp_stride(w)
    buf = np.zeros((h, stride), np.uint8)               # padding bytes = 0
    buf[:, :w * 3] = arr[::-1, :, ::-1].reshape(h, w * 3)   # bottom-up, BGR
    size = 54 + stride * h
    header = b"BM" + struct.pack("<IHHI", size, 0, 0, 54)
    header += struct.pack("<IiiHHIIiiII", 40, w, h, 1, 24, 0, stride * h, 2835, 2835, 0, 0)
    assert len(header) == 54
    with open(path, "wb") as f:
        f.write(header)
        f.write(buf.tobytes())


def read_bmp24(path: Path):
    raw = Path(path).read_bytes()
    if len(raw) < 54 or raw[:2] != b"BM":
        raise StageError("BMP", "%s: not a BMP" % path.name)
    fsize, _, _, off = struct.unpack("<IHHI", raw[2:14])
    (dib, w, h, planes, bpp, comp, isize, _, _, _, _) = struct.unpack("<IiiHHIIiiII", raw[14:54])
    if dib != 40 or planes != 1 or bpp != 24 or comp != 0 or off != 54:
        raise StageError("BMP", "%s: not an uncompressed 24-bit BITMAPINFOHEADER BMP" % path.name)
    top_down = h < 0
    h = abs(h)
    stride = bmp_stride(w)
    if fsize != len(raw) or len(raw) != 54 + stride * h:
        raise StageError("BMP", "%s: header size %d, file %d, expected %d" %
                         (path.name, fsize, len(raw), 54 + stride * h))
    rows = np.frombuffer(raw, np.uint8, count=stride * h, offset=54).reshape(h, stride)
    if np.any(rows[:, w * 3:]):
        raise StageError("BMP", "%s: non-zero padding bytes" % path.name)
    arr = rows[:, :w * 3].reshape(h, w, 3)[:, :, ::-1]
    if not top_down:
        arr = arr[::-1]
    info = dict(width=w, height=h, bpp=bpp, file_size=len(raw), header_bytes=54,
                row_stride=stride, compression=comp)
    return np.ascontiguousarray(arr), info


# ----------------------------------------------------------------------------
# 2D Lorenz permutation
#   x1(n+1) = (1 + A*hL) x1(n) - hL x1(n) x2(n)
#   x2(n+1) = (1 - hL) x1(n) + hL x1(n)^2
#   (hL = Lorenz parameter "h"; unrelated to the BG block size)
# ----------------------------------------------------------------------------
def lorenz_key_sequence(N: int, A=LORENZ_A, hL=LORENZ_H, x1=LORENZ_X1, x2=LORENZ_X2,
                        transient=LORENZ_TRANSIENT) -> np.ndarray:
    c1, c2 = 1.0 + A * hL, 1.0 - hL
    for _ in range(transient):
        x1, x2 = c1 * x1 - hL * x1 * x2, c2 * x1 + hL * x1 * x1
    from array import array
    seq = array("d")
    app = seq.append
    for _ in range(N):
        app(x1 + x2)                                   # ordering key s = x1 + x2
        x1, x2 = c1 * x1 - hL * x1 * x2, c2 * x1 + hL * x1 * x1
    key = np.frombuffer(seq, dtype=np.float64)
    if not np.isfinite(key).all() or not (math.isfinite(x1) and math.isfinite(x2)):
        raise StageError("Lorenz", "non-finite values produced by the map (N=%d)" % N)
    return key


def lorenz_permutation(N: int):
    """perm such that scrambled_pixel[i] = original_pixel[perm[i]]."""
    key = lorenz_key_sequence(N)
    perm = np.argsort(key, kind="stable")     # ties broken by index -> deterministic
    sk = key[perm]
    ties = int(np.count_nonzero(sk[1:] == sk[:-1])) if N > 1 else 0
    seen = np.zeros(N, dtype=bool)
    seen[perm] = True
    if not seen.all():
        raise StageError("Lorenz", "permutation is not a bijection")
    if N > 100 and ties / N > MAX_TIE_FRACTION:
        raise StageError("Lorenz", "degenerate key sequence: %d ties among %d values" % (ties, N))
    info = dict(N=N, ties=ties, key_min=float(key.min()), key_max=float(key.max()))
    return perm, info


def forward_scramble(arr: np.ndarray, perm: np.ndarray) -> np.ndarray:
    h, w, _ = arr.shape
    return np.ascontiguousarray(arr.reshape(-1, 3)[perm].reshape(h, w, 3))


def inverse_scramble(scr: np.ndarray, perm: np.ndarray) -> np.ndarray:
    h, w, _ = scr.shape
    out = np.empty_like(scr.reshape(-1, 3))
    out[perm] = scr.reshape(-1, 3)
    return out.reshape(h, w, 3)


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def shannon_entropy(arr: np.ndarray) -> float:
    counts = np.bincount(arr.reshape(-1), minlength=256).astype(np.float64)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def neighbor_corr(arr: np.ndarray, direction: str, samples: int = 30000, seed: int = 12345):
    h, w, _ = arr.shape
    if direction == "horizontal":
        if w < 2:
            return None
        a, b = arr[:, :-1, :], arr[:, 1:, :]
    elif direction == "vertical":
        if h < 2:
            return None
        a, b = arr[:-1, :, :], arr[1:, :, :]
    else:
        if h < 2 or w < 2:
            return None
        a, b = arr[:-1, :-1, :], arr[1:, 1:, :]
    total = a.size
    rng = np.random.default_rng(seed)
    flat = rng.integers(0, total, size=min(samples, total)) if total > samples else np.arange(total)
    idx = np.unravel_index(flat, a.shape)
    x, y = a[idx].astype(np.float64), b[idx].astype(np.float64)
    if x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ----------------------------------------------------------------------------
# Per-image pipeline
# ----------------------------------------------------------------------------
def to_rgb(im: Image.Image) -> Image.Image:
    """Any Pillow mode -> 8-bit RGB (alpha composited on white, 16/32-bit scaled to 8-bit)."""
    if im.mode in ("I;16", "I;16L", "I;16B", "I", "F"):
        a = np.asarray(im, dtype=np.float64)
        lo, hi = float(a.min()), float(a.max())
        a8 = np.zeros(a.shape, np.uint8) if hi == lo else np.round((a - lo) * 255.0 / (hi - lo)).astype(np.uint8)
        return Image.fromarray(a8, "L").convert("RGB")
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        bg.alpha_composite(rgba)
        return bg.convert("RGB")
    return im.convert("RGB")


def load_and_resize(path: Path, size: int):
    with Image.open(path) as im:
        im.load()
        fmt, mode, src_size = im.format, im.mode, im.size
        rgb = to_rgb(im)
    if size is not None and rgb.size != (size, size):
        rgb = rgb.resize((size, size), Image.Resampling.LANCZOS)
    arr = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
    if arr.shape != (rgb.size[1], rgb.size[0], 3) or (size is not None and arr.shape != (size, size, 3)):
        raise StageError("load", "unexpected array shape %s" % (arr.shape,))
    return arr, fmt, mode, src_size


def bit_diff(a: bytes, b: bytes) -> int:
    if len(a) != len(b):
        return 8 * max(len(a), len(b))
    x = np.bitwise_xor(np.frombuffer(a, np.uint8), np.frombuffer(b, np.uint8))
    return int(np.unpackbits(x).sum())


def save_meta(root: Path, meta: dict) -> None:
    (root / METADATA_FILE).write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


def process_image(path: Path, base: str, ctx: dict) -> dict:
    root, B, S = ctx["root"], ctx["block_bytes"], ctx["size"]
    priv, rng, meta = ctx["priv"], ctx["rng"], ctx["meta"]
    pub = PublicKey(int(meta["bg_public"]["n_hex"], 16))
    V = ctx["values"].setdefault(base, {})          # secret values: terminal only, never in metadata.json
    A = ctx.setdefault("arrays", {}).setdefault(base, {})   # in-memory arrays handed to the report (no reloading)
    rec = {"source_file": path.name, "base": base, "status": "FAIL", "failed_stage": None,
           "checks": {}, "timings_s": {}}
    T, C = rec["timings_s"], rec["checks"]
    meta["images"][base] = rec
    stage = "load"
    t_all = time.perf_counter()
    P = {s: root / s / (base + ".bmp") for s in ("input", "scrambled", "encrypted", "decrypted", "descrambled")}
    try:
        t0 = time.perf_counter()
        arr, src_fmt, src_mode, src_size = load_and_resize(path, S)
        T["resize"] = time.perf_counter() - t0
        A["input"] = arr
        h, w, _ = arr.shape
        N, L = w * h, w * h * 3
        t_blocks, last_len = block_layout(L, B)
        rec.update(source_format=src_fmt, source_mode=src_mode, source_size=list(src_size),
                   input_size=[w, h], pixels=N, rgb_bytes=L, rgb_bits=8 * L,
                   bg_block_bytes=B, bg_blocks=t_blocks, bg_final_block_bytes=last_len)
        if S is None:
            C["source_dims_preserved"] = (w, h) == tuple(src_size)
            if not C["source_dims_preserved"]:
                raise StageError("load", "native dimensions changed during RGB conversion")
        else:
            C["resize_target_met"] = (w, h) == (S, S)
        stage = "input BMP"
        write_bmp24(P["input"], arr)
        back, _ = read_bmp24(P["input"])
        C["input_bmp_readback"] = bool(np.array_equal(back, arr))
        if not C["input_bmp_readback"]:
            raise StageError(stage, "input BMP does not read back identical")

        stage = "Lorenz scrambling"
        t0 = time.perf_counter()
        perm, linfo = lorenz_permutation(N)
        T["lorenz_generation"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        scr = forward_scramble(arr, perm)
        T["scrambling"] = time.perf_counter() - t0
        A["scrambled"], A["perm"] = scr, perm
        C["lorenz_inverse"] = bool(np.array_equal(inverse_scramble(scr, perm), arr))
        if not C["lorenz_inverse"]:
            raise StageError(stage, "inverse permutation does not reproduce the input")
        rec["lorenz"] = dict(A=LORENZ_A, h_lorenz=LORENZ_H, x1_initial=LORENZ_X1, x2_initial=LORENZ_X2,
                             transient=LORENZ_TRANSIENT, ordering="stable argsort of x1+x2", **linfo)
        write_bmp24(P["scrambled"], scr)
        plain = scr.tobytes()

        stage = "BG encryption"
        x0 = generate_x0(pub, rng)
        validate_x0(x0, pub, priv)
        C["x0_quadratic_residue"] = True
        V["x0"] = x0
        t0 = time.perf_counter()
        ct, xfinal, t_used = bg_encrypt(plain, pub, x0, B)
        T["bg_encryption"] = time.perf_counter() - t0
        V["xfinal"], V["t"] = xfinal, t_used
        if t_used != t_blocks or len(ct) != L:
            raise StageError("partial-block handling", "block count / ciphertext length mismatch")
        write_bmp24(P["encrypted"], np.frombuffer(ct, np.uint8).reshape(h, w, 3))
        A["encrypted"] = np.frombuffer(ct, np.uint8).reshape(h, w, 3)
        rec["bg"] = dict(final_state_index="x_(t+1)", final_state_hex=hex(xfinal), blocks_t=t_used,
                         block_bytes=B, payload_bytes=L, byte_order="big-endian")
        save_meta(root, meta)

        stage = "BG decryption"
        ct_file = read_bmp24(P["encrypted"])[0].tobytes()
        stored = json.loads((root / METADATA_FILE).read_text(encoding="utf-8"))["images"][base]["bg"]
        xf_file = int(stored["final_state_hex"], 16)
        t0 = time.perf_counter()
        dec, x0_rec = bg_decrypt(ct_file, xf_file, pub, priv, stored["block_bytes"])
        T["bg_decryption"] = time.perf_counter() - t0
        det = recover_x0_detail(xf_file, stored["blocks_t"], pub, priv)
        V.update(x0_rec=x0_rec, dp=det["dp"], dq=det["dq"], u=det["u"], v=det["v"])
        C["crt_residues"] = (det["u"] == x0 % priv.p and det["v"] == x0 % priv.q)
        C["bg_state_recovery"] = (x0_rec == x0)
        if not C["bg_state_recovery"]:
            raise StageError("CRT reconstruction", "recovered x0 != original x0")
        dec_arr = np.frombuffer(dec, np.uint8).reshape(h, w, 3).copy()
        A["decrypted"] = dec_arr
        write_bmp24(P["decrypted"], dec_arr)
        dec_bits = bit_diff(read_bmp24(P["decrypted"])[0].tobytes(), read_bmp24(P["scrambled"])[0].tobytes())
        rec["decrypted_vs_scrambled_bit_diff"] = dec_bits
        C["decrypted_equals_scrambled"] = (dec_bits == 0)
        if dec_bits:
            raise StageError("BG decryption", "%d bits differ between decrypted and scrambled" % dec_bits)

        stage = "inverse Lorenz"
        t0 = time.perf_counter()
        perm2, _ = lorenz_permutation(N)
        T["lorenz_regeneration"] = time.perf_counter() - t0
        if not np.array_equal(perm, perm2):
            raise StageError(stage, "regenerated permutation differs")
        t0 = time.perf_counter()
        desc = inverse_scramble(dec_arr, perm2)
        T["descrambling"] = time.perf_counter() - t0
        A["descrambled"] = desc
        write_bmp24(P["descrambled"], desc)
        file_in = read_bmp24(P["input"])[0]
        file_desc = read_bmp24(P["descrambled"])[0]
        desc_bits = bit_diff(file_desc.tobytes(), file_in.tobytes())
        mse = float(np.mean((file_desc.astype(np.float64) - file_in.astype(np.float64)) ** 2))
        rec.update(descrambled_vs_input_bit_diff=desc_bits,
                   different_values=int(np.count_nonzero(file_desc != file_in)),
                   mse=mse, psnr="INF" if mse == 0 else float(10 * math.log10(255.0 ** 2 / mse)))
        C["descrambled_equals_input"] = (desc_bits == 0 and mse == 0)

        stage = "BMP verification"
        exp = bmp_expected_size(w, h)
        info = {}
        for name, p in P.items():
            a, inf = read_bmp24(p)
            with Image.open(p) as im:
                if im.mode != "RGB" or im.size != (w, h) or not np.array_equal(np.asarray(im), a):
                    raise StageError(stage, "%s: Pillow decode mismatch" % name)
            info[name] = inf
        rec["bmp_stages"] = info
        C["bmp_size_equality"] = {v["file_size"] for v in info.values()} == {exp}
        C["dimension_checks"] = {(v["width"], v["height"], v["bpp"]) for v in info.values()} == {(w, h, 24)}
        C["rgb_bytes_equal"] = {v["width"] * v["height"] * 3 for v in info.values()} == {L}
        sizes = [dict(stage="original", width=src_size[0], height=src_size[1], pixels=src_size[0] * src_size[1],
                      rgb_bytes=src_size[0] * src_size[1] * 3, file_bytes=path.stat().st_size,
                      kind="%s, mode %s" % (src_fmt, src_mode))]
        for name in ("input", "scrambled", "encrypted", "decrypted", "descrambled"):
            v = info[name]
            sizes.append(dict(stage=name, width=v["width"], height=v["height"], pixels=v["width"] * v["height"],
                              rgb_bytes=v["width"] * v["height"] * 3, file_bytes=v["file_size"], kind="BMP 24-bit"))
        rec["sizes"] = sizes
        rec["bmp_layout"] = dict(header_bytes=54, row_stride=info["input"]["row_stride"],
                                 padding_per_row=info["input"]["row_stride"] - w * 3, rows=h)
        rec["bmp_file_size"] = exp

        rec["hashes_sha256"] = dict(input=sha256(file_in.tobytes()), scrambled=sha256(plain), encrypted=sha256(ct),
                                    decrypted=sha256(dec), descrambled=sha256(file_desc.tobytes()))
        H = rec["hashes_sha256"]
        C["hash_chain"] = (H["decrypted"] == H["scrambled"]) and (H["descrambled"] == H["input"])
        ct_arr = np.frombuffer(ct, np.uint8).reshape(h, w, 3)
        rec["entropy_encrypted_bits_per_byte"] = shannon_entropy(ct_arr)
        rec["correlation_encrypted"] = {d: neighbor_corr(ct_arr, d) for d in ("horizontal", "vertical", "diagonal")}
        rec["correlation_input"] = {d: neighbor_corr(arr, d) for d in ("horizontal", "vertical", "diagonal")}

        if all(C.values()):
            rec["status"] = "PASS"
        else:
            rec["failed_stage"] = [k for k, v in C.items() if not v]
    except StageError as e:
        rec["failed_stage"], rec["error"] = e.stage, e.message
    except Exception as e:
        rec["failed_stage"], rec["error"] = stage, "%s: %s" % (type(e).__name__, e)
    T["total"] = time.perf_counter() - t_all
    save_meta(root, meta)
    return rec


# ----------------------------------------------------------------------------
# Reporting support (analysis only - uses its OWN random generator, never the encryption one)
# ----------------------------------------------------------------------------
def plaintext_sensitivity(A: dict, pub: PublicKey, priv: PrivateKey, x0: int, block_bytes: int, rng) -> dict:
    """NPCR/UACI from two controlled plaintexts P1, P2 (P2 = P1 with ONE bit flipped), same key and Lorenz
    permutation. Cases: A same x0 | B fresh x0 | Control: same plaintext, fresh x0."""
    arr, scr1, ct1, perm = A["input"], A["scrambled"], A["encrypted"], A["perm"]
    h, w, _ = arr.shape
    y, x, ch = h // 2, w // 2, 0
    old = int(arr[y, x, ch])
    arr2 = arr.copy()
    arr2[y, x, ch] = old ^ 1
    scr2 = forward_scramble(arr2, perm)
    del arr2
    plain1, plain2 = scr1.tobytes(), scr2.tobytes()
    bits = bit_diff(plain1, plain2)
    ct2a, _, _ = bg_encrypt(plain2, pub, x0, block_bytes)                 # case A: same key, same x0
    x0b = generate_x0(pub, rng)                                            # fresh, independent state
    validate_x0(x0b, pub, priv)
    ks, _, _ = bg_keystream(x0b, pub.n, len(plain1), block_bytes)
    ct1b, ct2b = _xor(plain1, ks), _xor(plain2, ks)
    shp = (h, w, 3)
    fr = lambda b: np.frombuffer(b, np.uint8).reshape(shp)
    return dict(modification=dict(pixel_x=x, pixel_y=y, channel="R", old=old, new=old ^ 1, plaintext_bits_changed=bits),
                fresh_state_differs=(x0b != x0),
                same_key_same_state=bg_analysis.npcr_uaci(ct1, fr(ct2a)),
                same_key_fresh_state=bg_analysis.npcr_uaci(ct1, fr(ct2b)),
                control_same_plaintext_fresh_state=bg_analysis.npcr_uaci(ct1, fr(ct1b)))


def build_run_keys(pub: PublicKey, priv: PrivateKey, safe: bool) -> dict:
    ii = bg_report.int_info
    return dict(p=ii(priv.p, True, safe), q=ii(priv.q, True, safe), n=ii(pub.n, False, safe),
                a=ii(priv.a, True, safe), b=ii(priv.b, True, safe))


def key_checks(pub: PublicKey, priv: PrivateKey, prime_bits: int, block_bytes: int) -> list:
    vr = random.Random()      # separate generator: re-testing primality must not touch the main RNG
    p, q = priv.p, priv.q
    return [("p mod 4 == 3", p % 4 == 3), ("q mod 4 == 3", q % 4 == 3), ("p != q", p != q),
            ("gcd(p, q) == 1", math.gcd(p, q) == 1), ("n == p*q", pub.n == p * q),
            ("a*p + b*q == 1 (Bezout)", priv.a * p + priv.b * q == 1),
            ("p is a probable prime (Miller-Rabin, 40 fresh rounds)", is_probable_prime(p, vr)),
            ("q is a probable prime (Miller-Rabin, 40 fresh rounds)", is_probable_prime(q, vr)),
            ("p has exactly %d bits" % prime_bits, p.bit_length() == prime_bits),
            ("q has exactly %d bits" % prime_bits, q.bit_length() == prime_bits),
            ("n bit length in {2*bits-1, 2*bits}", pub.n.bit_length() in (2 * prime_bits - 1, 2 * prime_bits)),
            ("block size h (%d bits) < modulus bits" % (8 * block_bytes), 8 * block_bytes < pub.n.bit_length())]


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(argv=None) -> int:
    t_run = time.perf_counter()
    ap = argparse.ArgumentParser(description="Lorenz + Blum-Goldwasser image encryption with research report")
    ap.add_argument("--root", type=Path, default=PROJECT_ROOT, help="project folder (default: script folder)")
    ap.add_argument("--prime-bits", type=int, default=1024)
    ap.add_argument("--block-bytes", type=int, default=64, help="BG block size in bytes (64 = 512 bits)")
    if RESIZE_TO is not None:
        ap.add_argument("--size", type=int, default=RESIZE_TO, help="resize every image to SIZE x SIZE")
    ap.add_argument("--hide-secrets", "--safe", dest="hide_secrets", action="store_true",
                    help="safe/shareable mode: hide p, q, a, b, x0, d_p, d_q, u, v in terminal AND report files")
    ap.add_argument("--test-only", action="store_true")
    ap.add_argument("--test-seed", type=int, default=20240601)
    ap.add_argument("--terminal", choices=("full", "compact", "pager"), default="full",
                    help="terminal verbosity; the saved report is always complete")
    ap.add_argument("--unicode", action="store_true", help="Unicode rules/dashes (default ASCII-only)")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--scatter-points", type=int, default=8000, help="pairs drawn per scatter panel (statistics use ALL pairs)")
    ap.add_argument("--skip-sensitivity", action="store_true", help="skip the NPCR/UACI controlled-pair test")
    ap.add_argument("--reports-dir", type=Path, default=None)
    ap.add_argument("--viz-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    args.size = getattr(args, "size", None)
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass

    if args.test_only:
        return 0 if run_unit_tests(args.test_seed) else 2
    if 8 * args.block_bytes >= 2 * args.prime_bits - 1:
        print("ERROR: block size must be smaller than the modulus.")
        return 2

    root = args.root.resolve()
    for d in STAGE_DIRS:
        (root / d).mkdir(parents=True, exist_ok=True)
    reports_dir = (args.reports_dir or root / "reports").resolve()
    viz_dir = (args.viz_dir or root / "visualizations").resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_plots:
        viz_dir.mkdir(parents=True, exist_ok=True)

    junk = {"thumbs.db", "desktop.ini", ".ds_store"}
    files = sorted((p for p in (root / "original").iterdir()
                    if p.is_file() and not p.name.startswith(".") and p.name.lower() not in junk),
                   key=lambda p: p.name.lower())
    inputs = [p for p in files if p.suffix.lower() in SUPPORTED_EXT]
    unsupported = [p for p in files if p.suffix.lower() not in SUPPORTED_EXT]
    for p in unsupported:
        print("UNSUPPORTED (not processed): %s" % p.name)
    if not inputs:
        print("No supported images found. Put pictures in:\n  %s\nand run again." % (root / "original"))
        return 1

    safe = args.hide_secrets
    print("Running self-tests (textbook example, randomized round trips, BMP, Lorenz) ...")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tests_ok = run_unit_tests(args.test_seed)
    log_text = buf.getvalue()
    if not tests_ok:
        print(log_text)
        return 2
    pass_lines = [l.strip()[len("[PASS] "):] for l in log_text.splitlines() if l.strip().startswith("[PASS]")]

    software = dict(python=sys.version.split()[0], platform=platform.platform(), numpy=np.__version__,
                    pillow=PIL.__version__, matplotlib=bg_visuals.MPL_VERSION or "not installed")
    sysrng = random.SystemRandom()
    t0 = time.perf_counter()
    pub, priv = generate_keypair(args.prime_bits, sysrng)
    t_key = time.perf_counter() - t0

    started = datetime.now().astimezone()
    run = dict(run_id=started.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2),
               started=started.strftime("%Y-%m-%d %H:%M:%S") + " (UTC" + started.strftime("%z")[:3] + ":" + started.strftime("%z")[3:] + ")", software=software,
               args=dict(prime_bits=args.prime_bits, block_bytes=args.block_bytes, terminal=args.terminal,
                         safe_mode=safe, dpi=args.dpi, scatter_points=args.scatter_points,
                         plots=not args.no_plots, sensitivity=not args.skip_sensitivity),
               bg=dict(prime_bits=args.prime_bits, modulus_bits=pub.n.bit_length(), block_bytes=args.block_bytes,
                       block_bits=8 * args.block_bytes, textbook_bound=int(math.log2(pub.n.bit_length())),
                       byte_order="big-endian"),
               lorenz=dict(A=LORENZ_A, h=LORENZ_H, x1_0=LORENZ_X1, x2_0=LORENZ_X2, transient=LORENZ_TRANSIENT),
               keys=build_run_keys(pub, priv, safe), keygen_seconds=t_key,
               key_checks=key_checks(pub, priv, args.prime_bits, args.block_bytes),
               unit_tests=dict(ok=True, pass_lines=pass_lines), n_images=len(inputs), unsupported=len(unsupported))

    rep = bg_report.Reporter(safe=safe, unicode_ok=args.unicode, terminal=args.terminal)
    rep.header(run)
    if 8 * args.block_bytes > run["bg"]["textbook_bound"]:
        rep.line()
        rep.line("  NOTE: block size h = %d bits exceeds the textbook provable-security bound (%d bits); disclose in the paper." % (
            8 * args.block_bytes, run["bg"]["textbook_bound"]))

    meta = dict(software=software, bg_algorithm=BG_ALGORITHM,
                bg_public=dict(n_hex=hex(pub.n), modulus_bits=pub.n.bit_length(), block_bytes=args.block_bytes,
                               block_bits=8 * args.block_bytes, keygen_seconds=t_key),
                resize=(dict(size=args.size, method="Lanczos") if args.size else dict(enabled=False)),
                unsupported_files=[p.name for p in unsupported], images={})
    ctx = dict(root=root, block_bytes=args.block_bytes, size=args.size, priv=priv, rng=sysrng, meta=meta,
               values={}, arrays={})
    sens_rng = random.SystemRandom()      # independent of the encryption RNG
    imgs, used = [], set()
    t_batch = time.perf_counter()
    for i, p in enumerate(inputs, 1):
        base = "%s__%s" % (p.stem, p.suffix.lower().lstrip("."))
        k = 2
        while base.lower() in used:                 # unique output names (case-insensitive filesystems)
            base = "%s__%s_%d" % (p.stem, p.suffix.lower().lstrip("."), k)
            k += 1
        used.add(base.lower())
        rec = process_image(p, base, ctx)
        A = ctx["arrays"].pop(base, {})
        V = ctx["values"].get(base, {})
        img = dict(index=i, base=base, rec=rec, values=V, have=set(k for k in A if k != "perm"), warnings=[], timings={},
                   src=dict(path=str(p.resolve()), bytes=p.stat().st_size, sha256=bg_analysis.sha256_file(p)),
                   paths={s: str(root / s / (base + ".bmp")) for s in STAGE_DIRS[1:]})
        try:
            t0 = time.perf_counter()
            if "perm" in A:
                img["perm"] = bg_analysis.array_summary(A["perm"])
            img["analysis"], hist = bg_analysis.analyze_image(A, rec)
            img["histograms"] = {s: h.tolist() for s, h in hist.items()}
            img["warnings"] += img["analysis"]["warnings"]
            img["timings"]["analysis"] = time.perf_counter() - t0
            if not args.skip_sensitivity and rec["status"] == "PASS":
                if rec["rgb_bytes"] > SENS_MAX_BYTES:
                    img["warnings"].append("NPCR/UACI skipped: payload %s bytes exceeds %s (use --skip-sensitivity to silence)" % (
                        format(rec["rgb_bytes"], ","), format(SENS_MAX_BYTES, ",")))
                else:
                    img["sensitivity"] = plaintext_sensitivity(A, pub, priv, V["x0"], args.block_bytes, sens_rng)
                    sA = img["sensitivity"]["same_key_same_state"]["npcr"]["mean"]
                    if sA < 99.0:
                        img["warnings"].append(
                            "NPCR (case A, same key and x0) = %.6f %%, far below the ideal %.4f %%: see interpretation note "
                            "in the final summary." % (sA, bg_analysis.NPCR_IDEAL))
            elif not args.skip_sensitivity:
                img["warnings"].append("NPCR/UACI not computed: pipeline did not PASS")
        except Exception as e:
            img["analysis"] = img.get("analysis") or dict(error="%s: %s" % (type(e).__name__, e))
            img["warnings"].append("analysis failed: %s: %s" % (type(e).__name__, e))
            traceback.print_exc(file=sys.stderr)
            hist = {}
        if not args.no_plots and hist and "stages" in (img["analysis"] or {}):
            t0 = time.perf_counter()
            img["plots"] = bg_visuals.make_plots(base, A, img["analysis"], hist, viz_dir, args.dpi, args.scatter_points)
            img["warnings"] += img["plots"]["warnings"]
            img["timings"]["plots"] = time.perf_counter() - t0
        rep.image_report(i, len(inputs), img)
        imgs.append(img)
        del A, hist

    npass = sum(r["rec"]["status"] == "PASS" for r in imgs)
    status = "PASS" if npass == len(imgs) else "FAIL (%d of %d images failed)" % (len(imgs) - npass, len(imgs))
    meta["batch"] = dict(images=len(imgs), passed=npass, failed=len(imgs) - npass, seconds=time.perf_counter() - t_batch)
    save_meta(root, meta)
    paths = bg_report.Reporter.paths(reports_dir, run["run_id"], safe)
    nplots = sum(len((i.get("plots") or {}).get("files", [])) for i in imgs)
    out_files = {"text report": paths["txt"], "JSON results": paths["json"], "summary CSV": paths["csv_summary"],
                 "long-format metrics CSV": paths["csv_metrics"], "plots": "%d PNG files in %s" % (nplots, viz_dir if not args.no_plots else "(disabled)"),
                 "pipeline metadata": root / METADATA_FILE, "stage folders": ", ".join(STAGE_DIRS[1:])}
    rep.finish(run, imgs, status, time.perf_counter() - t_run, {k: str(v) for k, v in out_files.items()})
    rep.write_all(paths, run, imgs)
    if args.terminal == "pager":
        import pydoc
        pydoc.pager(rep.text())
    return 0 if npass == len(imgs) else 1


if __name__ == "__main__":
    sys.exit(main())
