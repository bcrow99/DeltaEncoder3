#!/usr/bin/env python3
"""
comparison.py -- Comparison.java for Python: a rate-distortion comparison of
JPEG and the delta coder, both made from the same lossless original and
both measured against it.

    python3 comparison.py image.png [scale] [display width]
      scale          1 = full size (default), 2 = half size (faster), ...
      display width  width the image is viewed at, for the "display" columns
                     (default 1024). The original and each decoded image are
                     shrunk to it (box average) before comparing, so detail
                     too fine to see there no longer counts as error.

JPEG: qualities 5-95. Delta coder: Color Resolution 0-6 x Pixel Resolution
0-10, block map (size 12, Scanline 16) with Context coding, sized as
block_writer saves it.

The output matches the Java's. The delta-coder results are the same
computation. For JPEG, Java's ImageIO and libjpeg (used here through Pillow
to encode and OpenCV to decode) do the same arithmetic; the one difference
is how they scale the quantization tables for a quality setting, so this
builds Java's tables (java_qtables) and gives them to the encoder. With
them the coded data is byte-for-byte the Java's, so sizes and errors match.

Pillow is optional: without it this falls back to OpenCV's own quality
scaling, which gives the same tables at most qualities but not all (q30, for
one), so a few JPEG rows then differ slightly from the Java's.
"""

import math
import sys

import cv2
import numpy as np

try:
    from PIL import Image as _PILImage
except ImportError:                     # optional: see the module docstring
    _PILImage = None

import code_mapper as cm
import delta_mapper as dm
import resize_mapper as rm
import viewer_support as vs

GUIDE = (
    "How to read this:\n"
    "  bpp      bits per pixel of the file (file bytes x 8 / pixels). Smaller = smaller file.\n"
    "  PSNR     error against the original, in dB. Higher = closer. Roughly: 30 visible damage,\n"
    "           35 good, 40 hard to tell apart, 50+ near-perfect. 99 = identical.\n"
    "  SSIM     structural similarity, 0-1. 1 = identical; above 0.95 is usually very good.\n"
    "  max err  largest error of any pixel value (0-255).\n"
    "  blocky   how much stronger edges are on the 8-pixel grid than elsewhere; 1.00 = no blocks.\n"
    "  display  the same PSNR/SSIM after shrinking to the display width.\n")


# =============================================================================
# Images
# =============================================================================

class Img:
    """Three channels in the toolkit's order (red, green, blue), each a flat
    int64 array, w x h."""

    def __init__(self, w, h, c):
        self.w, self.h, self.c = w, h, c

    @staticmethod
    def from_rgb(rgb, scale):
        """From an (h, w, 3) uint8 RGB array, box-averaged down by scale
        (rounded)."""
        H, W = rgb.shape[:2]
        w, h = W // scale, H // scale
        a = rgb[:h * scale, :w * scale].astype(np.int64)
        n = scale * scale
        s = a.reshape(h, scale, w, scale, 3).sum(axis=(1, 3))
        avg = (s + n // 2) // n
        return Img(w, h, [avg[:, :, k].reshape(-1).copy() for k in range(3)])

    def shrink(self, factor):
        """Box average over factor x factor squares (rounded)."""
        if factor == 1:
            return self
        w, h = self.w // factor, self.h // factor
        n = factor * factor
        c = []
        for ch in self.c:
            a = ch.reshape(self.h, self.w)[:h * factor, :w * factor]
            s = a.reshape(h, factor, w, factor).sum(axis=(1, 3))
            c.append(((s + n // 2) // n).reshape(-1))
        return Img(w, h, c)

    def luma(self):
        return 0.299 * self.c[0] + 0.587 * self.c[1] + 0.114 * self.c[2]

    def rgb(self):
        return np.stack([ch.reshape(self.h, self.w) for ch in self.c], axis=2).astype(np.uint8)


# =============================================================================
# The delta coder, as block_writer
# =============================================================================

def quantized(m, pixel_quant, pixel_shift, size):
    q = []
    for ch in m.c:
        if pixel_quant != 0:
            ch = rm.resize(ch, m.w, size[0], size[1])
        q.append(dm.quantize_channel(ch, pixel_shift))
    return q


def delta_bytes(m, pixel_quant, pixel_shift):
    size = dm.get_quantized_size(m.w, m.h, pixel_quant)
    q = quantized(m, pixel_quant, pixel_shift, size)
    qc, _ = dm.get_candidate_channels(q[0], q[1], q[2])
    # Java's getIdealFrequency; get_ideal_frequency2 gives the identical histogram.
    s = [int(math.floor(cm.get_shannon_limit(dm.get_ideal_frequency2(qc[i], size[0], size[1])))) for i in range(6)]
    best, best_sum = 0, 2147483647
    for k in range(10):
        c = dm.get_channels(k)
        t = s[c[0]] + s[c[1]] + s[c[2]]
        if t < best_sum:
            best_sum, best = t, k
    ids = dm.get_channels(best)
    return 9 + 3 * 8 + dm.get_block_map_bytes([qc[i] for i in ids], size[0], size[1], 12, 0)   # + block_writer's header


def reconstruct(m, pixel_quant, pixel_shift):
    """What the reader shows: the coding is lossless, so the channels come
    back as quantized; resize back, shift back."""
    size = dm.get_quantized_size(m.w, m.h, pixel_quant)
    q = quantized(m, pixel_quant, pixel_shift, size)
    c = []
    for ch in q:
        if pixel_quant != 0:
            ch = rm.resize(ch, size[0], m.w, m.h)
        c.append(dm.shift(ch, pixel_shift) if pixel_shift != 0 else ch)
    return Img(m.w, m.h, [np.asarray(ch, dtype=np.int64) for ch in c])


# =============================================================================
# JPEG
# =============================================================================

# The standard JPEG quantization tables (Annex K), natural order.
_STD_LUMINANCE = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99]
_STD_CHROMINANCE = [
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99, 99, 99, 99, 99, 99] + [99] * 32


def java_qtables(quality):
    """The tables Java's ImageIO uses at quality/100: JPEG.convertToLinearQuality
    then JPEGQTable.getScaledInstance(scale, true), in float arithmetic."""
    q = np.float32(quality) / np.float32(100)
    q = max(q, np.float32(0.01))
    scale = np.float32(0.5) / q if q < np.float32(0.5) else np.float32(2.0) - q * np.float32(2.0)

    def scaled(table):
        return [int(min(255, max(1, np.floor(np.float32(v) * scale + np.float32(0.5))))) for v in table]
    return [scaled(_STD_LUMINANCE), scaled(_STD_CHROMINANCE)]


def write_jpeg(m, quality):
    """JPEG bytes at quality 1-100 (baseline, 4:2:0), with Java's tables when
    Pillow is available."""
    if _PILImage is not None:
        import io
        buf = io.BytesIO()
        _PILImage.fromarray(m.rgb()).save(buf, "JPEG", qtables=java_qtables(quality), subsampling=2)
        return buf.getvalue()
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(m.rgb(), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise IOError("JPEG encoding failed")
    return buf.tobytes()


def read_jpeg(data):
    bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    return Img.from_rgb(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), 1)


# =============================================================================
# Metrics
# =============================================================================

def psnr(a, b):
    se = sum(int(((x - y) ** 2).sum()) for x, y in zip(a.c, b.c))    # exact, as the Java's sum of integer squares
    mse = se / (3.0 * len(a.c[0]))
    return 99.0 if mse == 0 else 10 * math.log10(255.0 * 255.0 / mse)


def max_error(a, b):
    return float(max(int(np.abs(x - y).max()) for x, y in zip(a.c, b.c)))


def ssim(ia, ib):
    """SSIM of the luma over 8x8 windows."""
    w, h = ia.w, ia.h
    nh, nw = h // 8, w // 8
    if nh == 0 or nw == 0:
        return float("nan")
    a = ia.luma().reshape(h, w)[:nh * 8, :nw * 8].reshape(nh, 8, nw, 8)
    b = ib.luma().reshape(h, w)[:nh * 8, :nw * 8].reshape(nh, 8, nw, 8)
    ma = a.sum(axis=(1, 3)) / 64
    mb = b.sum(axis=(1, 3)) / 64
    da = a - ma[:, None, :, None]
    db = b - mb[:, None, :, None]
    va = (da * da).sum(axis=(1, 3)) / 63
    vb = (db * db).sum(axis=(1, 3)) / 63
    cov = (da * db).sum(axis=(1, 3)) / 63
    c1, c2 = 6.5025, 58.5225
    s = ((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2))
    return float(s.sum() / s.size)


def blockiness(m):
    """Mean luma jump across 8-pixel boundaries / mean jump elsewhere."""
    y = m.luma().reshape(m.h, m.w)
    dx = np.abs(np.diff(y, axis=1))           # jump into column x, for x = 1..w-1
    dy = np.abs(np.diff(y, axis=0))           # jump into row r, for r = 1..h-1
    on_x = (np.arange(1, m.w) % 8 == 0)
    on_y = (np.arange(1, m.h) % 8 == 0)
    on = dx[:, on_x].sum() + dy[on_y, :].sum()
    off = dx[:, ~on_x].sum() + dy[~on_y, :].sum()
    n_on = dx[:, on_x].size + dy[on_y, :].size
    n_off = dx[:, ~on_x].size + dy[~on_y, :].size
    with np.errstate(divide="ignore", invalid="ignore"):
        return float((on / n_on) / max(1e-9, off / n_off)) if n_on and n_off else float("nan")


# =============================================================================
# One measured result
# =============================================================================

class Point:
    __slots__ = ("name", "bytes", "bpp", "psnr", "ssim", "max", "blocky", "psnr_display", "ssim_display")

    def full_columns(self):
        return "%5.2f  %6.4f  %7.0f  %6.2f" % (self.psnr, self.ssim, self.max, self.blocky)

    def display_columns(self):
        return "%5.2f  %6.4f" % (self.psnr_display, self.ssim_display)


def measure(name, nbytes, pixels, orig, orig_display, rec, factor):
    p = Point()
    p.name, p.bytes, p.bpp = name, nbytes, nbytes * 8 / pixels
    p.psnr, p.ssim, p.max, p.blocky = psnr(orig, rec), ssim(orig, rec), max_error(orig, rec), blockiness(rec)
    rd = rec.shrink(factor)
    p.psnr_display, p.ssim_display = psnr(orig_display, rd), ssim(orig_display, rd)
    return p


def frontier(sorted_by_bpp, at_display):
    """Settings no other setting beats with a smaller file and higher PSNR."""
    f, best = [], -1.0
    for p in sorted_by_bpp:
        v = p.psnr_display if at_display else p.psnr
        if v > best:
            f.append(p)
            best = v
    return f


def jpeg_psnr_at(jpeg, bpp, at_display):
    for a, b in zip(jpeg, jpeg[1:]):
        if a.bpp <= bpp <= b.bpp:
            pa = a.psnr_display if at_display else a.psnr
            pb = b.psnr_display if at_display else b.psnr
            return pa + (bpp - a.bpp) / (b.bpp - a.bpp) * (pb - pa)
    return float("nan")


def compare(label, front, jpeg, at_display):
    """The best delta settings against JPEG at the same file size."""
    def ps(p):
        return p.psnr_display if at_display else p.psnr
    print("Delta coder vs JPEG at the same file size, error measured at " + label)
    print("  (best delta settings only; JPEG's PSNR interpolated between its quality steps)")
    print("  setting                bpp  |  delta PSNR   JPEG PSNR   difference")
    cross_bpp = cross_psnr = previous = float("nan")
    previous_point = None
    top = jpeg[-1]
    for p in front:
        mine = ps(p)
        jp = jpeg_psnr_at(jpeg, p.bpp, at_display)
        if math.isnan(jp):
            if p.bpp < jpeg[0].bpp:
                verdict = "smaller than JPEG's smallest file"
            elif mine > ps(top):
                verdict = "closer than JPEG's best (%s, %.2f dB)" % (top.name, ps(top))
            else:
                verdict = "bigger than %s, yet %s is closer (%.2f dB)" % (top.name, top.name, ps(top))
        else:
            d = mine - jp
            verdict = "%+6.2f dB  %s" % (d, "delta better" if d >= 0 else "JPEG better")
            if not math.isnan(previous) and previous < 0 and d >= 0:
                t = -previous / (d - previous)
                cross_bpp = previous_point.bpp + t * (p.bpp - previous_point.bpp)
                cross_psnr = ps(previous_point) + t * (mine - ps(previous_point))
            previous, previous_point = d, p
        print("  %-18s %6.2f  |  %9.2f   %9s   %s" % (p.name, p.bpp, mine, "-" if math.isnan(jp) else "%.2f" % jp, verdict))
    if not math.isnan(cross_bpp):
        print("  => Crossover near %.2f bpp / %.1f dB: below that, JPEG gives less error for the same size;" % (cross_bpp, cross_psnr)
              + "\n     above it, the delta coder does.")
    else:
        all_jpeg = True
        for p in front:
            jp = jpeg_psnr_at(jpeg, p.bpp, at_display)
            if not math.isnan(jp) and ps(p) >= jp:
                all_jpeg = False
        print("  => JPEG is better wherever the two overlap; the delta coder is only useful beyond JPEG's best quality."
              if all_jpeg else "  => The delta coder is better wherever the two overlap.")
    print()


def main(argv):
    if len(argv) < 2:
        print("Usage: python3 comparison.py image.png [scale] [display width]")
        return
    scale = int(argv[2]) if len(argv) > 2 else 1
    display = int(argv[3]) if len(argv) > 3 else 1024
    orig = Img.from_rgb(vs.read_image(argv[1]), scale)
    factor = max(1, math.ceil(orig.w / display))
    orig_display = orig.shrink(factor)
    pixels = float(orig.w * orig.h)

    print("Image: %s, %d x %d%s" % (argv[1], orig.w, orig.h, " (1/%d size)" % scale if scale > 1 else ""))
    print("Display comparison at %d x %d%s" % (orig_display.w, orig_display.h,
          " (1/%d)" % factor if factor > 1 else " (no shrinking needed)"))
    print()
    print(GUIDE, end="")
    print()

    lossless = delta_bytes(orig, 0, 0)
    print("Lossless (delta coder, no quantization): {:,} bytes = {:.2f} bpp".format(lossless, lossless * 8 / pixels))
    print()

    # ---- JPEG
    print("JPEG, by quality setting")
    print("  quality      bytes    bpp  |  PSNR    SSIM  max err  blocky  |  display: PSNR    SSIM  |  decoded JPEG re-coded losslessly")
    jpeg = []
    for q in (5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 95):
        data = write_jpeg(orig, q)
        dec = read_jpeg(data)
        p = measure("q%d" % q, len(data), pixels, orig, orig_display, dec, factor)
        jpeg.append(p)
        recode = delta_bytes(dec, 0, 0)
        print("  {:<7} {:>10,}  {:5.2f}  | {}  | {}  |  {:>10,} bytes = {:.1f} x the JPEG".format(
            p.name, p.bytes, p.bpp, p.full_columns(), p.display_columns(), recode, recode / len(data)))
    print("  (The last column is why delta-coding a decoded JPEG is misleading: its artifacts cost")
    print("   several times the JPEG's own size to store losslessly.)")
    print()

    # ---- Delta coder
    color, pixel = [0, 1, 2, 3, 4, 5, 6], [0, 2, 4, 6, 8, 10]
    delta = [None] * (len(color) * len(pixel))

    def one(t):
        c, px = color[t // len(pixel)], pixel[t % len(pixel)]
        delta[t] = measure("color %d, pixel %d" % (c, px), delta_bytes(orig, px, c), pixels, orig, orig_display,
                           reconstruct(orig, px, c), factor)
    vs.parallel(len(delta), one)
    delta.sort(key=lambda p: p.bpp)
    front_full, front_display = frontier(delta, False), frontier(delta, True)
    print("Delta coder, by Color Resolution and Pixel Resolution (smallest file first)")
    print("  * = no other setting gives a smaller file with less error (full size); d = same, at display size")
    print("  setting               best      bytes    bpp  |  PSNR    SSIM  max err  blocky  |  display: PSNR    SSIM")
    for p in delta:
        print("  {:<18}    {}{}  {:>10,}  {:5.2f}  | {}  | {}".format(p.name, "*" if p in front_full else " ",
              "d" if p in front_display else " ", p.bytes, p.bpp, p.full_columns(), p.display_columns()))
    print()

    # ---- Comparison
    compare("full size", front_full, jpeg, False)
    compare("display size", front_display, jpeg, True)

    # ---- Bottom line
    print("Bottom line")
    for at_display in (False, True):
        top = jpeg[-1]
        top_psnr = top.psnr_display if at_display else top.psnr
        cheapest = next((p for p in delta if (p.psnr_display if at_display else p.psnr) > top_psnr), None)
        where = "at display size" if at_display else "at full size   "
        if cheapest is None:
            print("  %s: no delta setting is closer than JPEG %s (%.2f dB)." % (where, top.name, top_psnr))
        else:
            print(("  %s: JPEG %s reaches %.2f dB at %.2f bpp. The smallest delta setting that does better is\n"
                   "                   %s: %.2f dB at %.2f bpp (%.1f x the size of %s). Anything coarser, use JPEG.")
                  % (where, top.name, top_psnr, top.bpp, cheapest.name,
                     cheapest.psnr_display if at_display else cheapest.psnr, cheapest.bpp, cheapest.bpp / top.bpp, top.name))


if __name__ == "__main__":
    main(sys.argv)
