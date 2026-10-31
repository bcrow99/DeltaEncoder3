#!/usr/bin/env python3
"""
simple_writer.py version 3.3 -- SimpleWriter.java for Python (PySide6).

A writer with one predictor and one entropy coder: every interior pixel is
predicted by the least-squares fit (at each pixel, the linear predictor that
best fits a causal window of the pixels before it, clamped to 2 beyond the
range of its eight neighbours; delta_mapper's _ls_*), with no blocks and no
map, and the deltas are mixed-context coded with the second opinions
(delta_mapper.pack_mixed_deltas). The analysis picks the channel set, then,
channel by channel, the window radius (4, 8 or 16: delta_mapper.WINDOW_RADII)
that codes the channel smallest, by the real coded size
(delta_mapper.survey_windows). It runs at startup and again after any change
to the quantization (Pixel Resolution or Color Resolution). Pixel Resolution
resizes with resize_mapper.resize2 (centred). The Bands menu splits the image
into up to MAX_BANDS bands of rows that a reader can decode at the same time,
for a small cost per band boundary (the fit and the contexts start again at
each band's top); the reader also decodes a band's three channels at the same
time. Save analyses and codes the image twice (encode), printing both
times (the first includes compiling the code) and the decode check's
apart. Saves (to the file "foo", as the Java version does) in Simple format
'S' version 8, read by simple_reader.py and SimpleReader.java.

    python3 simple_writer.py [image]

The coding (class SimpleCoder) is separate from the window:

    coder = SimpleCoder(viewer_support.read_image("photo.png"))
    coder.analyse(); coder.apply(); coder.save("photo.smp")
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import code_mapper as cm
import delta_mapper as dm
import resize_mapper as rm
import viewer_support as vs
from java_io import DataOutput

FORMAT_ID = ord('S')
FORMAT_VERSION = 8   # 3 to 5: unary strings or block maps with a choice of delta type and entropy coder (no longer read); 6: least squares alone, a window radius and clamp per channel, in bands; 7: 6 with centred resizing (resize_mapper.resize2); 8: 7 with the wide clamp only (no clamp byte)
MAX_BANDS = 16       # the most bands Save offers
BAND_MIN_ROWS = 64   # the fewest rows (of the quantized image) a band may have
TIE_SET = 7          # the channel set best_set picks when it ties for the smallest estimate: green, blue-green, red-green

QUANT_SETTLE_MS = 400   # how long a quantization slider must rest before the analysis runs again


def best_set(channel_sum):
    """The channel set with the smallest summed estimate: on a tie, TIE_SET
    if it is one of those tied (on an image where the estimates can't tell
    the sets apart, such as a gray one, its difference channels code
    smaller), otherwise the first (SimpleWriter.bestSet)."""
    c = dm.get_channels(TIE_SET)
    best, best_sum = TIE_SET, channel_sum[c[0]] + channel_sum[c[1]] + channel_sum[c[2]]
    for s in range(10):
        c = dm.get_channels(s)
        t = channel_sum[c[0]] + channel_sum[c[1]] + channel_sum[c[2]]
        if t < best_sum:
            best_sum, best = t, s
    return best


def band_start(b, n, rows):
    """Band b's first row, of rows split into n bands (SimpleWriter.bandStart)."""
    return b * rows // n


def describe_bands(n, rows):
    """"8 bands of 375 rows" (or "of 374 to 375 rows")."""
    lo, hi = rows // n, -(-rows // n)
    return "%d bands of %s rows" % (n, lo if lo == hi else "%d to %d" % (lo, hi))


def describe_windows(set_id, s):
    """The channel set and each channel's window, as apply prints them
    (SimpleWriter.describeWindows)."""
    t = "Channel set: " + dm.SET_NAMES[set_id] + "\n"
    for i in range(3):
        t += "Channel %d - %s: radius %d (%s bytes)\n" % (i, dm.channel_name(set_id, i), s.radius[i], format(s.bytes[i], ","))
    return t


# =============================================================================
# The coding, without the window
# =============================================================================

class SimpleCoder:
    """Holds an image and the quantization; analyse() picks each channel's
    window radius, apply() picks the channels (and makes the
    preview), save() codes and writes the file."""

    def __init__(self, rgb, file_length=0):
        self.image_ydim, self.image_xdim = rgb.shape[:2]
        self.source = [rgb[:, :, c].reshape(-1).astype(np.int64) for c in range(3)]
        self.file_length = file_length

        self.pixel_quant = 4
        self.pixel_shift = 3
        self.correction = 0         # preview only
        self.min_set_id = 0
        self.bands = 1              # independently decodable bands Save writes

        # The last analysis (a delta_mapper.WindowSurvey; None before the
        # first) and the quantization it was made for.
        self.survey = None
        self.survey_quant = self.survey_shift = -1
        self.survey_time = None     # the analysis's seconds, printed with the next apply's listing

        self.channel_sum = [0] * 6
        self.channel_min = [0] * 6
        self.channel_init = [0] * 6
        self.applied = False

    def quantized_channels(self, size, quant=None, shift=None):
        """(the six candidate channels after resizing and quantizing, their
        minimums, their first values), at this quantization (the coder's by
        default)."""
        quant = self.pixel_quant if quant is None else quant
        shift = self.pixel_shift if shift is None else shift
        q = [None] * 3

        def one(i):
            ch = self.source[i]
            if quant != 0:
                ch = rm.resize2(ch, self.image_xdim, size[0], size[1])
            q[i] = dm.quantize_channel(ch, shift)
        vs.parallel(3, one)
        qc, mins = dm.get_candidate_channels(q[0], q[1], q[2])
        return qc, list(mins), [int(c[0]) for c in qc]

    @staticmethod
    def channel_sums(qc, size):
        # Java's getIdealFrequency; get_ideal_frequency2 gives the identical histogram.
        s = [0] * 6

        def one(i):
            s[i] = int(np.floor(cm.get_shannon_limit(dm.get_ideal_frequency2(qc[i], size[0], size[1]))))
        vs.parallel(6, one)
        return s

    def survey_current(self):
        """Whether the last analysis was made for the current quantization."""
        return self.survey is not None and self.survey_quant == self.pixel_quant and self.survey_shift == self.pixel_shift

    def analyse(self, quant=None, shift=None):
        """Picks the channel set, then each channel's window radius, for this
        quantization (the coder's by default) (dm.survey_windows;
        Java: analyse). Returns the survey."""
        quant = self.pixel_quant if quant is None else quant
        shift = self.pixel_shift if shift is None else shift
        start = time.perf_counter()
        size = dm.get_quantized_size(self.image_xdim, self.image_ydim, quant)
        qc, _, _ = self.quantized_channels(size, quant, shift)
        ids = dm.get_channels(best_set(self.channel_sums(qc, size)))
        self.survey = dm.survey_windows([qc[i] for i in ids], size[0], size[1], dm.WINDOW_RADII)
        self.survey_quant, self.survey_shift = quant, shift
        self.survey_time = time.perf_counter() - start
        return self.survey

    def bands_used(self):
        """The bands Save will write: bands, but no more than leave
        BAND_MIN_ROWS rows in each."""
        rows = dm.get_quantized_size(self.image_xdim, self.image_ydim, self.pixel_quant)[1]
        return max(1, min(self.bands, rows // BAND_MIN_ROWS))

    def band_text(self):
        """The Bands dialog's status line."""
        n = self.bands_used()
        if n == 1:
            return "1 band: the image is too small for more" if self.bands > 1 else "1 band"
        return describe_bands(n, dm.get_quantized_size(self.image_xdim, self.image_ydim, self.pixel_quant)[1])

    def window_text(self):
        """Each channel's window, and the bands Save will write (the Window menu's Radius)."""
        if not self.applied:
            return "No analysis yet."
        t = describe_windows(self.min_set_id, self.survey)
        if self.bands_used() > 1:
            t += describe_bands(self.bands_used(), len(self.value_list[0]) // self.delta_xdim) + "\n"
        return t

    def sizes_text(self):
        """Each channel's coded size at every radius tried; the chosen one
        starred (the Window menu's Sizes)."""
        if not self.applied:
            return "No analysis yet."
        s = self.survey
        parts = []
        for i in range(3):
            t = "Channel %d - %s:\n" % (i, dm.channel_name(self.min_set_id, i))
            t += "  %6s %12s\n" % ("radius", "bytes")
            for r, radius in enumerate(s.radii):
                t += "  %6d %11s%s\n" % (radius, format(s.tried[i][r], ","), "*" if radius == s.radius[i] else " ")
            parts.append(t)
        return "\n".join(parts)

    def apply(self):
        """Quantizes and picks the channel set (the channels Save codes), then
        rebuilds the image for the preview. After a new analysis it prints the
        channel set and each channel's window. Returns the preview as an
        (ydim, xdim, 3) uint8 array."""
        self.applied = False
        # An analysis for this quantization first, if there isn't one (a coder
        # whose fields were set directly, or an analysis that failed).
        if not self.survey_current():
            self.analyse()
        size = dm.get_quantized_size(self.image_xdim, self.image_ydim, self.pixel_quant)
        w, h = size
        qc, self.channel_min, self.channel_init = self.quantized_channels(size)
        self.channel_sum = self.channel_sums(qc, size)
        self.min_set_id = best_set(self.channel_sum)
        ids = dm.get_channels(self.min_set_id)
        # The reader rebuilds each channel exactly as quantized (before the
        # minimum is added back), so the preview uses the quantized channels
        # as they are: no need to decode the deltas here.
        raw = [np.asarray(qc[j], dtype=np.int64) for j in ids]
        decoded = [raw[i] + self.channel_min[ids[i]] if ids[i] > 2 else raw[i] for i in range(3)]
        self.value_list, self.delta_xdim = raw, w

        if self.survey_time is not None:
            print(describe_windows(self.min_set_id, self.survey), end="")
            print("Analysis took %s (%d x %d, %s pixels)" % (vs.format_duration(self.survey_time), w, h, format(w * h, ",")))
            print()
            self.survey_time = None

        # As the reader: recombine the channel set first, then resize (centred: resize2), then shift.
        bgr = dm.get_blue_green_red(self.min_set_id, *decoded)

        def restore(c):
            v = bgr[c]
            if self.pixel_quant != 0:
                v = rm.resize2(v, w, self.image_xdim, self.image_ydim)
            if self.pixel_shift != 0:
                v = dm.shift(v, self.pixel_shift)
            if self.correction != 0:
                f = self.correction / 10.0
                v = v + ((self.source[c] - v) * f).astype(np.int64)     # (int) truncates toward zero
            bgr[c] = v
        vs.parallel(3, restore)
        rgb = np.stack([np.clip(v, 0, 255).reshape(self.image_ydim, self.image_xdim) for v in bgr], axis=2).astype(np.uint8)
        self.applied = True
        return rgb

    def encode(self):
        """(The file's bytes, the seconds the decode check took): the
        analysis and the coding at the current quantization (encode)."""
        size = dm.get_quantized_size(self.image_xdim, self.image_ydim, self.pixel_quant)
        qc, mins, _ = self.quantized_channels(size)
        return encode(qc, mins, size[0], self.image_xdim, self.image_ydim, self.pixel_shift, self.pixel_quant, self.bands_used())

    def save(self, filename="foo"):
        """Analyses and codes the image (encode), twice, and writes it;
        prints both times (the first includes compiling the code: Numba's,
        the first time it runs, or loading it from the cache), without the
        decode check, and the check's time (SimpleWriter.SaveHandler).
        Returns the file size."""
        took, check = [], []
        data = None
        for _ in range(2):
            start = time.perf_counter()
            got, c = self.encode()
            took.append(time.perf_counter() - start - c)
            check.append(c)
            if data is not None and got != data:
                print("WARNING: the second coding differs from the first.")
            data = got
        with open(filename, "wb") as f:
            f.write(data)
        n = self.bands_used()
        raw = self.image_xdim * self.image_ydim * 3
        w, rows = dm.get_quantized_size(self.image_xdim, self.image_ydim, self.pixel_quant)
        if n > 1:
            print("Saved %s: %s bytes (%s)" % (os.path.basename(filename), format(len(data), ","), describe_bands(n, rows)))
        else:
            print("Saved %s: %s bytes" % (os.path.basename(filename), format(len(data), ",")))
        print("Coded %d x %d (%s pixels)" % (w, rows, format(w * rows, ",")))
        print("Run time plus compile time: " + vs.format_duration(took[0]))
        print("Run time only:              " + vs.format_duration(took[1]))
        print("Decode check:               " + vs.format_duration(check[1]))
        print("Original compression rate: %.4f" % (self.file_length / raw))
        print("Output  compression rate:  %.4f" % (len(data) / raw))
        print()
        return len(data)


def encode(qc, mins, xdim, image_xdim, image_ydim, pixel_shift, pixel_quant, n):
    """Analyses and codes the image and returns (the file's bytes, the
    seconds the decode check took) (SimpleWriter.encode); qc: the six
    candidate channels, quantized (get_candidate_channels), mins their
    minimums. Format 8. Header: FORMAT_ID, FORMAT_VERSION, width, height
    (unsigned shorts), pixel_shift, pixel_quant, channel set; per channel
    (coding order) int min and the window radius (an unsigned byte); the
    number of bands n (an unsigned byte), n int lengths, and the n bands.
    Band b is rows band_start(b) to band_start(b + 1) of the quantized image,
    coded as an image of its own (the fit and the contexts start again at
    its top): per channel, int init, then the least-squares deltas
    mixed-context coded with the second opinions (pack_mixed_deltas).
    Channels 1 and 2 use the ones before them (their values in the fit,
    their deltas as context), so a band's channels are decoded in order,
    each a few rows behind the one before; the bands can be decoded at the
    same time.

    The analysis as the preview's: the channel set with the smallest
    estimate (best_set), then each channel's window radius by the real coded
    size (delta_mapper.survey_windows). One band is what the survey already
    coded; more bands reuse its deltas past each band's first rows
    (delta_mapper.get_least_squares_band_deltas) and are coded at the same
    time. Once everything is coded, each band's channels are checked by
    decoding them back, at the same time; the time that took is returned, so
    the coding can be timed without it."""
    rows = len(qc[0]) // xdim
    set_id = best_set(SimpleCoder.channel_sums(qc, (xdim, rows)))
    ids = dm.get_channels(set_id)
    value = [np.asarray(qc[j], dtype=np.int64) for j in ids]
    s = dm.survey_windows(value, xdim, rows, dm.WINDOW_RADII)
    ch = [[None] * 3 for _ in range(n)]      # [band][channel]
    d = [[None] * 3 for _ in range(n)]
    coded = [[None] * 3 for _ in range(n)]

    def code(b, i):
        coded[b][i] = s.coded[i] if n == 1 else dm.pack_mixed_deltas(d[b][i], ch[b][i], d[b][:i], xdim, True)

    def check(b, i):
        back = dm.unpack_mixed_deltas(coded[b][i], ch[b][i], d[b][:i], xdim, ch[b][i] - np.asarray(d[b][i], dtype=np.int64))
        if not np.array_equal(back, d[b][i]):
            print("WARNING: band %d channel %d deltas do not decode back." % (b, i))

    for b in range(n):
        y0, y1 = band_start(b, n, rows), band_start(b + 1, n, rows)
        for i in range(3):
            ch[b][i] = value[i][y0 * xdim:y1 * xdim]
            d[b][i] = s.deltas[i] if n == 1 else dm.get_least_squares_band_deltas(s.deltas[i], value[i], value[i - 1] if i > 0 else None,
                                                                                  xdim, y0, y1, s.radius[i])
    with ThreadPoolExecutor(max(1, min(3 * n, os.cpu_count() or 1))) as pool:
        for j in [pool.submit(code, b, i) for b in range(n) for i in range(3)]:
            j.result()
        # The check, once everything is coded (so its time can be told apart).
        start = time.perf_counter()
        for j in [pool.submit(check, b, i) for b in range(n) for i in range(3)]:
            j.result()
        check_time = time.perf_counter() - start

    out = DataOutput()
    out.write_byte(FORMAT_ID); out.write_byte(FORMAT_VERSION)
    out.write_short(image_xdim); out.write_short(image_ydim)
    out.write_byte(pixel_shift); out.write_byte(pixel_quant); out.write_byte(set_id)
    for i in range(3):
        out.write_int(int(mins[ids[i]]))
        out.write_byte(s.radius[i])
    out.write_byte(n)
    band = []
    for b in range(n):
        bo = DataOutput()
        for i in range(3):
            bo.write_int(int(ch[b][i][0]))
            bo.write(coded[b][i])
        band.append(bo.to_bytes())
    for bb in band:
        out.write_int(len(bb))
    for bb in band:
        out.write(bb)
    return out.to_bytes(), check_time


# =============================================================================
# The window
# =============================================================================

def open_image(parent=None):
    from PySide6.QtWidgets import QFileDialog
    path, _ = QFileDialog.getOpenFileName(parent, "Open Image", "",
                                          "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.pgm *.ppm);;All files (*)")
    if path:
        SimpleWriter(path)


_open_writers = []


class SimpleWriter:
    def __init__(self, filename):
        from PySide6.QtCore import QTimer
        from PySide6.QtGui import QAction, QKeySequence

        self.filename = filename
        try:
            rgb = vs.read_image(filename)
        except IOError as e:
            vs.show_error(None, str(e))
            return
        self.coder = c = SimpleCoder(rgb, os.path.getsize(filename))
        self.updating = False
        self.analysing = False
        print("Loaded %s, %d x %d (%s pixels)" % (filename, c.image_xdim, c.image_ydim, format(c.image_xdim * c.image_ydim, ",")))

        self.view = view = vs.ImageWindow("Simple Writer  " + filename, c.image_xdim, c.image_ydim)
        bar = view.menuBar()
        _open_writers.append(self)
        view.destroyed.connect(lambda *_: _open_writers.remove(self) if self in _open_writers else None)

        # A quantization change analyses again, once the slider has stopped
        # moving for QUANT_SETTLE_MS.
        self.quant_timer = QTimer(view)
        self.quant_timer.setSingleShot(True)
        self.quant_timer.setInterval(QUANT_SETTLE_MS)
        self.quant_timer.timeout.connect(self.run_analysis)

        file_menu = bar.addMenu("File")
        a = QAction("Open...", view, shortcut=QKeySequence("Ctrl+O"))
        a.triggered.connect(lambda: open_image(view))
        file_menu.addAction(a)
        file_menu.addSeparator()
        a = QAction("Reset", view)
        a.triggered.connect(self.reset)
        file_menu.addAction(a)
        a = QAction("Save", view)
        a.triggered.connect(self.save)
        file_menu.addAction(a)

        view.make_view_menu()

        quant = bar.addMenu("Quantization")
        self.sliders = {}
        for key, title, lo, hi in (("pixel_quant", "Pixel Resolution", 0, 10), ("pixel_shift", "Color Resolution", 0, 7)):
            act, self.sliders[key] = vs.make_slider_dialog(view, title, lo, hi, getattr(c, key), self._quant_setter(key))
            quant.addAction(act)
        # Error Correction blends the preview back toward the original by
        # correction/10. Preview only; the saved file is unaffected, and there
        # is nothing to analyse again.
        quant.addSeparator()
        act, self.sliders["correction"] = vs.make_slider_dialog(view, "Error Correction", 0, 10, c.correction, self._set_correction)
        quant.addAction(act)

        # Bands menu: Save codes the image as this many bands of rows, each
        # decodable on its own (so a reader can decode them at the same time),
        # for a small cost per band boundary. Like Save itself, it never
        # changes the preview.
        bands_menu = bar.addMenu("Bands")
        act, self.bands_slider = vs.make_slider_dialog(view, "Number of Bands", 1, MAX_BANDS, c.bands, self._set_bands)
        bands_menu.addAction(act)

        # Window menu: what the analysis chose (nothing to set by hand).
        window_menu = bar.addMenu("Window")
        a = QAction("Radius", view)
        a.triggered.connect(lambda: self.show_text("Window", self.coder.window_text()))
        window_menu.addAction(a)
        a = QAction("Sizes", view)
        a.triggered.connect(lambda: self.show_text("Coded Size by Radius", self.coder.sizes_text()))
        window_menu.addAction(a)

        view.set_image(rgb)
        view.show_window()
        QTimer.singleShot(0, self.run_analysis)    # the preview shows the original until the analysis is done

    def _quant_setter(self, key):
        """A quantization slider's change: analyses again once the sliders settle."""
        def set_value(v):
            setattr(self.coder, key, v)
            if not self.updating:
                self.quant_changed()
        return set_value

    def _set_bands(self, v):
        self.coder.bands = v
        self.view.set_status(self.coder.band_text())

    def _set_correction(self, v):
        self.coder.correction = v
        if not self.updating and self.coder.survey_current():
            self.apply()

    def quant_changed(self):
        self.view.set_status("waiting to analyse…")
        self.quant_timer.start()

    def reset(self):
        c = self.coder
        c.pixel_quant = c.pixel_shift = c.correction = 0
        self.updating = True
        for key, slider in self.sliders.items():
            slider.setValue(getattr(c, key))
        self.updating = False
        self.quant_changed()

    def show_text(self, title, text):
        """Shows text in a dialog, in a fixed-width font so the columns line up."""
        from PySide6.QtGui import QFontDatabase
        from PySide6.QtWidgets import QDialog, QDialogButtonBox, QPlainTextEdit, QVBoxLayout
        dialog = QDialog(self.view)
        dialog.setWindowTitle(title)
        edit = QPlainTextEdit(text.strip())
        edit.setReadOnly(True)
        edit.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        metrics = edit.fontMetrics()
        lines = text.strip().split("\n")
        edit.setMinimumSize(min(900, metrics.horizontalAdvance(max(lines, key=len)) + 40), min(600, metrics.lineSpacing() * (len(lines) + 2)))
        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(dialog.accept)
        layout = QVBoxLayout(dialog)
        layout.addWidget(edit)
        layout.addWidget(buttons)
        dialog.exec()

    def apply(self):
        try:
            self.view.set_image(self.coder.apply())
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("Apply: %r" % e)

    def run_analysis(self):
        """Analyses in the background with the menus disabled, then applies
        (which prints the listing). If the quantization changed meanwhile, it
        analyses again."""
        if self.analysing:
            return      # done() checks the quantization again
        self.analysing = True
        self.view.set_menus_enabled(False)
        self.view.set_status("analysing…")
        c = self.coder
        quant, shift = c.pixel_quant, c.pixel_shift

        def done(result, error):
            self.analysing = False
            if error is not None:
                print("analysis: %r" % error)
            changed = (quant, shift) != (c.pixel_quant, c.pixel_shift)
            if changed and not self.quant_timer.isActive():
                self.run_analysis()
                return
            self.view.set_menus_enabled(True)
            if error is not None:
                self.view.set_status("analysis failed")
                return
            self.view.set_status(None)
            if c.survey_current():
                self.apply()
        vs.run_in_background(lambda: c.analyse(quant, shift), done)

    def save(self):
        if not self.coder.applied:
            self.apply()
        if not self.coder.applied:
            vs.show_error(self.view, "Nothing saved: the last Apply failed.")
            return
        try:
            self.coder.save("foo")
        except Exception as e:
            import traceback
            traceback.print_exc()
            if os.path.exists("foo"):
                os.remove("foo")
            vs.show_error(self.view, "Save failed: %r" % e)


def main():
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    if len(sys.argv) > 1:
        SimpleWriter(sys.argv[1])
    else:
        open_image()
    if not _open_writers:
        return
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
