#!/usr/bin/env python3
"""
block_writer.py version 2.3 -- BlockWriter.java for Python (PySide6).

A writer with one delta type and one entropy coder: the block map (delta
type 13) with mixed-context-coded deltas (delta_mapper.pack_mixed_deltas)
and a coded map. Each block uses one predictor from its channel's set, and
a set can be any of the 256 predictors delta_mapper has ids for (id 255 is
the least-squares predictor, LS). There are no block settings to choose:
the survey makes them from the predictor table alone. It picks the channel
set, then, channel by channel, the block size (a square of 4, 6, 8, 12 or 16) and
the predictor set that code the channel smallest
(delta_mapper.survey_channels: predictors added in greedy order by
estimated coded bits, the estimate's reference deltas from the whole table,
blocks assigned by the estimate, and the real coded size measured at each
count). It runs at startup and again after any change to the quantization
(Pixel Resolution or Color Resolution). Saves (to the file "foo", as the
Java version does) in Block format 'B' version 5, read by block_reader.py
and BlockReader.java. The survey measures sizes with the Context coder
(pack_context_deltas); the file uses the mixed coder, about 1% smaller.
File > Bands splits the image into up to MAX_BANDS bands of rows that a
reader can decode at the same time (format 6), for about a byte per column
per band boundary; with 1 (the default) Save writes format 5.

    python3 block_writer.py [image]

The coding (class BlockCoder) is separate from the window:

    coder = BlockCoder(viewer_support.read_image("photo.png"))
    coder.survey(); coder.apply(); coder.save("photo.blk")
"""

import os
import sys
import time

import numpy as np

import code_mapper as cm
import delta_mapper as dm
import resize_mapper as rm
import viewer_support as vs
from java_io import DataOutput

FORMAT_ID = ord('B')
FORMAT_VERSION = 6   # 3: one predictor table (delta_mapper.PREDICTOR_TABLE) and cross-channel block maps; 4: mixed context coding; 5: a block size per channel; 6: independently decodable bands
ONE_BAND_VERSION = 5   # the version written with one band: the layout before bands
MAX_BANDS = 16         # the most bands Save offers
BAND_MIN_ROWS = 64     # the fewest rows (of the quantized image) a band may have
BAND_COST_TENTHS = 13  # roughly what a band boundary costs, tenths of a byte per column (for the estimate only)

QUANT_SETTLE_MS = 400   # how long a quantization slider must rest before the survey runs again


def mixed_code(delta, values, xdim):
    """Each channel's deltas mixed-context coded (pack_mixed_deltas), channel
    i's contexts using channels 0..i-1. Checks each decodes back."""
    coded = [None] * 3

    def one(i):
        coded[i] = dm.pack_mixed_deltas(delta[i], values[i], delta[:i], xdim)
        back = dm.unpack_mixed_deltas(coded[i], values[i], delta[:i], xdim)
        if not np.array_equal(back, delta[i]):
            print("WARNING: channel %d mixed-coded deltas do not decode back." % i)

    vs.parallel(3, one)
    return coded


def best_set(channel_sum):
    """The channel set with the smallest summed estimate (first on a tie)."""
    best, best_sum = 0, 2147483647
    for s in range(10):
        c = dm.get_channels(s)
        t = channel_sum[c[0]] + channel_sum[c[1]] + channel_sum[c[2]]
        if t < best_sum:
            best_sum, best = t, s
    return best


# =============================================================================
# The coding, without the window
# =============================================================================

class BlockCoder:
    """Holds an image and the quantization; survey() picks each channel's
    block size and predictor set, apply() makes the deltas (and the
    preview), save() codes and writes the file."""

    def __init__(self, rgb, file_length=0):
        self.image_ydim, self.image_xdim = rgb.shape[:2]
        self.source = [rgb[:, :, c].reshape(-1).astype(np.int64) for c in range(3)]
        self.file_length = file_length

        self.pixel_quant = 4
        self.pixel_shift = 3
        self.correction = 0         # preview only
        self.min_set_id = 0
        self.bands = 1              # independently decodable bands Save writes (1: one, format 5)

        # The last survey: each channel's block size and predictor ids, in
        # coding order (None before the first), and the quantization it was
        # made for.
        self.survey_blocks = None
        self.survey_ids = None
        self.survey_deltas = self.survey_maps = None
        self.survey_quant = self.survey_shift = -1
        self.survey_time = None     # the survey's seconds, printed with the next apply's listing
        self.survey_bytes = 0       # the survey's coded size, for the Bands estimate

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
                ch = rm.resize(ch, self.image_xdim, size[0], size[1])
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
        """Whether the last survey was made for the current quantization."""
        return self.survey_blocks is not None and self.survey_quant == self.pixel_quant and self.survey_shift == self.pixel_shift

    def survey(self, quant=None, shift=None):
        """Surveys the channel set, then each channel's block size and
        predictor set, for this quantization (the coder's by default)
        (dm.survey_channels; Java: survey). Returns each channel's block size."""
        quant = self.pixel_quant if quant is None else quant
        shift = self.pixel_shift if shift is None else shift
        start = time.perf_counter()
        size = dm.get_quantized_size(self.image_xdim, self.image_ydim, quant)
        qc, _, _ = self.quantized_channels(size, quant, shift)
        ids = dm.get_channels(best_set(self.channel_sums(qc, size)))
        survey = dm.survey_channels([qc[i] for i in ids], size[0], size[1], dm.SURVEY_CHANNEL_SIZES, whole_table=True)
        self.survey_blocks, self.survey_ids = list(survey.blocks), survey.ids
        self.survey_deltas, self.survey_maps = survey.deltas, survey.maps     # what apply codes
        self.survey_quant, self.survey_shift = quant, shift
        self.survey_time = time.perf_counter() - start
        self.survey_bytes = survey.bytes
        return self.survey_blocks

    def size_text(self):
        """Each channel's block size and number of blocks (the Block menu's Size)."""
        if not self.applied:
            return "No survey yet."
        t = "".join(dm.describe_block_size(self.min_set_id, i, self.map_list[i], self.delta_xdim) + "\n" for i in range(3))
        if self.bands_used() > 1:
            t += describe_bands(self.bands_used(), len(self.value_list[0]) // self.delta_xdim) + "\n"
        return t

    def bands_used(self):
        """The bands Save will write: bands, but no more than leave
        BAND_MIN_ROWS rows in each."""
        rows = dm.get_quantized_size(self.image_xdim, self.image_ydim, self.pixel_quant)[1]
        return max(1, min(self.bands, rows // BAND_MIN_ROWS))

    def band_estimate(self):
        """The Bands dialog's estimate of what the bands cost."""
        n = self.bands_used()
        if n == 1:
            return "1 band: the image is too small for more" if self.bands > 1 else "1 band"
        w, h = dm.get_quantized_size(self.image_xdim, self.image_ydim, self.pixel_quant)
        cost = (n - 1) * w * BAND_COST_TENTHS // 10
        pct = " (%+.2f%%)" % (100.0 * cost / self.survey_bytes) if self.survey_bytes > 0 else ""
        return describe_bands(n, h) + ": about +%s bytes" % format(cost, ",") + pct

    def predictor_text(self):
        """Each channel's predictors, most used first, with their numbers of
        blocks (the Block menu's Predictors)."""
        if not self.applied:
            return "No survey yet."
        return "\n".join("Channel %d - %s:\n" % (i, dm.channel_name(self.min_set_id, i)) + dm.describe_block_predictors(self.map_list[i], "  ")
                         for i in range(3))

    def apply(self):
        """Quantizes, picks the channel set, makes the block map deltas (what
        Save codes), then rebuilds the image from them. After a new survey it
        prints the channel set and each channel's blocks and predictors.
        Returns the preview as an (ydim, xdim, 3) uint8 array."""
        self.applied = False
        # A survey for this quantization first, if there isn't one (a coder
        # whose fields were set directly, or a survey that failed).
        if not self.survey_current():
            self.survey()
        size = dm.get_quantized_size(self.image_xdim, self.image_ydim, self.pixel_quant)
        w, h = size
        qc, self.channel_min, self.channel_init = self.quantized_channels(size)
        self.channel_sum = self.channel_sums(qc, size)
        self.min_set_id = best_set(self.channel_sum)
        ids = dm.get_channels(self.min_set_id)
        deltas, maps, decoded = ([None] * 3 for _ in range(3))
        # The deltas and maps the survey chose (for this quantization, so for
        # these channels): each channel's block size and set, blocks assigned
        # by estimated coded size. The reader rebuilds each channel exactly as quantized (before the
        # minimum is added back), so the preview and the mixed coder use the
        # quantized channels as they are: no need to decode the deltas here.
        raw = [np.asarray(qc[j], dtype=np.int64) for j in ids]
        for i in range(3):
            j = ids[i]
            deltas[i], maps[i] = self.survey_deltas[i], self.survey_maps[i]
            decoded[i] = raw[i] + self.channel_min[j] if j > 2 else raw[i]
        self.delta_list, self.map_list, self.delta_xdim = deltas, maps, w
        self.value_list = raw       # the channels as the reader rebuilds them (the mixed coder's contexts use them)

        if self.survey_time is not None:
            print(dm.describe_block_maps(self.min_set_id, self.map_list, self.delta_xdim), end="")
            print("Analysis took " + vs.format_duration(self.survey_time))
            print()
            self.survey_time = None

        # As the reader: recombine the channel set first, then resize, then shift.
        bgr = dm.get_blue_green_red(self.min_set_id, *decoded)

        def restore(c):
            v = bgr[c]
            if self.pixel_quant != 0:
                v = rm.resize(v, w, self.image_xdim, self.image_ydim)
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

    def save(self, filename="foo"):
        """One band: format 5 (one_band). Bands: format 6 (in_bands), and the
        size it would have been as one band for the report. Returns the file
        size."""
        n = self.bands_used()
        data = self.one_band()
        one = len(data)
        if n > 1:
            data = self.in_bands(n)
        with open(filename, "wb") as f:
            f.write(data)
        raw = self.image_xdim * self.image_ydim * 3
        if n > 1:
            print("Saved %s: %s bytes (%s; %s as one band, %+.2f%%)" % (os.path.basename(filename), format(len(data), ","),
                  describe_bands(n, len(self.value_list[0]) // self.delta_xdim), format(one, ","), 100.0 * (len(data) - one) / one))
        else:
            print("Saved %s: %s bytes" % (os.path.basename(filename), format(len(data), ",")))
        print("Original compression rate: %.4f" % (self.file_length / raw))
        print("Output  compression rate:  %.4f" % (len(data) / raw))
        print()
        return len(data)

    def one_band(self):
        """Format 5. Header: FORMAT_ID, ONE_BAND_VERSION, width, height
        (unsigned shorts), pixel_shift, pixel_quant, channel set. Then per
        channel: int min, int init, the block map (write_map, type 13; it
        carries the channel's block size and predictor ids; its context lines
        up the previous channel's map, align_map), then the deltas,
        mixed-context coded (pack_mixed_deltas). Channels 1 and 2 use the ones
        before them as context, so the reader decodes in order. Returns the
        bytes."""
        coded = mixed_code(self.delta_list, self.value_list, self.delta_xdim)

        ids = dm.get_channels(self.min_set_id)
        out = DataOutput()
        out.write_byte(FORMAT_ID); out.write_byte(ONE_BAND_VERSION)
        out.write_short(self.image_xdim); out.write_short(self.image_ydim)
        out.write_byte(self.pixel_shift); out.write_byte(self.pixel_quant); out.write_byte(self.min_set_id)
        for i in range(3):
            j = ids[i]
            out.write_int(self.channel_min[j]); out.write_int(self.channel_init[j])
            dm.write_map(out, 13, self.map_list[i], self.map_list[i - 1] if i > 0 else None, self.delta_xdim)
            out.write(coded[i])
        return out.to_bytes()

    def in_bands(self, n):
        """Format 6: the header as format 5's, then the number of bands n (an
        unsigned byte), each channel's int min (coding order), then n int
        lengths and the n bands. Band b is rows band_start(b) to
        band_start(b + 1) of the quantized image, coded as an image of its own
        with the survey's block sizes and predictor sets (blocks assigned by
        cost in the band; the predictors, least squares and contexts start
        again at its top): per channel, int init, the block map, then the
        mixed-coded deltas. A reader can decode the bands at the same time.
        Returns the bytes."""
        w = self.delta_xdim
        rows = len(self.value_list[0]) // w
        band = []
        for b in range(n):
            y0, y1 = band_start(b, n, rows), band_start(b + 1, n, rows)
            ch = [np.asarray(v[y0 * w:y1 * w], dtype=np.int64) for v in self.value_list]
            got = dm.get_block_deltas_by_cost(ch, w, y1 - y0, self.survey_blocks, self.survey_ids, whole_table=True)
            d = [g[0] for g in got]
            m = [g[1] for g in got]
            coded = mixed_code(d, ch, w)
            bo = DataOutput()
            for i in range(3):
                bo.write_int(int(ch[i][0]))
                dm.write_map(bo, 13, m[i], m[i - 1] if i > 0 else None, w)
                bo.write(coded[i])
            band.append(bo.to_bytes())

        ids = dm.get_channels(self.min_set_id)
        out = DataOutput()
        out.write_byte(FORMAT_ID); out.write_byte(FORMAT_VERSION)
        out.write_short(self.image_xdim); out.write_short(self.image_ydim)
        out.write_byte(self.pixel_shift); out.write_byte(self.pixel_quant); out.write_byte(self.min_set_id)
        out.write_byte(n)
        for i in range(3):
            out.write_int(self.channel_min[ids[i]])
        for b in band:
            out.write_int(len(b))
        for b in band:
            out.write(b)
        return out.to_bytes()


def band_start(b, n, rows):
    """Band b's first row, of rows split into n bands (BlockWriter.bandStart)."""
    return b * rows // n


def describe_bands(n, rows):
    """"8 bands of 375 rows" (or "of 374 to 375 rows")."""
    lo, hi = rows // n, -(-rows // n)
    return "%d bands of %s rows" % (n, lo if lo == hi else "%d to %d" % (lo, hi))


# =============================================================================
# The window
# =============================================================================

def open_image(parent=None):
    from PySide6.QtWidgets import QFileDialog
    path, _ = QFileDialog.getOpenFileName(parent, "Open Image", "",
                                          "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.pgm *.ppm);;All files (*)")
    if path:
        BlockWriter(path)


_open_writers = []


class BlockWriter:
    def __init__(self, filename):
        from PySide6.QtCore import QTimer
        from PySide6.QtGui import QAction, QKeySequence

        self.filename = filename
        try:
            rgb = vs.read_image(filename)
        except IOError as e:
            vs.show_error(None, str(e))
            return
        self.coder = c = BlockCoder(rgb, os.path.getsize(filename))
        self.updating = False
        self.analysing = False
        print("Loaded %s, %d x %d" % (filename, c.image_xdim, c.image_ydim))

        self.view = view = vs.ImageWindow("Block Writer  " + filename, c.image_xdim, c.image_ydim)
        bar = view.menuBar()
        _open_writers.append(self)
        view.destroyed.connect(lambda *_: _open_writers.remove(self) if self in _open_writers else None)

        # A quantization change surveys again, once the slider has stopped
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
        # Bands: Save codes the image as this many bands of rows, each
        # decodable on its own (so a reader can decode them at the same time),
        # for a small cost per band boundary. Like Save itself, it never
        # changes the preview.
        act, self.bands_slider = vs.make_slider_dialog(view, "Bands", 1, MAX_BANDS, c.bands, self._set_bands)
        file_menu.addAction(act)
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
        # is nothing to survey again.
        quant.addSeparator()
        act, self.sliders["correction"] = vs.make_slider_dialog(view, "Error Correction", 0, 10, c.correction, self._set_correction)
        quant.addAction(act)

        # Block menu: what the survey chose (nothing to set by hand).
        block_menu = bar.addMenu("Block")
        a = QAction("Size", view)
        a.triggered.connect(lambda: self.show_text("Block Size", self.coder.size_text()))
        block_menu.addAction(a)
        a = QAction("Predictors", view)
        a.triggered.connect(lambda: self.show_text("Block Predictors", self.coder.predictor_text()))
        block_menu.addAction(a)

        view.set_image(rgb)
        view.show_window()
        QTimer.singleShot(0, self.run_analysis)    # the preview shows the original until the survey has chosen the blocks

    def _quant_setter(self, key):
        """A quantization slider's change: surveys again once the sliders settle."""
        def set_value(v):
            setattr(self.coder, key, v)
            if not self.updating:
                self.quant_changed()
        return set_value

    def _set_bands(self, v):
        self.coder.bands = v
        self.view.set_status(self.coder.band_estimate())

    def _set_correction(self, v):
        self.coder.correction = v
        if not self.updating and self.coder.survey_current():
            self.apply()

    def quant_changed(self):
        self.view.set_status("waiting to survey…")
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
        """Shows text in a dialog, in a fixed-width font so the counts line up."""
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
        """Surveys in the background with the menus disabled, then applies
        (which prints the listing). If the quantization changed meanwhile, it
        surveys again."""
        if self.analysing:
            return      # done() checks the quantization again
        self.analysing = True
        self.view.set_menus_enabled(False)
        self.view.set_status("surveying…")
        c = self.coder
        quant, shift = c.pixel_quant, c.pixel_shift

        def done(result, error):
            self.analysing = False
            if error is not None:
                print("survey: %r" % error)
            changed = (quant, shift) != (c.pixel_quant, c.pixel_shift)
            if changed and not self.quant_timer.isActive():
                self.run_analysis()
                return
            self.view.set_menus_enabled(True)
            if error is not None:
                self.view.set_status("survey failed")
                return
            self.view.set_status(None)
            if c.survey_current():
                self.apply()
        vs.run_in_background(lambda: c.survey(quant, shift), done)

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
        BlockWriter(sys.argv[1])
    else:
        open_image()
    if not _open_writers:
        return
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
