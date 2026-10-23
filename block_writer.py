#!/usr/bin/env python3
"""
block_writer.py version 2.0 -- BlockWriter.java for Python (PySide6).

A writer with one delta type and one entropy coder: the block map (delta
type 13) with Context-coded deltas and a coded map. Each block uses one
predictor from its channel's set, and a set can be any of the 255
predictors delta_mapper has ids for. The startup analysis (and Find Best)
picks the channel set, then surveys the sets (delta_mapper.survey_block_sets:
per channel, predictors added in greedy order by estimated coded bits,
blocks assigned by the estimate, and the real coded size measured), and
also codes with each named set (find_best_block); it keeps whichever is
smallest -- actual sizes, not estimates. The sets are surveyed per block
shape: a shape picked by hand uses its own. Saves (to the file
"foo", as the Java version does) in Block format 'B' version 3, read by
block_reader.py and BlockReader.java.

    python3 block_writer.py [image]

The coding (class BlockCoder) is separate from the window:

    coder = BlockCoder(viewer_support.read_image("photo.png"))
    coder.survey(); coder.apply(); coder.save("photo.blk")
"""

import os
import sys

import numpy as np

import code_mapper as cm
import delta_mapper as dm
import resize_mapper as rm
import viewer_support as vs
from delta_writer import context_code
from java_io import DataOutput

SURVEYED = -1
FORMAT_ID = ord('B')
FORMAT_VERSION = 3   # 3: one predictor table (delta_mapper.PREDICTOR_TABLE) and cross-channel block maps


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
    """Holds an image and the settings; survey() picks the block size and
    predictor set, apply() makes the deltas (and the preview), save() codes
    and writes the file."""

    def __init__(self, rgb, file_length=0):
        self.image_ydim, self.image_xdim = rgb.shape[:2]
        self.source = [rgb[:, :, c].reshape(-1).astype(np.int64) for c in range(3)]
        self.file_length = file_length

        self.pixel_quant = 4
        self.pixel_shift = 3
        self.correction = 0         # preview only
        self.min_set_id = 0
        self.block_size = dm.BLOCK_DEFAULT   # the square's size
        self.block_shape = 0        # 0: square (block_size); i + 1: dm.BLOCK_SHAPES[i]
        self.survey_shapes = True   # Find Best also tries the rectangles and lines
        self.block_set = 0          # a named set (dm.BLOCK_SET_NAMES), or SURVEYED
        self.survey_sets = None     # each channel's surveyed predictor ids, by block shape: every
                                    # shape the last analysis tried, plus any surveyed since

        self.channel_sum = [0] * 6
        self.channel_min = [0] * 6
        self.channel_init = [0] * 6
        self.applied = False

    def block(self):
        """The block map's shape code (dm.block_width / dm.block_height)."""
        return self.block_size if self.block_shape == 0 else dm.SHAPE_FIRST + self.block_shape - 1

    def quantized_channels(self, size):
        """(the six candidate channels after resizing and quantizing, their
        minimums, their first values)."""
        q = [None] * 3

        def one(i):
            ch = self.source[i]
            if self.pixel_quant != 0:
                ch = rm.resize(ch, self.image_xdim, size[0], size[1])
            q[i] = dm.quantize_channel(ch, self.pixel_shift)
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

    def survey(self):
        """Picks the channel set with the smallest entropy estimate, then the
        block size and predictor sets that code smallest: the surveyed sets
        (dm.survey_block_sets, kept in survey_sets) or a named set
        (dm.find_best_block) (Java: init). Sets and returns (block size,
        named set or SURVEYED)."""
        timer = vs.Timer()
        size = dm.get_quantized_size(self.image_xdim, self.image_ydim, self.pixel_quant)
        qc, _, _ = self.quantized_channels(size)
        set_id = best_set(self.channel_sums(qc, size))
        ids = dm.get_channels(set_id)
        best, table = dm.find_best_block([qc[i] for i in ids], size[0], size[1])
        print("Channel set: " + dm.SET_NAMES[set_id])
        print(dm.get_block_table(table, best), end="")
        named = table[best[1]][dm.BLOCK_SEARCH_SIZES.index(best[0])]
        print("Named sets took " + timer.elapsed())
        print()
        t = vs.Timer()
        survey = dm.survey_block_sets([qc[i] for i in ids], size[0], size[1], self.survey_shapes)
        self.survey_sets = {b: t.ids for b, t in survey.tried.items()}
        print(survey.report, end="")
        print(dm.describe_group_use(survey, dm.get_block_deltas_by_cost([qc[i] for i in ids], size[0], size[1], survey.block, survey.ids)), end="")
        print("Survey took " + t.elapsed())
        use_survey = survey.bytes <= named
        print("Surveyed sets (block %s) %s bytes, best named set (%s, block %d) %s bytes: using the %s."
              % (dm.shape_name(survey.block), format(survey.bytes, ","), dm.BLOCK_SET_NAMES[best[1]], best[0], format(named, ","),
                 "surveyed sets" if use_survey else "named set"))
        print("Analysis took " + timer.elapsed())
        print()
        best = (survey.block, SURVEYED) if use_survey else best
        block, self.block_set = best
        if block < dm.SHAPE_FIRST:
            self.block_size, self.block_shape = block, 0
        else:
            self.block_shape = block - dm.SHAPE_FIRST + 1
        return best

    def needs_survey(self):
        """Whether the surveyed sets are chosen but none exist yet for this shape."""
        return self.block_set == SURVEYED and self.survey_sets is not None and self.block() not in self.survey_sets

    def survey_shape(self, block=None):
        """Surveys each channel's predictor ids at one block shape (the
        current one by default), for the current quantization and channel
        set, and keeps them in survey_sets (Java: surveyShape)."""
        block = self.block() if block is None else block
        timer = vs.Timer()
        size = dm.get_quantized_size(self.image_xdim, self.image_ydim, self.pixel_quant)
        qc, _, _ = self.quantized_channels(size)
        ids = dm.get_channels(best_set(self.channel_sums(qc, size)))
        survey = dm.survey_at_block_size([qc[i] for i in ids], size[0], size[1], block)
        print("Surveyed sets for block %s: %s bytes, %d, %d and %d predictors (took %s)" % (dm.shape_name(block), format(survey.bytes, ","),
              len(survey.ids[0]), len(survey.ids[1]), len(survey.ids[2]), timer.elapsed()))
        self.survey_sets[block] = survey.ids
        return survey.ids

    def apply(self):
        """Quantizes, picks the channel set, makes the block map deltas (what
        Save codes), then rebuilds the image from them. Returns the preview
        as an (ydim, xdim, 3) uint8 array."""
        self.applied = False
        size = dm.get_quantized_size(self.image_xdim, self.image_ydim, self.pixel_quant)
        w, h = size
        qc, self.channel_min, self.channel_init = self.quantized_channels(size)
        self.channel_sum = self.channel_sums(qc, size)
        self.min_set_id = best_set(self.channel_sum)
        ids = dm.get_channels(self.min_set_id)
        deltas, maps, decoded = ([None] * 3 for _ in range(3))
        # Surveyed sets: blocks assigned by estimated coded size, as the
        # survey does (channels in order). Named sets: by smallest |delta|.
        surveyed = None
        survey_ids = self.survey_sets.get(self.block()) if self.survey_sets is not None else None
        if self.block_set == SURVEYED and survey_ids is not None:
            surveyed = dm.get_block_deltas_by_cost([qc[j] for j in ids], w, h, self.block(), survey_ids)

        raw = [None] * 3   # decoded, before the minimum: the next channel's reference

        def channel(i):
            j = ids[i]
            if surveyed is not None:
                d, m = surveyed[i]
            else:
                d, m, _ = dm.get_deltas(qc[j], w, h, 13, 0, self.block(), max(0, self.block_set),
                                       ref=qc[ids[i - 1]] if i > 0 else None)
            deltas[i], maps[i] = d, m
            ch = dm.get_values_from_deltas(d, w, h, self.channel_init[j], 13, m, 0, ref=raw[i - 1] if i > 0 else None)
            raw[i] = ch
            if j > 2:
                ch = ch + self.channel_min[j]
            decoded[i] = ch
        vs.channels(True, 3, channel)
        self.delta_list, self.map_list, self.delta_xdim = deltas, maps, w

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
        """Header: FORMAT_ID, FORMAT_VERSION, width, height (unsigned shorts),
        pixel_shift, pixel_quant, channel set. Then per channel: int min, int
        init, the block map (write_map, type 13; it carries the block size and
        the channel's predictor ids), then the context-coded deltas. Channels 1 and 2 use
        the ones before them as context, so the reader decodes in order.
        Returns the file size."""
        timer = vs.Timer()
        coded = context_code(self.delta_list, self.delta_xdim)
        print("Entropy coding took " + timer.elapsed())

        ids = dm.get_channels(self.min_set_id)
        out = DataOutput()
        out.write_byte(FORMAT_ID); out.write_byte(FORMAT_VERSION)
        out.write_short(self.image_xdim); out.write_short(self.image_ydim)
        out.write_byte(self.pixel_shift); out.write_byte(self.pixel_quant); out.write_byte(self.min_set_id)
        for i in range(3):
            j = ids[i]
            out.write_int(self.channel_min[j]); out.write_int(self.channel_init[j])
            dm.write_map(out, 13, self.map_list[i], self.map_list[i - 1] if i > 0 else None, self.delta_xdim)
            out.write(coded[i])

        with open(filename, "wb") as f:
            f.write(out.to_bytes())
        raw = self.image_xdim * self.image_ydim * 3
        print("Block %s, predictors %s: %d bytes" % (dm.shape_name(self.block()), ", ".join(dm.describe_map_set(m) for m in self.map_list), out.size()))
        print("Original compression rate: %.4f" % (self.file_length / raw))
        print("Output  compression rate:  %.4f" % (out.size() / raw))
        print()
        return out.size()


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
        from PySide6.QtGui import QAction, QKeySequence

        self.filename = filename
        try:
            rgb = vs.read_image(filename)
        except IOError as e:
            vs.show_error(None, str(e))
            return
        self.coder = c = BlockCoder(rgb, os.path.getsize(filename))
        self.updating = False
        print("Loaded %s, %d x %d" % (filename, c.image_xdim, c.image_ydim))

        self.view = view = vs.ImageWindow("Block Writer  " + filename, c.image_xdim, c.image_ydim)
        bar = view.menuBar()
        _open_writers.append(self)
        view.destroyed.connect(lambda *_: _open_writers.remove(self) if self in _open_writers else None)

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
            act, self.sliders[key] = vs.make_slider_dialog(view, title, lo, hi, getattr(c, key), self._setter(key))
            quant.addAction(act)
        # Error Correction blends the preview back toward the original by
        # correction/10. Preview only; the saved file is unaffected.
        quant.addSeparator()
        act, self.sliders["correction"] = vs.make_slider_dialog(view, "Error Correction", 0, 10, c.correction, self._setter("correction"))
        quant.addAction(act)

        # Block menu: the settings, and Find Best to rerun the analysis
        # (worth doing after changing the quantization: the surveyed sets are
        # kept until then). Block Predictors offers the surveyed sets
        # (available once an analysis has run) and the named sets.
        block_menu = bar.addMenu("Block")
        act, self.block_spinner = vs.make_spinner_dialog(view, "Block Size", dm.BLOCK_MIN, dm.BLOCK_MAX, c.block_size, self._block_setter("block_size"))
        block_menu.addAction(act)
        # Block Shape: squares (of Block Size), or a rectangle or line.
        shape_names = ["Square (Block Size)"] + [dm.shape_name(dm.SHAPE_FIRST + i) + (" (line)" if h == 1 else "")
                                                 for i, (_, h) in enumerate(dm.BLOCK_SHAPES)]
        act, self.shape_buttons = vs.make_radio_dialog(view, "Block Shape", shape_names, c.block_shape, self._block_setter("block_shape"))
        block_menu.addAction(act)
        set_names = ["Surveyed (per channel)"] + list(dm.BLOCK_SET_NAMES)
        set_block = self._block_setter("block_set")
        act, self.set_buttons = vs.make_radio_dialog(view, "Block Predictors", set_names, c.block_set + 1, lambda v: set_block(v - 1))
        self.set_buttons[0].setEnabled(False)      # until the first analysis
        block_menu.addAction(act)
        block_menu.addSeparator()
        a = QAction("Find Best Tries Rectangles", view, checkable=True, checked=c.survey_shapes)
        a.toggled.connect(lambda on: setattr(c, "survey_shapes", on))
        block_menu.addAction(a)
        a = QAction("Find Best", view)
        a.triggered.connect(self.run_analysis)
        block_menu.addAction(a)

        view.set_image(rgb)
        view.show_window()
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, self.show_initial_image)

    def _setter(self, key):
        def set_value(v):
            setattr(self.coder, key, v)
            if not self.updating:
                self.apply()
        return set_value

    def _block_setter(self, key):
        """As _setter, for the block settings: see update."""
        def set_value(v):
            setattr(self.coder, key, v)
            if not self.updating:
                self.update()
        return set_value

    def update(self):
        """After a change to the block settings: with the surveyed sets chosen
        and none yet for this shape, surveys them first (in the background),
        so a shape picked by hand gets the sets that suit it; otherwise just
        applies."""
        if not self.coder.needs_survey():
            self.apply()
            return
        block = self.coder.block()
        self.view.set_menus_enabled(False)
        self.view.set_status("surveying block %s…" % dm.shape_name(block))

        def done(result, error):
            if error is not None:
                print("survey: %r" % error)
            self.view.set_menus_enabled(True)
            self.view.set_status(None)
            self.apply()
        vs.run_in_background(lambda: self.coder.survey_shape(block), done)

    def reset(self):
        c = self.coder
        c.pixel_quant = c.pixel_shift = c.correction = 0
        self.updating = True
        for key, slider in self.sliders.items():
            slider.setValue(getattr(c, key))
        self.updating = False
        self.apply()

    def apply(self):
        try:
            self.view.set_image(self.coder.apply())
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("Apply: %r" % e)

    def show_initial_image(self):
        self.apply()
        self.run_analysis()

    def run_analysis(self):
        """Runs the analysis in the background with the menus disabled, then
        shows the chosen settings in the controls and applies them."""
        self.view.set_menus_enabled(False)
        self.view.set_status("analysing…")

        def done(result, error):
            if error is not None:
                print("init: %r" % error)
            else:
                self.updating = True
                self.block_spinner.setValue(self.coder.block_size)
                self.shape_buttons[self.coder.block_shape].setChecked(True)
                self.set_buttons[self.coder.block_set + 1].setChecked(True)
                self.set_buttons[0].setEnabled(self.coder.survey_sets is not None)
                self.updating = False
            self.view.set_menus_enabled(True)
            self.view.set_status(None)
            self.apply()
        vs.run_in_background(self.coder.survey, done)

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
