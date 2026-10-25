#!/usr/bin/env python3
"""
pixel_writer.py version 1.4 -- PixelWriter.java for Python (PySide6).

Codes the red, green and blue channels block by block with context-adaptive
arithmetic coding (see pixel_mapper.py) and saves (to the file "foo", as the
Java version does) in Pixel format 'X' version 4: the same file layout as
PixelWriter.java, so either program's files open in either reader.

    python3 pixel_writer.py [image]

Menus: File, View, Quantization (Pixel Resolution and Color Resolution) and
Block (Block Size; Coding: the pixel values with an exact prediction or with
brightness bands as context, or the prediction errors; Channels: red, green
and blue, or green and the differences red - green and blue - green).
Quantization > Average shrinks the red, green and blue channels by averaging
2 x 2 squares (1 or 2 levels) and codes only the smallest level, plus one
sign bit per pixel per level to restore detail when expanding (the averaging
pyramid in pixel_mapper.py). At start-up the block size that codes the image
smallest is found (up to one block for the whole frame); changing the
coding, the channels or the averaging finds it again.

The coding (class PixelCoder) is separate from the window:

    coder = PixelCoder(viewer_support.read_image("photo.png"))
    coder.survey(); coder.apply(); coder.save("photo.pix")
"""

import math
import os
import sys

import numpy as np

import delta_mapper as dm
import pixel_mapper as px
import resize_mapper as rm
import viewer_support as vs
from java_io import DataOutput

FORMAT_ID = ord('X')   # it was 'P' until that turned out to be the Packet format's id too
FORMAT_VERSION = 4      # 2 added the coding byte, 3 the channels byte, 4 the averaging pyramid


class PixelCoder:
    """Holds an image and the settings; survey() finds the block size,
    apply() codes, save() checks the coding decodes back and writes the file."""

    def __init__(self, rgb, file_length=0):
        self.image_ydim, self.image_xdim = rgb.shape[:2]
        self.source = [rgb[:, :, c].reshape(-1).astype(np.int64) for c in range(3)]   # red, green, blue
        self.file_length = file_length

        self.pixel_quant = 4
        self.pixel_shift = 3
        self.block_size = 64            # replaced by survey()
        self.coding = px.VALUES
        self.channels = px.PRIMARY
        self.pixel_pyramid = 0          # Average: levels of 2 x 2 averaging (0..pixel_mapper.PYRAMID_MAX)
        self.use_saddle = False         # expand with the cross-derivative term too
        self.coded = None
        self.applied = False
        self.quantize()

    def quantize(self):
        """The channels resized (Pixel Resolution) and shifted (Color
        Resolution); with Average, each shrunk to the top level (top, txdim x
        tydim: what is coded), with its sign bits, and expanded back as the
        reader will (shown)."""
        size = dm.get_quantized_size(self.image_xdim, self.image_ydim, self.pixel_quant)
        q = []
        for ch in self.source:
            if self.pixel_quant != 0:
                ch = px.clamp(rm.resize(ch, self.image_xdim, size[0], size[1]))
            q.append(np.asarray(dm.quantize_channel(ch, self.pixel_shift), dtype=np.int64))
        self.quantized, (self.qxdim, self.qydim) = q, size

        max_value = 255 >> self.pixel_shift
        self.txdim, self.tydim = px.pyramid_size(self.qxdim, self.qydim, self.pixel_pyramid)
        if self.pixel_pyramid == 0:
            self.top, self.sign_bits, self.shown = q, [None] * 3, q
            return
        top, bits, shown = [None] * 3, [None] * 3, [None] * 3

        def one(i):
            top[i], bits[i] = px.shrink_pyramid(q[i], self.qxdim, self.qydim, self.pixel_pyramid)
            shown[i] = px.expand_pyramid(top[i], self.qxdim, self.qydim, bits[i], max_value, self.use_saddle)
        vs.parallel(3, one)
        self.top, self.sign_bits, self.shown = top, bits, shown

    def preview(self):
        """The quantized image as pixel_reader will show it."""
        return px.get_rgb(self.shown, self.qxdim, self.image_xdim, self.image_ydim, self.pixel_quant, self.pixel_shift)

    def average_error(self):
        """How far the averaged and expanded channels are from the quantized
        ones (on the 0-255 scale)."""
        r = "  Average error (expanded vs. quantized, 0-255 scale):"
        for i in range(3):
            e = np.abs(self.shown[i] - self.quantized[i]) << self.pixel_shift
            sq = float(np.sum(e.astype(np.float64) * e))
            r += "  %s RMS %.2f, max %d" % (px.CHANNEL_NAMES[i], math.sqrt(sq / len(e)), int(e.max()) if len(e) else 0)
        return r

    def survey(self):
        """Picks the block size that codes the image smallest at the current
        quantization, coding and channels, and prints the sizes tried (Java:
        init)."""
        timer = vs.Timer()
        self.quantize()
        tried = []
        self.block_size = px.find_best_block(self.top, self.txdim, self.tydim, 255 >> self.pixel_shift,
                                             self.pixel_shift, self.coding, self.channels, tried)
        print("Block sizes tried (%s; %s; Pixel Resolution %d, Color Resolution %d, Average %d):"
              % (px.CODING_NAMES[self.coding], px.CHANNEL_SET_NAMES[self.channels], self.pixel_quant, self.pixel_shift,
                 self.pixel_pyramid))
        print("bytes for the 3 channels.")
        print("<= marks the size chosen.")
        for size, nbytes in tried:
            print("  %4d  %12s%s" % (size, "{:,}".format(nbytes), "  <=" if size == self.block_size else ""))
        print("Block size search took " + timer.elapsed())
        print()
        return self.block_size

    def apply(self):
        """Quantizes and codes (what Save writes). Returns the preview."""
        self.applied = False
        self.quantize()
        timer = vs.Timer()
        self.coded = px.encode(self.top, self.txdim, self.tydim, 255 >> self.pixel_shift, self.pixel_shift,
                               self.block_size, self.coding, self.channels)
        nbytes = sum(len(c) for c in self.coded)
        sign = sum(px.sign_bit_bytes(b) for b in self.sign_bits) if self.pixel_pyramid > 0 else 0
        print("%s; %s; Pixel Resolution %d, Color Resolution %d, Average %d%s, block size %d: %s bytes coded%s (%.4f of the raw size) in %s"
              % (px.CODING_NAMES[self.coding], px.CHANNEL_SET_NAMES[self.channels], self.pixel_quant, self.pixel_shift,
                 self.pixel_pyramid, " (saddle)" if self.pixel_pyramid > 0 and self.use_saddle else "",
                 self.block_size, "{:,}".format(nbytes + sign),
                 " ({:,} pixels + {:,} sign bits)".format(nbytes, sign) if self.pixel_pyramid > 0 else "",
                 (nbytes + sign) / (3.0 * self.image_xdim * self.image_ydim), timer.elapsed()))
        if self.pixel_pyramid > 0:
            print(self.average_error())
        self.applied = True
        return self.preview()

    def save(self, filename="foo"):
        """Checks the channels decode back, then writes: FORMAT_ID,
        FORMAT_VERSION, width, height (unsigned shorts), pixel_shift,
        pixel_quant (bytes), block size (short), coding and channels (bytes),
        Average levels and saddle (bytes), then each coded channel in order
        (green, then red and blue or their differences from green): its
        length (int) and bytes; then with Average, the sign bits of red, green
        and blue (pixel_mapper.write_sign_bits). Returns the file size;
        raises IOError if a channel doesn't decode back."""
        timer = vs.Timer()
        back = px.decode(self.coded, self.txdim, self.tydim, 255 >> self.pixel_shift, self.pixel_shift,
                         self.block_size, self.coding, self.channels)
        for i in range(3):
            if not np.array_equal(back[i], self.top[i]):
                raise IOError("Nothing saved: the %s channel does not decode back." % px.CHANNEL_NAMES[i])
        print("Decode check took " + timer.elapsed())

        out = DataOutput()
        out.write_byte(FORMAT_ID); out.write_byte(FORMAT_VERSION)
        out.write_short(self.image_xdim); out.write_short(self.image_ydim)
        out.write_byte(self.pixel_shift); out.write_byte(self.pixel_quant)
        out.write_short(self.block_size)
        out.write_byte(self.coding)
        out.write_byte(self.channels)
        out.write_byte(self.pixel_pyramid); out.write_byte(1 if self.use_saddle else 0)
        for c in self.coded:
            out.write_int(len(c))
            out.write(np.asarray(c, dtype=np.uint8).tobytes())
        if self.pixel_pyramid > 0:
            for bits in self.sign_bits:
                px.write_sign_bits(out, bits)
        with open(filename, "wb") as f:
            f.write(out.to_bytes())
        raw = self.image_xdim * self.image_ydim * 3
        print("Original compression rate: %.4f" % (self.file_length / raw))
        print("Output  compression rate:  %.4f" % (out.size() / raw))
        return out.size()


# =============================================================================
# The window
# =============================================================================

def open_image(parent=None):
    from PySide6.QtWidgets import QFileDialog
    path, _ = QFileDialog.getOpenFileName(parent, "Open Image", "",
                                          "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.pgm *.ppm);;All files (*)")
    if path:
        PixelWriter(path)


_open_writers = []


class PixelWriter:
    def __init__(self, filename):
        from PySide6.QtGui import QAction, QKeySequence

        self.filename = filename
        try:
            rgb = vs.read_image(filename)
        except IOError as e:
            vs.show_error(None, str(e))
            return
        self.coder = c = PixelCoder(rgb, os.path.getsize(filename))
        self.updating = False
        print("Loaded %s, %d x %d" % (filename, c.image_xdim, c.image_ydim))

        self.view = view = vs.ImageWindow("Pixel Writer  " + filename, c.image_xdim, c.image_ydim)
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
        quant.addAction(self._make_pyramid_dialog())

        # Block size, coding and channels: the coded size changes, the picture
        # doesn't. A new coding or channel choice has its own best block size,
        # so choosing one runs the block size search again.
        block_menu = bar.addMenu("Block")
        act, self.block_slider = vs.make_slider_dialog(view, "Block Size", px.BLOCK_MIN, px.block_limit(c.image_xdim, c.image_ydim), c.block_size, self._setter("block_size"))
        block_menu.addAction(act)
        act, _ = vs.make_radio_dialog(view, "Coding", px.CODING_NAMES, c.coding, self.set_coding)
        block_menu.addAction(act)
        act, _ = vs.make_radio_dialog(view, "Channels", px.CHANNEL_SET_NAMES, c.channels, self.set_channels)
        block_menu.addAction(act)

        view.set_image(rgb)
        view.show_window()
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, self.show_initial_image)

    def _make_pyramid_dialog(self):
        """Quantization > Average: the number of averaging levels (- / +)
        and Use Saddle. A new number of levels finds the block size again,
        since the coded image changes size."""
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QAction
        from PySide6.QtWidgets import QCheckBox, QDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout
        c = self.coder
        dialog = QDialog(self.view)
        dialog.setWindowTitle("Average")
        field = QLineEdit(str(c.pixel_pyramid))
        field.setReadOnly(True)
        field.setAlignment(Qt.AlignCenter)
        field.setFixedWidth(field.fontMetrics().horizontalAdvance("000") + 12)
        minus, plus = QPushButton("-"), QPushButton("+")
        saddle = QCheckBox("Use Saddle")
        saddle.setChecked(c.use_saddle)

        def set_levels(v):
            if v == c.pixel_pyramid:
                return
            c.pixel_pyramid = v
            field.setText(str(v))
            self.find_block_size()

        def set_saddle(on):
            c.use_saddle = on
            if c.pixel_pyramid > 0:
                self.apply()
        minus.clicked.connect(lambda: set_levels(c.pixel_pyramid - 1) if c.pixel_pyramid > 0 else None)
        plus.clicked.connect(lambda: set_levels(c.pixel_pyramid + 1) if c.pixel_pyramid < px.PYRAMID_MAX else None)
        saddle.toggled.connect(set_saddle)
        layout = QVBoxLayout(dialog)
        row = QHBoxLayout()
        row.addWidget(QLabel("Levels"))
        row.addWidget(field)
        layout.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(minus)
        row.addWidget(plus)
        layout.addLayout(row)
        layout.addWidget(saddle)
        action = QAction("Average", self.view)
        action.triggered.connect(lambda: vs._open_dialog(self.view, dialog))
        return action

    def _setter(self, key):
        def set_value(v):
            setattr(self.coder, key, v)
            if not self.updating:
                self.apply()
        return set_value

    def set_coding(self, v):
        if v != self.coder.coding:
            self.coder.coding = v
            self.find_block_size()

    def set_channels(self, v):
        if v != self.coder.channels:
            self.coder.channels = v
            self.find_block_size()

    def reset(self):
        c = self.coder
        c.pixel_quant = c.pixel_shift = 0
        self.updating = True
        for key, slider in self.sliders.items():
            slider.setValue(getattr(c, key))
        self.updating = False
        self.apply()

    def block_range(self):
        """Sets the Block Size slider's maximum, the whole frame of the image
        as coded (pixel_mapper.block_limit) or the block size if that is
        larger, and its value, the block size."""
        c = self.coder
        b = c.block_size
        was, self.updating = self.updating, True
        self.block_slider.setMaximum(max(px.block_limit(c.txdim, c.tydim), b))
        self.block_slider.setValue(b)
        c.block_size = b
        self.updating = was

    def apply(self):
        try:
            self.view.set_image(self.coder.apply())
            self.block_range()
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("Apply: %r" % e)

    def show_initial_image(self):
        """Shows the default quantization, then finds the block size."""
        self.coder.quantize()
        self.view.set_image(self.coder.preview())
        self.find_block_size()

    def find_block_size(self):
        """Runs the block size search in the background with the menus
        disabled, then applies the size it found."""
        self.view.set_menus_enabled(False)
        self.view.set_status("finding the best block size...")

        def done(result, error):
            if error is not None:
                print("init: %r" % error)
            self.view.set_status(None)
            self.view.set_menus_enabled(True)
            self.block_range()
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
        except IOError as e:
            vs.show_error(self.view, str(e))
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
        PixelWriter(sys.argv[1])
    else:
        open_image()
    if not _open_writers:
        return
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
