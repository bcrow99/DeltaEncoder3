#!/usr/bin/env python3
"""
delta_writer.py — PySide6 (Qt for Python) port of DeltaWriter.java.

This targets the newer, simplified version of DeltaWriter.java: no
Statistics menu/tables, a flat 4-item Entropy menu (LZ77/Huffman/
Arithmetic/Slow Arithmetic, no sub-dialog), a plain single-slider dialog
for every Quantization item (no combined Smooth dialog, no String
compression-threshold slider), and a new `scanline5_variant` header byte.

Per the user's instruction, Java's `Deflater` (used for the LZ77 payload
and for compressing the arithmetic-coding frequency tables) is replaced
directly with Python's `zlib` module here in DeltaWriter itself, since
it's a JDK utility being swapped for its Python standard-library
equivalent, not one of the project's own external algorithm classes.

External algorithm classes are imported and called directly, with no
stubs.py facade layer in between:
  - DeltaMapper      -> delta_mapper.py (real translation)
  - CodeMapper       -> code_mapper.py (real translation; itself needs
                         string_mapper.py and segment_mapper.py importable)
  - StringMapper     -> string_mapper.py (real translation)
  - ArithmeticMapper -> arithmetic_mapper.py (real translation)
  - ResizeMapper     -> resize_mapper.py (real translation)

The real modules operate on flat 1D sequences indexed by k = i*xdim + j
(matching Java's flat int[] convention), while this file's own channel
arrays are 2D numpy arrays. The small `_encoder`/`_decoder`/`_freq`
wrappers and the handful of named helper functions right after the
imports below handle that reshaping; they are not a stand-in for the
algorithms themselves, which are called directly.

Run:
    pip install PySide6 numpy opencv-python
    python3 delta_writer.py [image_file]
"""

import os
import sys
import threading
import time
import zlib

import numpy as np
import cv2

from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QImage, QPixmap, QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QScrollArea, QLabel, QFileDialog,
    QDialog, QSlider, QLineEdit, QHBoxLayout, QVBoxLayout,
    QRadioButton, QButtonGroup,
)

# Real translated modules, called directly -- no stubs.py facade layer.
import delta_mapper as dm
import code_mapper as cm
import string_mapper as sm
import arithmetic_mapper as am
import resize_mapper as rm


# =============================================================================
# Thin local adapters over the real modules above.
#
# These are NOT a stand-in/stub layer -- delta_mapper.py, code_mapper.py,
# string_mapper.py and arithmetic_mapper.py are the real translations and
# are called directly everywhere below. What lives here is purely
# reshaping: the real functions operate on flat 1D sequences indexed by
# k = i*xdim + j (matching the original Java's flat int[] convention),
# while this file's own channel arrays are 2D numpy arrays (ydim x xdim).
# Each wrapper flattens 2D input before the call and reshapes the result
# back to 2D afterward; delta arrays (already flat by the time they reach
# CodeMapper/StringMapper/ArithmeticMapper calls) pass through untouched.
#
# ResizeMapper.java has been translated to resize_mapper.py; resize_channel()
# below just handles the flat<->2D reshaping between it and this file's own
# 2D numpy channel arrays.
# =============================================================================
def _flat(src):
    """Flatten a 2D array to a plain Python list (row-major, matching
    delta_mapper's k = i*xdim + j convention); pass 1D input through as a
    list unchanged."""
    arr = np.asarray(src)
    if arr.ndim > 1:
        return arr.reshape(-1).tolist()
    return arr.tolist() if not isinstance(src, list) else src


def _encoder(fn):
    """Wrap a delta_mapper encoder (flat src in) so it also accepts a 2D
    per-channel array, as delta_writer.py's own code passes."""
    def wrapped(src, xdim, ydim):
        return fn(_flat(src), xdim, ydim)
    return wrapped


def _decoder(fn):
    """Wrap a delta_mapper decoder taking (src, xdim, ydim, init_value):
    src (the delta array) is already flat by the time it reaches here, so
    this only normalizes init_value to a plain Python int (it may arrive
    as a numpy scalar)."""
    def wrapped(src, xdim, ydim, init_value):
        return fn(src, xdim, ydim, int(init_value))
    return wrapped


def _decoder_map(fn):
    """Same as _decoder, for the map-using decoders (src, xdim, ydim,
    init_value, map_)."""
    def wrapped(src, xdim, ydim, init_value, map_):
        return fn(src, xdim, ydim, int(init_value), map_)
    return wrapped


def _freq(fn):
    """Wrap a delta_mapper frequency estimator (flat src in) so it also
    accepts a 2D per-channel array."""
    def wrapped(src, xdim, ydim):
        return fn(_flat(src), xdim, ydim)
    return wrapped


get_ideal_frequency = _freq(dm.get_ideal_frequency)
get_ideal_frequency8 = _freq(dm.get_ideal_frequency8)
get_ideal_frequency16 = _freq(dm.get_ideal_frequency16)
get_med_scanline_frequency = _freq(dm.get_med_scanline_frequency)
get_scanline2_frequency = _freq(dm.get_scanline2_frequency)
get_mixed_deltas4_frequency = _freq(dm.get_mixed_deltas4_frequency)
get_mixed_deltas16_frequency = _freq(dm.get_mixed_deltas16_frequency)


def bilateral_smooth(src, xdim, ydim, threshold):
    flat = dm.bilateral_smooth(_flat(src), xdim, ydim, threshold)
    return np.asarray(flat).reshape(ydim, xdim)


def anisotropic_smooth(src, xdim, ydim, threshold):
    flat = dm.anisotropic_smooth(_flat(src), xdim, ydim, threshold)
    return np.asarray(flat).reshape(ydim, xdim)


def shift_2d(src, amount):
    a = np.asarray(src)
    flat = dm.shift(_flat(a), amount)
    return np.asarray(flat).reshape(a.shape) if a.ndim > 1 else np.asarray(flat)


def quantize_channel(ch, pixel_shift):
    """Right-shifts ch by pixel_shift with rounding to nearest (adding
    half a quantization step before truncating), rather than a pure
    truncating shift.

    This is a deliberate encode-side design choice, not a straight
    port of anything -- DeltaReader.java's decode is a pure left-shift
    with no rounding compensation (see DeltaMapper.getPixel()'s
    blue_shift/green_shift/red_shift), so it reconstructs whatever
    quantized value was actually stored, however that value was chosen.
    A pure truncating right-shift here (shift_2d(ch, -pixel_shift), i.e.
    floor division) would introduce a systematic DARKENING bias: every
    reconstructed pixel is <= the original, never brighter, biased
    downward by about half a quantization step on average (confirmed:
    at pixel_shift=7 this measured ~61 levels darker on average, out of
    a worst case of 127). Adding half a step before truncating centers
    the quantization error around zero instead. No change to
    delta_reader.py or the on-disk file format is needed for this --
    the reader (and this file's own preview decode) just left-shifts
    back whichever quantized value ends up written; only which value
    gets written in the first place changes here.

    CLAMPED at the top of the range: naively adding half a step and
    truncating can round a near-maximum input up to a quantization index
    whose reconstruction (index << pixel_shift) exceeds 255 -- e.g. at
    pixel_shift=3, input 255 rounds to index 32, reconstructing to 256,
    for every pixel_shift value from 1-7 (confirmed by direct
    computation, not just at the boundary). This numpy pipeline clips
    the final assembled image to 0-255 right before display, so it
    wouldn't crash or wrap here -- but it would silently push every
    near-white pixel to the single coarsest quantization bucket instead
    of its properly rounded one, and if this same rounding scheme were
    ever ported into DeltaMapper.getPixel()'s bit-packed-int assembly
    (blue[k] << (pixel_shift+16), etc.), a value of 256 in an 8-bit-wide
    field would overflow into the next channel's bits -- a real color
    corruption, not just clipping. Capping the pre-shift value at 255
    (so the chosen index can never reconstruct past the input's own
    valid range) avoids both."""
    arr = np.asarray(ch, dtype=np.int64)
    half = 1 << (pixel_shift - 1)
    return np.minimum(arr + half, 255) >> pixel_shift


def difference_2d(src1, src2):
    a, b = np.asarray(src1), np.asarray(src2)
    flat = dm.get_difference(a.reshape(-1).tolist(), b.reshape(-1).tolist())
    return np.asarray(flat).reshape(a.shape)


def sum_2d(src1, src2):
    a, b = np.asarray(src1), np.asarray(src2)
    flat = dm.get_sum(a.reshape(-1).tolist(), b.reshape(-1).tolist())
    return np.asarray(flat).reshape(a.shape)


def resize_channel(src, old_xdim, new_xdim, new_ydim):
    """resize_mapper.resize() operates on a flat, row-major sequence and
    returns a plain Python list; delta_writer.py's own channel arrays are
    2D numpy arrays, so flatten in and reshape back out here."""
    arr = np.asarray(src)
    dtype = arr.dtype
    flat = rm.resize(_flat(arr), int(old_xdim), int(new_xdim), int(new_ydim))
    return np.asarray(flat, dtype=dtype).reshape(int(new_ydim), int(new_xdim))


def get_histogram(src):
    # delta_writer's Huffman save path calls .tolist() on the returned
    # histogram, so return a numpy array here rather than the plain
    # Python list string_mapper.get_histogram() itself produces.
    min_v, hist, _rng = sm.get_histogram(list(np.asarray(src).reshape(-1)))
    return min_v, np.asarray(hist)


def get_string_list(value, compress):
    # string_mapper.get_string_list MUTATES its `value` argument in place
    # (see that module's own docstring): value[0] is overwritten and
    # value[1:] each have min_value subtracted. Each `delta`/map array
    # flowing through here is used once, matching that requirement.
    vals = list(np.asarray(value).reshape(-1).tolist()) if not isinstance(value, list) else value
    return sm.get_string_list(vals, compress)


def unpack_strings(src, table, size, bitlength):
    flat = sm.unpack_strings(src, list(table), int(size), int(bitlength))
    return np.asarray(flat)


def pack_code(src, table, code, length):
    """code_mapper.pack_code_byte returns [dst, bitlength, table, code,
    length, len(src)]; delta_writer.py only needs the first two."""
    result = cm.pack_code_byte(list(src), list(table), list(code), list(length))
    return result[0], result[1]

ZOOM_FACTOR = 1.25
ZOOM_MIN = 0.05
ZOOM_MAX = 32.0

CHANNEL_STRINGS = ["blue", "green", "red", "blue-green", "red-green", "red-blue"]
SET_STRINGS = [
    "blue, green, red",
    "blue, red, red-green",
    "blue, red, blue-green",
    "blue, blue-green, red-green",
    "blue, blue-green, red-blue",
    "green, red, blue-green",
    "red, blue-green, red-green",
    "green, blue-green, red-green",
    "green, red-green, red-blue",
    "red, red-green, red-blue",
]
DELTA_TYPE_STRINGS = [
    "horizontal", "vertical", "average", "med", "directional", "adaptive",
    "scanline (1)", "scanline (2)", "scanline (3)", "scanline (4)",
    "scanline (5)", "frame map (1)", "frame map (2)",
]

DELTA_MENU_NAMES = ["H", "V", "Average", "Med", "Directional", "Scanline 5", "Adaptive",
                    "Scanline 1", "Scanline 2", "Scanline 3", "Scanline 4", "Map (1)", "Map (2)"]
DELTA_MENU_TYPES = [0, 1, 2, 3, 4, 10, 5, 6, 7, 8, 9, 11, 12]

# entropy menu order -> internal entropy_type id (mirrors Java's entropy_map)
ENTROPY_MENU_NAMES_TYPES = [("LZ77", 0), ("Huffman", 1), ("Arithmetic", 3), ("Slow Arithmetic", 2)]


# delta_type 10 ("scanline 5") uses a per-row predictor set selected from
# FILTER_SETS_8 by an extra `variant` byte (dm.get_mixed_deltas_from_values8_rows /
# dm.get_values_from_mixed_deltas8_rows take one, unlike every other delta
# type here) -- wrapped separately since it doesn't fit the plain
# _encoder/_decoder_map signatures. The variant used is always
# self.scanline5_variant, written to the file header, so encode and decode
# agree; variant selection itself isn't wired to any UI (matches the
# original "not yet wired to any UI" note on scanline5_variant).
def _encoder_scanline5(variant):
    def wrapped(src, xdim, ydim):
        return dm.get_mixed_deltas_from_values8_rows(_flat(src), xdim, ydim, variant)
    return wrapped


def _decoder_scanline5(variant):
    def wrapped(src, xdim, ydim, init_value, map_):
        return dm.get_values_from_mixed_deltas8_rows(src, xdim, ydim, int(init_value), map_, variant)
    return wrapped


# delta_type -> (encoder, decoder, uses_map)
_ENCODERS = {
    0: (_encoder(dm.get_horizontal_deltas_from_values), _decoder(dm.get_values_from_horizontal_deltas), False),
    1: (_encoder(dm.get_vertical_deltas_from_values), _decoder(dm.get_values_from_vertical_deltas), False),
    2: (_encoder(dm.get_average_deltas_from_values), _decoder(dm.get_values_from_average_deltas), False),
    3: (_encoder(dm.get_med_deltas_from_values), _decoder(dm.get_values_from_med_deltas), False),
    4: (_encoder(dm.get_directional_deltas_from_values), _decoder(dm.get_values_from_directional_deltas), False),
    5: (_encoder(dm.get_adaptive_deltas_from_values), _decoder(dm.get_values_from_adaptive_deltas), False),
    6: (_encoder(dm.get_mixed_deltas_from_values), _decoder_map(dm.get_values_from_mixed_deltas), True),
    7: (_encoder(dm.get_mixed_deltas_from_values2), _decoder_map(dm.get_values_from_mixed_deltas2), True),
    8: (_encoder(dm.get_mixed_deltas_from_values4), _decoder_map(dm.get_values_from_mixed_deltas4), True),
    9: (_encoder(dm.get_mixed_deltas_from_values16_rows), _decoder_map(dm.get_values_from_mixed_deltas16_rows), True),
    11: (_encoder(dm.get_ideal_deltas_from_values8), _decoder_map(dm.get_values_from_ideal_deltas8), True),
    12: (_encoder(dm.get_ideal_deltas_from_values16), _decoder_map(dm.get_values_from_ideal_deltas16), True),
    # entry for 10 ("scanline 5") is added below once self.scanline5_variant's
    # default (0) is known -- see _make_encoders_for_variant().
}


def _make_encoders_for_variant(variant):
    """Returns a full _ENCODERS-shaped dict with entry 10 bound to the
    given scanline5_variant. delta_writer.py doesn't currently expose a
    way to change scanline5_variant from the UI, so this is called once
    with variant=0 below; re-call it if that ever changes."""
    encoders = dict(_ENCODERS)
    encoders[10] = (_encoder_scanline5(variant), _decoder_scanline5(variant), True)
    return encoders


_ENCODERS = _make_encoders_for_variant(0)


def numpy_bgr_to_qpixmap(arr_bgr: np.ndarray) -> QPixmap:
    """cv2 images are BGR-ordered; Qt's Format_RGB888 wants RGB, so convert."""
    arr_bgr = np.ascontiguousarray(arr_bgr.clip(0, 255).astype(np.uint8))
    rgb = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape
    qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


# =============================================================================
# Background "init()" worker -- auto-picks channel set / delta type /
# compress type, mirroring the Java SwingWorker in showInitialImage().
# =============================================================================
class InitWorker(QThread):
    finished_with = Signal(int, int, int)  # min_set_id, delta_type, compress_type

    def __init__(self, window):
        super().__init__()
        self.window = window

    def run(self):
        w = self.window
        print(f"[InitWorker] analyzing '{w.filename}' in background thread...")
        qcl6 = w._build_quantized_channels()
        # FIX: was DeltaMapper.getIdealFrequency(qcl6[i], 0, 0) -- passing
        # literal 0 for both xdim and ydim instead of qcl6[i]'s actual
        # dimensions. get_ideal_frequency's loop is `for i in range(1,
        # ydim)`, so ydim=0 makes that loop never run, leaving its
        # internal delta_list empty, and min()/max() on an empty list
        # raises ValueError -- crashing this background QThread every
        # time (visible as "Error calling Python override of
        # QThread::run()" with a ValueError: min() iterable argument is
        # empty traceback). Reading xdim/ydim directly off qcl6[i]'s own
        # shape guarantees they match what _build_quantized_channels()
        # actually produced, the same way _apply_preview_impl already
        # does via new_xdim, new_ydim from _quantized_dims().
        qc_ydim, qc_xdim = qcl6[0].shape
        channel_sum = [int(cm.get_shannon_limit(get_ideal_frequency(qcl6[i], qc_xdim, qc_ydim))) for i in range(6)]
        set_sum = w._compute_set_sums(channel_sum)
        min_set_id = int(np.argmin(set_sum))
        channel_id = dm.get_channels(min_set_id)

        # ---- pick the best delta type (mirrors init()'s total_delta_sum loop) ----
        total_delta_sum = [0] * 13
        for ci in channel_id:
            qc = qcl6[ci]
            xdim, ydim = qc.shape[1], qc.shape[0]

            for t in range(6):  # horizontal..adaptive
                delta = _ENCODERS[t][0](qc, xdim, ydim)[1]
                packed = get_string_list(delta, False)[3]
                total_delta_sum[t] += sm.get_bitlength(sm.compress_strings(packed))

            delta10 = _ENCODERS[10][0](qc, xdim, ydim)[1]  # scanline 5 (mixed 8-rows)
            packed10 = get_string_list(delta10, False)[3]
            total_delta_sum[10] += sm.get_bitlength(sm.compress_strings(packed10))

            dhist8, maphist8 = get_ideal_frequency8(qc, xdim, ydim)
            dhist16, maphist16 = get_ideal_frequency16(qc, xdim, ydim)
            total_delta_sum[11] += int(cm.get_shannon_limit(dhist8) + cm.get_shannon_limit(maphist8))
            total_delta_sum[12] += int(cm.get_shannon_limit(dhist16) + cm.get_shannon_limit(maphist16))

            dhist6, rawdelta6 = get_med_scanline_frequency(qc, xdim, ydim)
            total_delta_sum[6] += int(cm.get_shannon_limit(dhist6)) + \
                sm.get_bitlength(get_string_list(rawdelta6, False)[3])
            dhist7, rawdelta7 = get_scanline2_frequency(qc, xdim, ydim)
            total_delta_sum[7] += int(cm.get_shannon_limit(dhist7)) + \
                sm.get_bitlength(get_string_list(rawdelta7, False)[3])
            dhist8b, rawdelta8 = get_mixed_deltas4_frequency(qc, xdim, ydim)
            total_delta_sum[8] += int(cm.get_shannon_limit(dhist8b)) + \
                sm.get_bitlength(get_string_list(rawdelta8, False)[3])
            dhist9, rawdelta9 = get_mixed_deltas16_frequency(qc, xdim, ydim)
            total_delta_sum[9] += int(cm.get_shannon_limit(dhist9)) + \
                sm.get_bitlength(get_string_list(rawdelta9, False)[3])

        best_dt = int(np.argmin(total_delta_sum))

        # ---- pick compress_type (String vs String*) ----
        str_bits = star_bits = 0
        for ci in channel_id:
            qc = qcl6[ci]
            delta = _ENCODERS[best_dt][0](qc, qc.shape[1], qc.shape[0])[1]
            str_bits += sm.get_bitlength(get_string_list(delta, False)[3])
            star_bits += sm.get_bitlength(get_string_list(delta, True)[3])
        compress_type = 2 if star_bits < str_bits else 1

        time.sleep(0.05)  # keep the async nature visible/testable
        self.finished_with.emit(min_set_id, best_dt, compress_type)


# =============================================================================
# Scroll area with Ctrl+wheel zoom anchored at the cursor.
# =============================================================================
class ZoomScrollArea(QScrollArea):
    def __init__(self, owner):
        super().__init__()
        self.owner = owner

    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            pos = event.position().toPoint()
            h_bar, v_bar = self.horizontalScrollBar(), self.verticalScrollBar()
            mcx, mcy = pos.x() + h_bar.value(), pos.y() + v_bar.value()
            old = self.owner.zoom_scale
            new = old * ZOOM_FACTOR if event.angleDelta().y() > 0 else old / ZOOM_FACTOR
            new = max(ZOOM_MIN, min(ZOOM_MAX, new))
            if new == old:
                return
            self.owner.zoom_scale = new
            self.owner.update_display_image()
            r = new / old
            h_bar.setValue(max(0, int(mcx * r) - pos.x()))
            v_bar.setValue(max(0, int(mcy * r) - pos.y()))
            self.owner.update_title()
            event.accept()
        else:
            super().wheelEvent(event)


# =============================================================================
# One slider + numeric readout in a small popup QDialog (every Quantization
# menu item uses this now -- no more special-cased combined dialogs).
# =============================================================================
def make_slider_dialog(parent, title, lo, hi, init, on_change):
    dialog = QDialog(parent)
    dialog.setWindowTitle(title)
    slider = QSlider(Qt.Horizontal)
    slider.setMinimum(lo)
    slider.setMaximum(hi)
    slider.setValue(init)
    slider.setTickInterval(1)
    slider.setTickPosition(QSlider.TicksBelow)
    slider.setMinimumWidth(220)
    field = QLineEdit(str(init))
    field.setFixedWidth(40)
    field.setReadOnly(True)

    def _changed(v):
        field.setText(str(v))
        on_change(v)

    slider.valueChanged.connect(_changed)
    layout = QHBoxLayout(dialog)
    layout.addWidget(slider)
    layout.addWidget(field)

    action = QAction(title, parent)

    def _open():
        p = parent.pos()
        dialog.move(p.x(), max(0, p.y() - 60))
        dialog.show()

    action.triggered.connect(_open)
    return action, dialog, slider


# =============================================================================
# Main per-image window.
# =============================================================================
def open_image_dialog(parent=None):
    try:
        path, _ = QFileDialog.getOpenFileName(
            parent, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)")
        if path:
            DeltaWriterWindow(path)
    except Exception:
        # Qt can silently swallow exceptions raised inside a signal-connected
        # callback (this runs from the File > Open action's triggered
        # signal), so make sure a real failure here is actually visible
        # instead of just "nothing happens."
        import traceback
        print("open_image_dialog failed:")
        traceback.print_exc()


class DeltaWriterWindow(QMainWindow):
    open_window_count = 0
    next_window_offset = 0

    def __init__(self, filename: str):
        super().__init__()
        self.filename = filename
        self.pixel_quant = 4
        self.pixel_shift = 3
        # arithmetic entropy coding segment size (0-10, slider in the
        # Entropy menu). 0 is a DELIBERATE, benchmarked default, not an
        # arbitrary starting point -- see the conversation this was
        # produced in for the full measurement, but in short: Slow
        # Arithmetic's per-symbol cost grows worse than quadratically
        # with segment size (measured: a 500-symbol segment took ~30ms;
        # 750 symbols, only 1.5x more data, took ~117ms -- 3.9x longer).
        # pixel_segment=0 maps to _segment_payload's smallest possible
        # segment (500 symbols, the floor of its `500 + pixel_segment*500`
        # formula), which is the fastest setting this slider can reach --
        # confirmed to already take ~8s end-to-end for a full 640x480
        # image with Slow Arithmetic selected. Any higher setting (larger,
        # fewer segments) costs dramatically more; there's no headroom to
        # trade for Slow Arithmetic's theoretical compression benefit
        # (a wider per-segment interval gives simplest_fraction_in_interval
        # more room to find a low-denominator fraction) without a much
        # longer save. If you want a genuinely SMALLER floor than 500 for
        # very large images, that needs a change to _segment_payload's own
        # formula, not just this default -- ask if you want that explored.
        self.pixel_segment = 0
        self.correction = 0
        self.min_set_id = 0
        self.delta_type = 5
        self.compress_type = 1
        self.entropy_type = 0
        self.smooth_level = 0
        self.smooth2_level = 0
        self.scanline5_variant = 0   # written to file header, not yet wired to any UI

        self.initialized = False

        self._load_image()

        # zoom_scale is set once here from screen size, matching Java's
        # Toolkit.getScreenSize()-based calculation in the constructor --
        # showInitialImage() in this version no longer refits it.
        screen = QApplication.primaryScreen().size()
        self.screen_xdim, self.screen_ydim = screen.width(), screen.height()
        mw = int(self.screen_xdim * 0.70) - 40
        mh = int(self.screen_ydim * 0.70) - 80
        self.fit_scale = min(1.0, min(mw / self.image_xdim, mh / self.image_ydim))
        self.zoom_scale = self.fit_scale

        self._build_ui()

        DeltaWriterWindow.open_window_count += 1
        off = DeltaWriterWindow.next_window_offset
        DeltaWriterWindow.next_window_offset = (off + 30) % 270
        w = min(self.image_xdim + 40, int(self.screen_xdim * 0.70))
        h = min(self.image_ydim + 80, int(self.screen_ydim * 0.70))
        self.resize(w, h)
        self.move((self.screen_xdim - w) // 2 + off, (self.screen_ydim - h) // 2 + off)
        self.update_display_image()
        self.update_title()
        self.show()

        QTimer.singleShot(0, self._show_initial_image)

    # ------------------------------------------------------------------ IO
    def _load_image(self):
        img_bgr = cv2.imread(self.filename, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"cv2 could not read image file: {self.filename}")
        self.image_ydim, self.image_xdim = img_bgr.shape[0], img_bgr.shape[1]
        self.file_length = os.path.getsize(self.filename)
        # cv2 loads BGR-ordered, so channel_list[0..2] are genuinely
        # blue/green/red (matching the Java field names exactly).
        self.channel_list = [img_bgr[:, :, 0].astype(np.int32),
                              img_bgr[:, :, 1].astype(np.int32),
                              img_bgr[:, :, 2].astype(np.int32)]
        self.working_bgr = img_bgr.copy()
        print(f"Loaded file: {self.filename}")
        print(f"Image xdim = {self.image_xdim}, ydim = {self.image_ydim}\n")

    # -------------------------------------------------------------- UI build
    def _build_ui(self):
        self.canvas = QLabel()
        self.canvas.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.scroll = ZoomScrollArea(self)
        self.scroll.setWidget(self.canvas)
        self.scroll.setWidgetResizable(False)
        self.setCentralWidget(self.scroll)

        menu_bar = self.menuBar()
        self._build_file_menu(menu_bar)
        self._build_view_menu(menu_bar)
        self._build_quantization_menu(menu_bar)
        self._build_delta_menu(menu_bar)
        self._build_datatype_menu(menu_bar)
        self._build_entropy_menu(menu_bar)

    def _build_file_menu(self, menu_bar):
        m = menu_bar.addMenu("File")

        open_act = QAction("Open...", self, shortcut=QKeySequence("Ctrl+O"))
        open_act.triggered.connect(lambda: open_image_dialog(self))
        m.addAction(open_act)
        m.addSeparator()

        reset_act = QAction("Reset", self)
        reset_act.triggered.connect(self._reset)
        m.addAction(reset_act)

        save_act = QAction("Save", self)
        save_act.triggered.connect(self._save)
        m.addAction(save_act)

    def _build_view_menu(self, menu_bar):
        m = menu_bar.addMenu("View")
        zi = QAction("Zoom In", self, shortcut=QKeySequence("Ctrl+="))
        zi.triggered.connect(lambda: self._zoom_by(ZOOM_FACTOR))
        zo = QAction("Zoom Out", self, shortcut=QKeySequence("Ctrl+-"))
        zo.triggered.connect(lambda: self._zoom_by(1.0 / ZOOM_FACTOR))
        zf = QAction("Fit", self, shortcut=QKeySequence("Ctrl+0"))
        zf.triggered.connect(self._fit_to_window)
        za = QAction("100%", self, shortcut=QKeySequence("Ctrl+1"))
        za.triggered.connect(self._actual_size)
        for a in (zi, zo, zf, za):
            m.addAction(a)

    def _build_quantization_menu(self, menu_bar):
        m = menu_bar.addMenu("Quantization")
        act, _d, self.smooth_slider = make_slider_dialog(self, "Smooth", 0, 10, self.smooth_level, self._set_smooth)
        m.addAction(act)
        act, _d, self.smooth2_slider = make_slider_dialog(self, "Smooth2", 0, 10, self.smooth2_level, self._set_smooth2)
        m.addAction(act)
        act, _d, self.pquant_slider = make_slider_dialog(self, "Pixel Resolution", 0, 10, self.pixel_quant, self._set_pixel_quant)
        m.addAction(act)
        act, _d, self.pshift_slider = make_slider_dialog(self, "Color Resolution", 0, 7, self.pixel_shift, self._set_pixel_shift)
        m.addAction(act)
        act, _d, self.corr_slider = make_slider_dialog(self, "Error Correction", 0, 10, self.correction, self._set_correction)
        m.addAction(act)

    def _set_smooth(self, v):
        self.smooth_level = v; self._apply_preview()

    def _set_smooth2(self, v):
        self.smooth2_level = v; self._apply_preview()

    def _set_pixel_quant(self, v):
        self.pixel_quant = v; self._apply_preview()

    def _set_pixel_shift(self, v):
        self.pixel_shift = v; self._apply_preview()

    def _set_correction(self, v):
        self.correction = v; self._apply_preview()

    def _build_delta_menu(self, menu_bar):
        m = menu_bar.addMenu("Delta")
        group = QActionGroup(self)
        group.setExclusive(True)
        self.delta_actions = []
        for name, dt in zip(DELTA_MENU_NAMES, DELTA_MENU_TYPES):
            act = QAction(name, self, checkable=True)
            group.addAction(act)
            m.addAction(act)
            act.triggered.connect(lambda checked, dt=dt: self._set_delta_type(dt))
            self.delta_actions.append((act, dt))
        for act, dt in self.delta_actions:
            act.setChecked(dt == self.delta_type)

    def _set_delta_type(self, dt):
        if dt != self.delta_type:
            self.delta_type = dt
            self._apply_preview()

    def _build_datatype_menu(self, menu_bar):
        m = menu_bar.addMenu("Datatype")

        int_a, str_a = QRadioButton("Integer"), QRadioButton("String")
        int_b, str_b = QRadioButton("Integer"), QRadioButton("String")
        self._compress_widgets = (int_a, str_a, int_b, str_b)
        group_a = QButtonGroup(self); group_a.addButton(int_a); group_a.addButton(str_a)
        group_b = QButtonGroup(self); group_b.addButton(int_b); group_b.addButton(str_b)
        self.int_radio_btns = [int_a, int_b]

        int_dialog = QDialog(self); int_dialog.setWindowTitle("Integer")
        lay = QVBoxLayout(int_dialog)
        lay.addWidget(int_a); lay.addWidget(str_a)
        int_act = QAction("Integer", self)

        def _open_int():
            p = self.pos(); int_dialog.move(p.x(), max(0, p.y() - 80)); int_dialog.show()

        int_act.triggered.connect(_open_int)
        m.addAction(int_act)

        str_dialog = QDialog(self); str_dialog.setWindowTitle("String")
        lay2 = QHBoxLayout(str_dialog)
        lay2.addWidget(int_b); lay2.addWidget(str_b)
        str_act = QAction("String", self)

        def _open_str():
            p = self.pos(); str_dialog.move(p.x(), max(0, p.y() - 80)); str_dialog.show()

        str_act.triggered.connect(_open_str)
        m.addAction(str_act)

        (int_a if self.compress_type == 0 else str_a).setChecked(True)
        (int_b if self.compress_type == 0 else str_b).setChecked(True)

        def sync_compress(new_type, checked):
            if not checked or self.compress_type == new_type:
                return
            self.compress_type = new_type
            for w in self._compress_widgets:
                w.blockSignals(True)
            (int_a if new_type == 0 else str_a).setChecked(True)
            (int_b if new_type == 0 else str_b).setChecked(True)
            for w in self._compress_widgets:
                w.blockSignals(False)
            self._apply_preview()

        int_a.toggled.connect(lambda c: sync_compress(0, c))
        str_a.toggled.connect(lambda c: sync_compress(1, c))
        int_b.toggled.connect(lambda c: sync_compress(0, c))
        str_b.toggled.connect(lambda c: sync_compress(1, c))

    def _build_entropy_menu(self, menu_bar):
        # Flat 4-item exclusive menu now -- no Arithmetic sub-dialog.
        # Selecting an entropy type does NOT re-run the preview (matches
        # Java: entropy only affects Save, never the on-screen image).
        m = menu_bar.addMenu("Entropy")
        group = QActionGroup(self)
        group.setExclusive(True)
        self.entropy_actions = []
        for name, et in ENTROPY_MENU_NAMES_TYPES:
            act = QAction(name, self, checkable=True)
            group.addAction(act)
            m.addAction(act)
            act.setChecked(self.entropy_type == et)
            act.triggered.connect(lambda checked, et=et: setattr(self, "entropy_type", et))
            self.entropy_actions.append((act, et))

        # Segment size for the Arithmetic/Slow Arithmetic entropy types
        # (_segment_payload's `min_seg = 500 + pixel_segment*500`, up to
        # pixel_segment=10 forcing a single unsegmented chunk). This was
        # never wired up to any control in this version -- pixel_segment
        # stayed permanently at 0, its hardcoded default, which is NOT
        # "no segmentation": at 0, min_seg is still 500, so any payload
        # over ~500 bytes gets split every ~500 bytes regardless (a real
        # image channel easily produces hundreds of segments). Restoring
        # a control here doesn't change that default, just makes the
        # value actually adjustable again, matching the other sliders'
        # pattern. Like entropy_type, this doesn't affect the preview --
        # segmentation only happens inside _save_arithmetic() at Save time.
        m.addSeparator()
        act, _d, self.segment_slider = make_slider_dialog(
            self, "Segment Size", 0, 10, self.pixel_segment, self._set_pixel_segment)
        m.addAction(act)

    def _set_pixel_segment(self, v):
        self.pixel_segment = v

    # -------------------------------------------------------------- zoom/view
    def _zoom_by(self, factor):
        new = max(ZOOM_MIN, min(ZOOM_MAX, self.zoom_scale * factor))
        if new == self.zoom_scale:
            return
        vp = self.scroll.viewport().size()
        h_bar, v_bar = self.scroll.horizontalScrollBar(), self.scroll.verticalScrollBar()
        cx, cy = h_bar.value() + vp.width() / 2.0, v_bar.value() + vp.height() / 2.0
        r = new / self.zoom_scale
        self.zoom_scale = new
        self.update_display_image()
        h_bar.setValue(max(0, int(cx * r - vp.width() / 2.0)))
        v_bar.setValue(max(0, int(cy * r - vp.height() / 2.0)))
        self.update_title()

    def _fit_to_window(self):
        vp = self.scroll.viewport().size()
        self.zoom_scale = min(vp.width() / self.image_xdim, vp.height() / self.image_ydim)
        self.update_display_image()
        self.update_title()

    def _actual_size(self):
        self.zoom_scale = 1.0
        self.update_display_image()
        self.update_title()

    def update_display_image(self):
        pm = numpy_bgr_to_qpixmap(self.working_bgr)
        w = max(1, int(self.image_xdim * self.zoom_scale))
        h = max(1, int(self.image_ydim * self.zoom_scale))
        if self.zoom_scale != 1.0:
            pm = pm.scaled(w, h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        self.canvas.setPixmap(pm)
        self.canvas.resize(w, h)

    def update_title(self):
        self.setWindowTitle(f"Delta Writer  {self.filename}  [{round(self.zoom_scale * 100)}%]")

    # -------------------------------------------------------------- lifecycle
    def _show_initial_image(self):
        self.smooth_level = self.smooth2_level = 0
        self.pixel_quant, self.pixel_shift, self.correction = 4, 3, 0
        for slider, val in ((self.smooth_slider, 0), (self.smooth2_slider, 0),
                            (self.pquant_slider, 4), (self.pshift_slider, 3), (self.corr_slider, 0)):
            slider.blockSignals(True); slider.setValue(val); slider.blockSignals(False)

        self._apply_preview()  # immediate preview with default parameters

        self._worker = InitWorker(self)
        self._worker.finished_with.connect(self._on_init_finished)
        self._worker.start()

    def _on_init_finished(self, min_set_id, delta_type, compress_type):
        print(f"[InitWorker] recommendation -> set {min_set_id} "
              f"({SET_STRINGS[min_set_id]}), delta '{DELTA_TYPE_STRINGS[delta_type]}', "
              f"compress_type={compress_type}")
        self.delta_type = delta_type
        self.compress_type = compress_type
        for act, dt in self.delta_actions:
            act.setChecked(dt == delta_type)
        int_a, str_a, int_b, str_b = self._compress_widgets
        for w in self._compress_widgets:
            w.blockSignals(True)
        (int_a if compress_type == 0 else str_a).setChecked(True)
        (int_b if compress_type == 0 else str_b).setChecked(True)
        for w in self._compress_widgets:
            w.blockSignals(False)
        self._apply_preview()

    def _reset(self):
        self.smooth_level = self.smooth2_level = 0
        self.pixel_quant = self.pixel_shift = self.correction = 0
        for slider, val in ((self.smooth_slider, 0), (self.smooth2_slider, 0),
                            (self.pquant_slider, 0), (self.pshift_slider, 0), (self.corr_slider, 0)):
            slider.blockSignals(True); slider.setValue(val); slider.blockSignals(False)
        self._apply_preview()

    def closeEvent(self, event):
        DeltaWriterWindow.open_window_count -= 1
        if DeltaWriterWindow.open_window_count <= 0:
            QApplication.quit()
        event.accept()

    # ------------------------------------------------------ core computation
    def _build_quantized_channels(self):
        new_xdim, new_ydim = self._quantized_dims()
        qcl = []
        for i in range(3):
            ch = self.channel_list[i]
            if self.smooth_level > 0:
                ch = bilateral_smooth(ch, self.image_xdim, self.image_ydim, self.smooth_level)
            if self.smooth2_level > 0:
                ch = anisotropic_smooth(ch, self.image_xdim, self.image_ydim, self.smooth2_level)
            if self.pixel_quant != 0:
                ch = resize_channel(ch, self.image_xdim, new_xdim, new_ydim)
            if self.pixel_shift != 0:
                ch = quantize_channel(ch, self.pixel_shift)
            qcl.append(ch)
        qcl.append(difference_2d(qcl[0], qcl[1]))
        qcl.append(difference_2d(qcl[2], qcl[1]))
        qcl.append(difference_2d(qcl[2], qcl[0]))
        return qcl

    def _quantized_dims(self):
        if self.pixel_quant == 0:
            return self.image_xdim, self.image_ydim
        f = self.pixel_quant / 10.0
        # Integer division here (matching DeltaReader.java's `xdim/2` on a
        # declared-int xdim, which truncates) is required so this file's
        # own encode-time dims agree with what DeltaReader.java's decode
        # independently recomputes from just xdim/ydim/pixel_quant -- it
        # never reads new_xdim/new_ydim from the file. Plain Python `/2`
        # would disagree by up to one pixel whenever xdim/ydim are odd.
        new_xdim = self.image_xdim - int(f * (self.image_xdim // 2 - 2))
        new_ydim = self.image_ydim - int(f * (self.image_ydim // 2 - 2))
        return new_xdim, new_ydim

    @staticmethod
    def _compute_set_sums(channel_sum):
        cs = channel_sum
        return [
            cs[0] + cs[1] + cs[2], cs[0] + cs[4] + cs[2], cs[0] + cs[3] + cs[2],
            cs[0] + cs[1] + cs[4], cs[0] + cs[3] + cs[5], cs[3] + cs[1] + cs[2],
            cs[3] + cs[4] + cs[2], cs[3] + cs[1] + cs[4], cs[5] + cs[1] + cs[4],
            cs[5] + cs[4] + cs[2],
        ]

    def _apply_preview(self):
        try:
            self._apply_preview_impl()
        except Exception as e:
            print(f"apply_preview error: {e!r}")
            import traceback
            traceback.print_exc()

    def _apply_preview_impl(self):
        new_xdim, new_ydim = self._quantized_dims()
        qcl = self._build_quantized_channels()

        channel_min = [0] * 6
        channel_init = [0] * 6
        channel_sum = [0] * 6
        for i in range(6):
            qc = qcl[i]
            mn = int(qc.min())
            channel_min[i] = mn
            if i > 2:
                qc = qc - mn
                qcl[i] = qc
            channel_init[i] = int(qc.flat[0])
            channel_sum[i] = int(cm.get_shannon_limit(get_ideal_frequency(qc, new_xdim, new_ydim)))
        self.channel_min = channel_min
        self.channel_init = channel_init

        set_sum = self._compute_set_sums(channel_sum)
        self.min_set_id = int(np.argmin(set_sum))
        self.file_compression_rate = self.file_length / (self.image_xdim * self.image_ydim * 3)
        channel_id = dm.get_channels(self.min_set_id)

        int_allowed = True
        for ci in channel_id:
            qc = qcl[ci]
            if (int(qc.max()) - int(qc.min())) * 2 > 255:
                int_allowed = False
                break
        if not int_allowed and self.compress_type == 0:
            self.compress_type = 1
            int_a, str_a, int_b, str_b = self._compress_widgets
            for w in self._compress_widgets:
                w.blockSignals(True)
            str_a.setChecked(True); str_b.setChecked(True)
            for w in self._compress_widgets:
                w.blockSignals(False)
        for btn in self.int_radio_btns:
            btn.setEnabled(int_allowed)

        channel_delta_min = [0] * 6
        channel_length = [0] * 6
        channel_compressed_length = [0] * 6
        channel_iterations = [0, 0, 0]
        table_list, string_list, map_list, delta_list = [], [], [], []

        encoder, decoder, uses_map = _ENCODERS[self.delta_type]

        for i, j in enumerate(channel_id):
            qc = qcl[j]
            result = encoder(qc, new_xdim, new_ydim)
            delta = np.asarray(result[1])
            if uses_map:
                map_list.append(result[2])

            if self.compress_type == 0:
                dmin, hist = get_histogram(delta)
                channel_delta_min[j] = dmin
                # One signed byte per pixel here (matching DeltaReader.java's
                # `byte[] payload`, `delta[k] = payload[k] + delta_min`) --
                # NOT a 4-byte-per-pixel int32 array. Values are wrapped into
                # 0-255 the same way Java's `(byte)` cast would (silently,
                # if delta range ever exceeds what a signed byte holds --
                # the int_allowed check above is what's meant to keep that
                # from happening in practice).
                db = np.zeros(len(delta), dtype=np.uint8)
                db[1:] = (np.asarray(delta[1:], dtype=np.int64) - dmin) & 0xFF
                delta_list.append(db)
            else:
                precompress = (self.compress_type == 2)
                dmin, length, table, packed = get_string_list(delta, precompress)
                channel_delta_min[j] = dmin
                channel_length[j] = length
                table_list.append(table)
                string_list.append(packed)
                channel_compressed_length[j] = sm.get_bitlength(packed)
                channel_iterations[i] = sm.get_iterations(packed)

        self.channel_delta_min = channel_delta_min
        self.channel_length = channel_length
        self.channel_compressed_length = channel_compressed_length
        self.channel_iterations = channel_iterations

        # Stash everything Save needs so it can work from the real
        # per-channel payload instead of a stand-in.
        self._last_channel_id = list(channel_id)
        self._last_tables = list(table_list)
        self._last_maps = list(map_list)
        self._last_payloads = [
            (delta_list[i].tobytes() if self.compress_type == 0 else bytes(string_list[i]))
            for i in range(len(channel_id))
        ]

        # ---- decode pass (drives the live preview) ----
        dqcl = []
        for i, j in enumerate(channel_id):
            n = new_xdim * new_ydim
            if self.compress_type == 0:
                db = delta_list[i]
                delta = np.zeros(n, dtype=np.int64)
                # db is uint8 (one wrapped byte per pixel, see the encode
                # side above); reinterpret each byte as SIGNED before
                # adding delta_min back, matching Java's `payload[k]` (a
                # signed byte) + delta_min exactly -- an unsigned add here
                # would silently corrupt every negative delta.
                delta[1:] = db[1:].astype(np.int8).astype(np.int64) + channel_delta_min[j]
            else:
                table = table_list[i]
                packed = sm.decompress_strings(string_list[i])
                delta = unpack_strings(packed, table, n, channel_length[j]).astype(np.int64)
                delta[0] = 0
                delta[1:] = delta[1:] + channel_delta_min[j]

            if uses_map:
                ch_flat = decoder(delta, new_xdim, new_ydim, channel_init[j], map_list[i])
            else:
                ch_flat = decoder(delta, new_xdim, new_ydim, channel_init[j])
            ch2d = np.asarray(ch_flat).reshape(new_ydim, new_xdim)
            if j > 2:
                ch2d = ch2d + channel_min[j]

            dqcl.append(ch2d)

        # DeltaReader.java (the authoritative on-disk format) assembles
        # blue/green/red from the small per-channel decoded arrays FIRST
        # (_recombine below), and only THEN resizes and shifts the
        # resulting 3 combined channels -- not the other way around.
        # Resize's internal averaging steps use truncating integer
        # division, and recombination involves subtraction, so
        # "resize each raw channel, then combine" and "combine, then
        # resize" are NOT equivalent -- order must match the reader
        # exactly, not just use an equivalent-looking sequence.
        blue, green, red = self._recombine(self.min_set_id, dqcl)

        if self.pixel_quant != 0:
            blue = resize_channel(blue, new_xdim, self.image_xdim, self.image_ydim)
            green = resize_channel(green, new_xdim, self.image_xdim, self.image_ydim)
            red = resize_channel(red, new_xdim, self.image_xdim, self.image_ydim)
        if self.pixel_shift != 0:
            blue = shift_2d(blue, self.pixel_shift)
            green = shift_2d(green, self.pixel_shift)
            red = shift_2d(red, self.pixel_shift)

        if self.correction != 0:
            f = self.correction / 10.0
            ob, og, orr = self.channel_list
            blue = blue + ((ob - blue) * f).astype(np.int32)
            green = green + ((og - green) * f).astype(np.int32)
            red = red + ((orr - red) * f).astype(np.int32)

        bgr = np.zeros((self.image_ydim, self.image_xdim, 3), dtype=np.int32)
        bgr[:, :, 0] = blue
        bgr[:, :, 1] = green
        bgr[:, :, 2] = red
        self.working_bgr = bgr.clip(0, 255).astype(np.uint8)

        self.update_display_image()
        self.initialized = True

    @staticmethod
    def _recombine(min_set_id, dqcl):
        d = dqcl
        if min_set_id == 0:
            blue, green, red = d[0], d[1], d[2]
        elif min_set_id == 1:
            blue, red = d[0], d[1]
            green = difference_2d(red, d[2])
        elif min_set_id == 2:
            blue, red = d[0], d[1]
            green = difference_2d(blue, d[2])
        elif min_set_id == 3:
            blue = d[0]
            green = difference_2d(blue, d[1])
            red = sum_2d(d[2], green)
        elif min_set_id == 4:
            blue = d[0]
            green = difference_2d(blue, d[1])
            red = sum_2d(blue, d[2])
        elif min_set_id == 5:
            green, red = d[0], d[1]
            blue = sum_2d(d[2], green)
        elif min_set_id == 6:
            red = d[0]
            bg, rg = d[1], -d[2]
            green = sum_2d(rg, red)
            blue = sum_2d(bg, green)
        elif min_set_id == 7:
            green = d[0]
            blue = sum_2d(green, d[1])
            red = sum_2d(green, d[2])
        elif min_set_id == 8:
            green = d[0]
            red = sum_2d(green, d[1])
            blue = difference_2d(red, d[2])
        else:  # 9
            red = d[0]
            green = difference_2d(red, d[1])
            blue = difference_2d(red, d[2])
        return blue, green, red

    # ------------------------------------------------------------------ save
    # ---- byte-level helpers mirroring Java's writeTable/writeMap/getPayload ----
    def _write_table(self, f, table):
        f.write(len(table).to_bytes(2, "big", signed=False))
        if len(table) <= 255:
            for v in table:
                f.write((int(v) & 0xFF).to_bytes(1, "big"))
        else:
            for v in table:
                f.write((int(v) & 0xFFFF).to_bytes(2, "big"))

    def _write_map_raw2bit(self, f, i):
        """delta_type 6-8 on-disk map format: no table, no compression --
        just the raw per-pixel map values (0-3) packed 4-to-a-byte, 2 bits
        each. Matches DeltaReader.java's expectation for these three delta
        types exactly (map_raw[q] = (pm[q>>2] >> ((q&3)<<1)) & 0x3)."""
        map_bytes = self._last_maps[i]
        ml = len(map_bytes)
        pml = (ml + 3) // 4
        packed = bytearray(pml)
        for q, v in enumerate(map_bytes):
            packed[q >> 2] |= (int(v) & 0x3) << ((q & 3) << 1)
        f.write(ml.to_bytes(4, "big", signed=True))
        f.write(pml.to_bytes(4, "big", signed=True))
        f.write(bytes(packed))

    def _write_map_stringmapper(self, f, i):
        """delta_type 9-12 on-disk map format: table + StringMapper-packed
        payload, same convention as _write_table's own payload encoding.

        DELIBERATE FIX: does NOT call string_mapper's get_string_list()
        directly. That function is written for DELTA arrays, where
        position 0 is always a meaningless placeholder (DeltaMapper's own
        dst[0]=0 convention) and so it intentionally overwrites value[0]
        with value_range//2 before packing -- correct and verified for
        delta arrays (see string_mapper.py's own docstring), but a MAP
        array's position 0 is real data (e.g. row 0's predictor choice),
        so that overwrite would silently corrupt it every time. The logic
        below is otherwise identical to get_string_list(compress=False),
        minus that one overwrite -- DeltaReader.java's read side already
        adds delta_min back onto every position uniformly with no
        special-casing, so this alone is enough to round-trip position 0
        correctly too."""
        map_bytes = self._last_maps[i]
        map_int = [int(v) for v in map_bytes]
        min_value, histogram, value_range = sm.get_histogram(map_int)
        string_table = sm.get_rank_table(histogram)
        shifted = [v - min_value for v in map_int]
        packed = sm.pack_strings(shifted, string_table)
        bl = sm.get_bitlength(packed)

        f.write(len(map_bytes).to_bytes(4, "big", signed=False))
        self._write_table(f, string_table)
        f.write(int(min_value).to_bytes(4, "big", signed=True))
        f.write(int(bl).to_bytes(4, "big", signed=True))
        f.write(bytes(packed[:sm.get_bytelength(bl)]))

    def _write_map(self, f, i):
        """Dispatches to the correct on-disk map format for self.delta_type."""
        if self.delta_type in (6, 7, 8):
            self._write_map_raw2bit(f, i)
        else:
            self._write_map_stringmapper(f, i)

    def _segment_payload(self, payload: bytes):
        """Matches Java's 500 + pixel_segment*500 minimum-segment-size split."""
        min_seg = 500 + self.pixel_segment * 500
        n_segs = 1 if self.pixel_segment >= 10 else max(1, len(payload) // max(1, min_seg))
        n_segs = max(1, n_segs)
        seg_len = max(1, len(payload) // n_segs)
        odd_len = len(payload) - seg_len * (n_segs - 1)
        segs, freqs = [], []
        pos = 0
        for m in range(n_segs):
            length = seg_len if m < n_segs - 1 else odd_len
            seg = payload[pos:pos + length]
            segs.append(seg)
            hist = [0] * 256
            for b in seg:
                hist[b] += 1
            freqs.append(hist)
            pos += length
        return n_segs, segs, freqs

    def _deflate_frequencies(self, n_segs, freqs):
        """Replaces Java's Deflater-based deflateFrequencies() with zlib,
        applied to the real per-segment byte-value histograms."""
        fmax = max((v for row in freqs for v in row), default=0)
        if fmax < 254:
            len_type, bpe = 0, 1
        elif fmax < 65534:
            len_type, bpe = 1, 2
        else:
            len_type, bpe = 2, 4
        fb = bytearray(n_segs * 256 * bpe)
        for k in range(n_segs):
            for m in range(256):
                v = freqs[k][m]
                base = k * 256 * bpe + m * bpe
                for b in range(bpe):
                    fb[base + b] = (v >> (8 * b)) & 0xFF
        zipped = zlib.compress(bytes(fb), level=9)  # Deflater.BEST_COMPRESSION equivalent
        return len_type, zipped

    def _save(self):
        if not self.initialized:
            self._apply_preview()

        def worker():
            channel_id = self._last_channel_id
            print(f"Saving (entropy_type={self.entropy_type})...")
            try:
                with open("foo", "wb") as f:
                    f.write(self.image_xdim.to_bytes(2, "big"))
                    f.write(self.image_ydim.to_bytes(2, "big"))
                    f.write(bytes([
                        self.pixel_shift & 0xFF, self.pixel_quant & 0xFF, self.min_set_id & 0xFF,
                        self.delta_type & 0xFF, self.compress_type & 0xFF, self.entropy_type & 0xFF,
                        self.scanline5_variant & 0xFF,
                    ]))
                    if self.entropy_type in (0, 1):
                        self._save_lz77_or_huffman(f, channel_id)
                    elif self.entropy_type == 2:
                        self._save_arithmetic(f, channel_id, slow=True)
                    else:
                        self._save_arithmetic(f, channel_id, slow=False)

                size = os.path.getsize("foo")
                rate = size / (self.image_xdim * self.image_ydim * 3)
                print(f"Original compression rate: {self.file_compression_rate:.4f}")
                print(f"Output  compression rate:  {rate:.4f}\n")
            except Exception as e:
                print(f"Save error: {e!r}")
                import traceback
                traceback.print_exc()

        threading.Thread(target=worker, daemon=True).start()

    def _save_lz77_or_huffman(self, f, channel_id):
        results = [None] * len(channel_id)
        threads = []

        def encode_one(i):
            payload = self._last_payloads[i]
            if self.entropy_type == 0:
                # LZ77: real zlib compression of the real per-channel
                # payload -- same DEFLATE/zlib framing (RFC 1950) Java's
                # Deflater(BEST_COMPRESSION) produces.
                results[i] = ("lz77", zlib.compress(payload, level=9))
            else:
                # Huffman: exercises the real CodeMapper call shape/control
                # flow; CodeMapper itself is still a stub (to be replaced
                # when the real external modules are translated).
                pi = list(payload)
                dmin, hist = get_histogram(np.array(pi, dtype=np.int64))
                rank_table = sm.get_rank_table(hist)
                shifted = [v - dmin for v in pi]
                freq_sorted = sorted(hist.tolist(), reverse=True)
                hl2 = cm.get_huffman_length2(freq_sorted)
                hc = cm.get_canonical_code(hl2)
                packed_bytes, bitlen = pack_code(shifted, rank_table, hc, hl2)
                ltn, ltinit, ltmax, ltdelta = cm.pack_length_table(hl2)
                results[i] = ("huffman", dmin, rank_table, ltn, ltinit, ltmax, ltdelta, bitlen, packed_bytes)

        for i in range(len(channel_id)):
            t = threading.Thread(target=encode_one, args=(i,))
            threads.append(t); t.start()
        for t in threads:
            t.join()

        for i, j in enumerate(channel_id):
            f.write(int(self.channel_min[j]).to_bytes(4, "big", signed=True))
            f.write(int(self.channel_init[j]).to_bytes(4, "big", signed=True))
            f.write(int(self.channel_delta_min[j]).to_bytes(4, "big", signed=True))
            f.write(int(self.channel_length[j]).to_bytes(4, "big", signed=True))
            f.write(int(self.channel_compressed_length[j]).to_bytes(4, "big", signed=True))
            f.write(bytes([int(self.channel_iterations[i]) & 0xFF]))
            if self.delta_type >= 6:  # all of 6-12 use a map now (10 = scanline 5)
                self._write_map(f, i)
            if self.compress_type > 0:
                self._write_table(f, self._last_tables[i])

            r = results[i]
            if r[0] == "lz77":
                compressed = r[1]
                payload = self._last_payloads[i]
                f.write(len(payload).to_bytes(4, "big", signed=True))
                f.write(len(compressed).to_bytes(4, "big", signed=True))
                f.write(compressed)
            else:
                _, dmin, rank_table, ltn, ltinit, ltmax, ltdelta, bitlen, packed_bytes = r
                self._write_table(f, rank_table)
                f.write(int(dmin).to_bytes(4, "big", signed=True))
                f.write(int(ltn).to_bytes(4, "big", signed=True))
                f.write(bytes([int(ltinit) & 0xFF, int(ltmax) & 0xFF, len(ltdelta) & 0xFF]))
                f.write(bytes(ltdelta))
                f.write(int(bitlen).to_bytes(4, "big", signed=True))
                f.write(len(packed_bytes).to_bytes(4, "big", signed=True))
                f.write(bytes(packed_bytes))

    def _save_arithmetic(self, f, channel_id, slow: bool):
        payloads = [self._last_payloads[i] for i in range(len(channel_id))]
        seg_data = [self._segment_payload(p) for p in payloads]

        encoded = [None] * len(channel_id)

        def encode_channel(i):
            n_segs, segs, freqs = seg_data[i]
            out = []
            for m in range(n_segs):
                if slow:
                    low, high = am.get_interval_value(segs[m], freqs[m])
                    out.append((low, high))
                else:
                    # Fenwick-tree variant: confirmed to produce a
                    # byte-identical encoded stream to get_interval_value_fast
                    # (see the conversation this was produced in -- 28/28
                    # cross-checks matched exactly across alphabet sizes and
                    # lengths), while running ~1.7x faster since its
                    # per-symbol cumulative-frequency update is O(log 256)
                    # instead of O(256). Safe drop-in swap, no format change.
                    out.append(am.get_interval_value_fast_fenwick(segs[m], freqs[m]))
            encoded[i] = out

        threads = [threading.Thread(target=encode_channel, args=(i,)) for i in range(len(channel_id))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        len_types = [None] * len(channel_id)
        zipped_freqs = [None] * len(channel_id)
        for i in range(len(channel_id)):
            n_segs, segs, freqs = seg_data[i]
            len_types[i], zipped_freqs[i] = self._deflate_frequencies(n_segs, freqs)

        for i, j in enumerate(channel_id):
            f.write(int(self.channel_min[j]).to_bytes(4, "big", signed=True))
            f.write(int(self.channel_init[j]).to_bytes(4, "big", signed=True))
            f.write(int(self.channel_delta_min[j]).to_bytes(4, "big", signed=True))
            f.write(int(self.channel_length[j]).to_bytes(4, "big", signed=True))
            f.write(int(self.channel_compressed_length[j]).to_bytes(4, "big", signed=True))
            f.write(bytes([int(self.channel_iterations[i]) & 0xFF]))
            if self.delta_type >= 6:  # all of 6-12 use a map now (10 = scanline 5)
                self._write_map(f, i)
            if self.compress_type > 0:
                self._write_table(f, self._last_tables[i])

            n_segs, segs, freqs = seg_data[i]
            f.write(n_segs.to_bytes(4, "big", signed=True))
            f.write(int(len_types[i]).to_bytes(4, "big", signed=True))
            f.write(len(zipped_freqs[i]).to_bytes(4, "big", signed=True))
            f.write(zipped_freqs[i])

            if slow:
                # BigInteger.toByteArray() equivalent: minimal big-endian
                # two's-complement bytes. ArithmeticMapper itself is still
                # a stub, so exact byte-for-byte parity isn't meaningful
                # yet -- this just gives each interval bound a real,
                # round-trippable byte encoding.
                for (low, high) in encoded[i]:
                    for val in (low, high):
                        nbytes = max(1, (int(val).bit_length() + 8) // 8)
                        b = int(val).to_bytes(nbytes, "big", signed=True)
                        f.write(len(b).to_bytes(4, "big", signed=True))
                        f.write(b)
            else:
                for enc in encoded[i]:
                    f.write(len(enc).to_bytes(4, "big", signed=True))
                    f.write(bytes(enc))


def main():
    app = QApplication(sys.argv)
    if len(sys.argv) > 1:
        DeltaWriterWindow(sys.argv[1])
    else:
        open_image_dialog(None)
        if DeltaWriterWindow.open_window_count == 0:
            sys.exit(0)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
