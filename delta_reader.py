#!/usr/bin/env python3
"""
delta_reader.py — PySide6 (Qt for Python) port of DeltaReader.java.

Reads a file written by DeltaWriter.java's format and displays the
decoded image in a zoomable window, mirroring DeltaReader.java's Swing
UI feature-for-feature (Ctrl+wheel zoom anchored at the cursor, Fit,
Actual Size, per-channel decode threads, an optional resize thread when
xdim > 600).

External algorithm classes are imported and called directly (no stubs
layer), same as delta_writer.py:
  - DeltaMapper      -> delta_mapper.py
  - CodeMapper       -> code_mapper.py
  - StringMapper     -> string_mapper.py
  - ArithmeticMapper -> arithmetic_mapper.py
  - ResizeMapper     -> resize_mapper.py

Java's `Inflater` is replaced with Python's stdlib `zlib.decompress`,
mirroring how delta_writer.py already replaces `Deflater` with
`zlib.compress` -- a JDK-utility swap, not one of the project's own
external algorithm classes.

COMPATIBILITY NOTE (read before assuming this round-trips with our
delta_writer.py): this file is a faithful, literal translation of the
DeltaReader.java given, and that Java does not appear to match
delta_writer.py's own on-disk conventions in two places:
  1. delta_type == 10 here means "scanline (5)" (decoded via
     DeltaMapper.getValuesFromMixedDeltas8Rows with scanline5_variant).
     In delta_writer.py, delta_type == 10 is *gradient* -- a different
     algorithm entirely.
  2. For delta_type 6-8, this reader expects the per-row map to be a raw
     2-bit-packed array (4 map values per byte, no table, no
     compression). delta_writer.py's own _write_map() always writes maps
     through the StringMapper-compressed format instead (the format this
     reader expects only for delta_type 9-12).
  3. The quantized-dimension formula also differs: this file uses
     integer division for `xdim // 2` (matching Java's `int` arithmetic
     exactly, since xdim is declared `int` there), while
     delta_writer.py's _quantized_dims() uses Python's true division
     (`self.image_xdim / 2`). These agree whenever xdim/ydim are even,
     and can differ by the rounding of a single pixel when they're odd.
Together these mean a file saved by our delta_writer.py will only be
read correctly by this module for delta_type 0-5 (which have no map and
are unaffected by point 2) and even image dimensions (point 3); delta
types 6-12, or odd dimensions with pixel_quant != 0, are a genuine
format mismatch between the two files as given, not a translation bug
here. See the conversation this was produced in for what to do about it
(reconcile the two, or get the real newer DeltaWriter.java this reader
was written against).

Run:
    pip install PySide6 numpy opencv-python
    python3 delta_reader.py <filename>
"""

import struct
import sys
import threading
import zlib

import numpy as np
import cv2

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QAction, QKeySequence
from PySide6.QtWidgets import QApplication, QMainWindow, QScrollArea, QLabel

import delta_mapper as dm
import code_mapper as cm
import string_mapper as sm
import resize_mapper as rm
import arithmetic_mapper as am

ZOOM_FACTOR = 1.25
ZOOM_MIN = 0.05
ZOOM_MAX = 32.0

SET_STRING = [
    "blue, green, and red.",
    "blue, red, and red-green.",
    "blue, red, and blue-green.",
    "blue, blue-green, and red-green.",
    "blue, blue-green, and red-blue.",
    "green, red, and blue-green.",
    "red, blue-green, and red-green.",
    "green, blue-green, and red-green.",
    "green, red-green, and red-blue.",
    "red, red-green, red-blue.",
]

DELTA_TYPE_STRING = [
    "horizontal", "vertical", "average", "med", "directional",
    "adaptive", "scanline (1)", "scanline (2)", "scanline (3)", "scanline (4)",
    "scanline (5)", "frame map", "frame map (2)",
]

ENTROPY_TYPE_STRING = ["LZ77", "Huffman", "Arithmetic", "Fast Arithmetic"]


# =============================================================================
# Delta-decoder dispatch, mirroring DeltaReader.java's Decompressor.run()
# if/else chain by delta_type. Values are (decoder, uses_map, uses_variant).
# =============================================================================
_DECODERS = {
    0: (dm.get_values_from_horizontal_deltas, False, False),
    1: (dm.get_values_from_vertical_deltas, False, False),
    2: (dm.get_values_from_average_deltas, False, False),
    3: (dm.get_values_from_med_deltas, False, False),
    4: (dm.get_values_from_directional_deltas, False, False),
    5: (dm.get_values_from_adaptive_deltas, False, False),
    6: (dm.get_values_from_mixed_deltas, True, False),
    7: (dm.get_values_from_mixed_deltas2, True, False),
    8: (dm.get_values_from_mixed_deltas4, True, False),
    9: (dm.get_values_from_mixed_deltas16_rows, True, False),
    11: (dm.get_values_from_ideal_deltas8, True, False),
    12: (dm.get_values_from_ideal_deltas16, True, False),
    10: (dm.get_values_from_mixed_deltas8_rows, True, True),  # "scanline (5)"
}


# =============================================================================
# Binary reader -- mirrors java.io.DataInputStream's big-endian, signed
# read*() methods used throughout DeltaReader.java.
# =============================================================================
class BinReader:
    def __init__(self, f):
        self.f = f

    def read_byte(self) -> int:
        """Java DataInputStream.readByte(): signed 8-bit."""
        return struct.unpack(">b", self._read_n(1))[0]

    def read_ubyte(self) -> int:
        """Java: `in.readByte() & 0xFF` -- unsigned 8-bit."""
        return self._read_n(1)[0]

    def read_short(self) -> int:
        """Java DataInputStream.readShort(): signed 16-bit, big-endian."""
        return struct.unpack(">h", self._read_n(2))[0]

    def read_int(self) -> int:
        """Java DataInputStream.readInt(): signed 32-bit, big-endian."""
        return struct.unpack(">i", self._read_n(4))[0]

    def read_fully(self, n: int) -> bytes:
        return self._read_n(n)

    def _read_n(self, n: int) -> bytes:
        data = self.f.read(n)
        if len(data) != n:
            raise EOFError(f"expected {n} bytes, got {len(data)}")
        return data


def read_table(r: BinReader):
    """Java: readTable(DataInputStream) -> int[]."""
    tl = r.read_short()
    max_byte = 255  # Byte.MAX_VALUE * 2 + 1
    tbl = [0] * tl
    if tl <= max_byte:
        for k in range(tl):
            v = r.read_byte()
            if v < 0:
                v = max_byte + 1 + v
            tbl[k] = v
    else:
        for k in range(tl):
            tbl[k] = r.read_short()
    return tbl


def _inflate(zdata: bytes, expected_len: int) -> bytes:
    """Java's Inflater(); Python's zlib.decompress covers the same
    zlib/DEFLATE (RFC 1950) stream Deflater(BEST_COMPRESSION) produces."""
    out = zlib.decompress(zdata)
    if len(out) != expected_len:
        # Java's Inflater.inflate(byte[]) silently accepts a short fill if
        # the buffer is larger than needed; here we only warn, since a
        # length mismatch beyond that would indicate a genuinely different
        # problem worth seeing rather than silently truncating/padding.
        print(f"Warning: inflated {len(out)} bytes, expected {expected_len}")
    return out


def _read_frequency_tables(r: BinReader):
    """Shared by entropy_type 2 and 3: n_segs deflated per-segment
    256-entry frequency tables, packed as 1/2/4 bytes per entry depending
    on len_type."""
    n_segs = r.read_int()
    len_type = r.read_int()
    zfl = r.read_int()
    zfd = r.read_fully(zfl)

    bpe = 1 if len_type == 0 else 2 if len_type == 1 else 4
    n_bytes = n_segs * 256 * bpe
    fb = _inflate(zfd, n_bytes)

    freqs = [[0] * 256 for _ in range(n_segs)]
    if len_type == 0:
        for k in range(n_segs):
            base = k * 256
            for m in range(256):
                freqs[k][m] = fb[base + m]
    elif len_type == 1:
        for k in range(n_segs):
            base = k * 512
            for m in range(256):
                a = fb[base + 2 * m]
                b = fb[base + 2 * m + 1]
                freqs[k][m] = a | (b << 8)
    else:
        for k in range(n_segs):
            base = k * 1024
            for m in range(256):
                a = fb[base + 4 * m]
                b = fb[base + 4 * m + 1]
                c = fb[base + 4 * m + 2]
                d = fb[base + 4 * m + 3]
                freqs[k][m] = a | (b << 8) | (c << 16) | (d << 24)

    return n_segs, freqs


def numpy_bgr_to_qpixmap(arr_bgr: np.ndarray) -> QPixmap:
    arr_bgr = np.ascontiguousarray(arr_bgr.clip(0, 255).astype(np.uint8))
    rgb = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape
    qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


# =============================================================================
# Scroll area with Ctrl+wheel zoom anchored at the cursor (same pattern as
# delta_writer.py's ZoomScrollArea).
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


class DeltaReaderWindow(QMainWindow):
    def __init__(self, filename: str):
        super().__init__()
        self.filename = filename
        self.decoded_bgr = None   # final assembled image, set once decoding finishes
        self.zoom_scale = 1.0
        self.fit_scale = 1.0

        # ---- header / per-channel scalars, mirroring the Java fields ----
        self.xdim = self.ydim = 0
        self.intermediate_xdim = self.intermediate_ydim = 0
        self.pixel_shift = self.pixel_quant = self.set_id = 0
        self.delta_type = self.compress_type = self.entropy_type = 0
        self.scanline5_variant = 0

        self.min = [0, 0, 0]
        self.init = [0, 0, 0]
        self.delta_min = [0, 0, 0]
        self.length = [0, 0, 0]
        self.compressed_length = [0, 0, 0]
        self.channel_iterations = [0, 0, 0]

        self.table_list = []
        self.map_list = []

        self.channel_array = [None, None, None]
        self.resize_array = [None, None, None]

        self.lz77_data_list = []
        self.lz77_orig_length = [0, 0, 0]

        self.huff_rank_list = []
        self.huff_cl_list = []
        self.huff_pay_list = []
        self.huff_bl = [0, 0, 0]
        self.huff_pay_min = [0, 0, 0]

        self.freq_list = []
        self.offset_list = []
        self.segment_list = []
        self.fast_enc_list = []

        self._read_file(filename)
        self._build_ui()

        QTimer.singleShot(0, self._decode_and_show)

    # ------------------------------------------------------------------ IO
    def _read_file(self, filename):
        with open(filename, "rb") as fh:
            r = BinReader(fh)

            self.xdim = r.read_short()
            self.ydim = r.read_short()
            self.pixel_shift = r.read_byte()
            self.pixel_quant = r.read_byte()
            self.set_id = r.read_byte()
            self.delta_type = r.read_byte()
            self.compress_type = r.read_byte()
            self.entropy_type = r.read_byte()
            self.scanline5_variant = r.read_byte()

            print(f"Image:        {self.xdim} x {self.ydim}")
            print(f"Channel set:  {SET_STRING[self.set_id & 0xFF]}")
            print(f"Delta type:   {DELTA_TYPE_STRING[self.delta_type & 0xFF]}")
            print(f"Entropy type: {ENTROPY_TYPE_STRING[self.entropy_type & 0xFF]}")
            print()

            for i in range(3):
                print(f"Reading channel {i}")

                self.min[i] = r.read_int()
                self.init[i] = r.read_int()
                self.delta_min[i] = r.read_int()
                self.length[i] = r.read_int()
                self.compressed_length[i] = r.read_int()
                self.channel_iterations[i] = r.read_byte()

                if self.delta_type in (6, 7, 8):
                    ml = r.read_int()
                    pml = r.read_int()
                    pm = r.read_fully(pml)
                    map_raw = bytearray(ml)
                    for q in range(ml):
                        map_raw[q] = (pm[q >> 2] >> ((q & 3) << 1)) & 0x3
                    self.map_list.append(bytes(map_raw))
                elif self.delta_type in (9, 10, 11, 12):
                    # These four cases are byte-for-byte identical in the
                    # Java source (each its own separate if/elif there);
                    # merged here since there's no behavioral difference.
                    ml = r.read_int()
                    tbl = read_table(r)
                    dmin = r.read_int()
                    bl = r.read_int()
                    str_bytes = r.read_fully(sm.get_bytelength(bl))
                    decomp = sm.decompress_strings(str_bytes)
                    vals = sm.unpack_strings(decomp, tbl, ml, sm.get_bitlength(decomp))
                    map_ = bytearray(ml)
                    for q in range(ml):
                        map_[q] = (vals[q] + dmin) & 0xFF
                    self.map_list.append(bytes(map_))
                else:
                    self.map_list.append(None)

                if self.compress_type > 0:
                    self.table_list.append(read_table(r))
                else:
                    self.table_list.append(None)

                if self.entropy_type == 0:
                    orig_len = r.read_int()
                    zip_len = r.read_int()
                    zip_data = r.read_fully(zip_len)
                    self.lz77_orig_length[i] = orig_len
                    self.lz77_data_list.append(zip_data)

                elif self.entropy_type == 1:
                    rank_table = read_table(r)
                    pay_min = r.read_int()
                    n = r.read_int()
                    init_val = r.read_byte()
                    max_delta = r.read_byte()
                    pdt_len = r.read_ubyte()
                    pdt = r.read_fully(pdt_len)
                    code_length = cm.unpack_length_table(n, init_val, max_delta, pdt)
                    bl = r.read_int()
                    pay_len = r.read_int()
                    pay_bytes = r.read_fully(pay_len)

                    self.huff_rank_list.append(rank_table)
                    self.huff_cl_list.append(code_length)
                    self.huff_pay_list.append(pay_bytes)
                    self.huff_bl[i] = bl
                    self.huff_pay_min[i] = pay_min

                elif self.entropy_type == 2:
                    n_segs, freqs = _read_frequency_tables(r)
                    offsets = []
                    for _k in range(n_segs):
                        ll = r.read_int(); bb = r.read_fully(ll)
                        num = int.from_bytes(bb, "big", signed=True)
                        ll = r.read_int(); bb = r.read_fully(ll)
                        den = int.from_bytes(bb, "big", signed=True)
                        offsets.append([num, den])
                    self.freq_list.append(freqs)
                    self.offset_list.append(offsets)
                    self.segment_list.append([b""] * n_segs)

                else:  # entropy_type == 3
                    n_segs, freqs = _read_frequency_tables(r)
                    self.freq_list.append(freqs)
                    fast_enc = []
                    for _k in range(n_segs):
                        enc_len = r.read_int()
                        fast_enc.append(r.read_fully(enc_len))
                    self.fast_enc_list.append(fast_enc)

        print(f"File read: {filename}")

    # -------------------------------------------------------------- UI build
    def _build_ui(self):
        self.canvas = QLabel()
        self.canvas.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.scroll = ZoomScrollArea(self)
        self.scroll.setWidget(self.canvas)
        self.scroll.setWidgetResizable(False)
        self.setCentralWidget(self.scroll)

        menu_bar = self.menuBar()
        m = menu_bar.addMenu("View")

        zi = QAction("Zoom In (+)", self, shortcut=QKeySequence("Ctrl+="))
        zi.triggered.connect(lambda: self._zoom_by(ZOOM_FACTOR))
        m.addAction(zi)

        zo = QAction("Zoom Out (-)", self, shortcut=QKeySequence("Ctrl+-"))
        zo.triggered.connect(lambda: self._zoom_by(1.0 / ZOOM_FACTOR))
        m.addAction(zo)

        zf = QAction("Fit to Window", self, shortcut=QKeySequence("Ctrl+0"))
        zf.triggered.connect(self._fit_to_window)
        m.addAction(zf)

        za = QAction("Actual Size (100%)", self, shortcut=QKeySequence("Ctrl+1"))
        za.triggered.connect(self._actual_size)
        m.addAction(za)

        screen = QApplication.primaryScreen().size()
        sw, sh = screen.width(), screen.height()
        self.fit_scale = min(1.0, min((sw * 70 // 100 - 40) / self.xdim, (sh * 70 // 100 - 80) / self.ydim))
        self.zoom_scale = self.fit_scale

        self.canvas.resize(max(1, int(self.xdim * self.zoom_scale)), max(1, int(self.ydim * self.zoom_scale)))
        self.resize(min(self.xdim + 40, int(sw * 0.70)), min(self.ydim + 80, int(sh * 0.70)))
        self.move(5, 5)
        self.update_title()
        self.show()

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
        self.zoom_scale = min(vp.width() / self.xdim, vp.height() / self.ydim)
        self.update_display_image()
        self.update_title()

    def _actual_size(self):
        self.zoom_scale = 1.0
        self.update_display_image()
        self.update_title()

    def update_display_image(self):
        if self.decoded_bgr is None:
            return
        pm = numpy_bgr_to_qpixmap(self.decoded_bgr)
        w = max(1, int(self.xdim * self.zoom_scale))
        h = max(1, int(self.ydim * self.zoom_scale))
        if self.zoom_scale != 1.0:
            pm = pm.scaled(w, h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        self.canvas.setPixmap(pm)
        self.canvas.resize(w, h)

    def update_title(self):
        if self.decoded_bgr is None:
            self.setWindowTitle("Delta Reader  [decoding\u2026]")
        else:
            self.setWindowTitle(f"Delta Reader  [{round(self.zoom_scale * 100)}%]")

    # -------------------------------------------------------------- decode
    def _decode_and_show(self):
        try:
            self._decode()
        except Exception as e:
            print(f"Decode error: {e!r}")
            import traceback
            traceback.print_exc()
            return
        self.update_display_image()
        self.update_title()

    def _decode(self):
        channel_id = dm.get_channels(self.set_id)

        # ---- per-channel decode, threaded like Java's Decompressor ----
        threads = [threading.Thread(target=self._decode_channel, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # ---- assemble RGB from the 3 decoded channels per set_id ----
        ch = [None, None, None]
        a = self.channel_array
        if self.set_id == 0:
            ch[0], ch[1], ch[2] = a[0], a[1], a[2]
        elif self.set_id == 1:
            ch[0] = a[0]; ch[1] = a[1] - a[2]; ch[2] = a[1]
        elif self.set_id == 2:
            ch[0] = a[0]; ch[1] = a[0] - a[2]; ch[2] = a[1]
        elif self.set_id == 3:
            ch[0] = a[0]; ch[1] = a[0] - a[1]; ch[2] = a[2] + ch[1]
        elif self.set_id == 4:
            ch[0] = a[0]; ch[1] = a[0] - a[1]; ch[2] = a[0] + a[2]
        elif self.set_id == 5:
            ch[0] = a[2] + a[0]; ch[1] = a[0]; ch[2] = a[1]
        elif self.set_id == 6:
            neg2 = -a[2]
            ch[1] = neg2 + a[0]
            ch[0] = a[1] + ch[1]
            ch[2] = a[0]
        elif self.set_id == 7:
            ch[0] = a[0] + a[1]; ch[1] = a[0]; ch[2] = a[0] + a[2]
        elif self.set_id == 8:
            ch[2] = a[0] + a[1]; ch[0] = ch[2] - a[2]; ch[1] = a[0]
        else:  # 9
            ch[0] = a[0] - a[2]; ch[1] = a[0] - a[1]; ch[2] = a[0]

        # ---- resize (if pixel_quant != 0) and pack into BGR ----
        if self.pixel_quant == 0:
            blue, green, red = ch[0], ch[1], ch[2]
        else:
            if self.xdim > 600:
                threads = [
                    threading.Thread(target=self._resize_channel, args=(ch[i], self.intermediate_xdim, self.xdim, self.ydim, i))
                    for i in range(3)
                ]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                blue, green, red = self.resize_array
            else:
                blue = np.asarray(rm.resize(list(ch[0]), self.intermediate_xdim, self.xdim, self.ydim))
                green = np.asarray(rm.resize(list(ch[1]), self.intermediate_xdim, self.xdim, self.ydim))
                red = np.asarray(rm.resize(list(ch[2]), self.intermediate_xdim, self.xdim, self.ydim))

        # DeltaMapper.getPixel() in the Java packs channel0 into bits
        # 16-23, channel1 into 8-15, channel2 into 0-7 of a single int
        # (blue_shift = pixel_shift+16, etc.) purely so BufferedImage's
        # setRGB() sees a standard 0x00RRGGBB word -- that packing detail
        # is irrelevant here since we build a BGR numpy array directly
        # (matching delta_writer.py's own approach) rather than a packed
        # AWT pixel int. What DOES matter and is preserved exactly is the
        # actual resolution-restoring shift: left by pixel_shift, undoing
        # DeltaMapper.shift(ch, -pixel_shift) applied at encode time.
        # pixel_shift is always non-negative in practice (slider range
        # 0-7), matching Java's unconditional `<<` in getPixel().
        blue = blue.reshape(self.ydim, self.xdim) << self.pixel_shift
        green = green.reshape(self.ydim, self.xdim) << self.pixel_shift
        red = red.reshape(self.ydim, self.xdim) << self.pixel_shift

        bgr = np.zeros((self.ydim, self.xdim, 3), dtype=np.int64)
        bgr[:, :, 0] = blue
        bgr[:, :, 1] = green
        bgr[:, :, 2] = red
        self.decoded_bgr = bgr.clip(0, 255).astype(np.uint8)

    def _resize_channel(self, ch, old_xdim, new_xdim, new_ydim, i):
        flat = rm.resize(list(np.asarray(ch).reshape(-1)), old_xdim, new_xdim, new_ydim)
        self.resize_array[i] = np.asarray(flat)

    def _decode_channel(self, i):
        # ---- current (possibly quantized) dimensions ----
        if self.pixel_quant == 0:
            cur_xdim, cur_ydim = self.xdim, self.ydim
        else:
            f = self.pixel_quant / 10.0
            # Integer division here matches Java's `xdim/2` on a
            # declared-int xdim exactly (see module docstring's
            # COMPATIBILITY NOTE point 3).
            self.intermediate_xdim = self.xdim - int(f * (self.xdim // 2 - 2))
            self.intermediate_ydim = self.ydim - int(f * (self.ydim // 2 - 2))
            cur_xdim, cur_ydim = self.intermediate_xdim, self.intermediate_ydim
        size = cur_xdim * cur_ydim

        # ---- entropy decode -> payload bytes (unsigned 0..255 ints) ----
        if self.entropy_type == 0:
            zip_data = self.lz77_data_list[i]
            orig_len = self.lz77_orig_length[i]
            payload = zlib.decompress(zip_data)
            if len(payload) != orig_len:
                print(f"Warning: LZ77 payload length {len(payload)} != expected {orig_len}")

        elif self.entropy_type == 1:
            rank_table = self.huff_rank_list[i]
            code_len = self.huff_cl_list[i]
            packed = self.huff_pay_list[i]
            bl = self.huff_bl[i]
            pay_min = self.huff_pay_min[i]
            hcode = cm.get_canonical_code(code_len)

            num_sym = size if self.compress_type == 0 else sm.get_bytelength(self.compressed_length[i])

            decoded = [0] * num_sym
            cm.unpack_code_int_dst(packed, rank_table, hcode, code_len, bl, decoded)

            payload = bytes((v + pay_min) & 0xFF for v in decoded)

        elif self.entropy_type == 2:
            expected = size if self.compress_type == 0 else sm.get_bytelength(self.compressed_length[i])
            freqs = self.freq_list[i]
            offsets = self.offset_list[i]
            n_segs = len(freqs)
            seg_len = expected // n_segs
            odd_len = seg_len + expected % n_segs

            segs = [None] * n_segs

            def decode_seg(k):
                length = seg_len if k < n_segs - 1 else odd_len
                segs[k] = am.get_arithmetic_values(offsets[k], freqs[k], length)

            threads = [threading.Thread(target=decode_seg, args=(k,)) for k in range(n_segs)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            payload = b"".join(bytes(s) for s in segs)

        else:  # entropy_type == 3
            expected = size if self.compress_type == 0 else sm.get_bytelength(self.compressed_length[i])
            freqs = self.freq_list[i]
            fast_enc = self.fast_enc_list[i]
            n_segs = len(freqs)
            seg_len = expected // n_segs
            odd_len = seg_len + expected % n_segs

            segs = [None] * n_segs

            def decode_seg_fast(k):
                length = seg_len if k < n_segs - 1 else odd_len
                segs[k] = am.get_arithmetic_values_fast(fast_enc[k], freqs[k], length)

            threads = [threading.Thread(target=decode_seg_fast, args=(k,)) for k in range(n_segs)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            payload = b"".join(bytes(s) for s in segs)

        # ---- payload -> delta values ----
        if self.compress_type == 0:
            delta = [0] * size
            for k in range(1, size):
                v = payload[k]
                if v >= 128:
                    v -= 256  # payload bytes are signed (delta[k] - delta_min)
                delta[k] = v + self.delta_min[i]
        else:
            tbl = self.table_list[i]
            decomp = sm.decompress_strings(payload)
            delta = sm.unpack_strings(decomp, tbl, size, self.length[i])
            delta = list(delta)
            delta[0] = 0
            for k in range(1, len(delta)):
                delta[k] += self.delta_min[i]

        # ---- delta -> channel values ----
        decoder, uses_map, uses_variant = _DECODERS.get(self.delta_type, _DECODERS[0])
        if uses_variant:
            cur_ch = decoder(delta, cur_xdim, cur_ydim, self.init[i], self.map_list[i], self.scanline5_variant)
        elif uses_map:
            cur_ch = decoder(delta, cur_xdim, cur_ydim, self.init[i], self.map_list[i])
        else:
            cur_ch = decoder(delta, cur_xdim, cur_ydim, self.init[i])

        cur_ch = np.asarray(cur_ch, dtype=np.int64)

        # Restore difference-channel offset.
        channel_id = dm.get_channels(self.set_id)
        if channel_id[i] > 2:
            cur_ch = cur_ch + self.min[i]

        self.channel_array[i] = cur_ch


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 delta_reader.py <filename>")
        sys.exit(0)
    app = QApplication(sys.argv)
    win = DeltaReaderWindow(sys.argv[1])
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
