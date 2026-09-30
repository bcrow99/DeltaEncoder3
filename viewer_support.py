"""
viewer_support.py -- ViewerSupport.java for PySide6: the window code shared
by the writers and readers.

  - ImageWindow: a window showing one image, with zoom (View menu, Ctrl+wheel
    anchored at the cursor), a status shown in the title, and a way to grey
    out the other menus while a background job runs.
  - Dialog helpers: slider, spinner and radio-button dialogs opened from a
    menu item.
  - read_image, show_error, format_duration, parallel, run_in_background.

HiDPI: Qt 6 scales the whole interface itself on Linux, Windows and macOS
(from the screen's scale factor, or QT_SCALE_FACTOR), so the Java version's
font-scaling fallback isn't needed. Zoom 100% shows one image pixel per
logical pixel, which is what the Java version's "Actual Size" does.

The program exits when its last window closes (Qt's default).
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from PySide6.QtCore import Qt, QObject, Signal, QSize, QRectF
from PySide6.QtGui import QImage, QPainter, QAction, QKeySequence, QGuiApplication
from PySide6.QtWidgets import (QMainWindow, QScrollArea, QWidget, QDialog, QSlider, QLineEdit,
                               QHBoxLayout, QVBoxLayout, QLabel, QSpinBox, QRadioButton,
                               QButtonGroup, QMessageBox, QApplication)

ZOOM_FACTOR = 1.25
ZOOM_MIN = 0.05
ZOOM_MAX = 32.0
MAX_DIM = 65535          # largest dimension the file formats store (unsigned short)
MIN_DIM = 4


# =============================================================================
# Small helpers
# =============================================================================

def read_image(filename):
    """Reads an image as an (ydim, xdim, 3) uint8 array in R, G, B order
    (gray images are copied to all three; alpha is dropped). Raises IOError
    with a message for the user if the file can't be used."""
    import cv2
    try:
        data = np.fromfile(filename, dtype=np.uint8)      # works with any path on Windows too
    except OSError as e:
        raise IOError("Can't read %s: %s" % (filename, e))
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError(filename + " is not an image format this program can read.")
    h, w = bgr.shape[:2]
    if w < MIN_DIM or h < MIN_DIM:
        raise IOError("%s is %d x %d; images must be at least %d x %d." % (filename, w, h, MIN_DIM, MIN_DIM))
    if w > MAX_DIM or h > MAX_DIM:
        raise IOError("%s is %d x %d; the largest supported dimension is %d." % (filename, w, h, MAX_DIM))
    return np.ascontiguousarray(bgr[:, :, ::-1])


def show_error(parent, message):
    """Prints the message and shows it in a dialog."""
    print(message)
    if QApplication.instance() is not None:
        QMessageBox.critical(parent, "Error", message)


def format_duration(seconds):
    """3 decimals in the smallest unit that keeps the whole part under 1000,
    e.g. "543.210 usecs"."""
    nanos = seconds * 1e9
    units = ["ns", "usecs", "ms", "secs", "min"]
    divisors = [1.0, 1e3, 1e6, 1e9, 60e9]
    idx = len(units) - 1
    for i, d in enumerate(divisors):
        if nanos / d < 1000.0:
            idx = i
            break
    value = nanos / divisors[idx]
    if value >= 999.9995 and idx < len(units) - 1:
        idx += 1
        value = nanos / divisors[idx]
    return "%.3f %s" % (value, units[idx])


class Timer:
    def __init__(self):
        self.start = time.perf_counter()

    def elapsed(self):
        return format_duration(time.perf_counter() - self.start)


_pool = ThreadPoolExecutor()


def parallel(n, body):
    """Runs body(0..n-1) on the shared thread pool and waits; exceptions reach
    the caller. (The Numba-compiled work releases the GIL, so this really
    runs in parallel.)"""
    for f in [_pool.submit(body, i) for i in range(n)]:
        f.result()


class _Relay(QObject):
    done = Signal(object)


def run_in_background(job, on_done):
    """Runs job() on a worker thread, then on_done(result, error) on the GUI
    thread (error is the exception, or None)."""
    relay = _Relay()

    def finished(payload):
        relay.deleteLater()
        on_done(*payload)

    relay.done.connect(finished, Qt.QueuedConnection)

    def run():
        try:
            relay.done.emit((job(), None))
        except Exception as e:
            import traceback
            traceback.print_exc()
            relay.done.emit((None, e))

    threading.Thread(target=run, daemon=True).start()


# =============================================================================
# Dialogs opened from a menu item
# =============================================================================

def _open_dialog(parent, dialog):
    p = parent.pos()
    dialog.move(p.x(), max(0, p.y() - 60))
    dialog.adjustSize()
    dialog.show()
    dialog.raise_()


def make_slider_dialog(parent, title, lo, hi, init, on_change):
    """A menu action opening a small dialog with a slider (lo..hi) and its
    value. on_change gets every new value. Returns (action, slider)."""
    dialog = QDialog(parent)
    dialog.setWindowTitle(title)
    slider = QSlider(Qt.Horizontal)
    slider.setRange(lo, hi)
    slider.setValue(init)
    slider.setTickInterval(1)
    slider.setTickPosition(QSlider.TicksBelow)
    slider.setMinimumWidth(220)
    field = QLineEdit(str(init))
    field.setReadOnly(True)
    field.setFixedWidth(field.fontMetrics().horizontalAdvance("0000") + 12)

    def changed(v):
        field.setText(str(v))
        on_change(v)

    slider.valueChanged.connect(changed)
    layout = QHBoxLayout(dialog)
    layout.addWidget(slider)
    layout.addWidget(field)
    action = QAction(title, parent)
    action.triggered.connect(lambda: _open_dialog(parent, dialog))
    return action, slider


def make_spinner_dialog(parent, title, lo, hi, init, on_change):
    """A menu action opening a small dialog with a number field and up/down
    arrows (lo..hi). Returns (action, spin_box)."""
    dialog = QDialog(parent)
    dialog.setWindowTitle(title)
    spin = QSpinBox()
    spin.setRange(lo, hi)
    spin.setValue(init)
    spin.valueChanged.connect(on_change)
    layout = QHBoxLayout(dialog)
    layout.addWidget(QLabel(title + ":"))
    layout.addWidget(spin)
    action = QAction(title, parent)
    action.triggered.connect(lambda: _open_dialog(parent, dialog))
    return action, spin


def make_radio_dialog(parent, title, names, init, on_change):
    """A menu action opening a small dialog of radio buttons, one per name.
    on_change gets the index chosen. Returns (action, buttons)."""
    dialog = QDialog(parent)
    dialog.setWindowTitle(title)
    layout = QVBoxLayout(dialog)
    layout.setContentsMargins(12, 8, 12, 8)
    group = QButtonGroup(dialog)
    buttons = []
    for i, name in enumerate(names):
        b = QRadioButton(name)
        b.setChecked(i == init)
        group.addButton(b, i)
        layout.addWidget(b)
        buttons.append(b)
    group.idClicked.connect(on_change)
    action = QAction(title, parent)
    action.triggered.connect(lambda: _open_dialog(parent, dialog))
    return action, buttons


def make_button_dialog(parent, title, widgets, vertical=True):
    """A menu action opening a small dialog holding the given widgets."""
    dialog = QDialog(parent)
    dialog.setWindowTitle(title)
    layout = QVBoxLayout(dialog) if vertical else QHBoxLayout(dialog)
    for w in widgets:
        layout.addWidget(w)
    action = QAction(title, parent)
    action.triggered.connect(lambda: _open_dialog(parent, dialog))
    return action


# =============================================================================
# The image window
# =============================================================================

class _Canvas(QWidget):
    def __init__(self, window):
        super().__init__()
        self.window = window
        self.cache = None                 # scaled copy, when zoomed out

    def sizeHint(self):
        w = self.window
        return QSize(max(1, int(w.xdim * w.zoom_scale)), max(1, int(w.ydim * w.zoom_scale)))

    def paintEvent(self, event):
        w = self.window
        if w.qimage is None:
            return
        tw, th = max(1, int(w.xdim * w.zoom_scale)), max(1, int(w.ydim * w.zoom_scale))
        painter = QPainter(self)
        if w.zoom_scale == 1.0:
            painter.drawImage(0, 0, w.qimage)
        elif w.zoom_scale < 1.0:
            # Zoomed out: cache a smoothly scaled copy (smaller than the image).
            dpr = self.devicePixelRatioF()
            if self.cache is None or self.cache[0] != (tw, th, dpr):
                img = w.qimage.scaled(int(tw * dpr), int(th * dpr), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
                img.setDevicePixelRatio(dpr)
                self.cache = ((tw, th, dpr), img)
            painter.drawImage(0, 0, self.cache[1])
        else:
            # Zoomed in: scale only what is drawn, so zooming never allocates
            # a huge image.
            painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
            src = event.rect()
            s = w.zoom_scale
            painter.drawImage(QRectF(src), w.qimage, QRectF(src.x() / s, src.y() / s, src.width() / s, src.height() / s))
        painter.end()


class _Scroll(QScrollArea):
    def __init__(self, window):
        super().__init__()
        self.window = window

    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            p = event.position().toPoint()
            self.window.zoom_around(ZOOM_FACTOR if event.angleDelta().y() > 0 else 1.0 / ZOOM_FACTOR, p.x(), p.y())
            event.accept()
        else:
            super().wheelEvent(event)


class ImageWindow(QMainWindow):
    """A window titled `title` for an xdim x ydim image, sized to fit 70% of
    the screen. Add menus to menuBar() (see make_view_menu), then call
    show_window()."""
    _next_offset = 0

    def __init__(self, title, xdim, ydim):
        super().__init__()
        self.title, self.xdim, self.ydim = title, xdim, ydim
        self.qimage = None
        self.status = None
        screen = QGuiApplication.primaryScreen().availableGeometry()
        max_w = int(screen.width() * 0.70) - 40
        max_h = int(screen.height() * 0.70) - 80
        self.zoom_scale = min(1.0, max_w / xdim, max_h / ydim)
        self.canvas = _Canvas(self)
        self.scroll = _Scroll(self)
        self.scroll.setWidget(self.canvas)
        self.scroll.setWidgetResizable(False)
        self.scroll.horizontalScrollBar().setSingleStep(16)
        self.scroll.verticalScrollBar().setSingleStep(16)
        self.setCentralWidget(self.scroll)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self._resize_canvas()
        self._update_title()

    def show_window(self):
        """Sizes the window to the image, places it (each new window a little
        offset from the last) and shows it."""
        screen = QGuiApplication.primaryScreen().availableGeometry()
        w = min(int(self.xdim * self.zoom_scale) + 40, int(screen.width() * 0.70))
        h = min(int(self.ydim * self.zoom_scale) + 80, int(screen.height() * 0.70))
        self.resize(w, h)
        offset = ImageWindow._next_offset
        ImageWindow._next_offset = (offset + 30) % 270
        self.move(screen.x() + (screen.width() - w) // 2 + offset, screen.y() + (screen.height() - h) // 2 + offset)
        self.show()

    def make_view_menu(self):
        menu = self.menuBar().addMenu("View")
        for name, keys, fn in (("Zoom In", "Ctrl+=", lambda: self.zoom_by(ZOOM_FACTOR)),
                               ("Zoom Out", "Ctrl+-", lambda: self.zoom_by(1.0 / ZOOM_FACTOR)),
                               ("Fit to Window", "Ctrl+0", self.fit_to_window),
                               ("Actual Size (100%)", "Ctrl+1", lambda: self.set_zoom(1.0))):
            a = QAction(name, self, shortcut=QKeySequence(keys))
            a.triggered.connect(fn)
            menu.addAction(a)
        return menu

    def set_menus_enabled(self, enabled):
        """Enables or disables every menu except View (while a background job
        runs that Apply or Save must not overlap)."""
        for action in self.menuBar().actions():
            if action.text() != "View":
                action.setEnabled(enabled)

    def set_image(self, rgb):
        """rgb: (ydim, xdim, 3) uint8, R, G, B."""
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        h, w = rgb.shape[:2]
        self.qimage = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self.canvas.cache = None
        self.canvas.update()

    def set_status(self, status):
        """Shown in the title in place of the zoom level; None shows the zoom."""
        self.status = status
        self._update_title()

    def fit_and_shrink(self):
        """Zooms so the whole image fits (at most 100%), then shrinks the
        window if it is larger than the image needs."""
        view = self.scroll.viewport().size()
        if view.width() > 0 and view.height() > 0:
            self.set_zoom(min(1.0, view.width() / self.xdim, view.height() / self.ydim))
        extra_w = self.width() - view.width()
        extra_h = self.height() - view.height()
        screen = QGuiApplication.primaryScreen().availableGeometry()
        self.resize(min(int(self.xdim * self.zoom_scale) + extra_w, int(screen.width() * 0.70)),
                    min(int(self.ydim * self.zoom_scale) + extra_h, int(screen.height() * 0.70)))

    def fit_to_window(self):
        view = self.scroll.viewport().size()
        if view.width() > 0 and view.height() > 0:
            self.set_zoom(min(view.width() / self.xdim, view.height() / self.ydim))

    def zoom_by(self, factor):
        view = self.scroll.viewport().size()
        self.zoom_around(factor, view.width() // 2, view.height() // 2)

    def set_zoom(self, zoom):
        self.zoom_scale = max(ZOOM_MIN, min(ZOOM_MAX, zoom))
        self.canvas.cache = None
        self._resize_canvas()
        self.canvas.update()
        self._update_title()

    def zoom_around(self, factor, x, y):
        """Zooms keeping the image point under (x, y) -- viewport coordinates --
        in place."""
        old = self.zoom_scale
        zoom = max(ZOOM_MIN, min(ZOOM_MAX, old * factor))
        if zoom == old:
            return
        h, v = self.scroll.horizontalScrollBar(), self.scroll.verticalScrollBar()
        px, py = h.value(), v.value()
        r = zoom / old
        self.set_zoom(zoom)
        h.setValue(max(0, int((px + x) * r) - x))
        v.setValue(max(0, int((py + y) * r) - y))

    def _resize_canvas(self):
        self.canvas.resize(max(1, int(self.xdim * self.zoom_scale)), max(1, int(self.ydim * self.zoom_scale)))

    def _update_title(self):
        shown = self.status if self.status is not None else "%d%%" % round(self.zoom_scale * 100)
        self.setWindowTitle("%s  [%s]" % (self.title, shown))
