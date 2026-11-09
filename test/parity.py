"""Java/Python parity for the writers' reference jobs.

    python3 test/parity.py <java_dir> <image> [job ...]

<java_dir> holds the Java files <job>.bin, as DeltaEncoder/test/run_jobs.sh
writes them. For each job (all of them if none are named) this builds the
same file with the Python coders as <java_dir>/<job>.py.bin, compares the
bytes with the Java file, and decodes both with the Python decoders.

For coffee.png (skimage.data.coffee()) the sizes are checked against EXPECTED
and the decoded image's md5 (B, G, R bytes per pixel) against EXPECTED_MD5
(EXPECTED_MD5_CENTRED for the Simple, Huffman and Pixel jobs, which resize with resize2);
for other images the Java and Python files must still be identical and decode
to the same image. Exit status 1 if anything differs.

Needs what the Delta programs need (numpy, PySide6, opencv-python, numba).
Without a display, set QT_QPA_PLATFORM=offscreen.
"""
import contextlib
import hashlib
import io
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import viewer_support as vs                       # noqa: E402
from block_writer import BlockCoder               # noqa: E402
from delta_writer import DeltaCoder               # noqa: E402
from delta_writer2 import DeltaCoder2             # noqa: E402
from packet_writer import PacketCoder             # noqa: E402
from simple_writer import SimpleCoder             # noqa: E402
from simple_writer2 import SimpleCoder2           # noqa: E402
from huffman_writer import HuffmanCoder           # noqa: E402
from pixel_writer import PixelCoder               # noqa: E402
from block_reader import BlockDecoder             # noqa: E402
from delta_reader import DeltaDecoder             # noqa: E402
from packet_reader import PacketDecoder           # noqa: E402
from simple_reader import SimpleDecoder           # noqa: E402
from simple_reader2 import SimpleDecoder2         # noqa: E402
from huffman_reader import HuffmanDecoder         # noqa: E402
from pixel_reader import PixelDecoder             # noqa: E402

# name, coder, decoder, fields set after the survey or analysis, if the coder has one (as Drive sets them in Java)
JOBS = [
    ("b",     BlockCoder,  BlockDecoder,  {}),
    ("b4",    BlockCoder,  BlockDecoder,  dict(bands=4)),
    ("d13",   DeltaCoder,  DeltaDecoder,  dict(delta_type=13)),
    ("d13c",  DeltaCoder,  DeltaDecoder,  dict(delta_type=13, entropy_type=4)),
    ("d2",    DeltaCoder,  DeltaDecoder,  dict(delta_type=2)),
    ("d2_13", DeltaCoder2, DeltaDecoder,  dict(delta_type=13)),
    ("p13",   PacketCoder, PacketDecoder, dict(delta_type=13)),
    ("s",     SimpleCoder, SimpleDecoder, {}),
    ("s4",    SimpleCoder, SimpleDecoder, dict(bands=4)),
    ("s2",    SimpleCoder2, SimpleDecoder2, {}),
    ("s2_4",  SimpleCoder2, SimpleDecoder2, dict(bands=4)),
    ("h",     HuffmanCoder, HuffmanDecoder, {}),
    ("h4",    HuffmanCoder, HuffmanDecoder, dict(bands=4)),
    ("px",    PixelCoder,  PixelDecoder,  {}),
    ("px_ls", PixelCoder,  PixelDecoder,  dict(prediction=1)),
    ("px_rgb", PixelCoder, PixelDecoder,  dict(channels=0)),
    ("px_ls0", PixelCoder, PixelDecoder,  dict(pixel_quant=0, pixel_shift=0, prediction=1, coding=2)),
]

# coffee.png, with LS (predictor id 255); b with Block format 9: the 225-entry predictor table, mixed context coding with second opinions, each channel's block size (4, 6, 8, 12 or 16) and set surveyed from the whole table; b4 the same in 4 bands (format 10); the delta type 13 jobs (D and P version 5) use the same survey's block maps; s with Simple format 8: least squares alone with the wide clamp, each channel's window radius (4, 8 or 16) chosen by the coded size, mixed context coding with second opinions, centred resizing (resize2); s4 the same in 4 bands; s2 with Simple 2 format 2: the same coding with the channels green, blue-green, red-green and window radius 10, the fit from the rows above only, from 6 neighbours (W, N, NW, NE, WW, NNE), solved at every 4th pixel, no analysis; s2_4 the same in 4 bands; h with Huffman format 1: Simple 2's channels, fit and bands, each channel's deltas coded as unary strings (string_mapper, its bit-pair rule) and a canonical Huffman code (code_mapper); h4 the same in 4 bands; px with Pixel format 5: green, blue - green, red - green less each difference's minimum, the median edge predictor, values with an exact prediction, blocks found by the search, centred resizing (resize2); px_ls the same with least squares (radius 8); px_rgb with red, green, blue; px_ls0 lossless with least squares and the errors coding
EXPECTED = dict(b=85838, b4=88232, d13=96368, d13c=89190, d2=110560, d2_13=96368, p13=97606, s=85535, s4=87702, s2=85375, s2_4=87496, h=96439, h4=97954,
                px=103598, px_ls=90261, px_rgb=105549, px_ls0=332460)
EXPECTED_MD5 = "eddc7360b43c66e6c86cf84d82c2ea3f"
EXPECTED_MD5_CENTRED = dict(s="ac19bcb1b1f2a097bf1aca2272d6bc95", s4="ac19bcb1b1f2a097bf1aca2272d6bc95",
                            s2="ac19bcb1b1f2a097bf1aca2272d6bc95", s2_4="ac19bcb1b1f2a097bf1aca2272d6bc95",
                            h="ac19bcb1b1f2a097bf1aca2272d6bc95", h4="ac19bcb1b1f2a097bf1aca2272d6bc95",
                            px="ac19bcb1b1f2a097bf1aca2272d6bc95", px_ls="ac19bcb1b1f2a097bf1aca2272d6bc95",
                            px_rgb="ac19bcb1b1f2a097bf1aca2272d6bc95",
                            px_ls0="32bc35ebbf58295ec49a616391eb8267")    # the Simple and Pixel jobs resize with resize2, so decode to a different image (px_ls0: lossless, the image itself)


def decoded_md5(path, decoder):
    """md5 of the decoded image's B, G, R bytes per pixel."""
    with contextlib.redirect_stdout(io.StringIO()):
        rgb = decoder(path).decode()
    return hashlib.md5(rgb[:, :, ::-1].tobytes()).hexdigest()


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    java_dir, image, only = sys.argv[1], sys.argv[2], sys.argv[3:]
    unknown = set(only) - {j[0] for j in JOBS}
    if unknown:
        sys.exit("unknown jobs: " + " ".join(sorted(unknown)))
    check = os.path.basename(image) == "coffee.png"
    rgb = vs.read_image(image)
    failed = False

    for name, coder_class, decoder, fields in JOBS:
        if only and name not in only:
            continue
        start = time.time()
        java_file = os.path.join(java_dir, name + ".bin")
        py_file = os.path.join(java_dir, name + ".py.bin")
        if not os.path.exists(java_file):
            print("%-6s no Java file %s" % (name, java_file))
            failed = True
            continue

        with contextlib.redirect_stdout(io.StringIO()):
            coder = coder_class(rgb, os.path.getsize(image))
            for first in ("analyse", "survey"):
                if hasattr(coder, first):
                    getattr(coder, first)()
                    break
            for k, v in fields.items():
                if not hasattr(coder, k):
                    raise AttributeError("%s has no field %s" % (coder_class.__name__, k))
                setattr(coder, k, v)
            coder.apply()
            coder.save(py_file)

        with open(java_file, "rb") as f:
            java = f.read()
        with open(py_file, "rb") as f:
            py = f.read()
        md5_java, md5_py = decoded_md5(java_file, decoder), decoded_md5(py_file, decoder)

        problems = []
        if java != py:
            problems.append("files differ")
        if md5_java != md5_py:
            problems.append("decoded images differ")
        if check and len(java) != EXPECTED[name]:
            problems.append("size: expected %d" % EXPECTED[name])
        want = EXPECTED_MD5_CENTRED.get(name, EXPECTED_MD5)
        if check and md5_java != want:
            problems.append("md5: expected %s" % want)
        failed |= bool(problems)
        print("%-6s java %7d  python %7d  md5 %s  %s  (%.0f s)" % (
            name, len(java), len(py), md5_java, "; ".join(problems) or "ok", time.time() - start), flush=True)

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
