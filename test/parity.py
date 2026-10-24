"""Java/Python parity for the writers' reference jobs.

    python3 test/parity.py <java_dir> <image> [job ...]

<java_dir> holds the Java files <job>.bin, as DeltaEncoder/test/run_jobs.sh
writes them. For each job (all of them if none are named) this builds the
same file with the Python coders as <java_dir>/<job>.py.bin, compares the
bytes with the Java file, and decodes both with the Python decoders.

For coffee.png (skimage.data.coffee()) the sizes are checked against EXPECTED
and the decoded image's md5 (B, G, R bytes per pixel) against EXPECTED_MD5;
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
from block_reader import BlockDecoder             # noqa: E402
from delta_reader import DeltaDecoder             # noqa: E402
from packet_reader import PacketDecoder           # noqa: E402
from simple_reader import SimpleDecoder           # noqa: E402

# name, coder, decoder, fields set after the survey (as Drive sets them in Java)
JOBS = [
    ("b",     BlockCoder,  BlockDecoder,  {}),
    ("d13",   DeltaCoder,  DeltaDecoder,  dict(delta_type=13, block_set=3)),
    ("d13c",  DeltaCoder,  DeltaDecoder,  dict(delta_type=13, block_set=1, entropy_type=4)),
    ("d2",    DeltaCoder,  DeltaDecoder,  dict(delta_type=2)),
    ("d2_13", DeltaCoder2, DeltaDecoder,  dict(delta_type=13, block_set=3)),
    ("p13",   PacketCoder, PacketDecoder, dict(delta_type=13, block_set=3)),
    ("s13",   SimpleCoder, SimpleDecoder, dict(delta_type=13, block_set=3, entropy_type=4)),
    ("s13a",  SimpleCoder, SimpleDecoder, dict(delta_type=13, block_set=0, entropy_type=2)),
]

# coffee.png, with LS (predictor id 255)
EXPECTED = dict(b=89247, d13=101019, d13c=94427, d2=110560, d2_13=101019, p13=101910, s13=93715, s13a=104759)
EXPECTED_MD5 = "eddc7360b43c66e6c86cf84d82c2ea3f"


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
            coder.survey()
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
        if check and md5_java != EXPECTED_MD5:
            problems.append("md5: expected %s" % EXPECTED_MD5)
        failed |= bool(problems)
        print("%-6s java %7d  python %7d  md5 %s  %s  (%.0f s)" % (
            name, len(java), len(py), md5_java, "; ".join(problems) or "ok", time.time() - start), flush=True)

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
