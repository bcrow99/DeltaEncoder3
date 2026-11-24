"""A table of compression ratios and times for one image, from the three
versions of the generic writer and reader:

  Native  NativeEncoder's NativeWriter and NativeReader (Java with the C
          library, if it's built; the table says if it wasn't found)
  Java    DeltaEncoder's GenericWriter and GenericReader
  Python  this repository's generic_writer.py and generic_reader.py

for each quantization (Color Resolution / Pixel Resolution), coding (LS or
Avg deltas; context aware arithmetic (A), Context tables (T), Huffman (H) or
Huffman tables, neighbour Huffman (N) coding) and number of bands asked for:

    python test/results_table.py image [options]

    --java DIR      DeltaEncoder's folder      (default: ../DeltaEncoder beside this repository)
    --native DIR    NativeEncoder's folder     (default: ../NativeEncoder beside this repository)
    --quants LIST   color/pixel pairs          (default 0/0,2/2,3/4)
    --bands LIST    band counts                (default 1,2,4,8)
    --methods LIST  codings                    (default LS-A,LS-T,LS-H,LS-N,Avg-A,Avg-T,Avg-H,Avg-N)
    --runs N        timed runs per setting     (default 2)
    --skip LIST     versions to leave out, e.g. --skip python  (native, java, python)

The ratio is the file's size over the raw image's (3 bytes a pixel), so
smaller is better. The times are in milliseconds, as the writers and readers
print them once their code is compiled: each setting is run once to warm up,
then run N times in one timing, divided by N. Encoding is Save without its
decode check; decoding is from reading the file to the finished image. The
versions are run one after another, never at the same time.

The check column says "ok" when all the versions wrote the same file and
decoded it to the same image, and (at 0/0) that image is the original;
otherwise what differs.

Java's half is test/TableBench.java, compiled here with each Java
repository's classes (javac and java must be on the PATH). The Native
column needs NativeEncoder's library built (as NativeWriter needs it).
Other Java options can be given in JAVA_OPTS, e.g. JAVA_OPTS=-Xmx8g.

The table is printed and saved, with a CSV of the same numbers, as
results_<image>_<machine>.txt and .csv in the current folder.
"""
import argparse
import contextlib
import glob
import hashlib
import io
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

VERSIONS = ("native", "java", "python")
TITLES = {"native": "Native", "java": "Java", "python": "Python"}


# =============================================================================
# The Java versions
# =============================================================================

def java_sources(version, folder):
    """The .java files of a repository's programs (not its tests or benchmarks)."""
    if version == "native":
        return sorted(glob.glob(os.path.join(folder, "java", "src", "*.java")))
    return sorted(glob.glob(os.path.join(folder, "*.java")))


def run_java(version, folder, image, args):
    """Compiles TableBench with the repository's classes and runs it.
    Returns (the INFO line's fields, {(quant, method, bands): row}), or
    (None, reason) if it couldn't be run."""
    sources = java_sources(version, folder)
    if not sources:
        return None, "no Java sources in " + folder
    for tool in ("javac", "java"):
        if shutil.which(tool) is None:
            return None, "no %s on the PATH" % tool
    classes = tempfile.mkdtemp(prefix="TableBench_")
    try:
        javac = ["javac", "--release", "8", "-nowarn", "-Xlint:-options", "-Xlint:-unchecked", "-encoding", "UTF-8", "-d", classes]
        done = subprocess.run(javac + sources + [os.path.join(HERE, "TableBench.java")], capture_output=True, text=True)
        if done.returncode != 0:
            return None, "javac failed:\n" + done.stdout + done.stderr

        writer, reader = ("NativeWriter", "NativeReader") if version == "native" else ("GenericWriter", "GenericReader")
        options = []
        # Java 17 and later warn when a library is loaded unless native
        # access is enabled; Java 8 doesn't know the option.
        if version == "native" and subprocess.run(["java", "--enable-native-access=ALL-UNNAMED", "-version"],
                                                  capture_output=True).returncode == 0:
            options.append("--enable-native-access=ALL-UNNAMED")
        options += os.environ.get("JAVA_OPTS", "").split()
        command = (["java"] + options + ["-cp", classes, "TableBench", writer, reader, image,
                   ",".join(args.quants), ",".join(str(b) for b in args.bands), ",".join(args.methods), str(args.runs)])
        # Run from the repository's folder: NativeEncoder's DeltaMapper
        # looks for its library in the folder native there.
        info, rows = {}, {}
        process = subprocess.Popen(command, cwd=folder, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        other = []
        for line in process.stdout:
            line = line.rstrip("\n")
            if line.startswith("INFO "):
                for field in line[5:].split(" ", 2):
                    key, _, value = field.partition("=")
                    info[key] = value
            elif line.startswith("ROW "):
                row = parse_row(line[4:].split())
                rows[row["key"]] = row
                progress(TITLES[version], row)
            elif "Picked up" not in line:
                other.append(line)
        if process.wait() != 0:
            return None, "TableBench failed:\n" + "\n".join(other[-30:])
        return info, rows
    finally:
        shutil.rmtree(classes, ignore_errors=True)


def parse_row(f):
    """A TableBench ROW line's fields."""
    return dict(key=(f[0], f[1], int(f[2])), bytes=int(f[3]), file_md5=f[4], image_md5=f[5], lossless=f[6] == "1",
                encode=float(f[7]), decode=float(f[8]))


def progress(title, row):
    q, m, n = row["key"]
    print("  %-6s %s %-5s %2d band%s  encode %8.1f ms  decode %8.1f ms" % (title, q, m, n, " " if n == 1 else "s", row["encode"], row["decode"]),
          flush=True)


# =============================================================================
# The Python version
# =============================================================================

def run_python(image, args):
    """Times generic_writer and generic_reader as TableBench times the Java
    ones. Returns (info, {(quant, method, bands): row}), or (None, reason)."""
    try:
        import numpy as np
        import viewer_support as vs
        import generic_writer as gw
        from generic_reader import GenericDecoder
    except ImportError as e:
        return None, "can't import the Python programs: %s" % e
    rgb = vs.read_image(image)
    original = rgb.tobytes()
    coder = gw.GenericCoder(rgb, os.path.getsize(image))
    coder.decode_check = False
    rows = {}
    handle, temp = tempfile.mkstemp(prefix="TableBench_", suffix=".bin")
    os.close(handle)
    try:
        for q in args.quants:
            shift, quant = (int(x) for x in q.split("/"))
            coder.pixel_shift, coder.pixel_quant = shift, quant
            with contextlib.redirect_stdout(io.StringIO()):
                coder.apply()
            for m in args.methods:
                coder.delta_type = gw.DELTA_LS if m.startswith("LS") else gw.DELTA_AVERAGE
                coder.entropy_type = {"A": gw.ENTROPY_MIXED, "T": gw.ENTROPY_TABLES, "H": gw.ENTROPY_HUFFMAN, "N": gw.ENTROPY_NEIGHBOUR}[m.split("-")[1]]
                last = -1
                for b in args.bands:
                    coder.bands = b
                    n = coder.bands_used()
                    if n == last:
                        continue   # the image is too small for more
                    last = n

                    data = bytes(coder.encode()[0])           # once to warm up (and compile)
                    start = time.perf_counter()
                    for _ in range(args.runs):
                        if bytes(coder.encode()[0]) != data:
                            raise RuntimeError("the runs coded different files: %s %s %d" % (q, m, n))
                    encode_ms = (time.perf_counter() - start) * 1000 / args.runs
                    with open(temp, "wb") as f:
                        f.write(data)

                    with contextlib.redirect_stdout(io.StringIO()):
                        decoder = GenericDecoder(temp)
                        decoded = decoder.decode()            # once to warm up
                        start = time.perf_counter()
                        for _ in range(args.runs):
                            decoder.decode()
                        decode_ms = (time.perf_counter() - start) * 1000 / args.runs
                    image_bytes = np.ascontiguousarray(decoded).tobytes()
                    row = dict(key=(q, m, n), bytes=len(data), file_md5=hashlib.md5(data).hexdigest(),
                               image_md5=hashlib.md5(image_bytes).hexdigest(), lossless=image_bytes == original,
                               encode=encode_ms, decode=decode_ms)
                    rows[row["key"]] = row
                    progress("Python", row)
    finally:
        os.remove(temp)
    return {"python": platform.python_version()}, rows


# =============================================================================
# The table
# =============================================================================

def processor_name():
    """The processor's name, if the system says."""
    try:
        if sys.platform == "win32":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        if sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown processor"


def ratio(n, raw):
    """A ratio as .1234 (or 1.234 if it's over 1)."""
    r = n / raw
    return ("%.4f" % r)[1:] if r < 1 else "%.3f" % r


def check(key, results):
    """The check column: whether the versions agree, and (at 0/0) the image
    came back as it was."""
    rows = [results[v][key] for v in results if key in results[v]]
    if not rows:
        return ""
    problems = []
    if len(set(r["file_md5"] for r in rows)) > 1:
        problems.append("files differ")
    if len(set(r["image_md5"] for r in rows)) > 1:
        problems.append("images differ")
    if key[0] == "0/0" and not all(r["lossless"] for r in rows):
        problems.append("not lossless")
    return ", ".join(problems) if problems else "ok"


def time_text(results, version, key, which):
    row = results.get(version, {}).get(key)
    return "%.0f" % row[which] if row else "-"


def make_table(image, xdim, ydim, args, results, infos, notes):
    raw = 3 * xdim * ydim
    import cv2
    import viewer_support as vs
    rgb = vs.read_image(image)
    png = len(cv2.imencode(".png", rgb[:, :, ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 9])[1])
    name = os.path.basename(image)
    ext = os.path.splitext(name)[1].lstrip(".").lower() or "file"
    if ext == "png":
        ext = "file"   # so it isn't confused with the png figure (OpenCV's, at level 9)

    shown = [v for v in VERSIONS if v in results]
    width = 7
    lines = []
    lines.append("%s   %d x %d   %s %s   png %s   (ratio = compressed / raw 24-bit, %s bytes)"
                 % (name, xdim, ydim, ext, ratio(os.path.getsize(image), raw), ratio(png, raw), format(raw, ",")))
    cores = os.cpu_count()
    lines.append("%s, %d logical processors; %s; times in ms, steady state (%d runs timed, divided by %d, after a warm-up)"
                 % (processor_name(), cores, platform.system() + " " + platform.release(), args.runs, args.runs))
    versions = []
    if "native" in infos:
        versions.append("Native: Java %s with %s" % (infos["native"].get("java", "?"), infos["native"].get("native", "?")))
    if "java" in infos:
        versions.append("Java: %s" % infos["java"].get("java", "?"))
    if "python" in infos:
        versions.append("Python: %s" % infos["python"]["python"])
    lines.append("; ".join(versions))
    for note in notes:
        lines.append(note)
    lines.append("")

    group = "".join("%*s" % (width, TITLES[v]) for v in shown)
    lead = "%-6s %-6s %5s  %-6s" % ("quant", "method", "bands", "ratio")
    span = len(group)
    lines.append(" " * len(lead) + "  " + "encode ms".center(span) + "   " + "decode ms".center(span))
    lines.append(lead + "  " + group + "   " + group + "   check")

    csv = ["image,quant,method,bands,bytes,ratio," + ",".join("encode_ms_" + v for v in shown) + "," + ",".join("decode_ms_" + v for v in shown) + ",check"]
    keys = []
    for q in args.quants:
        most = 0
        for m in args.methods:
            seen = sorted({k[2] for v in shown for k in results[v] if k[0] == q and k[1] == m})
            keys += [(q, m, n) for n in seen]
            most = max([most] + seen)
        if most and most < max(args.bands):
            lines.insert(-2, "At %s the image has room for %d bands at most (as Save allows: at least 64 rows a band)." % (q, most))
    last_quant = None
    for key in keys:
        q, m, n = key
        if last_quant is not None and q != last_quant:
            lines.append("")
        last_quant = q
        row = next(results[v][key] for v in shown if key in results[v])
        encode = "".join("%*s" % (width, time_text(results, v, key, "encode")) for v in shown)
        decode = "".join("%*s" % (width, time_text(results, v, key, "decode")) for v in shown)
        status = check(key, results)
        lines.append("%-6s %-6s %5d  %-6s  %s   %s   %s" % (q, m, n, ratio(row["bytes"], raw), encode, decode, status))
        csv.append(",".join([name, q, m, str(n), str(row["bytes"]), "%.6f" % (row["bytes"] / raw)]
                            + [time_text(results, v, key, "encode") for v in shown]
                            + [time_text(results, v, key, "decode") for v in shown] + [status]))
    return lines, csv


def main():
    parser = argparse.ArgumentParser(description="Compression ratios and times for one image (see the top of this file).")
    parser.add_argument("image")
    parser.add_argument("--java", default=os.path.join(os.path.dirname(REPO), "DeltaEncoder"))
    parser.add_argument("--native", default=os.path.join(os.path.dirname(REPO), "NativeEncoder"))
    parser.add_argument("--quants", default="0/0,2/2,3/4")
    parser.add_argument("--bands", default="1,2,4,8")
    parser.add_argument("--methods", default="LS-A,LS-T,LS-H,LS-N,Avg-A,Avg-T,Avg-H,Avg-N")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--skip", default="")
    args = parser.parse_args()
    args.quants = [q.strip() for q in args.quants.split(",") if q.strip()]
    args.bands = [int(b) for b in args.bands.split(",") if b.strip()]
    args.methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in args.methods:
        if m not in ("LS-A", "LS-T", "LS-H", "LS-N", "Avg-A", "Avg-T", "Avg-H", "Avg-N"):
            parser.error("unknown method %s: use LS-A, LS-T, LS-H, LS-N, Avg-A, Avg-T, Avg-H or Avg-N" % m)
    for q in args.quants:
        if len(q.split("/")) != 2:
            parser.error("quant %s: give it as color/pixel, e.g. 3/4" % q)
    skip = {s.strip().lower() for s in args.skip.split(",") if s.strip()}
    image = os.path.abspath(args.image)
    if not os.path.isfile(image):
        sys.exit("No image " + image)

    import viewer_support as vs
    rgb = vs.read_image(image)
    ydim, xdim = rgb.shape[:2]
    del rgb

    results, infos, notes = {}, {}, []
    for version in VERSIONS:
        if version in skip:
            continue
        print("%s ..." % TITLES[version], flush=True)
        if version == "python":
            info, rows = run_python(image, args)
        else:
            folder = os.path.abspath(args.native if version == "native" else args.java)
            info, rows = run_java(version, folder, image, args)
        if info is None:
            notes.append("%s left out: %s" % (TITLES[version], rows))
            print("  left out: " + rows, flush=True)
            continue
        if version == "native" and info.get("native", "").startswith("Java"):
            notes.append("NOTE: Native ran without its C library (%s): build it first, as for NativeWriter." % info["native"])
        results[version], infos[version] = rows, info
    if not results:
        sys.exit("Nothing was run.")

    lines, csv = make_table(image, xdim, ydim, args, results, infos, notes)
    print()
    print("\n".join(lines))
    stem = "results_%s_%s" % (os.path.splitext(os.path.basename(image))[0], socket.gethostname().split(".")[0])
    with open(stem + ".txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(stem + ".csv", "w", encoding="utf-8") as f:
        f.write("\n".join(csv) + "\n")
    print()
    print("Saved %s.txt and %s.csv in %s" % (stem, stem, os.getcwd()))


if __name__ == "__main__":
    main()
