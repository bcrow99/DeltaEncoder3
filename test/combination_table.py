"""A table of every delta / entropy combination of the generic writer and
reader on one image, each combination a column: LS and Avg deltas, each with
the four codings of the Entropy menu,

    A   Context aware
    T   Context tables
    H   Huffman
    N   Huffman tables (neighbour Huffman)

for each quantization (Color Resolution / Pixel Resolution) and number of
bands (the rows):

    python test/combination_table.py image [options]

    --versions LIST  native, java, python      (default native)
    --native DIR     NativeEncoder's folder    (default: ../NativeEncoder beside this repository)
    --java DIR       DeltaEncoder's folder     (default: ../DeltaEncoder beside this repository)
    --quants LIST    color/pixel pairs         (default 0/0,1/1,2/2,3/4)
    --bands LIST     band counts               (default 1,2,4,8)
    --deltas LIST    LS, Avg                   (default LS,Avg)
    --codings LIST   codings, in column order  (default A,T,H,N)
    --runs N         timed runs per setting    (default 2)

So that the eight columns fit, the table comes in blocks with the same rows:
the compression ratio (the file's size over the raw image's, 3 bytes a pixel:
smaller is better), then, for each version asked for (Native: NativeWriter and
NativeReader, Java with the C library; Java: DeltaEncoder's GenericWriter and
GenericReader; Python: generic_writer.py and generic_reader.py), its encoding
times and its decoding times, in milliseconds. The times are as
results_table.py measures them: each setting run once to warm up, then N
times in one timing, divided by N; encoding is Save without its decode check
(the fit included), decoding from reading the file to the finished image. The
versions are run one after another, never at the same time.

The check column (in the ratio block) says "ok" when every combination
decoded to the same image (at 0/0, the original) and, with more than one
version, they all wrote the same files; otherwise what differs.

The Java half is test/TableBench.java (see results_table.py, whose runs this
uses). The table is printed and saved, with a CSV of every number (one row a
quantization, delta, coding and number of bands), as
results_combinations_<image>_<machine>.txt and .csv in the current folder.
"""
import argparse
import os
import socket
import sys

import results_table as rt

CODINGS = ("A", "T", "H", "N")
CODING_NAMES = {"A": "Context aware", "T": "Context tables", "H": "Huffman", "N": "Huffman tables"}
WIDTH = 8   # each column's width


def main():
    parser = argparse.ArgumentParser(description="Every delta / entropy combination on one image (see the top of this file).")
    parser.add_argument("image")
    parser.add_argument("--versions", default="native")
    parser.add_argument("--native", default=os.path.join(os.path.dirname(rt.REPO), "NativeEncoder"))
    parser.add_argument("--java", default=os.path.join(os.path.dirname(rt.REPO), "DeltaEncoder"))
    parser.add_argument("--quants", default="0/0,1/1,2/2,3/4")
    parser.add_argument("--bands", default="1,2,4,8")
    parser.add_argument("--deltas", default="LS,Avg")
    parser.add_argument("--codings", default="A,T,H,N")
    parser.add_argument("--runs", type=int, default=2)
    args = parser.parse_args()
    versions = [v.strip().lower() for v in args.versions.split(",") if v.strip()]
    for v in versions:
        if v not in rt.VERSIONS:
            parser.error("unknown version %s: use native, java or python" % v)
    args.quants = [q.strip() for q in args.quants.split(",") if q.strip()]
    args.bands = [int(b) for b in args.bands.split(",") if b.strip()]
    deltas = [d.strip() for d in args.deltas.split(",") if d.strip()]
    codings = [c.strip().upper() for c in args.codings.split(",") if c.strip()]
    for d in deltas:
        if d not in ("LS", "Avg"):
            parser.error("unknown delta type %s: use LS or Avg" % d)
    for c in codings:
        if c not in CODINGS:
            parser.error("unknown coding %s: use A, T, H or N" % c)
    for q in args.quants:
        if len(q.split("/")) != 2:
            parser.error("quant %s: give it as color/pixel, e.g. 3/4" % q)
    methods = ["%s-%s" % (d, c) for d in deltas for c in codings]
    args.methods = methods
    image = os.path.abspath(args.image)
    if not os.path.isfile(image):
        sys.exit("No image " + image)

    import viewer_support as vs
    rgb = vs.read_image(image)
    ydim, xdim = rgb.shape[:2]
    raw = 3 * xdim * ydim

    # ---- The runs, one version after another ----
    results, infos, notes = {}, {}, []
    for v in versions:
        print("%s ..." % rt.TITLES[v], flush=True)
        if v == "python":
            info, rows = rt.run_python(image, args)
        else:
            info, rows = rt.run_java(v, os.path.abspath(args.native if v == "native" else args.java), image, args)
        if info is None:
            notes.append("%s left out: %s" % (rt.TITLES[v], rows))
            continue
        results[v], infos[v] = rows, info
        if v == "native" and info.get("native", "").startswith("Java"):
            notes.append("NOTE: NativeWriter ran without its C library: build it first, as for NativeWriter.")
    if not results:
        sys.exit("Nothing ran:\n" + "\n".join(notes))
    shown = [v for v in versions if v in results]

    # ---- The rows: each quantization's band counts (as many as the image has room for) ----
    keys = []   # (quant, bands)
    for q in args.quants:
        seen = sorted({k[2] for v in shown for k in results[v] if k[0] == q})
        keys += [(q, n) for n in seen]
        if seen and max(seen) < max(args.bands):
            notes.append("At %s the image has room for %d bands at most (as Save allows: at least 64 rows a band)." % (q, max(seen)))

    def row_of(v, q, m, n):
        return results[v].get((q, m, n))

    def any_row(q, m, n):
        return next((row_of(v, q, m, n) for v in shown if row_of(v, q, m, n)), None)

    def check(q, n):
        files, images, lossless = {}, set(), True
        for m in methods:
            for v in shown:
                r = row_of(v, q, m, n)
                if r is None:
                    continue
                files.setdefault(m, set()).add(r["file_md5"])
                images.add(r["image_md5"])
                lossless = lossless and r["lossless"]
        problems = []
        if any(len(f) > 1 for f in files.values()):
            problems.append("files differ")
        if len(images) > 1:
            problems.append("images differ")
        if q == "0/0" and not lossless:
            problems.append("not lossless")
        return ", ".join(problems) if problems else "ok"

    # ---- The table ----
    name = os.path.basename(image)
    lines = ["%s   %d x %d   (ratio = compressed / raw 24-bit, %s bytes)" % (name, xdim, ydim, format(raw, ",")),
             "%s, %d logical processors; %s %s; times in ms, steady state (%d runs timed, divided by %d, after a warm-up)"
             % (rt.processor_name(), os.cpu_count(), rt.platform.system(), rt.platform.release(), args.runs, args.runs)]
    described = []
    if "native" in infos:
        described.append("Native: NativeWriter / NativeReader, Java %s with %s" % (infos["native"].get("java", "?"), infos["native"].get("native", "?")))
    if "java" in infos:
        described.append("Java: GenericWriter / GenericReader, Java %s" % infos["java"].get("java", "?"))
    if "python" in infos:
        described.append("Python: generic_writer.py / generic_reader.py, Python %s" % infos["python"]["python"])
    lines.append("; ".join(described))
    lines.append("Codings: " + ", ".join("%s %s" % (c, CODING_NAMES[c]) for c in codings))
    lines += notes
    lines.append("")

    lead = "%-5s %5s " % ("quant", "bands")
    heads = "".join("%*s" % (WIDTH, m) for m in methods)

    def block(title, cell, with_check=False):
        lines.append(title)
        lines.append(lead + heads + ("   check" if with_check else ""))
        last = None
        for q, n in keys:
            if last is not None and q != last:
                lines.append("")
            last = q
            cells = "".join("%*s" % (WIDTH, cell(q, m, n)) for m in methods)
            lines.append("%-5s %5d %s%s" % (q, n, cells, ("   " + check(q, n)) if with_check else ""))
        lines.append("")

    def ratio_cell(q, m, n):
        r = any_row(q, m, n)
        return rt.ratio(r["bytes"], raw) if r else "-"

    def time_cell(v, which):
        def cell(q, m, n):
            r = row_of(v, q, m, n)
            return "%.0f" % r[which] if r else "-"
        return cell

    block("Ratio", ratio_cell, with_check=True)
    for v in shown:
        block("%s encode ms" % rt.TITLES[v], time_cell(v, "encode"))
        block("%s decode ms" % rt.TITLES[v], time_cell(v, "decode"))

    csv = ["image,quant,delta,coding,bands,bytes,ratio," + ",".join("encode_ms_%s,decode_ms_%s" % (v, v) for v in shown) + ",check"]
    for q, n in keys:
        status = check(q, n)
        for m in methods:
            r = any_row(q, m, n)
            if r is None:
                continue
            d, c = m.split("-")
            times = []
            for v in shown:
                rv = row_of(v, q, m, n)
                times += ["%.1f" % rv["encode"], "%.1f" % rv["decode"]] if rv else ["", ""]
            csv.append(",".join([name, q, d, c, str(n), str(r["bytes"]), "%.6f" % (r["bytes"] / raw)] + times + [status]))

    print()
    print("\n".join(lines))
    stem = "results_combinations_%s_%s" % (os.path.splitext(name)[0], socket.gethostname().split(".")[0])
    with open(stem + ".txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(stem + ".csv", "w", encoding="utf-8") as f:
        f.write("\n".join(csv) + "\n")
    print("Saved %s.txt and %s.csv in %s" % (stem, stem, os.getcwd()))


if __name__ == "__main__":
    main()
