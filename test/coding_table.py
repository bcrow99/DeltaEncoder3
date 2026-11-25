"""A table comparing the codings of one writer (NativeEncoder's NativeWriter
and NativeReader, or DeltaEncoder's GenericWriter and GenericReader) on one
image: compression ratio, encoding and decoding times for each coding, at
each quantization (Color Resolution / Pixel Resolution), delta type and
number of bands:

    python test/coding_table.py image [options]

    --native DIR    NativeEncoder's folder     (default: ../NativeEncoder beside this repository)
    --generic       time GenericWriter and GenericReader instead (Java only)
    --java DIR      DeltaEncoder's folder, for --generic (default: ../DeltaEncoder beside this repository)
    --quants LIST   color/pixel pairs          (default 0/0,1/1,2/2,3/4)
    --bands LIST    band counts                (default 1,2,4,8)
    --deltas LIST   LS, Avg                    (default LS,Avg)
    --codings LIST  codings, in column order   (default A,T,H,N)
    --runs N        timed runs per setting     (default 2)

The codings (the Entropy menu):
    A   Context aware
    T   Context tables: each channel's own contexts with fixed tables, rANS coded
    H   Huffman: unary strings, bit-pair compressed if it pays, Huffman coded
    N   Huffman tables (neighbour Huffman): Huffman tables chosen by the
        neighbours, a run mode and pairs (HuffmanWriter2's coding)

The ratio is the file's size over the raw image's (3 bytes a pixel), so
smaller is better. The times are in milliseconds, as the writer and reader
print them once their code is compiled: each setting is run once to warm up,
then run N times in one timing, divided by N. Encoding is Save without its
decode check (the fit included); decoding is from reading the file to the
finished image.

The check column says "ok" when every coding decoded to the same image (at
0/0, the original); otherwise what differs.

The Java half is test/TableBench.java, compiled with the repository's classes
(javac and java must be on the PATH; see results_table.py). The table is
printed and saved, with a CSV of the same numbers, as
results_codings_<image>_<machine>.txt and .csv in the current folder.
"""
import argparse
import os
import socket
import sys

import results_table as rt

CODING_NAMES = {"A": "Context aware", "T": "Context tables", "H": "Huffman", "N": "Huffman tables"}


def main():
    parser = argparse.ArgumentParser(description="Codings compared on one image (see the top of this file).")
    parser.add_argument("image")
    parser.add_argument("--native", default=os.path.join(os.path.dirname(rt.REPO), "NativeEncoder"))
    parser.add_argument("--generic", action="store_true")
    parser.add_argument("--java", default=os.path.join(os.path.dirname(rt.REPO), "DeltaEncoder"))
    parser.add_argument("--quants", default="0/0,1/1,2/2,3/4")
    parser.add_argument("--bands", default="1,2,4,8")
    parser.add_argument("--deltas", default="LS,Avg")
    parser.add_argument("--codings", default="A,T,H,N")
    parser.add_argument("--runs", type=int, default=2)
    args = parser.parse_args()
    args.quants = [q.strip() for q in args.quants.split(",") if q.strip()]
    args.bands = [int(b) for b in args.bands.split(",") if b.strip()]
    deltas = [d.strip() for d in args.deltas.split(",") if d.strip()]
    codings = [c.strip().upper() for c in args.codings.split(",") if c.strip()]
    for d in deltas:
        if d not in ("LS", "Avg"):
            parser.error("unknown delta type %s: use LS or Avg" % d)
    for c in codings:
        if c not in CODING_NAMES:
            parser.error("unknown coding %s: use %s" % (c, ", ".join(CODING_NAMES)))
    for q in args.quants:
        if len(q.split("/")) != 2:
            parser.error("quant %s: give it as color/pixel, e.g. 3/4" % q)
    args.methods = ["%s-%s" % (d, c) for d in deltas for c in codings]
    image = os.path.abspath(args.image)
    if not os.path.isfile(image):
        sys.exit("No image " + image)

    import viewer_support as vs
    rgb = vs.read_image(image)
    ydim, xdim = rgb.shape[:2]
    raw = 3 * xdim * ydim

    version = "java" if args.generic else "native"
    folder = os.path.abspath(args.java if args.generic else args.native)
    print("%s ..." % ("GenericWriter" if args.generic else "NativeWriter"), flush=True)
    info, rows = rt.run_java(version, folder, image, args)
    if info is None:
        sys.exit("Couldn't run it: " + rows)

    # ---- The table ----
    name = os.path.basename(image)
    lines = []
    lines.append("%s   %d x %d   (ratio = compressed / raw 24-bit, %s bytes)" % (name, xdim, ydim, format(raw, ",")))
    lines.append("%s, %d logical processors; %s %s; times in ms, steady state (%d runs timed, divided by %d, after a warm-up)"
                 % (rt.processor_name(), os.cpu_count(), rt.platform.system(), rt.platform.release(), args.runs, args.runs))
    if args.generic:
        lines.append("GenericWriter / GenericReader, Java %s" % info.get("java", "?"))
    else:
        lines.append("NativeWriter / NativeReader, Java %s with %s" % (info.get("java", "?"), info.get("native", "?")))
        if info.get("native", "").startswith("Java"):
            lines.append("NOTE: NativeWriter ran without its C library: build it first, as for NativeWriter.")
    lines.append("")
    group = "  ".join("%-20s" % CODING_NAMES[c] for c in codings)
    sub = "  ".join("%-20s" % "ratio   enc    dec" for c in codings)
    lead = "%-5s %-5s %5s" % ("quant", "delta", "bands")
    lines.append(" " * len(lead) + "   " + group + "  check")
    lines.append(lead + "   " + sub)

    csv = ["image,quant,delta,bands," + ",".join("%s_bytes,%s_ratio,%s_encode_ms,%s_decode_ms" % (c, c, c, c) for c in codings) + ",check"]
    for q in args.quants:
        mosts = []
        for d in deltas:
            seen = sorted({k[2] for k in rows if k[0] == q and k[1].startswith(d + "-")})
            mosts += seen
            for n in seen:
                cells, csv_cells, images, lossless = [], [], set(), True
                for c in codings:
                    r = rows.get((q, "%s-%s" % (d, c), n))
                    if r is None:
                        cells.append("%-20s" % "-")
                        csv_cells += ["", "", "", ""]
                        continue
                    cells.append("%-20s" % ("%-6s %5.0f %6.0f" % (rt.ratio(r["bytes"], raw), r["encode"], r["decode"])))
                    csv_cells += [str(r["bytes"]), "%.6f" % (r["bytes"] / raw), "%.1f" % r["encode"], "%.1f" % r["decode"]]
                    images.add(r["image_md5"])
                    lossless = lossless and r["lossless"]
                problems = []
                if len(images) > 1:
                    problems.append("images differ")
                if q == "0/0" and not lossless:
                    problems.append("not lossless")
                status = ", ".join(problems) if problems else "ok"
                lines.append("%-5s %-5s %5d   %s  %s" % (q, d, n, "  ".join(cells), status))
                csv.append(",".join([name, q, d, str(n)] + csv_cells + [status]))
        if mosts and max(mosts) < max(args.bands):
            lines.append("(at %s the image has room for %d bands at most: at least 64 rows a band)" % (q, max(mosts)))
        lines.append("")

    print()
    print("\n".join(lines))
    stem = "results_codings_%s_%s" % (os.path.splitext(name)[0], socket.gethostname().split(".")[0])
    with open(stem + ".txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(stem + ".csv", "w", encoding="utf-8") as f:
        f.write("\n".join(csv) + "\n")
    print("Saved %s.txt and %s.csv in %s" % (stem, stem, os.getcwd()))


if __name__ == "__main__":
    main()
