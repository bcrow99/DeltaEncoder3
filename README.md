# Delta Writer / Delta Reader (Python)

Python (PySide6) versions of DeltaWriter.java and DeltaReader.java, version 1.0.
They quantize an image, code it as deltas, and save and read Delta format 'D'
version 1 — the same file format as the Java programs. A file from either
writer opens in either reader.

```
python3 delta_writer.py [image]      # saves to the file "foo", like the Java version
python3 delta_reader.py foo
```

Requirements: `numpy`, `PySide6`, `opencv-python` (image loading), and `numba`.
Numba is optional but strongly recommended: without it everything still works,
identically, but slowly. The first run compiles the Numba code (about a minute);
after that it loads from the cache in `__pycache__`.

## Modules

| Module | Java | Contents |
|---|---|---|
| `delta_writer.py` | DeltaWriter | `DeltaCoder` (survey, apply, save — usable without a window) and the window |
| `delta_reader.py` | DeltaReader | `DeltaDecoder` (usable without a window) and the window |
| `viewer_support.py` | ViewerSupport | image window with zoom, dialogs, image loading, background jobs |
| `delta_mapper.py` | DeltaMapper | channel sets, delta types 0–13, maps, context-coded deltas, block search |
| `arithmetic_mapper.py` | ArithmeticMapper | range coder; static, adaptive and context coders; frequency tables |
| `string_mapper.py` | StringMapper | unary strings and their bit-run compression |
| `code_mapper.py` | CodeMapper | Huffman ("regular"), Shannon estimate, Deflate |
| `resize_mapper.py` | ResizeMapper | Pixel Resolution resizing |
| `java_io.py` | — | DataInputStream / DataOutputStream equivalents |
| `numba_support.py` | — | the optional `@njit` and Java-style integer division |

These modules contain only what the Delta programs use. They replace the older
modules of the same names, so any other Python program that still uses the old
versions (for functions not here) needs the old files until it is ported.

HiDPI: Qt 6 scales the interface itself on Linux, Windows and macOS, so the Java
version's font-scaling fallback isn't needed.

## Not ported yet

The pixel pyramid ("Average" in the Java Quantization menu) needs ImageMapper.
The Python writer saves without it, and the Python reader refuses files that
use it, with a message.

## Compatibility testing

The `test` folder has differential tests. A Java program writes random inputs
along with the Java outputs, and a Python script recomputes every output and
compares them:

```
java CoderCheck coders.bin && python3 check_coders.py coders.bin
java DeltaCheck delta.bin a.png b.png c.png d.png && python3 check_delta.py delta.bin
```

(Compile the Java with the Java programs' classes on the classpath; run the
Python from the `test` folder.)

Results when this was written:

- **Coders:** 1,010 cases, all identical. Covered: strings, the static,
  adaptive and context coders, Huffman, and resizing at every Pixel Resolution.
- **DeltaMapper:** about 8,000 cases, all identical. Covered: all 14 delta types
  and their decoders, the map forms, context deltas, smoothing, and the block
  search.
- **Whole files:** 700 files saved by both writers at the same settings, all
  byte-for-byte identical. Covered: 14 delta types × 5 entropy types × 3
  datatypes; lossless and quantized; odd image sizes; grey images.
- **Reading:** both readers decode every file (Java, Python and DeltaWriter2)
  to the same pixels.

One difference to expect: JPEG decoders differ, so the same JPEG can load with
slightly different pixel values in Java and Python, and the saved files then
differ too (each still reads correctly in both readers). PNGs load identically.
