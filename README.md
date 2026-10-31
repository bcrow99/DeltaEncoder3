# Delta Writer / Delta Reader (Python)

Python (PySide6) versions of DeltaWriter.java and DeltaReader.java, version 1.0.
They quantize an image, code it as deltas, and save and read Delta format 'D'
version 1 — the same file format as the Java programs. A file from either
writer opens in either reader.

```
python3 delta_writer.py [image]      # saves to the file "foo", like the Java version
python3 delta_reader.py foo
```

The other writer/reader pairs are ported the same way, with the same file
formats as their Java versions, and also save to "foo":

```
python3 delta_writer2.py [image]     # Delta format, read by delta_reader.py
python3 simple_writer.py [image]     # Simple format 'S' version 8
python3 simple_reader.py foo
python3 simple_writer2.py [image]    # Simple 2 format 'F' version 2
python3 simple_reader2.py foo
python3 block_writer.py [image]      # Block format 'B'
python3 block_reader.py foo
python3 packet_writer.py [image]     # Packet format 'P' version 1
python3 packet_reader.py foo
python3 pixel_writer.py [image]      # Pixel format 'X' versions 1-4
python3 pixel_reader.py foo
```

The Pixel format's ID byte is 'X'; it was 'P' until that turned out to be the
Packet format's too, so pixel_reader.py explains a 'P' file instead of
reading it.

Three smaller programs:

```
python3 comparison.py image.png [scale] [display width]   # delta coder vs JPEG, rate and distortion
python3 segment_steer_demo.py [dimension] [budget] [workers]  # steered arithmetic coding demo
python3 test_get_digits.py                                # repeating-decimal demo (asks for a and b)
```

Requirements: `numpy`, `PySide6`, `opencv-python` (image loading), and `numba`.
`comparison.py` also uses `pillow` if it's installed, to make JPEGs exactly as
Java's ImageIO does.
Numba is optional but strongly recommended: without it everything still works,
identically, but slowly. The first run compiles the Numba code (about a minute);
after that it loads from the cache in `__pycache__`.

## Modules

| Module | Java | Contents |
|---|---|---|
| `delta_writer.py` | DeltaWriter | `DeltaCoder` (survey, apply, save — usable without a window) and the window |
| `delta_reader.py` | DeltaReader | `DeltaDecoder` (usable without a window) and the window |
| `delta_writer2.py` | DeltaWriter2 | `DeltaCoder2`: delta types 6–13 only, String* datatype, map-coding report |
| `simple_writer.py` | SimpleWriter | `SimpleCoder`: least squares alone, mixed-context-coded deltas; each channel's window radius chosen by the analysis |
| `simple_reader.py` | SimpleReader | `SimpleDecoder` |
| `simple_writer2.py` | SimpleWriter2 | `SimpleCoder2`: the same coding with the channels green, blue-green, red-green and window radius 10, no analysis |
| `simple_reader2.py` | SimpleReader2 | `SimpleDecoder2` |
| `block_writer.py` | BlockWriter | `BlockCoder`: block map with mixed-context-coded deltas; each channel's block size and predictors chosen by the survey |
| `block_reader.py` | BlockReader | `BlockDecoder` |
| `packet_writer.py` | PacketWriter | `PacketCoder`: unary strings split into separately compressed packets |
| `packet_reader.py` | PacketReader | `PacketDecoder` |
| `pixel_writer.py` | PixelWriter | `PixelCoder`: context-adaptive arithmetic coding of the pixels, block size search |
| `pixel_reader.py` | PixelReader | `PixelDecoder` |
| `viewer_support.py` | ViewerSupport | image window with zoom, dialogs, image loading, background jobs |
| `delta_mapper.py` | DeltaMapper | channel sets, delta types 0–13, maps, context-coded and mixed-context-coded deltas, block search |
| `arithmetic_mapper.py` | ArithmeticMapper | range coder; static, adaptive and context coders; frequency tables |
| `string_mapper.py` | StringMapper | unary strings and their bit-run compression |
| `code_mapper.py` | CodeMapper | Huffman ("regular"), Shannon estimate, Deflate |
| `resize_mapper.py` | ResizeMapper | Pixel Resolution resizing |
| `segment_mapper.py` | SegmentMapper | the segmenting and packing the Packet programs use |
| `pixel_mapper.py` | PixelMapper | per-block context models for the pixels or prediction errors, red/green/blue or green-difference channels |
| `image_mapper.py` | ImageMapper | the pixel-pyramid operators, dilation, area resampling, registration |
| `probabilistic_mapper.py` | ProbabilisticMapper | steering an exact arithmetic-coding interval, combinadic and c-sequence coders |
| `fraction_mapper.py` | FractionMapper | `BigFraction` (exact fractions plus the wheel's infinity and bottom), repeating decimal and binary digits |
| `comparison.py` | Comparison | rate-distortion comparison of the delta coder and JPEG |
| `segment_steer_demo.py` | SegmentSteerDemo | races candidate targets for ProbabilisticMapper's steering, on separate processes |
| `test_get_digits.py` | TestGetDigits | interactive demo of the decimal digit functions |
| `java_io.py` | — | DataInputStream / DataOutputStream equivalents |
| `numba_support.py` | — | the optional `@njit` and Java-style integer division |

These modules contain only what the Delta programs use. They replace the older
modules of the same names, so any other Python program that still uses the old
versions (for functions not here) needs the old files until it is ported.

HiDPI: Qt 6 scales the interface itself on Linux, Windows and macOS, so the Java
version's font-scaling fallback isn't needed.

## The pixel pyramid

The pixel pyramid ("Average" in the old Java Quantization menu) has been taken
out of both the Java and the Python Delta programs. Both readers refuse old
files that used it, with a message. The Pixel programs have it, in Java and
Python: Quantization > Average in pixel_writer.py codes the red, green and
blue channels averaged over 1 or 2 levels of 2 x 2 squares, plus a sign bit
per pixel per level (pixel_mapper.py, with the operators in
`image_mapper.py`).

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
- **FractionMapper (version 2.1):** 20,500 cases plus 129 rounding ties,
  all identical, and every digit expansion checked against exact
  arithmetic. Covered: BigFraction arithmetic and comparison including
  infinity and bottom, to_double, and decimal and binary digit expansions
  in both directions, including the inputs that overflowed before 2.1.
- **Comparison, SegmentSteerDemo, TestGetDigits:** the same output as the
  Java, line for line: Comparison on 4 images plus a half-size run (every
  number in both the JPEG and delta-coder tables), SegmentSteerDemo's segment
  and every target's search at 5 segment sizes, and TestGetDigits on 23
  inputs, including bad and overflowing ones.
- **PixelMapper:** 640 random cases (sizes 1-139 in each direction, every
  Color Resolution, block sizes 8-256, all 3 codings and both channel sets),
  every coded channel byte-for-byte identical and decoding back in both
  versions, and the block size search choosing the same size from the same
  table in 160 of them.
- **Pixel programs:** 31 files from 4 PNGs (a photo, a cat, two synthetic
  images with odd sizes) saved by the Java and Python writers: the start-up
  settings plus every coding and channel set and a range of Pixel and Color
  Resolutions. All byte-for-byte identical, with identical block-size
  tables. Both readers decode all 62 files, plus 23 version 1 and 2 copies,
  to the same pixels.
- **DeltaWriter2, Simple and Block programs:** 420 files from 5 PNGs (odd and
  even sizes, tiny, grey) saved by the Java and Python writers: every delta
  type each writer offers, all entropy types, a range of block sizes and
  predictor sets, Pixel and Color Resolution, and arithmetic block sizes. All
  byte-for-byte identical, with identical previews, and the startup analysis
  chose the same delta type, block settings and channel set as the Java for
  every image. Both readers decode all 840 files to the same pixels.

One difference to expect: JPEG decoders differ, so the same JPEG can load with
slightly different pixel values in Java and Python, and the saved files then
differ too (each still reads correctly in both readers). PNGs load identically.
