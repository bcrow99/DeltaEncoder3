import java.awt.image.BufferedImage;
import java.io.BufferedInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.security.MessageDigest;

/**
 * The Java half of results_table.py: times one writer and its reader
 * (NativeWriter and NativeReader, or GenericWriter and GenericReader) on one
 * image, for each quantization, coding and number of bands asked for, and
 * prints a line per setting for results_table.py to read.
 *
 * <p>It is compiled with the classes of the repository whose writer it
 * times (NativeEncoder's java/src, or DeltaEncoder's), and uses that
 * repository's DeltaMapper, ResizeMapper and ViewerSupport. The writer and
 * reader are reached by name (their encode, readHeader and decodeFile
 * aren't public), so the same source serves both. The channels are made as
 * the writers' Apply makes them (resize2, quantizeChannel,
 * getCandidateChannels, the writer's CHANNELS).
 *
 * <p>Each setting is run once to warm up, then encoded runs times in one
 * timing and the time divided by runs; the same for the decoding (the
 * writers' and readers' "2 runs timed, halved"). The encoding is Save's
 * without its decode check; the decoding is the reader's, from reading the
 * file to the finished image.
 *
 * <pre>
 * java TableBench Writer Reader image quants bands methods runs
 *   e.g. java TableBench GenericWriter GenericReader scrubjay.jpg 0/0,2/2,3/4 1,2,4,8 LS-A,LS-H,Avg-A,Avg-H 2
 * </pre>
 *
 * quants are Color Resolution / Pixel Resolution pairs; methods are the
 * deltas (LS or Avg), a dash, and the coding (A context aware, H Huffman,
 * ZS Deflate (Strings), Z Deflate, N neighbour Huffman). Output lines:
 * <pre>
 * INFO java=... processors=... native=...
 * ROW color/pixel method bands bytes file_md5 image_md5 lossless encode_ms decode_ms
 * </pre>
 * image_md5 is the md5 of the decoded image's red, green and blue bytes,
 * pixel by pixel; lossless is 1 if the decoded image is the original, else 0.
 */
public class TableBench
{
	/** Not made: main only. */
	private TableBench() { }

	/**
	 * Runs the benchmark.
	 *
	 * @param args writer, reader, image, quants, bands, methods, runs
	 * @throws Exception if the image can't be read or a step fails
	 */
	public static void main(String[] args) throws Exception
	{
		if(args.length < 7)
		{
			System.out.println("Usage: java TableBench Writer Reader image quants bands methods runs");
			System.exit(1);
		}
		Class<?> writer = Class.forName(args[0]);
		Class<?> reader = Class.forName(args[1]);
		String   image_file = args[2];
		String[] quants  = args[3].split(",");
		String[] bands   = args[4].split(",");
		String[] methods = args[5].split(",");
		int      runs    = Integer.parseInt(args[6]);

		int[] channels    = (int[]) constant(writer, "CHANNELS");
		int   band_min    = (Integer) constant(writer, "BAND_MIN_ROWS");
		int   max_bands   = (Integer) constant(writer, "MAX_BANDS");
		Method encode = writer.getDeclaredMethod("encode", DataOutputStream.class, int[][].class, int[].class, int.class, int.class, int.class,
		                                         int.class, int.class, int.class, int.class, int.class, boolean.class);
		encode.setAccessible(true);

		String status = "Java";
		try { status = (String) constant(Class.forName("DeltaMapper"), "NATIVE_STATUS"); }
		catch(NoSuchFieldException e) { /* DeltaEncoder's DeltaMapper: Java only */ }
		System.out.println("INFO java=" + System.getProperty("java.version") + " processors=" + Runtime.getRuntime().availableProcessors()
		                   + " native=" + status);

		BufferedImage image = ViewerSupport.readImage(image_file);
		int xdim = image.getWidth(), ydim = image.getHeight();
		int[] original = image.getRGB(0, 0, xdim, ydim, null, 0, xdim);
		image = null;
		// The writers' source channels: the pixel's top byte first.
		int[][] bgr = new int[3][original.length];
		for(int k = 0; k < original.length; k++)
		{
			bgr[0][k] = (original[k] >> 16) & 0xff;
			bgr[1][k] = (original[k] >> 8) & 0xff;
			bgr[2][k] = original[k] & 0xff;
		}

		File temp = File.createTempFile("TableBench", ".bin");
		temp.deleteOnExit();
		for(String q : quants)
		{
			String[] pair = q.split("/");
			int shift = Integer.parseInt(pair[0]), quant = Integer.parseInt(pair[1]);

			// The channels Save codes, as Apply makes them.
			int[] size = DeltaMapper.getQuantizedSize(xdim, ydim, quant);
			int w = size[0], h = size[1];
			int[][] qch = new int[3][];
			for(int i = 0; i < 3; i++)
				qch[i] = DeltaMapper.quantizeChannel((quant != 0) ? ResizeMapper.resize2(bgr[i], xdim, w, h) : bgr[i], shift);
			int[] min6 = new int[6];
			int[][] candidates = DeltaMapper.getCandidateChannels(qch[0], qch[1], qch[2], min6);
			int[][] value = new int[3][];
			int[] min = new int[3];
			for(int i = 0; i < 3; i++) { value[i] = candidates[channels[i]]; min[i] = min6[channels[i]]; }

			for(String m : methods)
			{
				int delta_type   = m.startsWith("LS") ? (Integer) constant(writer, "DELTA_LS") : (Integer) constant(writer, "DELTA_AVERAGE");
				int entropy_type = (Integer) constant(writer, entropyConstant(m.substring(m.indexOf('-') + 1)));
				int last = -1;
				for(String b : bands)
				{
					// The bands Save writes (the writer's bandsUsed).
					int n = Math.max(1, Math.min(Math.min(Integer.parseInt(b.trim()), max_bands), h / band_min));
					if(n == last) continue;   // the image is too small for more
					last = n;

					// Encoding: once to warm up, then runs in one timing.
					byte[] file = encodeOnce(encode, value, min, w, xdim, ydim, shift, quant, n, delta_type, entropy_type);
					long t0 = System.nanoTime();
					for(int run = 0; run < runs; run++)
					{
						byte[] again = encodeOnce(encode, value, min, w, xdim, ydim, shift, quant, n, delta_type, entropy_type);
						if(!java.util.Arrays.equals(again, file)) throw new IllegalStateException("The runs coded different files: " + q + " " + m + " " + n);
					}
					double encode_ms = (System.nanoTime() - t0) / 1e6 / runs;
					try(FileOutputStream out = new FileOutputStream(temp)) { out.write(file); }

					// Decoding: the reader made without its window (as ScaleBench
					// does), its header read, then decodeFile: once to warm up,
					// then runs in one timing.
					Object r = newReader(reader, temp);
					Method decodeFile = reader.getDeclaredMethod("decodeFile", String.class);
					decodeFile.setAccessible(true);
					BufferedImage decoded = (BufferedImage) decodeFile.invoke(r, temp.getPath());
					t0 = System.nanoTime();
					for(int run = 0; run < runs; run++) decodeFile.invoke(r, temp.getPath());
					double decode_ms = (System.nanoTime() - t0) / 1e6 / runs;

					int[] rgb = decoded.getRGB(0, 0, xdim, ydim, null, 0, xdim);
					boolean lossless = true;
					for(int k = 0; k < rgb.length && lossless; k++) lossless = (rgb[k] & 0xffffff) == (original[k] & 0xffffff);
					System.out.println(String.format("ROW %s %s %d %d %s %s %d %.1f %.1f", q, m, n, file.length, md5(file), imageMd5(rgb),
					                                 lossless ? 1 : 0, encode_ms, decode_ms));
					System.out.flush();
				}
			}
		}
		temp.delete();
	}

	/**
	 * The writer's constant for a method's coding: A context aware, H
	 * Huffman, ZS Deflate (Strings), Z Deflate, N neighbour Huffman.
	 *
	 * @param coding the coding's letters
	 * @return the constant's name
	 */
	private static String entropyConstant(String coding)
	{
		switch(coding)
		{
			case "A":  return "ENTROPY_MIXED";
			case "H":  return "ENTROPY_HUFFMAN";
			case "ZS": return "ENTROPY_DEFLATE_STRINGS";
			case "Z":  return "ENTROPY_DEFLATE";
			case "N":  return "ENTROPY_NEIGHBOUR";
			default: throw new IllegalArgumentException("unknown coding " + coding + ": use A, H, ZS, Z or N");
		}
	}

	/**
	 * One encoding, without the decode check.
	 *
	 * @return the file's bytes
	 * @throws Exception if it fails
	 */
	private static byte[] encodeOnce(Method encode, int[][] value, int[] min, int w, int xdim, int ydim, int shift, int quant, int n,
	                                 int delta_type, int entropy_type) throws Exception
	{
		ByteArrayOutputStream bytes = new ByteArrayOutputStream();
		encode.invoke(null, new DataOutputStream(bytes), value, min, w, xdim, ydim, shift, quant, n, delta_type, entropy_type, false);
		return bytes.toByteArray();
	}

	/**
	 * A reader made without its window, with the file's header read.
	 *
	 * @param c    the reader's class
	 * @param file the file
	 * @return the reader
	 * @throws Exception if it fails
	 */
	private static Object newReader(Class<?> c, File file) throws Exception
	{
		Field uf = Class.forName("sun.misc.Unsafe").getDeclaredField("theUnsafe");
		uf.setAccessible(true);
		Object unsafe = uf.get(null);
		Object reader = unsafe.getClass().getMethod("allocateInstance", Class.class).invoke(unsafe, c);
		Field mf = c.getDeclaredField("min");   // the one field the readers initialize where it's declared
		mf.setAccessible(true);
		mf.set(reader, new int[3]);
		Method header = c.getDeclaredMethod("readHeader", DataInputStream.class, String.class);
		header.setAccessible(true);
		try(DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(file)))) { header.invoke(reader, in, file.getPath()); }
		return reader;
	}

	/**
	 * A static field's value, however it's declared.
	 *
	 * @return the value
	 * @throws Exception if there's no such field
	 */
	private static Object constant(Class<?> c, String name) throws Exception
	{
		Field f = c.getDeclaredField(name);
		f.setAccessible(true);
		return f.get(null);
	}

	/**
	 * The md5 of an image's red, green and blue bytes, pixel by pixel.
	 *
	 * @param rgb the pixels, as getRGB gives them
	 * @return the md5, in hex
	 * @throws Exception if there's no md5
	 */
	private static String imageMd5(int[] rgb) throws Exception
	{
		byte[] b = new byte[3 * rgb.length];
		for(int k = 0; k < rgb.length; k++)
		{
			b[3 * k]     = (byte) (rgb[k] >> 16);
			b[3 * k + 1] = (byte) (rgb[k] >> 8);
			b[3 * k + 2] = (byte) rgb[k];
		}
		return md5(b);
	}

	/**
	 * An md5, in hex.
	 *
	 * @param b the bytes
	 * @return the md5
	 * @throws Exception if there's no md5
	 */
	private static String md5(byte[] b) throws Exception
	{
		StringBuilder s = new StringBuilder();
		for(byte x : MessageDigest.getInstance("MD5").digest(b)) s.append(String.format("%02x", x & 0xff));
		return s.toString();
	}
}
