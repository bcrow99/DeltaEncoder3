"""
java_io.py -- java.io.DataOutputStream / DataInputStream equivalents:
big-endian ints and shorts, and Java's signed/unsigned byte reads.
"""

import struct


class DataOutput:
    """Collects bytes like a DataOutputStream over a ByteArrayOutputStream."""

    def __init__(self):
        self.buf = bytearray()

    def write_byte(self, v):
        self.buf.append(int(v) & 0xFF)

    def write_short(self, v):
        self.buf += struct.pack(">H", int(v) & 0xFFFF)

    def write_int(self, v):
        self.buf += struct.pack(">I", int(v) & 0xFFFFFFFF)

    def write(self, data):
        self.buf += bytes(data)

    def size(self):
        return len(self.buf)

    def to_bytes(self):
        return bytes(self.buf)


class DataInput:
    """Reads a bytes object (or a file) like a DataInputStream."""

    def __init__(self, source):
        if isinstance(source, (bytes, bytearray, memoryview)):
            import io
            source = io.BytesIO(bytes(source))
        self.f = source

    def read_fully(self, n):
        data = self.f.read(n)
        if len(data) != n:
            raise EOFError("end of file: wanted %d bytes, got %d" % (n, len(data)))
        return data

    def read_byte(self):
        return struct.unpack(">b", self.read_fully(1))[0]

    def read_unsigned_byte(self):
        return self.read_fully(1)[0]

    def read_short(self):
        return struct.unpack(">h", self.read_fully(2))[0]

    def read_unsigned_short(self):
        return struct.unpack(">H", self.read_fully(2))[0]

    def read_int(self):
        return struct.unpack(">i", self.read_fully(4))[0]
