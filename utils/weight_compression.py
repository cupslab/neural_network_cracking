#!/usr/bin/env python

import sys
import argparse
import struct
import numpy

import model_compression

read_chunksize_compress = struct.calcsize(
    model_compression.CompressActor.RECORD_FMT)
def read_chunk_compress(chunkbytes):
    return struct.unpack(
        model_compression.CompressActor.RECORD_FMT, chunkbytes)[0]

read_chunksize_decompress = 2
def read_chunk_decompress(chunkbytes):
    return numpy.fromstring(chunkbytes, dtype=numpy.float16)[0]

def read_input(fname, read_chunk_fn, chunksize):
    with open(fname, 'rb') as afile:
        chunkbytes = afile.read(chunksize)
        while chunkbytes != b'':
            yield read_chunk_fn(chunkbytes)
            chunkbytes = afile.read(chunksize)

def write_output_compress(next_float):
    out_bytes = numpy.float16(next_float)
    assert numpy.isfinite(out_bytes)
    return bytes(memoryview(out_bytes))

def write_output_decompress(next_float):
    return struct.pack(model_compression.CompressActor.RECORD_FMT, next_float)

def write_output(input_floats, outfile, write_fn):
    with open(outfile, 'wb') as ofile:
        for next_float in input_floats:
            ofile.write(write_fn(next_float))

def debug(input_floats, outfile):
    with open(outfile, 'w') as ofile:
        for afloat in input_floats:
            ofile.write(str(afloat) + '\n')

def main(args):
    assert args.decompress or args.compress, (
        'Must provide compress or decompress argument')
    if args.compress:
        read_fn = read_chunk_compress
        chunksize = read_chunksize_compress
        write_fn = write_output_compress
    else:
        read_fn = read_chunk_decompress
        chunksize = read_chunksize_decompress
        write_fn = write_output_decompress
    input_floats = read_input(args.ifile, read_fn, chunksize)
    write_output(input_floats, args.ofile, write_fn)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Compress a weight file. ')
    parser.add_argument('ifile', help = 'Input file. ')
    parser.add_argument('ofile', help = 'Output file.')
    parser.add_argument('-c', '--compress', action='store_true')
    parser.add_argument('-d', '--decompress', action='store_true')
    main(parser.parse_args())
