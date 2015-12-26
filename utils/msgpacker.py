import sys
import argparse
import json
import itertools
import struct

import simpleubjson
import msgpack
import bitstring

weight_first_list = [
  'timeDistributedDense', 'denseLayer', 'embeddingLayer',
  'batchNormalizationLayer', 'parametricReLULayer', 'parametricSoftplusLayer',
  'rLSTMLayer', 'rGRULayer', 'rJZS1Layer', 'rJZS2Layer', 'rJZS3Layer',
  'convolution2DLayer', 'convolution1DLayer']

def transform_to_binary(weights, bits):
    answer = {}
    for key in weights:
        value = weights[key]
        assert type(key) == str
        assert type(value) == list
        assert len(value) > 0
        size = [len(value)]
        flat = value
        if type(value[0]) == list:
            for sublist in value:
                assert len(sublist) == len(value[0])
            size.append(len(value[0]))
            flat = list(itertools.chain(*value))
        bits_flat = bitstring.BitArray()
        for item in flat:
            bits_flat.append(bitstring.BitArray(uint=item, length=bits))
        answer[key] = {
            'size' : size,
            'data' : bits_flat.tobytes()
        }
    return answer

def main(args):
    model = json.load(args.ifile)
    if args.transform_to_binary != -1:
        for layer in model:
            if layer['layerName'] in weight_first_list:
                layer['parameters'][0] = transform_to_binary(
                    layer['parameters'][0], args.transform_to_binary)
    if args.ubjson_format:
        args.ofile.write(simpleubjson.encode(model))
    else:
        msgpack.dump(model, args.ofile, use_bin_type=True,
                     use_single_float=args.single_precision_float)
    args.ofile.close()

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='Save a JSON file using the msgpack format')
    parser.add_argument('-i', '--ifile', type=argparse.FileType('r'),
                        help='Input file. Default is stdin. ',
                        default=sys.stdin)
    parser.add_argument('ofile', type=argparse.FileType('wb'),
                        help='Output file. ')
    parser.add_argument('-s', '--single-precision-float', action='store_true')
    parser.add_argument('-t', '--transform-to-binary', type=int, default=-1)
    parser.add_argument('-u', '--ubjson-format', action='store_true')
    main(parser.parse_args())
