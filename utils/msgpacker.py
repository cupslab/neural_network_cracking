import sys
import argparse
import json
import itertools
import struct
import logging

import simpleubjson
import msgpack
import bitstring
import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten

weight_first_list = [
  'timeDistributedDense', 'denseLayer', 'embeddingLayer',
  'batchNormalizationLayer', 'parametricReLULayer', 'parametricSoftplusLayer',
  'rLSTMLayer', 'rGRULayer', 'rJZS1Layer', 'rJZS2Layer', 'rJZS3Layer',
  'convolution2DLayer', 'convolution1DLayer']

class Transformer(object):
    needs_two_step = False

class BinaryTransformer(Transformer):
    def __init__(self, bits):
        self.bits = bits

    def __call__(self, flat_array, size):
        bits_flat = bitstring.BitArray()
        for item in flat_array:
            bits_flat.append(bitstring.BitArray(uint=item, length=self.bits))
        return {
            'data' : bits_flat.tobytes(),
            'size' : size
        }

class KmeansTransformer(Transformer):
    needs_two_step = True
    def __init__(self, k):
        self.k = k
        self.data_observations = []
        self.rme = 0

    def __call__(self, flat_array, size):
        codes, errs = vq(np.array(flat_array), self.codebook)
        self.rme += np.sum(np.square(errs))
        return codes.tolist()

    def consume(self, layer_params):
        for key in layer_params:
            self.data_observations.append(np.array(layer_params[key]))

    def done(self):
        elements = tuple(map(lambda f: f.flatten(), self.data_observations))
        all_elems = np.concatenate(elements)
        logging.info('Size of all elems %d', all_elems.shape[0])
        self.codebook, dist = kmeans(all_elems, self.k)
        logging.info('Kmeans done with distortion %f', dist)

    def serialize_codebook(self):
        logging.info('Kmeans RME is %f', self.rme**0.5)
        return self.codebook.tolist()

class SVDTransformer(Transformer):
    def __init__(self, num, reshape):
        self.error_ratio = num / 100
        self.reshaper = reshape != -1
        self.reshape_ratio = reshape

    def find_cum_percent(self, data):
        sum_percent = np.sum(data)
        if sum_percent == 0:
            return -1
        b = np.cumsum(data) / sum_percent
        for i in range(b.shape[0]):
            if b[i] > self.error_ratio:
                return i
        return -1

    def do_reshape(self, orig):
        temp = self.reshaper
        self.reshaper = False
        answer = []
        hsplits = np.array_split(orig, self.reshape_ratio, axis=0)
        for h in hsplits:
            vsplits = np.array_split(h, self.reshape_ratio, axis=1)
            for v in vsplits:
                answer.append(self(v.flatten(), v.shape))
        self.reshaper = temp
        return answer

    def __call__(self, flat_array, size):
        orig = np.array(flat_array).reshape(size)
        if len(size) == 1:
            return orig.tolist()
        assert len(size) == 2
        if size[0] != size[1]:
            return orig.tolist()
        if self.reshaper:
            return self.do_reshape(orig)
        A = np.matrix(orig)
        U, sigma, V = np.linalg.svd(A)
        i = self.find_cum_percent(sigma)
        if i == -1:
            return orig.tolist()
        for d in size:
            if i > (size[0] / 2):
                return orig.tolist()
        Usmall = np.matrix(U[:, :i])
        Vsmall = np.matrix(np.diag(sigma[:i]) * np.matrix(V[:i, :]))
        outputMat = Usmall * Vsmall
        # np.testing.assert_allclose(outputMat, A, atol=1 - self.error_ratio)
        logging.info('Using svd decomposition of %d for original size %d %d',
                     i, orig.shape[0], orig.shape[1])
        return {
            'U' : Usmall.tolist(),
            'V': Vsmall.tolist()
        }

def transform(weights, fn):
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
        answer[key] = fn(flat, size)
    return answer

def fn_from_args(args):
    if args.transform_to_binary != -1:
        return BinaryTransformer(args.transform_to_binary)
    if args.svd != -1:
        return SVDTransformer(args.svd, args.reshape_svd)
    if args.kmeans != -1:
        return KmeansTransformer(args.kmeans)
    return None

def main(args):
    logging.basicConfig(level=logging.INFO)
    model = json.load(args.ifile)
    fn = fn_from_args(args)
    if fn:
        weight_layers = [layer for layer in model
                         if layer['layerName'] in weight_first_list]
        if fn.needs_two_step:
            for weights in [layer['parameters'][0] for layer in weight_layers]:
                fn.consume(weights)
            fn.done()
        for i, layer in enumerate(weight_layers):
            layer['parameters'][0] = transform(layer['parameters'][0], fn)
        if fn.needs_two_step:
            model = {
                'codebook' : fn.serialize_codebook(),
                'model' : model
            }
    if args.ubjson_format:
        args.ofile.write(simpleubjson.encode(model))
    elif args.json:
        args.ofile.write(json.dumps(model).encode('utf8'))
    else:
        msgpack.dump(model, args.ofile, use_bin_type=True,
                     use_single_float=args.single_precision_float)
    args.ofile.close()
    args.ifile.close()

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
    parser.add_argument('-j', '--json', action='store_true')
    parser.add_argument('-v', '--svd', help='SVD decomposition',
                        type=int, default=-1)
    parser.add_argument('-r', '--reshape-svd', type=int, default=-1)
    parser.add_argument('-k', '--kmeans', type=int, default=-1)
    main(parser.parse_args())
