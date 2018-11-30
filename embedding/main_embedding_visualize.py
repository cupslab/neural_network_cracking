#!/usr/bin/env python3

import argparse
import sys

import pass_embedding as pe
import tensorflow as tf


def plot_with_labels(embedding, alphabet, filename):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    tsne = TSNE(
      perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
    low_dim_embs = tsne.fit_transform(embedding)
    labels = list(alphabet)

    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.savefig(filename)

def make_parser():
    parser = argparse.ArgumentParser(description='Run embedding analogies')
    parser.add_argument('config')
    parser.add_argument('embedding')
    parser.add_argument(
      '--analogy',
      help=('Input analogy separated by spaces. ex: "a A c" '
            'queries for a is to A as c is to _. Multiple queries can be made. '
            'Ex: "ab AB cd"'))
    parser.add_argument('-p', '--plot')
    return parser

def run_analogy(analogy, evaler, cfg):
    analogy_a, analogy_b, analogy_c = analogy.split(' ')
    with tf.Session() as session:
        ans = evaler.run_analogies(
          (analogy_a, analogy_b, analogy_c), session)

    for i, row in enumerate(ans):
        a, b, c = analogy_a[i], analogy_b[i], analogy_c[i]
        row_str = [cfg.alphabet[j] for j in row]
        sys.stdout.write('%s : %s :: %s : (%s)\n' % (a, b, c, row_str))

def main(args):
    cfg = pe.EmbeddingConfig.from_config_file(args.config)
    reader = pe.CharEmbeddingLoader(cfg)
    with open(args.embedding, 'r') as ifile:
        emb = reader.read_from_file(ifile)

    evaler = pe.EmbeddingAnalogy(cfg, emb)
    if args.analogy:
        run_analogy(args.analogy, evaler, cfg)

    if args.plot is not None:
        plot_with_labels(evaler.embedding, cfg.alphabet, args.plot)


if __name__ == '__main__':
    main(make_parser().parse_args())
