import argparse
import logging
import sys


import tensorflow as tf



def make_parser():
    parser = argparse.ArgumentParser(description='Train a character embedding')
    parser.add_argument('-i', '--input-file',
                        help="Input file of passwords, one per line.")
    parser.add_argument('-o', '--output-file', help='Output file. ',
                        type=argparse.FileType('w'))
    parser.add_argument('-c', '--config', help="Configuration file path.")
    parser.add_argument('-t', '--tensorboard-logdir',
                        help='Log directory for tensorboard')
    parser.add_argument('--help-config', action='store_true',
                        help='Show configuration options and exit')
    return parser


def main(args):
    if not args.input_file or not args.output_file:
        sys.stderr.write('--input-file and --output-file are required\n')
        return

    FORMAT = '%(asctime)s %(levelname)s %(message)s'
    logging.basicConfig(level=logging.INFO, format=FORMAT)

    import pass_embedding as pe
    import pass_dataset as pd

    if args.help_config:
        sys.stdout.write(pe.EmbeddingConfig.__init__.__doc__ + "\n")
        return

    logging.info('Called with %s', vars(args))

    if args.config is not None:
        logging.info('Reading configuration from %s', args.config)
        try:
            config = pe.EmbeddingConfig.from_config_file(args.config)

        except pe.ConfigurationException as e:
            logging.fatal('Error while reading configuration: %s', str(e))
            raise

        except ValueError as e:
            logging.fatal('Error while reading configuration: %s', str(e))
            raise

    else:
        config = pe.EmbeddingConfig()

    dataset = pd.PasswordDatasetMakerFromFile([args.input_file]).make()
    with tf.Graph().as_default():
        with tf.Session() as session:
            trainer = pe.EmbeddingTrainer(
              config, tensorboard_logdir=args.tensorboard_logdir)
            trainer.train_and_save(dataset, session, args.output_file)


if __name__ == '__main__':
    main(make_parser().parse_args())
