import sys
import argparse
import json

import pwd_guess

def main(args):
    with open(args.config, 'r') as config_args:
        c = json.load(config_args)
        config = pwd_guess.ModelDefaults(c['config'])
    ctable = pwd_guess.CharacterTable.fromConfig(config)
    with open(args.ofile, 'w') as ofile:
        json.dump({
            'char_bag': config.char_bag,
            'uppercase': config.uppercase_character_optimization,
            'rare_character_optimization': config.rare_character_optimization,
            'context_length': config.context_length,
            'rare_char_bag': config.get_intermediate_info('rare_character_bag'),
            'char_bag_real': ctable.chars,
            'character_frequencies': config.get_intermediate_info(
                'character_frequencies'),
            'beginning_character_frequencies': config.get_intermediate_info(
                'beginning_character_frequencies'),
            'end_character_frequencies': config.get_intermediate_info(
                'end_character_frequencies'),
            'train_backwards': config.train_backwards,
            'min_len': config.min_len
        }, ofile)

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='Extract information about the model for JS')
    parser.add_argument('config')
    parser.add_argument('ofile')
    main(parser.parse_args())
