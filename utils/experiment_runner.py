#!/usr/bin/env python

import sys
import argparse
import subprocess
import os
import logging
import collections
import json

CODE_SUCCESS = 'success'
CODE_RET_ERROR = 'return code error'
CODE_NOT_STARTED = 'not started'

FNAME_CONFIG = 'config.json'
FNAME_STDOUT = 'stdout.txt'
FNAME_STDERR = 'stderr.txt'

def run_experiment(logger, exp, base_config):
    name, config, cmd_template = exp['name'], exp['config'], exp['command']
    if os.path.exists(name):
        logger.error(('Path exists %s. Cowardly refusing to overwrite. '
                       'Skipping test.'), name)
        return CODE_NOT_STARTED
    os.mkdir(name)
    config_file_path = os.path.join(name, FNAME_CONFIG)
    real_config = base_config.copy()
    real_config.update(config)
    with open(config_file_path, 'w') as exp_config_file:
        json.dump(real_config, exp_config_file)
    command = cmd_template.format(config = FNAME_CONFIG)
    logger.info('Starting %s: %s', name, command)
    stdout_file = open(os.path.join(name, FNAME_STDOUT), 'w')
    stderr_file = open(os.path.join(name, FNAME_STDERR), 'w')
    try:
        ret = subprocess.call(command, cwd = name, stdout = stdout_file,
                              stderr = stderr_file, shell = True)
    finally:
        stdout_file.close()
        stderr_file.close()
    if ret != 0:
        logger.error('Experiment %s returned error code %s!', name, ret)
        return CODE_RET_ERROR
    return CODE_SUCCESS

def analytics(logger, ret_codes):
    logger.info('Experiment report')
    logger.info('-----------------')
    for c in ret_codes:
        logger.info('%s: %s', c[0], c[1])
    counter = collections.Counter(ret_codes)
    logger.info('Successes: %s', counter[CODE_SUCCESS])
    logger.info('Errors: %s', counter[CODE_RET_ERROR])
    logger.info('Not started: %s', counter[CODE_NOT_STARTED])

def main(args):
    logger = logging.getLogger('exp')
    logger.setLevel(logging.INFO)
    with open(args['experiment_config'], 'r') as config_file:
        config = json.load(config_file)
    experiment_list = config['experiments']
    base_config = config['base_config'] if 'base_config' in config else {}
    codes = []
    for experiment in experiment_list:
        codes.append((experiment['name'],
                      run_experiment(logger, experiment, base_config)))
    analytics(logger, codes)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('experiment_config')
    main(vars(parser.parse_args()))
