# Copyright (c) VUNO Inc. All rights reserved.

import argparse
import os

import mergedeep
import yaml

import algorithms


def parse() -> dict:
    parser = argparse.ArgumentParser('ECG segmentation testing')

    parser.add_argument(
        '-f',
        '--config_path',
        dest='config_path',
        required=True,
        type=str,
        metavar='FILE',
        help='YAML config file path',
    )
    parser.add_argument(
        '-o',
        '--override_config_path',
        dest='override_config_path',
        default=None,
        type=str,
        metavar='FILE',
        help='YAML config file path to override',
    )
    parser.add_argument(
        '--output_dir',
        default="",
        type=str,
        metavar='DIR',
        help='path where to save',
    )
    parser.add_argument(
        '--exp_name',
        default="",
        type=str,
        help='experiment name',
    )
    parser.add_argument(
        '--model_path',
        default="",
        type=str,
        metavar='PATH',
        help='saved from checkpoint',
    )

    args = parser.parse_args()
    with open(os.path.realpath(args.config_path), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    if args.override_config_path:
        with open(os.path.realpath(args.override_config_path), 'r') as f:
            override_config = yaml.load(f, Loader=yaml.FullLoader)
        config = mergedeep.merge(config, override_config)

    for k, v in vars(args).items():
        if v:
            if k == 'model_path':
                config['test']['model_path'] = v
            else:
                config[k] = v

    return config


if __name__ == "__main__":
    config = parse()
    algo_name = config.get('algorithm')
    if algo_name in algorithms.__dict__:
        algo = algorithms.__dict__[algo_name]
    else:
        raise ValueError(f"Invalid algorithm: {algo_name}")
    algo.test(config)
