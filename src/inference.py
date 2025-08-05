# Copyright (c) VUNO Inc. All rights reserved.

import argparse
import os

import mergedeep
import numpy as np
import torch
import yaml
from tqdm import tqdm

from algorithms.base import init_model_from_cfg
from utils.semi_dataset import build_seg_dataset, get_dataloader


def parse() -> dict:
    parser = argparse.ArgumentParser('ECG segmentation inference')

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


def inference(config):
    output_dir = os.path.join(config['output_dir'], config['exp_name'])
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(config['device'])

    dataset_test = build_seg_dataset(config['dataset'], split="test")
    data_loader_test = get_dataloader(
        dataset_test,
        is_distributed=False,
        mode='test',
        **config['dataloader'],
    )
    model = init_model_from_cfg(config, train=False)
    if config['test'].get('model_path', None):
        checkpoint_path = config['test']['model_path']
    else:
        target_metric = config['test'].get('target_metric', 'loss')
        checkpoint_path = os.path.join(output_dir, f'best-{target_metric}.pth')
    assert os.path.exists(checkpoint_path), \
        f"Checkpoint not found: {checkpoint_path}"
    state_dict = torch.load(checkpoint_path, map_location='cpu')['model']
    # drop the auxiliary head
    for k in list(state_dict.keys()):
        if k.startswith('auxiliary_head'):
            del state_dict[k]
    msg = model.load_state_dict(state_dict)
    print(msg)

    model.to(device)

    model.eval()

    use_amp = config['test'].get('use_amp', False)
    outputs_total = []
    for samples in tqdm(data_loader_test, desc="Inference"):
        inputs = samples['ecg'].to(device, non_blocking=True)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=use_amp):
                results = model(inputs, return_loss=False)
        outputs = torch.softmax(results['seg_logits'], dim=1)
        outputs_total.append(outputs.cpu())
    outputs = torch.cat(outputs_total, dim=0).numpy()

    np.save(
        os.path.join(output_dir, 'test_outputs.npy'),
        outputs,
    )
    print("Done!")


if __name__ == "__main__":
    config = parse()
    inference(config)
