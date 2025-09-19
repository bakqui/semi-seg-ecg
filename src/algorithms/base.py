# Copyright (c) VUNO Inc. All rights reserved.

import datetime
import json
import math
import os
import sys
import time
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import yaml
from torch.utils.tensorboard import SummaryWriter

import models.backbones as backbones
import models.decode_heads as decode_heads
import utils.lr_sched as lr_sched
import utils.misc as misc
from models.encoder_decoder import EncoderDecoder
from utils.lr_decay import param_groups_lrd
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from utils.optimizer import get_optimizer_from_config
from utils.perf_metrics import build_metric_fn, is_best_metric
from utils.semi_dataset import build_seg_dataset, get_dataloader


def init_model_from_cfg(config, train=True):
    # Init backbone
    backbone_name, backbone_kwargs = list(config['backbone'].items())[0]
    assert backbone_name in backbones.__dict__, \
        f"Unsupported model name: {backbone_name}"
    backbone = backbones.__dict__[backbone_name](**backbone_kwargs)

    # Init decode head
    decoder_name, decoder_kwargs = list(config['decode_head'].items())[0]
    assert decoder_name in decode_heads.__dict__, \
        f"Unsupported decode head name: {decoder_name}"
    decoder = decode_heads.__dict__[decoder_name](**decoder_kwargs)
    decoder_loss = nn.CrossEntropyLoss()

    # Init optional auxiliary heads
    if config.get('auxiliary_heads', None) and train:
        auxiliary_heads = nn.ModuleList()
        auxiliary_head_losses = nn.ModuleList()
        for aux_head_cfg in config['auxiliary_heads']:
            auxiliary_head_name, auxiliary_head_kwargs = list(aux_head_cfg.items())[0]
            assert auxiliary_head_name in decode_heads.__dict__, \
                f"Unsupported auxiliary head name: {auxiliary_head_name}"
            auxiliary_head = decode_heads.__dict__[auxiliary_head_name](
                **auxiliary_head_kwargs
            )
            auxiliary_heads.append(auxiliary_head)
            auxiliary_head_loss = nn.CrossEntropyLoss()
            auxiliary_head_losses.append(auxiliary_head_loss)
    else:
        auxiliary_heads = None
        auxiliary_head_losses = None

    # Optional latent projection layer (ReCo)
    use_latent_projection = config.get('use_latent_projection', False)
    projection_in_dim = config.get('projection_in_dim', None)
    projection_out_dim = config.get('projection_out_dim', None)

    # Wrap the model
    model = EncoderDecoder(
        backbone=backbone,
        decode_head=decoder,
        decode_head_loss=decoder_loss,
        auxiliary_heads=auxiliary_heads,
        auxiliary_head_losses=auxiliary_head_losses,
        use_latent_projection=use_latent_projection,
        projection_in_dim=projection_in_dim,
        projection_out_dim=projection_out_dim,
    )
    return model


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    use_amp=True,
    config: Optional[dict] = None,
):
    print_freq = 20
    accum_iter = config.get('accum_iter', 1)
    max_norm = config.get('max_norm', None)

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        'lr',
        misc.SmoothedValue(window_size=1, fmt='{value:.6f}'),
    )
    header = 'Epoch: [{}]'.format(epoch)
    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    model.train()
    optimizer.zero_grad()

    for data_iter_step, samples in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer,
                data_iter_step / len(data_loader) + epoch,
                config,
            )
        inputs = samples['ecg'].to(device, non_blocking=True)
        labels = samples['target'].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            results = model(inputs, labels, return_loss=True)
        loss = results.get('loss')
        if 'loss_aux' in results:
            aux_loss_weights: List[float] = config.get(
                'auxiliary_loss_weights', [0.4]
            )
            aux_losses: List[torch.Tensor] = results['loss_aux']
            loss += sum(
                w * l for w, l in zip(aux_loss_weights, aux_losses)
            )
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        loss = loss / accum_iter
        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        max_lr = 0.
        for group in optimizer.param_groups:
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int(
                (epoch + data_iter_step / len(data_loader)) * 1000
            )
            log_writer.add_scalar(
                'loss',
                loss_value_reduce,
                epoch_1000x,
            )
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('Averaged stats:', metric_logger)
    train_stat = {
        k: meter.global_avg for k, meter in metric_logger.meters.items()
    }

    return train_stat


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
    metric_fn,
    use_amp=True,
):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Eval:'

    outputs_total = []
    labels_total = []
    for samples in metric_logger.log_every(data_loader, 10, header):
        inputs = samples['ecg'].to(device, non_blocking=True)
        labels = samples['target'].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            results = model(inputs, labels, return_loss=True)
        loss = results.get('loss').item()
        logits = results['seg_logits']
        outputs = torch.softmax(logits, dim=1)

        outputs = misc.concat_all_gather(outputs)
        preds = F.one_hot(
            outputs.argmax(dim=1),
            num_classes=outputs.size(1),
        ).movedim(1, -1)
        labels = misc.concat_all_gather(labels)
        labels = F.one_hot(
            labels,
            num_classes=outputs.size(1),
        ).movedim(1, -1)
        metric_fn.update(preds, labels)
        metric_logger.meters['loss'].update(loss, n=inputs.size(0))
        outputs_total.append(outputs.cpu())
        labels_total.append(labels.cpu())

    metric_logger.synchronize_between_processes()
    valid_stats = {
        k: meter.global_avg for k, meter in metric_logger.meters.items()
    }
    metrics = metric_fn.compute()
    if not isinstance(metrics, dict):
        metrics = {metric_fn.__class__.__name__: metrics}
    metric_dict = dict()
    for k, v in metrics.items():
        v = v.tolist()
        if isinstance(v, list):
            for i, vi in enumerate(v):
                metric_dict[f"{k}_{i}"] = vi
        else:
            metric_dict[k] = v

    metric_str = "  ".join([f"{k}: {v:.3f}" for k, v in metric_dict.items()])
    metric_str = f"{metric_str}  loss: {metric_logger.loss.global_avg:.3f}"
    print(f"* {metric_str}")
    outputs = torch.cat(outputs_total, dim=0)
    labels = torch.cat(labels_total, dim=0)
    metric_fn.reset()
    return valid_stats, metric_dict, outputs, labels


def train(config):
    misc.init_distributed_mode(config['ddp'])

    print(f'job dir: {os.path.dirname(os.path.realpath(__file__))}')
    print(yaml.dump(config, default_flow_style=False, sort_keys=False))

    device = torch.device(config['device'])

    # fix the seed for reproducibility
    seed = config['seed'] + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # ECG dataset
    dataset_train = build_seg_dataset(config['dataset'], split='train_labeled')
    dataset_valid = build_seg_dataset(config['dataset'], split='valid')

    data_loader_train = get_dataloader(
        dataset_train,
        is_distributed=config['ddp']['distributed'],
        mode='train',
        **config['dataloader'],
    )
    data_loader_valid = get_dataloader(
        dataset_valid,
        is_distributed=config['ddp']['distributed'],
        mode='valid',
        **config['dataloader'],
    )

    if misc.is_main_process() and config['output_dir']:
        output_dir = os.path.join(config['output_dir'], config['exp_name'])
        os.makedirs(output_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=output_dir)
    else:
        output_dir = None
        log_writer = None

    # define the model
    model = init_model_from_cfg(config)
    if config['mode'] != "scratch":
        checkpoint = torch.load(
            config['pretrained_backbone'],
            map_location='cpu',
        )
        print(f"Load backbone from {config['pretrained_backbone']}")
        state_dict = checkpoint['model']
        msg = model.backbone.load_state_dict(state_dict, strict=False)
        print(msg)
        assert set(msg.missing_keys).issubset(
            {'mask_embedding', 'head.weight', 'head.bias'}
        )
        if config['mode'] == "freeze_backbone":
            for _, p in model.backbone.named_parameters():
                p.requires_grad = False
    model.to(device)

    model_without_ddp = model
    print(f"Model = {model_without_ddp}")

    eff_batch_size = config['dataloader']['batch_size']
    eff_batch_size *= config['train']['accum_iter']
    eff_batch_size *= misc.get_world_size()

    if config['train']['lr'] is None:
        config['train']['lr'] = config['train']['blr'] * eff_batch_size / 256

    print(f"base lr: {config['train']['lr'] * 256 / eff_batch_size}")
    print(f"actual lr: {config['train']['lr']}")
    print(f"accumulate grad iterations: {config['train']['accum_iter']}")
    print(f"effective batch size: {eff_batch_size}")

    if config['ddp']['distributed']:
        # SyncBatchNorm
        if config['ddp'].get('sync_bn', True):
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[config['ddp']['gpu']],
        )
        model_without_ddp = model.module

    layer_decay = config['train'].get('layer_decay', None)
    if layer_decay:
        param_groups = param_groups_lrd(
            model_without_ddp,
            weight_decay=config['train']['weight_decay'],
            no_weight_decay_list=model_without_ddp.no_weight_decay(),
            layer_decay=layer_decay,
        )
    else:
        param_groups = model_without_ddp.parameters()
    optimizer = get_optimizer_from_config(config['train'], param_groups)
    print(f"Optimizer = {optimizer}")
    loss_scaler = NativeScaler()

    best_loss = float('inf')
    metric_fn, best_metrics = build_metric_fn(config['metric'])
    metric_fn.to(device)

    misc.load_model(config, model_without_ddp, optimizer, loss_scaler)

    num_epochs = config['train']['epochs']
    print(f"Start training for {num_epochs} epochs")
    use_amp = config.get('use_amp', True)
    start_time = time.time()
    for epoch in range(config['start_epoch'], num_epochs):
        if config['ddp']['distributed']:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            log_writer,
            use_amp=use_amp,
            config=config['train'],
        )
        valid_stats, metrics, _, _ = evaluate(
            model,
            data_loader_valid,
            device,
            metric_fn,
            use_amp=use_amp,
        )
        curr_loss = valid_stats['loss']
        if output_dir and curr_loss < best_loss:
            best_loss = curr_loss
            misc.save_model(
                config,
                os.path.join(output_dir, 'best-loss.pth'),
                epoch,
                model_without_ddp,
                optimizer,
                loss_scaler,
                metrics={'loss': curr_loss, **metrics},
            )
        for metric_name, metric_class in metric_fn.items():
            curr_metric = metrics[metric_name]
            print(f"{metric_name}: {curr_metric:.3f}")
            if output_dir and is_best_metric(
                metric_class,
                best_metrics[metric_name],
                curr_metric,
            ):
                best_metrics[metric_name] = curr_metric
                misc.save_model(
                    config,
                    os.path.join(output_dir, f'best-{metric_name}.pth'),
                    epoch,
                    model_without_ddp,
                    optimizer,
                    loss_scaler,
                    metrics={'loss': valid_stats['loss'], **metrics},
                )
            print(f"Best {metric_name}: {best_metrics[metric_name]:.3f}")

        if log_writer is not None:
            log_writer.add_scalar('perf/valid_loss', curr_loss, epoch)
            for metric_name, curr_metric in metrics.items():
                log_writer.add_scalar(
                    f'perf/{metric_name}',
                    curr_metric,
                    epoch,
                )

        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'valid_{k}': v for k, v in valid_stats.items()},
            **metrics,
            'epoch': epoch,
        }

        if output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(
                os.path.join(output_dir, 'log.txt'),
                mode='a',
                encoding="utf-8",
            ) as f:
                f.write(json.dumps(log_stats) + '\n')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training time {total_time_str}')

    if log_writer is not None:
        log_writer.close()


def test(config):
    output_dir = os.path.join(config['output_dir'], config['exp_name'])
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(config['device'])

    dataset_test = build_seg_dataset(config['dataset'], split='test')
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

    metric_fn, _ = build_metric_fn(config['metric'])
    metric_fn.to(device)

    test_stats, metrics, outputs, labels = evaluate(
        model,
        data_loader_test,
        device,
        metric_fn,
        use_amp=config.get('use_amp', True),
    )

    metrics['loss'] = test_stats['loss']
    metric_df = pd.DataFrame([metrics])
    metric_df.to_csv(
        os.path.join(output_dir, 'test_metrics.csv'),
        index=False,
        float_format='%.4f',
    )
    np.save(
        os.path.join(output_dir, 'test_outputs.npy'),
        outputs.numpy(),
    )
    np.save(
        os.path.join(output_dir, 'test_labels.npy'),
        labels.numpy(),
    )
    print('Done!')
