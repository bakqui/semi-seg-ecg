# Copyright (c) VUNO Inc. All rights reserved.

import datetime
import json
import math
import os
import sys
import time
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import yaml
from torch.utils.tensorboard import SummaryWriter

import utils.lr_sched as lr_sched
import utils.misc as misc
from algorithms.base import evaluate, init_model_from_cfg, test
from utils.lr_decay import param_groups_lrd
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from utils.optimizer import get_optimizer_from_config
from utils.perf_metrics import build_metric_fn, is_best_metric
from utils.semi_dataset import build_seg_dataset, get_dataloader


def train_one_epoch(
    model: torch.nn.Module,
    labeled_data_loader: Iterable,
    unlabeled_data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    use_amp=True,
    config: Optional[dict] = None,
):
    """FixMatch training
    """
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

    num_steps = len(unlabeled_data_loader)
    assert len(labeled_data_loader) == num_steps, \
        "The number of labeled and unlabeled data should be the same"

    for data_iter_step, (labeled, unlabeled) in enumerate(
        metric_logger.log_every(
            zip(
                labeled_data_loader,
                unlabeled_data_loader,
            ),
            print_freq,
            header,
            length=len(labeled_data_loader)
        )
    ):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer,
                data_iter_step / num_steps + epoch,
                config,
            )
        # Labeled batch
        ecg_x = labeled['ecg'].to(device, non_blocking=True)
        mask_x = labeled['target'].to(device, non_blocking=True)

        ecg_u_w = unlabeled['ecg'].to(device, non_blocking=True)
        ecg_u_s = unlabeled['ecg_aug'].to(device, non_blocking=True)

        # pseudo-label generation
        with torch.no_grad():
            model.eval()
            pred_u_w = model(ecg_u_w, return_loss=False)['seg_logits']
            conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
            mask_u_w = pred_u_w.argmax(dim=1)

        model.train()

        num_lb, num_ulb = ecg_x.size(0), ecg_u_w.size(0)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(
                torch.cat((ecg_x, ecg_u_s)),
                return_loss=False,
            )
            pred_x, pred_u_s = outputs['seg_logits'].split([num_lb, num_ulb])

            # supervised loss
            loss_x = F.cross_entropy(pred_x, mask_x)
            if 'aux_seg_logits' in outputs:
                pred_aux_list = outputs['aux_seg_logits']
                aux_loss_weights = config.get('aux_loss_weights', [0.4] * len(pred_aux_list))
                for pred_aux, aux_loss_weight in zip(pred_aux_list, aux_loss_weights):
                    pred_x_aux, _ = pred_aux.split([num_lb, num_ulb])
                    loss_x += aux_loss_weight * F.cross_entropy(pred_x_aux, mask_x)

            # consistency regularization loss
            loss_u_s = F.cross_entropy(pred_u_s, mask_u_w, reduction='none')
            loss_u_s = loss_u_s * (conf_u_w >= config['conf_thresh'])
            loss_u_s = loss_u_s.mean()

            loss = (loss_x + loss_u_s) / 2.0

        loss_value = loss.item()
        loss_x_value = loss_x.item()
        loss_u_s_value = loss_u_s.item()
        mask_ratio = (conf_u_w >= config['conf_thresh']).float().mean().item()

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

        metric_logger.update(loss_total=loss_value)
        metric_logger.update(loss_x=loss_x_value)
        metric_logger.update(loss_u_s=loss_u_s_value)
        metric_logger.update(mask_ratio=mask_ratio)

        max_lr = 0.
        for group in optimizer.param_groups:
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        loss_x_value_reduce = misc.all_reduce_mean(loss_x_value)
        loss_u_s_value_reduce = misc.all_reduce_mean(loss_u_s_value)
        mask_ratio_reduce = misc.all_reduce_mean(mask_ratio)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int(
                (epoch + data_iter_step / num_steps) * 1000
            )
            log_writer.add_scalar(
                'loss_total',
                loss_value_reduce,
                epoch_1000x,
            )
            log_writer.add_scalar(
                'loss_x',
                loss_x_value_reduce,
                epoch_1000x,
            )
            log_writer.add_scalar(
                'loss_u_s',
                loss_u_s_value_reduce,
                epoch_1000x,
            )
            log_writer.add_scalar(
                'mask_ratio',
                mask_ratio_reduce,
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
    dataset_train_unlabeled = build_seg_dataset(config['dataset'], split='train_unlabeled')

    num_unlabeled = len(dataset_train_unlabeled)
    dataset_train_labeled = build_seg_dataset(
        config['dataset'],
        split='train_labeled',
        num_unlabeled=num_unlabeled,
    )
    dataset_valid = build_seg_dataset(config['dataset'], split='valid')

    # NOTE The labeled set is over-sampled to balance with the unlabeled set.
    data_loader_train_labeled = get_dataloader(
        dataset_train_labeled,
        is_distributed=config['ddp']['distributed'],
        mode='train',
        **config['dataloader'],
    )
    print(
        f"Labeled: {len(dataset_train_labeled)} samples / {len(data_loader_train_labeled)} batches"
    )
    data_loader_train_unlabeled = get_dataloader(
        dataset_train_unlabeled,
        is_distributed=config['ddp']['distributed'],
        mode='train',
        **config['dataloader'],
    )
    print(
        f"Unlabeled: {len(dataset_train_unlabeled)} samples / {len(data_loader_train_unlabeled)} batches"
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
            data_loader_train_labeled.sampler.set_epoch(epoch)
            data_loader_train_unlabeled.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model,
            data_loader_train_labeled,
            data_loader_train_unlabeled,
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
