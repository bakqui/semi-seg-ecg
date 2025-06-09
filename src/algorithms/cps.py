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
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from utils.optimizer import get_optimizer_from_config
from utils.perf_metrics import build_metric_fn, is_best_metric
from utils.semi_dataset import build_seg_dataset, get_dataloader


def train_one_epoch(
    model_1: torch.nn.Module,
    model_2: torch.nn.Module,
    labeled_data_loader: Iterable,
    unlabeled_data_loader: Iterable,
    optimizer_1: torch.optim.Optimizer,
    optimizer_2: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    use_amp=True,
    config: Optional[dict] = None,
):
    """Cross Pseudo Supervision (CPS) training
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

    model_1.train()
    model_2.train()
    optimizer_1.zero_grad()
    optimizer_2.zero_grad()

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
                optimizer_1,
                data_iter_step / num_steps + epoch,
                config,
            )
            lr_sched.adjust_learning_rate(
                optimizer_2,
                data_iter_step / num_steps + epoch,
                config,
            )
        # Labeled batch
        ecg_x = labeled['ecg'].to(device, non_blocking=True)
        mask_x = labeled['target'].to(device, non_blocking=True)

        ecg_u_w = unlabeled['ecg'].to(device, non_blocking=True)

        num_lb, num_ulb = ecg_x.size(0), ecg_u_w.size(0)

        # pseudo-label generation
        model_1.eval()
        model_2.eval()
        with torch.no_grad():
            pred_u_w_1 = model_1(ecg_u_w, return_loss=False)['seg_logits']
            mask_u_w_1 = pred_u_w_1.argmax(dim=1)
            pred_u_w_2 = model_2(ecg_u_w, return_loss=False)['seg_logits']
            mask_u_w_2 = pred_u_w_2.argmax(dim=1)

        mean_loss_value = 0.
        mean_loss_x_value = 0.
        mean_loss_u_s_value = 0.

        for model, optimizer, mask_u_w in zip(
            [model_1, model_2],
            [optimizer_1, optimizer_2],
            [mask_u_w_2, mask_u_w_1],
        ):
            model.train()

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(
                    torch.cat((ecg_x, ecg_u_w)),
                    return_loss=False,
                )
                pred_x, pred_u_w = outputs['seg_logits'].split([num_lb, num_ulb])

                # supervised loss
                loss_x = F.cross_entropy(pred_x, mask_x)
                if 'aux_seg_logits' in outputs:
                    pred_aux_list = outputs['aux_seg_logits']
                    aux_loss_weights = config.get('aux_loss_weights', [0.4] * len(pred_aux_list))
                    for pred_aux, aux_loss_weight in zip(pred_aux_list, aux_loss_weights):
                        pred_x_aux, _ = pred_aux.split([num_lb, num_ulb])
                        loss_x += aux_loss_weight * F.cross_entropy(pred_x_aux, mask_x)

                # consistency regularization loss
                loss_u_s = F.cross_entropy(pred_u_w, mask_u_w)

                loss = (loss_x + loss_u_s) / 2.0

            loss_value = loss.item()
            loss_x_value = loss_x.item()
            loss_u_s_value = loss_u_s.item()

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

            mean_loss_value += loss_value
            mean_loss_x_value += loss_x_value
            mean_loss_u_s_value += loss_u_s_value

        torch.cuda.synchronize()

        mean_loss_value /= 2
        mean_loss_x_value /= 2
        mean_loss_u_s_value /= 2

        metric_logger.update(loss_total=mean_loss_value)
        metric_logger.update(loss_x=mean_loss_x_value)
        metric_logger.update(loss_u_s=mean_loss_u_s_value)

        max_lr = 0.
        for group in optimizer.param_groups:
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(mean_loss_value)
        loss_x_value_reduce = misc.all_reduce_mean(mean_loss_x_value)
        loss_u_s_value_reduce = misc.all_reduce_mean(mean_loss_u_s_value)
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
    model_1 = init_model_from_cfg(config)
    model_2 = init_model_from_cfg(config)

    model_1.to(device)
    model_2.to(device)

    model_without_ddp_1 = model_1
    model_without_ddp_2 = model_2
    print(f"Model = {model_without_ddp_1}")

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
            model_1 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_1)
            model_2 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_2)
        model_1 = torch.nn.parallel.DistributedDataParallel(
            model_1,
            device_ids=[config['ddp']['gpu']],
        )
        model_2 = torch.nn.parallel.DistributedDataParallel(
            model_2,
            device_ids=[config['ddp']['gpu']],
        )
        model_without_ddp_1 = model_1.module
        model_without_ddp_2 = model_2.module

    optimizer_1 = get_optimizer_from_config(
        config['train'],
        model_without_ddp_1.parameters(),
    )
    optimizer_2 = get_optimizer_from_config(
        config['train'],
        model_without_ddp_2.parameters(),
    )
    print(f"Optimizer = {optimizer_1}")
    loss_scaler = NativeScaler()

    best_loss = float('inf')
    metric_fn, best_metrics = build_metric_fn(config['metric'])
    metric_fn.to(device)

    num_epochs = config['train']['epochs']
    print(f"Start training for {num_epochs} epochs")
    use_amp = config.get('use_amp', True)
    start_time = time.time()
    for epoch in range(config['start_epoch'], num_epochs):
        if config['ddp']['distributed']:
            data_loader_train_labeled.sampler.set_epoch(epoch)
            data_loader_train_unlabeled.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model_1,
            model_2,
            data_loader_train_labeled,
            data_loader_train_unlabeled,
            optimizer_1,
            optimizer_2,
            device,
            epoch,
            loss_scaler,
            log_writer,
            use_amp=use_amp,
            config=config['train'],
        )
        valid_stats, metrics, _, _ = evaluate(
            model_1,
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
                model_without_ddp_1,
                optimizer_1,
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
                    model_without_ddp_1,
                    optimizer_1,
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
