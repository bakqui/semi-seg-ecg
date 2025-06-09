import datetime
import json
import math
import os
import sys
import time
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import utils.lr_sched as lr_sched
import utils.misc as misc
from algorithms.base import evaluate, init_model_from_cfg, test
from algorithms.base import train_one_epoch as train_one_epoch_labeled
from utils.lr_decay import param_groups_lrd
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from utils.optimizer import get_optimizer_from_config
from utils.perf_metrics import build_metric_fn, is_best_metric
from utils.semi_dataset import build_seg_dataset, get_dataloader


def calculate_miou(onehot_preds, onehot_labels, ignore_background=False):
    if ignore_background:
        onehot_preds = onehot_preds[:, 1:]
        onehot_labels = onehot_labels[:, 1:]
    ious = []
    for i in range(onehot_preds.shape[1]):
        intersection = (onehot_preds[:, i] * onehot_labels[:, i]).sum()
        union = onehot_preds[:, i].sum() + onehot_labels[:, i].sum() - intersection
        iou = intersection / union if union > 0 else 0.0
        ious.append(iou)
    return np.mean(ious)


@torch.no_grad()
def select_reliable(models, dataloader, device):
    for model in models:
        model.eval()

    id_to_reliability = []
    for i, data in tqdm(
        enumerate(dataloader),
        total=len(dataloader),
    ):
        ecg = data['ecg'].to(device, non_blocking=True)
        assert ecg.shape[0] == 1, \
            "Batch size should be 1 for reliability estimation"

        onehot_preds = []
        for model in models:
            logits = model(ecg, return_loss=False)['seg_logits']
            pred = torch.argmax(logits, dim=1)

            onehot_preds.append(
                F.one_hot(
                    pred,
                    num_classes=logits.shape[1],
                ).movedim(-1, 1).cpu().numpy()
            )

        mious = []
        for i in range(len(onehot_preds) - 1):
            mious.append(
                calculate_miou(
                    onehot_preds[i],
                    onehot_preds[-1],
                )
            )

        reliability = sum(mious) / len(mious)
        id_to_reliability.append((i, reliability))

    id_to_reliability.sort(key=lambda elem: elem[1], reverse=True)

    reliable_indices = [elem[0] for elem in id_to_reliability[:len(id_to_reliability) // 2]]
    unreliable_indices = [elem[0] for elem in id_to_reliability[len(id_to_reliability) // 2:]]

    return reliable_indices, unreliable_indices


def train_one_epoch(
    model_student: torch.nn.Module,
    model_teacher: torch.nn.Module,
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
    """Self-training w/ reliable pseudo-labels.
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

    model_student.train()
    model_teacher.eval()
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

        model_student.train()

        num_lb, num_ulb = ecg_x.size(0), ecg_u_w.size(0)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model_student(
                torch.cat((ecg_x, ecg_u_w)),
                return_loss=False,
            )
            pred_x, pred_u_w = outputs['seg_logits'].split([num_lb, num_ulb])

            # pseudo-label generation
            with torch.no_grad():
                pred_u_w_teacher = model_teacher(ecg_u_w, return_loss=False)['seg_logits']
            mask_u_w = torch.argmax(pred_u_w_teacher, dim=1)

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
            parameters=model_student.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss_total=loss_value)
        metric_logger.update(loss_x=loss_x_value)
        metric_logger.update(loss_u_s=loss_u_s_value)

        max_lr = 0.
        for group in optimizer.param_groups:
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        loss_x_value_reduce = misc.all_reduce_mean(loss_x_value)
        loss_u_s_value_reduce = misc.all_reduce_mean(loss_u_s_value)
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


"""Stage 1: Train the student model with labeled data only."""
def train_sup(config):
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
        output_dir = os.path.join(config['output_dir'], config['exp_name'], "stage1")
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
        train_stats = train_one_epoch_labeled(
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
        if (epoch + 1) in [num_epochs // 3, num_epochs * 2 // 3, num_epochs]:
            misc.save_model(
                config,
                os.path.join(output_dir, f'checkpoint-{epoch + 1}.pth'),
                epoch,
                model_without_ddp,
                optimizer,
                loss_scaler,
                metrics={'loss': curr_loss, **metrics},
            )
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


def prepare_semisup(config):
    device = torch.device(config['device'])

    # ECG dataset
    dataset_unlabeled_eval = build_seg_dataset(config['dataset'], split='train_unlabeled', mode='eval')
    data_loader_unlabeled = DataLoader(
        dataset_unlabeled_eval,
        batch_size=1,
        shuffle=False,
        num_workers=config['dataloader'].get('num_workers', 2),
        pin_memory=config['dataloader'].get('pin_memory', False),
    )

    models = []
    num_epochs = config['train']['epochs']
    for epoch in [num_epochs // 3, num_epochs * 2 // 3, num_epochs]:
        checkpoint = torch.load(
            os.path.join(config['output_dir'], config['exp_name'], "stage1", f'checkpoint-{epoch}.pth'),
            map_location='cpu',
        )
        model = init_model_from_cfg(config)
        model.load_state_dict(checkpoint['model'])
        model.to(device)
        models.append(model)

    # Select reliable pseudo-labels
    reliable_ids, _ = select_reliable(
        models,
        data_loader_unlabeled,
        device,
    )

    return reliable_ids


"""Stage 2/3: Train the student model with unlabeled data"""
def train_semisup(config, stage_id, unlabeled_subset_ids=None):
    misc.init_distributed_mode(config['ddp'], with_time=False)

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
    if unlabeled_subset_ids is not None:
        dataset_train_unlabeled = torch.utils.data.Subset(
            dataset_train_unlabeled,
            unlabeled_subset_ids,
        )

    # NOTE The labeled set is over-sampled to balance with the unlabeled set.
    num_unlabeled = len(dataset_train_unlabeled)
    dataset_train_labeled = build_seg_dataset(
        config['dataset'],
        split='train_labeled',
        num_unlabeled=num_unlabeled,
    )
    dataset_valid = build_seg_dataset(config['dataset'], split='valid')

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
        if stage_id == 2:
            output_dir = os.path.join(config['output_dir'], config['exp_name'], f"stage{stage_id}")
        else:
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

    # teacher model
    model_teacher = init_model_from_cfg(config)
    model_teacher.to(device)
    target_metric = config.get('test', {}).get('target_metric', "MeanIoU")
    teacher_checkpoint_path = os.path.join(
        config['output_dir'],
        config['exp_name'],
        f"stage{stage_id - 1}",
        f'best-{target_metric}.pth',
    )
    teacher_checkpoint = torch.load(
        teacher_checkpoint_path,
        map_location='cpu',
    )
    print(f"Load teacher model from {teacher_checkpoint_path}")
    model_teacher.load_state_dict(teacher_checkpoint['model'])
    for p in model_teacher.parameters():
        p.requires_grad = False

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
            model_teacher = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_teacher)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[config['ddp']['gpu']],
        )
        model_without_ddp = model.module
        model_teacher = torch.nn.parallel.DistributedDataParallel(
            model_teacher,
            device_ids=[config['ddp']['gpu']],
        )

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
            model_teacher,
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


def train(config):
    train_sup(config)
    if misc.is_main_process() and config['ddp']['distributed']:
        torch.distributed.destroy_process_group()
    reliable_ids = prepare_semisup(config)
    train_semisup(
        config,
        stage_id=2,
        unlabeled_subset_ids=reliable_ids,
    )
    if misc.is_main_process() and config['ddp']['distributed']:
        torch.distributed.destroy_process_group()
    train_semisup(
        config,
        stage_id=3,
    )
