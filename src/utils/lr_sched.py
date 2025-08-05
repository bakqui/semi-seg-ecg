# Copyright (c) VUNO Inc. All rights reserved.

import math


def adjust_learning_rate(optimizer, epoch, config):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < config['warmup_epochs']:
        lr = config['lr'] * epoch / config['warmup_epochs']
    else:
        lr = config['min_lr'] + (config['lr'] - config['min_lr']) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - config['warmup_epochs']) / (config['epochs'] - config['warmup_epochs'])))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def linear_ramp(epoch, total_epochs, sleep_epochs=0, initial=0.0, final=1.0):
    if epoch < sleep_epochs:
        return initial
    else:
        return min(final, final * (epoch - sleep_epochs) / (total_epochs - sleep_epochs))


def power_decay(epoch, total_epochs, power=2.5, initial=1.0, final=0.0):
    decay = (1 - epoch / total_epochs) ** power
    return final + (initial - final) * decay
