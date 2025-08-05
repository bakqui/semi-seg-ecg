# Copyright (c) VUNO Inc. All rights reserved.

def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    if hasattr(model, "backbone"):
        num_layers = model.backbone.depth + 1
    else:
        num_layers = model.depth + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    return list(param_groups.values())


def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name in [
        'cls_embedding',
        'pos_embedding',
        'sep_embedding',
        'lead_embeddings',
        'backbone.cls_embedding',
        'backbone.pos_embedding',
        'backbone.sep_embedding',
        'backbone.lead_embeddings',
    ]:
        return 0
    elif (
        name.startswith('to_patch_embedding')
        or name.startswith('backbone.to_patch_embedding')
    ):
        return 0
    elif name.startswith('block'):
        # block0.~~, block1.~~, ...
        return int(name.split('.')[0][5:]) + 1
    elif name.startswith('backbone.block'):
        # backbone.block0.~~, backbone.block1.~~, ...
        return int(name.split('.')[1][5:]) + 1
    else:
        return num_layers
