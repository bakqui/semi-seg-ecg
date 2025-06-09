from typing import Tuple, Dict

import torchmetrics
import torchmetrics.segmentation


def build_metric_fn(config: dict) -> Tuple[torchmetrics.Metric, Dict[str, float]]:
    common_metric_fn_kwargs = {
        "compute_on_cpu": config.get("compute_on_cpu", True),
        "sync_on_compute": config.get("sync_on_compute", False),
    }
    lib = torchmetrics.segmentation
    if config["task"] == "segmentation":
        common_metric_fn_kwargs["num_classes"] = config["num_classes"]
        common_metric_fn_kwargs["include_background"] = config.get("include_background", True)
        common_metric_fn_kwargs["per_class"] = config.get("per_class", False)
        common_metric_fn_kwargs["input_format"] = config.get("input_format", "one-hot")
    else:
        raise ValueError(f"Invalid task: {config['task']}")

    metric_list = []
    for metric_class_name in config["target_metrics"]:
        if isinstance(metric_class_name, dict):
            # e.g., {"AUROC": {"average": macro}}
            assert len(metric_class_name) == 1, \
                f"Invalid metric name: {metric_class_name}"
            metric_class_name, metric_fn_kwargs = list(metric_class_name.items())[0]
            metric_fn_kwargs.update(common_metric_fn_kwargs)
        else:
            metric_fn_kwargs = common_metric_fn_kwargs
        assert isinstance(metric_class_name, str), \
            f"metric name must be a string: {metric_class_name}"
        assert hasattr(lib, metric_class_name), \
            f"Invalid metric name: {metric_class_name}"
        metric_class = getattr(lib, metric_class_name)
        metric_fn = metric_class(**metric_fn_kwargs)
        metric_list.append(metric_fn)
    metric_fn = torchmetrics.MetricCollection(metric_list)

    best_metrics = {
        k: -float("inf") if v.higher_is_better else float("inf")
        for k, v in metric_fn.items()
    }

    return metric_fn, best_metrics


def is_best_metric(
    metric_class: torchmetrics.Metric,
    prev_metric: float,
    curr_metric: float,
) -> bool:
    # check "higher_is_better" attribute of the metric class
    higher_is_better = metric_class.higher_is_better
    if higher_is_better:
        return curr_metric > prev_metric
    else:
        return curr_metric < prev_metric
