# Copyright (c) VUNO Inc. All rights reserved.

import math
import os
import pickle as pkl
from typing import Callable, Dict, Iterable, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

import utils.transforms as T
from utils.transforms import get_transforms_from_config
from utils.misc import get_rank, get_world_size


def _get_transform(cfg):
    if cfg is None:
        return None
    transforms = get_transforms_from_config(cfg)
    if len(transforms) == 1:
        return transforms[0]
    elif len(transforms) > 1:
        return T.Compose(transforms)
    else:
        return None


class ECGSemiSegDataset(Dataset):
    """Dataset for semi-supervised segmentation (delineation) tasks with ECG data.

    Args:
        ecg_dir (str): Directory to load ECG waveforms
        label_dir (str): Directory to load ECG segmentation labels
        filenames (list[str]): Waveform filenames in ecg_dir (.pkl)
        label_filenames (list[str]): Segmentation label filenames in label_dir (.pkl)
        fs_list (list[int]): Sampling rates of each waveform
        target_fs (int, optional): Target sampling rate for resampling
        target_length (int, optional): Target signal length for resampling
        filter_fn (callable, optional): Filter function to be applied on each waveform
        crop_fn (callable, optional): Crop function to be applied on each waveform
        aug_fn (callable, optional): (Weak) augmetations to be applied on each waveform
        strong_aug_fn (callable, optional): Strong augmentations to be applied on each waveform
        transform (callable): ToTensor (+ Standardize) (default: T.ToTensor(dtype="float"))
        label_transform (callable, optional): ToTensor
        mode (str): Dataset mode (train_labeled, train_unlabeled, valid, test)
        num_unlabeled (int, optional): Number of unlabeled samples to be used in training

    Notes:
        - The waveform filenames must be in .pkl format.
            Each file is assumed to be single-lead ECG waveform with np.ndarray
            which have shape (T, ) where T is the number of samples.
        - The label filenames must be in .pkl format.
            Each file is assumed to be single-lead ECG segmentation label
            of corresponding waveform with np.ndarray which have shape (T, )
            where T is the number of samples.
        - To resample, target_fs or target_length must be provided.
            In case of target_fs, fs_list (original sampling rates) must be provided.
            For label resampling, the method is set to be a nearest-neighbor interpolation.
    """
    def __init__(
        self,
        ecg_dir: str,
        label_dir: str,
        filenames: Iterable,
        label_filenames: Iterable = None,
        fs_list: Optional[Iterable] = None,
        target_fs: Optional[int] = None,
        target_length: Optional[int] = None,
        filter_fn: Optional[Callable] = None,
        crop_fn: Optional[Callable] = None,
        aug_fn: Optional[Callable] = None,
        strong_aug_fn: Optional[Callable] = None,
        transform: Callable = T.ToTensor(dtype="float"),
        label_transform: Optional[Callable] = None,
        mode: Literal['train_labeled', 'train_unlabeled', 'valid', 'test'] = 'train_labeled',
        num_unlabeled: Optional[int] = None,
    ):
        self.ecg_dir = ecg_dir
        self.label_dir = label_dir
        self.filenames = filenames
        self.label_filenames = label_filenames
        self.fs_list = fs_list
        self.mode = mode
        if mode == 'train_labeled' and num_unlabeled is not None:
            # Upsample the labeled dataset to match the number of unlabeled samples.
            num_labeled = len(self.filenames)
            self.filenames *= math.ceil(num_unlabeled / num_labeled)
            self.filenames = self.filenames[:num_unlabeled]
            self.label_filenames *= math.ceil(num_unlabeled / num_labeled)
            self.label_filenames = self.label_filenames[:num_unlabeled]
            if fs_list is not None:
                self.fs_list *= math.ceil(num_unlabeled / num_labeled)
                self.fs_list = self.fs_list[:num_unlabeled]
        self.check_dataset()

        # NOTE: to resample, target_fs or target_length must be provided.
        # in case of target_fs, fs_list (original sampling rates) must be provided.
        # for label resampling, the method is set to be a nearest-neighbor interpolation.
        if fs_list is not None:
            self.resample = T.Resample(target_fs=target_fs)
            self.label_resample = T.Resample(
                target_fs=target_fs,
                method="interp",
                kind="zero",
            )
        elif target_length is not None:
            self.resample = T.Resample(target_length=target_length)
            self.label_resample = T.Resample(
                target_length=target_length,
                method="interp",
                kind="zero",
            )
        else:
            self.resample = None
            self.label_resample = None

        # NOTE: filter/crop function are separated from transform to apply on both waveform and label.
        self.filter_fn = filter_fn
        self.crop_fn = crop_fn
        self.aug_fn = aug_fn
        self.strong_aug_fn = strong_aug_fn
        self.transform = transform
        self.label_transform = label_transform

    @property
    def with_resample(self) -> bool:
        return self.resample is not None

    @property
    def with_filter(self) -> bool:
        return self.filter_fn is not None

    @property
    def with_crop(self) -> bool:
        return self.crop_fn is not None

    @property
    def with_augmentation(self) -> bool:
        return self.aug_fn is not None

    @property
    def with_strong_augmentation(self) -> bool:
        return self.strong_aug_fn is not None

    @property
    def labeled(self) -> bool:
        return self.mode in ['train_labeled', 'valid', 'test'] and self.label_filenames is not None

    def __len__(self) -> int:
        return len(self.filenames)

    def check_dataset(self):
        fname_not_pkl = [f for f in self.filenames if not f.endswith('.pkl')]
        assert len(fname_not_pkl) == 0, \
            f"Some files are not pkl. (e.g. {fname_not_pkl[0]}...)"
        fpaths = [
            os.path.join(self.ecg_dir, fname) for fname in self.filenames
        ]
        assert all([os.path.exists(fpath) for fpath in fpaths]), \
            f"Some files do not exist. (e.g. {fpaths[0]}...)"
        if self.labeled:
            fpaths = [
                os.path.join(self.label_dir, fname)
                for fname in self.label_filenames
            ]
            assert all([os.path.exists(fpath) for fpath in fpaths]), \
                f"Some files do not exist. (e.g. {fpaths[0]}...)"
            assert len(self.filenames) == len(self.label_filenames), \
                "The number of filenames and label_filenames are different."
        if self.fs_list is not None:
            assert len(self.filenames) == len(self.fs_list), \
                "The number of filenames and fs_list are different."

    def _process_waveforms(
        self,
        ecg: np.ndarray,
        label: Optional[np.ndarray] = None,
        fs: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Resample, crop, and transform.
        if self.with_resample:
            ecg = self.resample(ecg, fs)
            if label is not None:
                label = self.label_resample(label, fs)
        # NOTE: the label is cropped with the same indices as the ECG.
        # This is to prevent misalignment between the ECG and the label.
        if self.with_filter:
            ecg = self.filter_fn(ecg)
        if self.with_crop:
            ecg, label = self.crop_fn(ecg, label)
        if self.with_augmentation:
            if label is not None:
                ecg, label = self.aug_fn(ecg, label)
            else:
                ecg = self.aug_fn(ecg)

        return ecg, label

    def __getitem__(
        self,
        idx: int,
    ) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        # Load ECG data.
        fname = self.filenames[idx]
        fpath = os.path.join(self.ecg_dir, fname)

        # NOTE: only works for pkl files.
        # Each file is assumed to be single-lead ECG waveform with np.ndarray
        # which have shape (T, ) where T is the number of samples.
        with open(fpath, 'rb') as f:
            x = pkl.load(f)
        # make x to be a 2D array (1, T)
        x = x[np.newaxis, :]

        if self.labeled:
            # Load segmentation label.
            label_fname = self.label_filenames[idx]
            label_fpath = os.path.join(self.label_dir, label_fname)

            # NOTE: only works for pkl files.
            # Each file is assumed to be single-lead ECG segmentation label
            # of corresponding waveform with np.ndarray which have shape (T, )
            with open(label_fpath, 'rb') as f:
                y = pkl.load(f)
            # make y to be a 2D array (1, T)
            y = y[np.newaxis, :]
        else:
            y = None

        # Process waveforms.
        fs = self.fs_list[idx] if self.fs_list is not None else None
        x, y = self._process_waveforms(x, y, fs)
        out = {
            "ecg": self.transform(x)
        }
        if self.labeled:
            out["target"] = self.label_transform(y).squeeze()
        if self.with_strong_augmentation:
            x_aug = self.strong_aug_fn(x)
            out["ecg_aug"] = self.transform(x_aug)

        return out


def build_seg_dataset(
    cfg: dict,
    split: Literal['train_labeled', 'train_unlabeled', 'valid', 'test'],
    mode: Optional[Literal['train', 'eval']] = None,
    num_unlabeled: Optional[int] = None,
) -> ECGSemiSegDataset:
    fname_col = cfg.get("filename_col", "waveform")
    fs_col = cfg.get("fs_col", None)
    target_fs = cfg.get("fs", None)
    target_length = cfg.get("signal_length", None)

    index_dir = os.path.realpath(cfg["index_dir"])
    ecg_dir = os.path.realpath(cfg["ecg_dir"])

    if split != 'train_unlabeled':
        label_fname_col = cfg["label_filename_col"]
        label_dir = os.path.realpath(cfg["label_dir"])
    else:
        label_fname_col = None
        label_dir = None

    df_name = cfg.get(f"{split}_csv", None)
    assert df_name is not None, f"{split}_csv is not defined in the config."
    if os.path.splitext(df_name)[1] == ".csv":
        df = pd.read_csv(os.path.join(index_dir, df_name))
    elif os.path.splitext(df_name)[1] == ".pkl":
        df = pd.read_pickle(os.path.join(index_dir, df_name))
    else:
        raise ValueError(f"Invalid extension: {df_name}")
    filenames = df[fname_col].tolist()
    label_filenames = df[label_fname_col].tolist() if split != 'train_unlabeled' else None
    fs_list = df[fs_col].astype(int).tolist() if fs_col is not None else None

    if mode is None:
        mode = split
    if mode.startswith("train"):
        crop_cfg = cfg.get("train_crop", None)
        aug_cfg = cfg.get("augmentations", None)
        augmentations = _get_transform(aug_cfg)
        strong_aug_cfg = cfg.get("strong_augmentations", None)
        strong_augmentations = _get_transform(strong_aug_cfg)
    else:
        crop_cfg = cfg.get("eval_crop", None)
        augmentations = None
        strong_augmentations = None
    filter_cfg = cfg.get("filter", None)
    filter_fn = _get_transform(filter_cfg)
    crop_fn = _get_transform(crop_cfg)
    transforms_cfg = cfg.get("transforms", None)
    if transforms_cfg is None:
        transforms = T.ToTensor(dtype="float")
    else:
        transforms = _get_transform(transforms_cfg)
    if label_fname_col is not None:
        label_transform = T.ToTensor(dtype="long")
    else:
        label_transform = None

    dataset = ECGSemiSegDataset(
        ecg_dir,
        label_dir,
        filenames=filenames,
        label_filenames=label_filenames,
        fs_list=fs_list,
        target_fs=target_fs,
        target_length=target_length,
        filter_fn=filter_fn,
        crop_fn=crop_fn,
        aug_fn=augmentations,
        strong_aug_fn=strong_augmentations,
        transform=transforms,
        label_transform=label_transform,
        mode=split,
        num_unlabeled=num_unlabeled,
    )

    return dataset


def get_dataloader(
    dataset: Dataset,
    is_distributed: bool = False,
    dist_eval: bool = False,
    mode: Literal["train", "eval"] = "train",
    **kwargs,
) -> DataLoader:
    is_train = mode == "train"
    if is_distributed and (is_train or dist_eval):
        num_tasks = get_world_size()
        global_rank = get_rank()
        if not is_train and len(dataset) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                  'This will slightly alter validation results as extra duplicate entries are added to achieve '
                  'equal num of samples per-process.')
        # shuffle=True to reduce monitor bias even if it is for validation.
        # https://github.com/facebookresearch/mae/blob/main/main_finetune.py#L189
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=num_tasks,
            rank=global_rank,
            shuffle=True,
        )
    elif is_train:
        sampler = torch.utils.data.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)

    drop_last = kwargs.pop("drop_last", None)
    if drop_last is None:
        drop_last = is_train
    return DataLoader(
        dataset,
        sampler=sampler,
        drop_last=drop_last,
        **kwargs,
    )
