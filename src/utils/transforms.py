from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.signal import butter, resample, sosfiltfilt, square


__all__ = [
    'AdaptivePowerlineNoise',
    'AmplitudeScaling',
    'CenterCrop',
    'Compose',
    'Cutout',
    'HighpassFilter',
    'LowpassFilter',
    'MovingWindowCrop',
    'NCrop',
    'RandAugment',
    'RandomApply',
    'RandomBaselineShift',
    'RandomCrop',
    'RandomMask',
    'RandomPartialSineNoise',
    'RandomPartialSquareNoise',
    'RandomPartialWhiteNoise',
    'RandomResizeCrop',
    'RandomShift',
    'Resample',
    'SineNoise',
    'SOSFilter',
    'SquareNoise',
    'Standardize',
    'ToTensor',
    'WhiteNoise',
    'XFlip',
    'YFlip',
    'get_transforms_from_config',
    'get_rand_augment_from_config',
]


"""Preprocessing1
"""
class Resample:
    """Resample the input sequence.
    """
    def __init__(
        self,
        target_length: Optional[int] = None,
        target_fs: Optional[int] = None,
        method: str = 'fourier',
        kind: str = 'nearest',
    ) -> None:
        self.target_length = target_length
        self.target_fs = target_fs
        self.method = method
        self.kind = kind

    def _resample(self, x, target_length, axis):
        if self.method == 'fourier':
            return resample(x, target_length, axis=axis)
        elif self.method == 'interp':
            f = interp1d(
                np.arange(x.shape[axis]),
                x,
                axis=axis,
                kind=self.kind,
                fill_value='extrapolate',
            )
            return f(np.linspace(0, x.shape[axis] - 1, target_length))

    def __call__(self, x: np.ndarray, fs: Optional[int] = None) -> np.ndarray:
        if fs and self.target_fs and fs != self.target_fs:
            x = self._resample(x, int(x.shape[1] * self.target_fs / fs), axis=1)
        elif self.target_length and x.shape[1] != self.target_length:
            x = self._resample(x, self.target_length, axis=1)
        return x

class RandomResizeCrop:
    def __init__(
        self,
        target_length: int,
        scale_min: float = 0.5,
        scale_max: float = 2.0,
    ):
        self.target_length = target_length
        self.scale_min = scale_min
        self.scale_max = scale_max

    def __call__(self, ecg: np.ndarray, label: Optional[np.ndarray] = None):
        sig_len = ecg.shape[1]

        ratio = np.random.uniform(self.scale_min, self.scale_max)
        size = int(sig_len * ratio)

        # resize
        ecg_resized = resample(ecg, size, axis=1)
        if label is not None:
            assert ecg.shape[1] == label.shape[1], \
                f"Length mismatch: ecg: {ecg.shape}, label: {label.shape}"
            f = interp1d(
                np.arange(sig_len),
                label,
                axis=1,
                kind='nearest',
                fill_value='extrapolate',
            )
            label_resized = f(np.linspace(0, sig_len - 1, size))

        pad_len = self.target_length - size
        if pad_len > 0:
            left_pad = pad_len // 2
            right_pad = pad_len - left_pad
            ecg_resized = np.pad(ecg_resized, ((0, 0), (left_pad, right_pad)), mode='constant')
            if label is not None:
                label_resized = np.pad(label_resized, ((0, 0), (left_pad, right_pad)), mode='constant')

        # random crop
        start_idx = np.random.randint(0, ecg_resized.shape[1] - self.target_length + 1)
        ecg_crop = ecg_resized[:, start_idx:start_idx + self.target_length]
        if label is not None:
            label_crop = label_resized[:, start_idx:start_idx + self.target_length]
            return ecg_crop, label_crop
        return ecg_crop

class _BaseCrop:
    """Base class for cropping.
    """
    def __init__(self, crop_length: int) -> None:
        self.crop_length = crop_length

    def _check(
        self,
        x: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        x_ndim = x.ndim
        if x_ndim == 1:
            x = np.expand_dims(x, axis=0)
        elif x_ndim > 2:
            raise ValueError(
                f"Invalid x shape: {x.shape}, must be 1D or 2D."
            )
        seq_len = x.shape[1]
        if self.crop_length > seq_len:
            raise ValueError(
                f"crop_length is larger than the length of x ({seq_len})."
            )
        if y is not None:
            y_ndim = y.ndim
            if y_ndim == 1:
                y = np.expand_dims(y, axis=0)
            elif y_ndim > 2:
                raise ValueError(
                    f"Invalid y shape: {y.shape}, must be 1D or 2D."
                )
            if y.shape[1] != seq_len:
                raise ValueError(
                    f"length mismatch: x: {x.shape}, y: {y.shape}"
                )
            return x, y
        return x, None

    def _crop(
        self,
        start_idx: Union[int, Iterable[int]],
        x: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        if isinstance(start_idx, int):
            start_idx = [start_idx]
        x_crop = np.stack(
            [x[:, i:i + self.crop_length] for i in start_idx], axis=0
        )
        x_crop = np.squeeze(x_crop) if x_crop.shape[0] == 1 else x_crop
        if y is not None:
            y_crop = np.stack(
                [y[:, i:i + self.crop_length] for i in start_idx], axis=0
            )
            y_crop = np.squeeze(y_crop) if y_crop.shape[0] == 1 else y_crop
            return x_crop, y_crop
        return x_crop

    def __call__(
        self,
        x: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError

class RandomCrop(_BaseCrop):
    """Crop randomly the input sequence.
    """
    def __call__(
        self,
        x: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        x, y = self._check(x, y)
        start_idx = np.random.randint(0, x.shape[1] - self.crop_length + 1)
        return self._crop(start_idx, x, y)

class CenterCrop(_BaseCrop):
    """Crop the input sequence at the center.
    """
    def __call__(
        self,
        x: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        x, y = self._check(x, y)
        start_idx = (x.shape[1] - self.crop_length) // 2
        return self._crop(start_idx, x, y)

class MovingWindowCrop(_BaseCrop):
    """Crop the input sequence with a moving window.
    """
    def __init__(self, crop_length: int, crop_stride: int) -> None:
        self.crop_length = crop_length
        self.crop_stride = crop_stride

    def __call__(
        self,
        x: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        x, y = self._check(x, y)
        start_idx = np.arange(
            start=0,
            stop=x.shape[1] - self.crop_length + 1,
            step=self.crop_stride,
        )
        return self._crop(start_idx, x, y)

class NCrop(_BaseCrop):
    """Crop the input sequence to N segments with equally spaced intervals.
    """
    def __init__(self, crop_length: int, num_segments: int) -> None:
        self.crop_length = crop_length
        self.num_segments = num_segments

    def __call__(
        self,
        x: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        x, y = self._check(x, y)
        seq_len = x.shape[1]
        start_idx = np.arange(
            start=0,
            stop=seq_len - self.crop_length + 1,
            step=(seq_len - self.crop_length) // (self.num_segments - 1),
        )
        return self._crop(start_idx, x, y)

class SOSFilter:
    """Apply SOS filter to the input sequence.
    """
    def __init__(
        self,
        fs: int,
        cutoff: float,
        order: int = 5,
        btype: str = 'highpass',
    ) -> None:
        self.sos = butter(order, cutoff, btype=btype, fs=fs, output='sos')

    def __call__(self, x):
        return sosfiltfilt(self.sos, x)

class HighpassFilter(SOSFilter):
    """Apply highpass filter to the input sequence.
    """
    def __init__(self, fs: int, cutoff: float, order: int = 5) -> None:
        super(HighpassFilter, self).__init__(
            fs, cutoff, order, btype='highpass'
        )

class LowpassFilter(SOSFilter):
    """Apply lowpass filter to the input sequence.
    """
    def __init__(self, fs: int, cutoff: float, order: int = 5) -> None:
        super(LowpassFilter, self).__init__(
            fs, cutoff, order, btype='lowpass'
        )

class Standardize:
    """Standardize the input sequence.
    """
    def __init__(
        self,
        axis: Union[int, Tuple[int, ...], List[int]] = (-1, -2),
    ) -> None:
        if isinstance(axis, list):
            axis = tuple(axis)
        self.axis = axis

    def __call__(self, x: np.ndarray) -> np.ndarray:
        loc = np.mean(x, axis=self.axis, keepdims=True)
        scale = np.std(x, axis=self.axis, keepdims=True)
        # Set rst = 0 if std = 0
        return np.divide(
            x - loc,
            scale,
            out=np.zeros_like(x),
            where=scale != 0,
        )


"""Augmentations
"""
class _BaseAugment:
    """Base class for augmentations.
    """
    _label_changeable = False

    def __call__(self, x: np.ndarray, y: Optional[np.ndarray] = None):
        if y is not None:
            # whether label can be changed by augmentations
            if self._label_changeable:
                x, y = self._augment(x, y)
            else:
                x = self._augment(x)
            return x, y
        else:
            return self._augment(x)

    def _augment(self, *args, **kwargs):
        raise NotImplementedError

    def _set_level(self, level: int, max_level: int = 10, **kwargs) -> None:
        pass


"""Group 2: Signal manipulation
"""
class AmplitudeScaling(_BaseAugment):
    """Scale the amplitude of the input sequence.
    """
    def __init__(self, sigma=0.5) -> None:
        self.sigma = sigma

    def _augment(self, x: np.ndarray) -> np.ndarray:
        scales = np.random.normal(1, self.sigma, size=x.shape)
        return x * scales

    def _set_level(self, level: int, max_level: int = 10) -> None:
        self.sigma = level / max_level * 0.5

class XFlip(_BaseAugment):
    _label_changeable = True
    """Flip the signal along the x-axis.
    """
    def _augment(
        self,
        x: np.ndarray,
        y: Optional[np.ndarray] = None,
    ):
        x = np.flip(x, axis=1)
        if y is not None:
            y = np.flip(y, axis=1)
            return x, y
        return x

class YFlip(_BaseAugment):
    """Flip the signal along the y-axis.
    """
    def _augment(self, x: np.ndarray) -> np.ndarray:
        return -x

class _Mask(_BaseAugment):
    """Base class for signal masking.
    """
    def __init__(self, mask_ratio: float = 0.3) -> None:
        self.mask_ratio = mask_ratio

    def _set_level(self, level: int, max_level: int = 10) -> None:
        # self.mask_ratio = level / max_level * 0.3
        pass

class RandomMask(_Mask):
    """Randomly mask the input sequence.
    """
    def _augment(self, x: np.ndarray) -> np.ndarray:
        rst = x.copy()
        count = np.random.randint(0, int(x.shape[-1] * self.mask_ratio))
        indices = np.random.choice(x.shape[-1], (1, count), replace=False)
        rst[:, indices] = 0
        return rst

class Cutout(_Mask):
    """Cutout the input sequence.
    """
    _label_changeable = True

    def _augment(self, x: np.ndarray, y: Optional[np.ndarray] = None):
        rst = x.copy()
        count = int(np.random.uniform(0, self.mask_ratio) * x.shape[-1])
        start_idx = np.random.randint(0, x.shape[-1] - count)
        rst[:, start_idx:start_idx + count] = 0
        if y is not None:
            y[:, start_idx:start_idx + count] = 0
            return rst, y
        return rst

class RandomShift(_Mask):
    """Randomly shift (left or right) the input sequence and pad zeros.
    """
    _label_changeable = True

    def _augment(self, x: np.ndarray, y: Optional[np.ndarray] = None):
        rst = x.copy()
        direction = np.random.choice([-1, 1])
        sig_len = x.shape[-1]
        shift = int(np.random.uniform(0, self.mask_ratio) * sig_len)
        if direction == 1:
            rst[:, shift:] = rst[:, :sig_len - shift]
            rst[:, :shift] = 0
            if y is not None:
                y[:, shift:] = y[:, :sig_len - shift]
                y[:, :shift] = 0
                return rst, y
        else:
            rst[:, :sig_len - shift] = rst[:, shift:]
            rst[:, sig_len - shift:] = 0
            if y is not None:
                y[:, :sig_len - shift] = y[:, shift:]
                y[:, sig_len - shift:] = 0
                return rst, y
        return rst


"""Group 3: Noise manipulation
"""
class _Noise(_BaseAugment):
    """Base class for noise manipulation.
    """
    def __init__(self, amplitude: float = 1.0, freq: float = 0.5) -> None:
        self.amplitude = amplitude
        self.freq = freq

    def _get_noise(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _augment(self, x: np.ndarray) -> np.ndarray:
        noise = self._get_noise(x)
        return x + noise

    def _set_level(self, level: int, max_level: int = 10) -> None:
        level = level / max_level
        self.amplitude = level * 1.0
        self.freq = 0.5 / level

class RandomBaselineShift(_Noise):
    def __init__(
        self,
        ratio: float = 0.5,
        scale: float = 3.0,
    ):
        self.ratio = ratio
        self.scale = scale

    def _get_noise(self, x: np.ndarray) -> np.ndarray:
        background = np.median(x, axis=1, keepdims=True)
        count = int(np.random.uniform(0, self.ratio) * x.shape[1])
        start_idx = np.random.randint(0, x.shape[1] - count)
        shift_scale = np.random.uniform(0, self.scale) * (1 - 2 * np.random.randint(2))
        shifts = np.zeros_like(x)
        shifts[:, start_idx:start_idx + count] = background * shift_scale
        return shifts

    def _set_level(self, level: int, max_level: int = 10) -> None:
        super(RandomBaselineShift, self)._set_level(level, max_level)
        self.ratio = level / max_level * 0.5
        self.scale = level / max_level * 3.0

class AdaptivePowerlineNoise(_Noise):
    """Add powerline noise to the input sequence.
    Noise amplitude -> amplitude of signal * 0.5.
    Noise frequency -> 50Hz or 60Hz.
    """
    def __init__(self, fs: int = 500):
        self.fs = fs

    def _get_amplitude(self, x: np.ndarray) -> np.ndarray:
        robust_max = np.percentile(x, 95, axis=1, keepdims=True)
        robust_min = np.percentile(x, 5, axis=1, keepdims=True)
        robust_range = robust_max - robust_min
        return robust_range / 2

    def _get_noise(self, x: np.ndarray) -> np.ndarray:
        t = np.expand_dims(np.arange(x.shape[-1]) / self.fs, axis=0)
        amplitude = self._get_amplitude(x)
        if np.random.rand() < 0.5:
            freq = 50
        else:
            freq = 60
        noise = amplitude * np.sin(2 * np.pi * freq * t)
        return noise

class SineNoise(_Noise):
    """Add sine noise to the input sequence.
    """
    def _get_noise(self, x: np.ndarray) -> np.ndarray:
        t = np.expand_dims(np.arange(x.shape[-1]) / x.shape[-1], axis=0)
        return self.amplitude * np.sin(2 * np.pi * t / self.freq)

class SquareNoise(_Noise):
    """Add square noise to the input sequence.
    """
    def _get_noise(self, x: np.ndarray) -> np.ndarray:
        t = np.expand_dims(np.arange(x.shape[-1]) / x.shape[-1], axis=0)
        return self.amplitude * square(2 * np.pi * t / self.freq)

class WhiteNoise(_Noise):
    """Add white noise to the input sequence.
    """
    def _get_noise(self, x: np.ndarray) -> np.ndarray:
        return self.amplitude * np.random.randn(*x.shape)

class _RandomPartialNoise(_Noise):
    """Base class for adding noise to the random part of the input sequence.
    """
    def __init__(
        self,
        amplitude: float = 1.0,
        freq: float = 0.5,
        ratio: float = 0.5,
    ) -> None:
        super(_RandomPartialNoise, self).__init__(amplitude, freq)
        self.ratio = ratio

    def _get_partial_noise(self, x: np.ndarray) -> np.ndarray:
        noise = self._get_noise(x)
        count = int(np.random.uniform(0, self.ratio) * x.shape[-1])
        start_idx = np.random.randint(0, x.shape[-1] - count)
        partial_noise = np.zeros_like(x)
        partial_noise[:, start_idx:start_idx + count] = noise[:, :count]
        return partial_noise

    def _augment(self, x: np.ndarray) -> np.ndarray:
        noise = self._get_partial_noise(x)
        return x + noise

    def _set_level(self, level: int, max_level: int = 10) -> None:
        super(_RandomPartialNoise, self)._set_level(level, max_level)
        self.ratio = level / max_level * 0.5

class RandomPartialSineNoise(_RandomPartialNoise, SineNoise):
    """Add sine noise to the random part of the input sequence.
    """

class RandomPartialSquareNoise(_RandomPartialNoise, SquareNoise):
    """Add square noise to the random part of the input sequence.
    """

class RandomPartialWhiteNoise(_RandomPartialNoise, WhiteNoise):
    """Add white noise to the random part of the input sequence.
    """


"""Etc
"""
class RandomApply:
    """Apply randomly the given transform.
    """
    def __init__(self, transform: _BaseAugment, prob: float = 0.5) -> None:
        self.transform = transform
        self.prob = prob

    def __call__(self, x: np.ndarray, y: Optional[np.ndarray] = None):
        if np.random.rand() < self.prob:
            if y is not None:
                x, y = self.transform(x, y)
            else:
                x = self.transform(x)
        if y is not None:
            return x, y
        else:
            return x

class Compose:
    """Compose several transforms together.
    """
    def __init__(self, transforms: List[Any]) -> None:
        self.transforms = transforms

    def __call__(self, x: np.ndarray, y: Optional[np.ndarray] = None):
        for transform in self.transforms:
            if y is not None:
                x, y = transform(x, y)
            else:
                x = transform(x)
        if y is not None:
            return x, y
        else:
            return x

class ToTensor:
    """Convert ndarrays in sample to Tensors.
    """
    _DTYPES = {
        "float": torch.float32,
        "double": torch.float64,
        "int": torch.int32,
        "long": torch.int64,
    }

    def __init__(
        self,
        dtype: Union[str, torch.dtype] = torch.float32,
    ) -> None:
        if isinstance(dtype, str):
            assert dtype in self._DTYPES, f"Invalid dtype: {dtype}"
            dtype = self._DTYPES[dtype]
        self.dtype = dtype

    def __call__(self, x: Any) -> torch.Tensor:
        x = x.copy()
        return torch.tensor(x, dtype=self.dtype)


"""Random augmentation
"""
class RandAugment:
    """RandAugment: Practical automated data augmentation with a reduced search space.
        ref: https://arxiv.org/abs/1909.13719
    """
    def __init__(
        self,
        ops: list,
        level: int = 10,
        num_layers: int = 2,
        prob: float = 0.5,
    ) -> None:
        self.ops = []
        for op in ops:
            if hasattr(op, '_set_level'):
                op._set_level(level=level)
            self.ops.append(RandomApply(op, prob=prob))
        self.num_layers = num_layers
        self.prob = prob

    def __call__(self, x: np.ndarray, y: Optional[np.ndarray] = None):
        ops = np.random.choice(self.ops, self.num_layers, replace=False)
        for op in ops:
            if y is not None:
                x, y = op(x, y)
            else:
                x = op(x)
        if y is not None:
            return x, y
        else:
            return x


MAPPING = {
    'adaptive_powerline_noise': AdaptivePowerlineNoise,
    'amplitude_scaling': AmplitudeScaling,
    'center_crop': CenterCrop,
    'cutout': Cutout,
    'drop': RandomMask,
    'highpass_filter': HighpassFilter,
    'lowpass_filter': LowpassFilter,
    'moving_window_crop': MovingWindowCrop,
    'n_crop': NCrop,
    'random_baseline_shift': RandomBaselineShift,
    'random_crop': RandomCrop,
    'partial_sine_noise': RandomPartialSineNoise,
    'partial_square_noise': RandomPartialSquareNoise,
    'partial_white_noise': RandomPartialWhiteNoise,
    'random_resize_crop': RandomResizeCrop,
    'resample': Resample,
    'shift': RandomShift,
    'sine_noise': SineNoise,
    'sos_filter': SOSFilter,
    'square_noise': SquareNoise,
    'standardize': Standardize,
    'to_tensor': ToTensor,
    'white_noise': WhiteNoise,
    'xflip': XFlip,
    'yflip': YFlip,
}

AUGMENTATIONS = {
    'adaptive_powerline_noise': AdaptivePowerlineNoise,
    'amplitude_scaling': AmplitudeScaling,
    'cutout': Cutout,
    'drop': RandomMask,
    'random_baseline_shift': RandomBaselineShift,
    'random_crop': RandomCrop,
    'partial_sine_noise': RandomPartialSineNoise,
    'partial_square_noise': RandomPartialSquareNoise,
    'partial_white_noise': RandomPartialWhiteNoise,
    'random_resize_crop': RandomResizeCrop,
    'shift': RandomShift,
    'sine_noise': SineNoise,
    'square_noise': SquareNoise,
    'white_noise': WhiteNoise,
    'xflip': XFlip,
    'yflip': YFlip,
}

LABEL_CHANGEABLE_OPS = {
    'center_crop': CenterCrop,
    'cutout': Cutout,
    'drop': RandomMask,
    'moving_window_crop': MovingWindowCrop,
    'n_crop': NCrop,
    'random_crop': RandomCrop,
    'random_resize_crop': RandomResizeCrop,
    'resample': Resample,
    'shift': RandomShift,
    'xflip': XFlip,
}


def get_transforms_from_config(
    config: List[Union[str, Dict[str, Any]]]
) -> List[_BaseAugment]:
    """Get transforms from config.
    """
    transforms = []
    for transform in config:
        if isinstance(transform, str):
            name = transform
            kwargs = {}
        elif isinstance(transform, dict):
            assert len(transform) == 1, \
                "Each transform must have only one key."
            name, kwargs = list(transform.items())[0]
        else:
            raise ValueError(
                f"Invalid transform: {transform}, must be a str or a dict."
            )
        if name in MAPPING:
            transforms.append(MAPPING[name](**kwargs))
        elif name == "RandomApply":
            assert 'transform' in kwargs, \
                "RandomApply must have 'transform' key."
            assert 'prob' in kwargs, \
                "RandomApply must have 'prob' key."
            transform = get_transforms_from_config([kwargs['transform']])
            transforms.append(RandomApply(transform[0], prob=kwargs['prob']))
        elif name == "RandAugment":
            assert 'ops' in kwargs, \
                "RandAugment must have 'ops' key."
            assert 'level' in kwargs, \
                "RandAugment must have 'level' key."
            assert 'num_layers' in kwargs, \
                "RandAugment must have 'num_layers' key."
            transforms.append(
                RandAugment(
                    ops=get_transforms_from_config(kwargs['ops']),
                    level=kwargs.get('level', 10),
                    num_layers=kwargs.get('num_layers', 2),
                    prob=kwargs.get('prob', 0.5),
                )
            )
        elif name in globals():
            transforms.append(globals()[name](**kwargs))
        else:
            raise ValueError(f"Invalid name: {name}")

    if not transforms:
        return None
    return transforms
