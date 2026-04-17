import sys
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from audiomentations import Normalize
from audiomentations.core.utils import (
    calculate_rms,
    convert_decibels_to_amplitude_ratio,
)


class PostGain:
    """
    Gain up or down the audio after the given transform (or set of transforms) has
    processed the audio. There are several methods that determine how the audio should
    be gained. PostGain can be useful for compensating for any gain differences introduced
    by a (set of) transform(s), or for preventing clipping in the output.
    """

    def __init__(
        self,
        transform: Callable[[NDArray[np.float32], int], NDArray[np.float32]],
        method: str,  # , **kwargs
    ):
        """
        :param transform: A callable to be applied. It should input
            samples (ndarray), sample_rate (int) and optionally some user-defined
            keyword arguments.
        :param method: "same_rms", "same_lufs", "peak_normalize_always" or "peak_normalize_always"
        """
        self.transform = transform
        self.method = method
        assert self.method in (
            "same_rms",
            "same_lufs",
            "peak_normalize_always",
            "peak_normalize_if_too_loud",
            # "target_rms",
            # "target_lufs",
            # "target_peak_dbfs",
            # "target_true_peak_dbfs",
        )
        # if self.method == "target_rms":
        #     self.target_rms = kwargs["target_rms"]
        # elif self.method == "target_lufs":
        #     self.target_lufs = kwargs["target_lufs"]
        # elif self.method == "target_peak_dbfs":
        #     self.target_peak_dbfs = kwargs["target_peak_dbfs"]

    def method_same_rms(
        self, samples: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        pass

    def method_same_lufs(
        self, samples: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        pass

    def method_peak_normalize_always(
        self, samples: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        pass

    def method_peak_normalize_if_too_loud(
        self, samples: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        pass

    def __call__(self, samples: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:
        if self.method == "same_rms":
            return self.method_same_rms(samples, sample_rate)
        elif self.method == "same_lufs":
            return self.method_same_lufs(samples, sample_rate)
        elif self.method == "peak_normalize_always":
            return self.method_peak_normalize_always(samples, sample_rate)
        elif self.method == "peak_normalize_if_too_loud":
            return self.method_peak_normalize_if_too_loud(samples, sample_rate)
        else:
            raise Exception()
