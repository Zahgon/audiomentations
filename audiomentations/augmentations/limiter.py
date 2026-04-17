import math
import random
import sys
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from audiomentations.core.transforms_interface import BaseWaveformTransform
from audiomentations.core.utils import (
    convert_decibels_to_amplitude_ratio,
    get_max_abs_amplitude,
)


class Limiter(BaseWaveformTransform):
    """
    A simple audio limiter (dynamic range compression).
    """

    supports_multichannel = True

    def __init__(
        self,
        min_threshold_db: float = -24.0,
        max_threshold_db: float = -2.0,
        min_attack: float = 0.0005,
        max_attack: float = 0.025,
        min_release: float = 0.05,
        max_release: float = 0.7,
        threshold_mode: Literal[
            "relative_to_signal_peak", "absolute"
        ] = "relative_to_signal_peak",
        p: float = 0.5,
    ):
        """
        The threshold determines the audio level above which the limiter kicks in.
        The attack time is how quickly the limiter kicks in once the audio signal starts exceeding the threshold.
        The release time determines how quickly the limiter stops working after the signal drops below the threshold.

        :param min_threshold_db: Minimum threshold in Decibels
        :param max_threshold_db: Maximum threshold in Decibels
        :param min_attack: Minimum attack time in seconds
        :param max_attack: Maximum attack time in seconds
        :param min_release: Minimum release time in seconds
        :param max_release: Maximum release time in seconds
        :param threshold_mode: "relative_to_signal_peak" or "absolute"
            "relative_to_signal_peak" means the threshold is relative to peak of the signal.
            "absolute" means the threshold is relative to 0 dBFS, so it doesn't depend
            on the peak of the signal.
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        assert min_attack > 0.0, "min_attack must be a positive number"
        assert max_attack > 0.0, "max_attack must be a positive number"
        assert (
            max_attack >= min_attack
        ), "max_attack must be greater than or equal to min_attack"
        assert min_release > 0.0, "min_release must be a positive number"
        assert max_release > 0.0, "max_release must be a positive number"
        assert (
            max_release >= min_release
        ), "max_release must be greater than or equal to min_release"
        assert (
            max_threshold_db >= min_threshold_db
        ), "max_threshold_db must be greater than or equal to min_threshold_db"
        assert threshold_mode in (
            "relative_to_signal_peak",
            "absolute",
        ), 'threshold_mode must be either "relative_to_signal_peak" or "absolute"'
        self.min_threshold_db = min_threshold_db
        self.max_threshold_db = max_threshold_db
        self.min_attack = min_attack
        self.max_attack = max_attack
        self.min_release = min_release
        self.max_release = max_release
        self.threshold_mode = threshold_mode

    @staticmethod
    def convert_time_to_coefficient(
        t: float, sample_rate: int, decay_threshold: float = None
    ) -> float:
        pass

    def randomize_parameters(self, samples: NDArray[np.float32], sample_rate: int):
        pass

    def apply(
        self, samples: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        pass
