import random
from typing import Literal

import librosa
import numpy as np
from numpy.typing import NDArray
import python_stretch

from audiomentations.core.transforms_interface import BaseWaveformTransform


class TimeStretch(BaseWaveformTransform):
    """Time stretch the signal without changing the pitch"""

    supports_multichannel = True

    def __init__(
        self,
        min_rate: float = 0.8,
        max_rate: float = 1.25,
        leave_length_unchanged: bool = True,
        method: Literal[
            "librosa_phase_vocoder", "signalsmith_stretch"
        ] = "signalsmith_stretch",
        p: float = 0.5,
    ):
        """
        :param min_rate: Minimum time-stretch rate. Values less than 1.0 slow down the audio (reduce the playback speed).
        :param max_rate: Maximum time-stretch rate. Values greater than 1.0 speed up the audio (increase the playback speed).
        :param leave_length_unchanged: If `True`, the output audio will have the same duration as the input audio.
            If `False`, the duration of the output audio will be altered by the time-stretch rate.
        :param method: "librosa_phase_vocoder" or "signalsmith_stretch"
        :param p: The probability of applying this transform.
        """
        super().__init__(p)
        if min_rate < 0.1:
            raise ValueError("min_rate must be >= 0.1")
        if max_rate > 10:
            raise ValueError("max_rate must be <= 10")
        if min_rate > max_rate:
            raise ValueError("min_rate must not be greater than max_rate")

        self.min_rate = min_rate
        self.max_rate = max_rate
        self.leave_length_unchanged = leave_length_unchanged

        if method not in ("librosa_phase_vocoder", "signalsmith_stretch"):
            raise ValueError(
                'method must be set to either "librosa_phase_vocoder" or "signalsmith_stretch"'
            )
        self.method = method

    def randomize_parameters(self, samples: NDArray[np.float32], sample_rate: int):
        pass

    def apply(
        self, samples: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        pass
