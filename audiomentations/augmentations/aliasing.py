import random

import numpy as np
from numpy.typing import NDArray

from audiomentations.core.transforms_interface import BaseWaveformTransform
from audiomentations.core.utils import (
    convert_frequency_to_mel,
    convert_mel_to_frequency,
)


class Aliasing(BaseWaveformTransform):
    """
    Apply an aliasing effect to the audio by downsampling to a lower
    sample rate without filtering and upsampling after that.
    """

    supports_multichannel = True

    def __init__(
        self, min_sample_rate: int = 8000, max_sample_rate: int = 30000, p: float = 0.5
    ):
        """
        :param min_sample_rate: Minimum target sample rate to downsample to
        :param max_sample_rate: Maximum target sample rate to downsample to
        :param p: The probability of applying this transform
        """
        super().__init__(p)

        if min_sample_rate < 2:
            raise ValueError("min_sample_rate must be greater than or equal to 2")

        if min_sample_rate > max_sample_rate:
            raise ValueError("min_sample_rate must not be larger than max_sample_rate")
        
        self.min_sample_rate = min_sample_rate
        self.max_sample_rate = max_sample_rate
        self.min_mel = convert_frequency_to_mel(min_sample_rate)
        self.max_mel = convert_frequency_to_mel(max_sample_rate)

    def randomize_parameters(self, samples: NDArray[np.float32], sample_rate: int):
        pass

    def apply(self, samples: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:
        pass
