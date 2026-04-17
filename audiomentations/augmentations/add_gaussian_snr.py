import random

import numpy as np
from numpy.typing import NDArray

from audiomentations.core.transforms_interface import BaseWaveformTransform
from audiomentations.core.utils import calculate_desired_noise_rms, calculate_rms


class AddGaussianSNR(BaseWaveformTransform):
    """
    Add gaussian noise to the input. A random Signal to Noise Ratio (SNR) will be picked
    uniformly in the Decibel scale. This aligns with human hearing, which is more
    logarithmic than linear.
    """

    supports_multichannel = True

    def __init__(
        self,
        min_snr_db: float = 5.0,
        max_snr_db: float = 40.0,
        p: float = 0.5,
    ):
        """
        :param min_snr_db: Minimum signal-to-noise ratio in dB. A lower number means more noise.
        :param max_snr_db: Maximum signal-to-noise ratio in dB. A greater number means less noise.
        :param p: The probability of applying this transform
        """
        super().__init__(p)

        if min_snr_db > max_snr_db:
            raise ValueError("min_snr_db must not be greater than max_snr_db")
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db

    def randomize_parameters(self, samples: NDArray[np.float32], sample_rate: int):
        pass

    def apply(
        self, samples: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        pass
