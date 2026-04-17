import random

import numpy as np
from numpy.typing import NDArray
from scipy.signal import sosfilt, sosfilt_zi

from audiomentations.core.transforms_interface import BaseWaveformTransform
from audiomentations.core.utils import (
    convert_frequency_to_mel,
    convert_mel_to_frequency,
)


class PeakingFilter(BaseWaveformTransform):
    """
    Peaking filter transform. Applies a peaking filter at a specific center frequency in hertz
    of a specific gain in dB (note: can be positive or negative!), and a quality factor
    parameter. Filter coefficients are taken from the W3 Audio EQ Cookbook:
    https://www.w3.org/TR/audio-eq-cookbook/
    """

    supports_multichannel = True

    def __init__(
        self,
        min_center_freq: float = 50.0,
        max_center_freq: float = 7500.0,
        min_gain_db: float = -24.0,
        max_gain_db: float = 24.0,
        min_q: float = 0.5,
        max_q: float = 5.0,
        p: float = 0.5,
    ):
        """
        :param min_center_freq: The minimum center frequency of the peaking filter
        :param max_center_freq: The maximum center frequency of the peaking filter
        :param min_gain_db: The minimum gain at center frequency in dB
        :param max_gain_db: The maximum gain at center frequency in dB
        :param min_q: The minimum quality factor Q. The higher the Q, the steeper the
            transition band will be.
        :param max_q: The maximum quality factor Q. The higher the Q, the steeper the
            transition band will be.
        """

        assert (
            min_center_freq <= max_center_freq
        ), "`min_center_freq` should be no greater than `max_center_freq`"
        assert (
            min_gain_db <= max_gain_db
        ), "`min_gain_db` should be no greater than `max_gain_db`"

        assert min_q > 0, "`min_q` should be greater than 0"
        assert max_q > 0, "`max_q` should be greater than 0"

        super().__init__(p)

        self.min_center_freq = min_center_freq
        self.max_center_freq = max_center_freq

        self.min_gain_db = min_gain_db
        self.max_gain_db = max_gain_db

        self.min_q = min_q
        self.max_q = max_q

    def _get_biquad_coefficients_from_input_parameters(
        self, center_freq, gain_db, q_factor, sample_rate
    ):
        pass

    def randomize_parameters(self, samples: NDArray[np.float32], sample_rate: int):
        pass

    def apply(self, samples: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:
        pass
