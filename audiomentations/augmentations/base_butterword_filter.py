import random

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfilt, sosfiltfilt, sosfilt_zi

from audiomentations.core.transforms_interface import BaseWaveformTransform
from audiomentations.core.utils import (
    convert_frequency_to_mel,
    convert_mel_to_frequency,
)


class BaseButterworthFilter(BaseWaveformTransform):
    """
    A `scipy.signal.butter`-based generic filter class.
    """

    supports_multichannel = True

    # The types below must be equal to the ones accepted by
    # the `btype` argument of `scipy.signal.butter`
    ALLOWED_ONE_SIDE_FILTER_TYPES = ("lowpass", "highpass")
    ALLOWED_TWO_SIDE_FILTER_TYPES = ("bandpass", "bandstop")
    ALLOWED_FILTER_TYPES = ALLOWED_ONE_SIDE_FILTER_TYPES + ALLOWED_TWO_SIDE_FILTER_TYPES

    def __init__(self, **kwargs):
        assert "p" in kwargs
        assert "min_rolloff" in kwargs
        assert "max_rolloff" in kwargs
        assert "filter_type" in kwargs
        assert "zero_phase" in kwargs

        super().__init__(kwargs["p"])

        self.filter_type = kwargs["filter_type"]
        self.min_rolloff = kwargs["min_rolloff"]
        self.max_rolloff = kwargs["max_rolloff"]
        self.zero_phase = kwargs["zero_phase"]

        if self.zero_phase:
            assert (
                self.min_rolloff % 12 == 0
            ), "Zero phase filters can only have a steepness which is a multiple of 12 dB/octave"
            assert (
                self.max_rolloff % 12 == 0
            ), "Zero phase filters can only have a steepness which is a multiple of 12 dB/octave"
        else:
            assert (
                self.min_rolloff % 6 == 0
            ), "Non zero phase filters can only have a steepness which is a multiple of 6 dB/octave"
            assert (
                self.max_rolloff % 6 == 0
            ), "Non zero phase filters can only have a steepness which is a multiple of 6 dB/octave"

        assert (
            self.filter_type in BaseButterworthFilter.ALLOWED_FILTER_TYPES
        ), "Filter type must be one of: " + ", ".join(
            BaseButterworthFilter.ALLOWED_FILTER_TYPES
        )

        assert ("min_cutoff_freq" in kwargs and "max_cutoff_freq" in kwargs) or (
            "min_center_freq" in kwargs
            and "max_center_freq" in kwargs
            and "min_bandwidth_fraction" in kwargs
            and "max_bandwidth_fraction" in kwargs
        ), "Arguments for either a one-sided, or a two-sided filter must be given"

        if "min_cutoff_freq" in kwargs:
            self.initialize_one_sided_filter(
                min_cutoff_freq=kwargs["min_cutoff_freq"],
                max_cutoff_freq=kwargs["max_cutoff_freq"],
            )
        elif "min_center_freq" in kwargs:
            self.initialize_two_sided_filter(
                min_center_freq=kwargs["min_center_freq"],
                max_center_freq=kwargs["max_center_freq"],
                min_bandwidth_fraction=kwargs["min_bandwidth_fraction"],
                max_bandwidth_fraction=kwargs["max_bandwidth_fraction"],
            )

    def initialize_one_sided_filter(
        self,
        min_cutoff_freq,
        max_cutoff_freq,
    ):
        """
        :param min_cutoff_freq: Minimum cutoff frequency in hertz
        :param max_cutoff_freq: Maximum cutoff frequency in hertz
        :param min_rolloff: Minimum filter roll-off (in dB/octave).
            Must be a multiple of 6
        :param max_rolloff: Maximum filter roll-off (in dB/octave)
            Must be a multiple of 6
        :param p: The probability of applying this transform
        """
        pass

    def initialize_two_sided_filter(
        self,
        min_center_freq,
        max_center_freq,
        min_bandwidth_fraction,
        max_bandwidth_fraction,
    ):
        """
        :param min_center_freq: Minimum center frequency in hertz
        :param max_center_freq: Maximum center frequency in hertz
        :param min_bandwidth_fraction: Minimum bandwidth fraction relative to center
            frequency (number between 0.0 and 2.0)
        :param max_bandwidth_fraction: Maximum bandwidth fraction relative to center
            frequency (number between 0.0 and 2.0)
        :param min_rolloff: Minimum filter roll-off (in dB/octave).
            Must be a multiple of 6
        :param max_rolloff: Maximum filter roll-off (in dB/octave)
            Must be a multiple of 6
        :param p: The probability of applying this transform
        """
        pass

    def randomize_parameters(self, samples: NDArray[np.float32], sample_rate: int = None):
        pass

    def apply(self, samples: NDArray[np.float32], sample_rate: int = None) -> NDArray[np.float32]:
        pass
