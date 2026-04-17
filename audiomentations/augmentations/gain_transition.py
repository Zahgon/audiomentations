import random
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from audiomentations.core.transforms_interface import BaseWaveformTransform
from audiomentations.core.utils import convert_decibels_to_amplitude_ratio


def get_fade_mask(
    start_level_db: float,
    end_level_db: float,
    fade_time_samples: int,
):
    """
    :param start_level_db:
    :param end_level_db:
    :param fade_time_samples: How long does the fade last?
    :return:
    """
    pass


class GainTransition(BaseWaveformTransform):
    """
    Gradually change the volume up or down over a random time span. Also known as
    fade in and fade out. The fade works on a logarithmic scale, which is natural to
    human hearing.

    The way this works is that it picks two gains: a first gain and a second gain.
    Then it picks a time range for the transition between those two gains.
    Note that this transition can start before the audio starts and/or end after the
    audio ends, so the output audio can start or end in the middle of a transition.
    The gain starts at the first gain and is held constant until the transition start.
    Then it transitions to the second gain. Then that gain is held constant until the
    end of the sound.
    """

    supports_multichannel = True

    def __init__(
        self,
        min_gain_db: float = -24.0,
        max_gain_db: float = 6.0,
        min_duration: float | int = 0.2,
        max_duration: float | int = 6.0,
        duration_unit: Literal["fraction", "samples", "seconds"] = "seconds",
        p: float = 0.5,
    ):
        """
        :param min_gain_db: Minimum gain in dB.
        :param max_gain_db: Maximum gain in dB.
        :param min_duration: Minimum length of transition. See also duration_unit.
        :param max_duration: Maximum length of transition. See also duration_unit.
        :param duration_unit: Defines the unit of the value of min_duration and max_duration.
            "fraction": Fraction of the total sound length
            "samples": Number of audio samples
            "seconds": Number of seconds
        :param p: The probability of applying this transform
        """
        super().__init__(p)

        if min_gain_db > max_gain_db:
            raise ValueError("min_gain_db must not be greater than max_gain_db")
        self.min_gain_db = min_gain_db
        self.max_gain_db = max_gain_db

        if min_duration <= 0:
            raise ValueError("min_duration must be greater than zero")
        if min_duration > max_duration:
            raise ValueError("min_duration must not be greater than max_duration")
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.duration_unit = duration_unit

    def randomize_parameters(self, samples: NDArray[np.float32], sample_rate: int):
        pass

    def apply(self, samples: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:
        pass
