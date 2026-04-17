from typing import Literal

import python_stretch
import random
import warnings

import librosa
import numpy as np
from numpy.typing import NDArray

from audiomentations.core.transforms_interface import BaseWaveformTransform


class PitchShift(BaseWaveformTransform):
    """Pitch shift the sound up or down without changing the tempo"""

    supports_multichannel = True

    def __init__(
        self,
        min_semitones: float = -4.0,
        max_semitones: float = 4.0,
        method: Literal[
            "librosa_phase_vocoder", "signalsmith_stretch"
        ] = "signalsmith_stretch",
        p: float = 0.5,
    ):
        """
        :param min_semitones: Minimum semitones to shift. A negative number means shift down.
        :param max_semitones: Maximum semitones to shift. A positive number means shift up.
        :param method:
            "librosa_phase_vocoder": slow, low quality, supports any number of channels
            "signalsmith_stretch" (default): fast, high quality, only supports mono and stereo
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        if min_semitones < -24:
            raise ValueError("min_semitones must be >= -24")
        if max_semitones > 24:
            raise ValueError("max_semitones must be <= 24")
        if min_semitones > max_semitones:
            raise ValueError("min_semitones must not be greater than max_semitones")
        self.min_semitones = min_semitones
        self.max_semitones = max_semitones
        if method not in ("librosa_phase_vocoder", "signalsmith_stretch"):
            raise ValueError(
                'method must be set to either "librosa_phase_vocoder" or'
                ' "signalsmith_stretch"'
            )
        self.method = method

    def randomize_parameters(self, samples: NDArray[np.float32], sample_rate: int):
        pass

    def apply(
        self, samples: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        pass
