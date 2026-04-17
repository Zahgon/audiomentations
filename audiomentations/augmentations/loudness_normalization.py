import random
import sys

import numpy as np
from numpy.typing import NDArray

from audiomentations.core.transforms_interface import BaseWaveformTransform


class LoudnessNormalization(BaseWaveformTransform):
    """
    Apply a constant amount of gain to match a specific loudness (in LUFS). This is an
    implementation of ITU-R BS.1770-4.

    For an explanation on LUFS, see https://en.wikipedia.org/wiki/LUFS

    See also the following web page for more info on audio loudness normalization:
        https://en.wikipedia.org/wiki/Audio_normalization

    Warning: This transform can return samples outside the [-1, 1] range, which may lead to
    clipping or wrap distortion, depending on what you do with the audio in a later stage.
    See also https://en.wikipedia.org/wiki/Clipping_(audio)#Digital_clipping
    """

    supports_multichannel = True

    def __init__(
        self,
        min_lufs: float = -31.0,
        max_lufs: float = -13.0,
        p: float = 0.5,
    ):
        super().__init__(p)

        if min_lufs > max_lufs:
            raise ValueError("min_lufs must not be greater than max_lufs")

        self.min_lufs = min_lufs
        self.max_lufs = max_lufs

    def randomize_parameters(self, samples: NDArray[np.float32], sample_rate: int):
        pass

    def apply(
        self, samples: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        # Guard against digital silence
        pass
