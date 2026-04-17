import random

import numpy as np
from numpy.typing import NDArray

from audiomentations.core.transforms_interface import BaseWaveformTransform
from audiomentations.core.utils import calculate_rms


class TanhDistortion(BaseWaveformTransform):
    """
    Apply tanh (hyperbolic tangent) distortion to the audio. This technique is sometimes
    used for adding distortion to guitar recordings. The tanh() function can give a rounded
    "soft clipping" kind of distortion, and the distortion amount is proportional to the
    loudness of the input and the pre-gain. Tanh is symmetric, so the positive and
    negative parts of the signal are squashed in the same way. This transform can be
    useful as data augmentation because it adds harmonics. In other words, it changes
    the timbre of the sound.

    See this page for examples: http://gdsp.hf.ntnu.no/lessons/3/17/
    """

    supports_multichannel = True

    def __init__(
        self, min_distortion: float = 0.01, max_distortion: float = 0.7, p: float = 0.5
    ):
        """
        :param min_distortion: Minimum amount of distortion (between 0 and 1)
        :param max_distortion: Maximum amount of distortion (between 0 and 1)
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        assert 0 <= min_distortion <= 1
        assert 0 <= max_distortion <= 1
        assert min_distortion <= max_distortion
        self.min_distortion = min_distortion
        self.max_distortion = max_distortion

    def randomize_parameters(self, samples: NDArray[np.float32], sample_rate: int):
        pass

    def apply(self, samples: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:
        # Find out how much to pre-gain the audio to get a given amount of distortion
        pass
