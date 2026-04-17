import random
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from audiomentations.core.transforms_interface import BaseWaveformTransform


class Padding(BaseWaveformTransform):
    """
    Apply padding to the audio signal by taking a fraction of the start or end and replacing that
    portion with padding. This can be useful for training ML models on padded inputs.
    """

    supports_multichannel = True

    def __init__(
        self,
        mode: Literal["silence", "wrap", "reflect"] = "silence",
        min_fraction: float = 0.01,
        max_fraction: float = 0.7,
        pad_section: Literal["start", "end"] = "end",
        p: float = 0.5,
    ):
        """
        :param mode: Padding mode. Must be one of "silence", "wrap", "reflect"
        :param min_fraction: Minimum fraction of the signal duration to be padded
        :param max_fraction: Maximum fraction of the signal duration to be padded
        :param pad_section: Which part of the signal should be replaced with padding:
            "start" or "end"
        :param p: The probability of applying this transform
        """
        super().__init__(p)

        assert mode in ("silence", "wrap", "reflect")
        self.mode = mode

        assert max_fraction <= 1.0
        assert min_fraction >= 0
        assert min_fraction <= max_fraction
        self.min_fraction = min_fraction
        self.max_fraction = max_fraction

        assert pad_section in ("start", "end")
        self.pad_section = pad_section

    def randomize_parameters(self, samples: NDArray[np.float32], sample_rate: int):
        pass

    def apply(self, samples: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:
        pass
