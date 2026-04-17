from typing import Literal

import numpy as np
from numpy.typing import NDArray

from audiomentations.core.transforms_interface import BaseWaveformTransform
from audiomentations.core.utils import get_max_abs_amplitude


class Normalize(BaseWaveformTransform):
    """
    Apply a constant amount of gain, so that highest signal level present in the sound becomes
    0 dBFS, i.e. the loudest level allowed if all samples must be between -1 and 1. Also known
    as peak normalization.
    """

    supports_multichannel = True

    def __init__(
        self, apply_to: Literal["all", "only_too_loud_sounds"] = "all", p: float = 0.5
    ):
        super().__init__(p)
        assert apply_to in ("all", "only_too_loud_sounds")
        self.apply_to = apply_to

    def randomize_parameters(self, samples: NDArray[np.float32], sample_rate: int):
        pass

    def apply(
        self, samples: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        pass
