from typing import Literal

import numpy as np
from numpy.typing import NDArray

from audiomentations.core.transforms_interface import BaseWaveformTransform


class AdjustDuration(BaseWaveformTransform):
    """
    Trim or pad the audio to the specified length/duration in samples or seconds. If the
    input sound is longer than the target duration, pick a random offset and crop the
    sound to the target duration. If the input sound is shorter than the target
    duration, pad the sound so the duration matches the target duration.
    """

    supports_multichannel = True

    def __init__(
        self,
        duration_samples: int = None,
        duration_seconds: float = None,
        padding_mode: Literal["silence", "wrap", "reflect"] = "silence",
        padding_position: Literal["start", "end"] = "end",
        p: float = 0.5,
    ):
        """
        :param duration_samples: Target duration in number of samples
        :param duration_seconds: Target duration in seconds
        :param padding_mode: Padding mode. Must be "silence", "wrap" or "reflect". Only
            used when audio input is shorter than the target duration.
        :param padding_position: The position of the inserted/added padding. Must be
            "start" or "end". Only used when audio input is shorter than the target duration.
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        assert padding_mode in ("silence", "wrap", "reflect")
        if padding_mode == "silence":
            padding_mode = "constant"  # for numpy.pad compatibility
        self.padding_mode = padding_mode

        assert padding_position in ("start", "end")
        self.padding_position = padding_position

        assert duration_samples is not None or duration_seconds is not None
        if duration_samples is not None and duration_seconds is not None:
            raise ValueError(
                "You must specify either duration_samples or duration_seconds, but not both."
            )
        elif duration_seconds is not None:
            if duration_seconds <= 0:
                raise ValueError("duration_seconds must be a positive float")
            self._get_target_samples_func = self._get_target_samples_from_seconds
        elif duration_samples is not None:
            if duration_samples <= 0:
                raise ValueError("duration_samples must be a positive int")
            self._get_target_samples_func = self._get_target_samples_from_samples

        self.duration_samples = duration_samples
        self.duration_seconds = duration_seconds

    def _get_target_samples_from_seconds(self, sr: int) -> int:
        pass

    def _get_target_samples_from_samples(self, sr: int) -> int:
        pass

    def apply(
        self, samples: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        pass
