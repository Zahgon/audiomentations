import random
import warnings
from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from audiomentations.core.transforms_interface import BaseWaveformTransform
from audiomentations.core.utils import get_crossfade_length, get_crossfade_mask_pair

# 0.00025 seconds corresponds to 2 samples at 8000 Hz
DURATION_EPSILON = 0.00025


class RepeatPart(BaseWaveformTransform):
    """
    Select a subsection (or "part") of the audio and repeat that part a number of times.
    This can be useful when simulating scenarios where a short audio snippet gets
    repeated, for example:

    * Repetitions of some musical note or sound in a rhythmical way
    * A person stutters or says the same word (with variations) multiple times in a row
    * A mechanical noise with periodic repetitions
    * A "skip in the record" or a "stuck needle" effect, reminiscent of vinyl records or
        CDs when they repeatedly play a short section due to a scratch or other
        imperfection.
    * Digital audio glitches, such as a buffer underrun in video games,
        where the current audio frame gets looped continuously due to system overloads
        or a software crash.

    Note that the length of inputs you give it must be compatible with the part
    duration range and crossfade duration. If you give it an input audio array that is
    too short, a `UserWarning` will be raised and no operation is applied to the signal.
    """

    supports_multichannel = True

    def __init__(
        self,
        min_repeats: int = 1,
        max_repeats: int = 3,
        min_part_duration: float = 0.25,
        max_part_duration: float = 1.2,
        mode: Literal["insert", "replace"] = "insert",
        crossfade_duration: float = 0.005,
        part_transform: Callable[[NDArray[np.float32], int], NDArray[np.float32]]
        | None = None,
        p: float = 0.5,
    ):
        """
        :param min_repeats: Minimum number of times a selected audio segment should be
            repeated in addition to the original. For instance, if the selected number
            of repeats is 1, the selected segment will be followed by one repeat.
        :param max_repeats: Maximum number of times a selected audio segment can be
            repeated in addition to the original.
        :param min_part_duration: Minimum duration (in seconds) of the audio segment
            that can be selected for repetition.
        :param max_part_duration: Maximum duration (in seconds) of the audio segment
            that can be selected for repetition.
        :param mode: This parameter has two options:
            "insert": Insert the repeat(s), making the array longer. After the last
                repeat there will be the last part of the original audio, offset in time
                compared to the input array.
            "replace": Have the repeats replace (as in overwrite) the original audio.
                Any remaining part at the end (if not overwritten by repeats) will be
                left untouched without offset. The length of the output array is the
                same as the input array.
        :param crossfade_duration: Duration (in seconds) for crossfading between repeated
            parts as well as potentially from the original audio to the repetitions and back.
            The crossfades will be equal-energy or equal-gain depending on the audio and/or the
            chosen parameters of the transform. The crossfading feature can be used to smooth
            transitions and avoid abrupt changes, which can lead to impulses/clicks in the audio.
            If you know what you're doing, and impulses/clicks are desired for your use case,
            you can disable the crossfading by setting this value to `0.0`.
        :param part_transform: An optional callable (audiomentations transform) that
            gets applied individually to each repeat. This can be used to make each
            repeat slightly different from the previous one. Note that a part_transform
            that makes the part shorter is only supported if the transformed part is at
            least two times the crossfade duration.
        :param p: The probability of applying this transform
        """
        super().__init__(p)

        if min_repeats < 1:
            raise ValueError("min_repeats must be >= 1")
        if max_repeats < min_repeats:
            raise ValueError("max_repeats must be >= min_repeats")
        self.min_repeats = min_repeats
        self.max_repeats = max_repeats
        if min_part_duration < DURATION_EPSILON:
            raise ValueError(f"min_part_duration must be >= {DURATION_EPSILON}")
        if max_part_duration < min_part_duration:
            raise ValueError("max_part_duration must be >= min_part_duration")
        self.min_part_duration = min_part_duration
        self.max_part_duration = max_part_duration
        if mode not in ("insert", "replace"):
            raise ValueError('mode must be set to either "insert" or "replace"')
        self.mode = mode

        if crossfade_duration == 0.0:
            self.crossfade = False
        elif crossfade_duration < 0.0:
            raise ValueError("crossfade_duration must not be negative")
        elif crossfade_duration < DURATION_EPSILON:
            raise ValueError(
                "When crossfade_duration is set to a positive number, it must be >="
                f" {DURATION_EPSILON}"
            )
        else:
            self.crossfade = True
        if crossfade_duration > (min_part_duration / 2):
            raise ValueError(
                "crossfade_duration must be <= 0.5 * min_part_duration. You can fix this"
                " error by increasing min_part_duration or by decreasing"
                " crossfade_duration."
            )
        self.crossfade_duration = crossfade_duration
        self.part_transform = part_transform

    def randomize_parameters(self, samples: NDArray[np.float32], sample_rate: int):
        pass

    def apply(
        self, samples: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        pass

    def freeze_parameters(self):
        pass

    def unfreeze_parameters(self):
        pass
