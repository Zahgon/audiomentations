import os
import random
import tempfile
import uuid
import warnings
from typing import Literal

import librosa
import numpy as np
import sys
from numpy.typing import NDArray

from audiomentations.core.transforms_interface import BaseWaveformTransform
from audiomentations.core.utils import (
    convert_float_samples_to_int16,
    get_max_abs_amplitude,
)


class Mp3Compression(BaseWaveformTransform):
    """Compress the audio using an MP3 encoder to lower the audio quality.
    This may help machine learning models deal with compressed, low-quality audio.

    This transform depends on either lameenc or pydub/ffmpeg.

    Note that bitrates below 32 kbps are only supported for low sample rates (up to 24000 Hz).

    Note: When using the lameenc backend, the output may be slightly longer than the input due
    to the fact that the LAME encoder inserts some silence at the beginning of the audio.

    Warning: This transform writes to disk, so it may be slow. Ideally, the work should be done
    in memory. Contributions are welcome.
    """

    supports_multichannel = True

    SUPPORTED_BITRATES = [
        8,
        16,
        24,
        32,
        40,
        48,
        56,
        64,
        80,
        96,
        112,
        128,
        144,
        160,
        192,
        224,
        256,
        320,
    ]

    def __init__(
        self,
        min_bitrate: int = 8,
        max_bitrate: int = 64,
        backend: Literal["pydub", "lameenc", "fast-mp3-augment"] = "fast-mp3-augment",
        preserve_delay: bool = False,
        quality: int = 7,
        p: float = 0.5,
    ):
        """
        :param min_bitrate: Minimum bitrate in kbps
        :param max_bitrate: Maximum bitrate in kbps
        :param backend: "fast-mp3-augment", "pydub" or "lameenc".
            "fast-mp3-augment":
                * Fast (in-memory compute, encoder and decoder in separate threads working concurrently)
                * LAME encoder and minimp3 decoder
            "pydub":
                * Uses ffmpeg under the hood
                * Slower than lameenc and fast-mp3-augment
                * Writes temporary files to disk, which makes it comparatively slow
            "lameenc":
                * Slightly delays and pads the audio due to the way MP3 encoding and decoding normally works
                * Writes a temporary file to disk, which makes is comparatively slow
        :param preserve_delay:
            If False (default), the output length and timing will match the input.
            If True, include LAME encoder delay + filter delay (a few tens of milliseconds) and padding in the output.
            This makes the output
            1) longer than the input
            2) delayed (out of sync) relative to the input
            Normally, it makes sense to set preserve_delay to False, but if you want outputs that include the
            short, almost silent part in the beginning, you here have the option to get that.
        :param quality: LAME-specific parameter that controls a trade-off between audio quality and speed.
            quality is an int in range [0, 9]:
            0: higher quality audio at the cost of slower processing
            9: faster processing at the cost of lower quality audio
            Note: If using backend=="pydub", this parameter gets silently ignored.
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        if min_bitrate < self.SUPPORTED_BITRATES[0]:
            raise ValueError(
                "min_bitrate must be greater than or equal to"
                f" {self.SUPPORTED_BITRATES[0]}"
            )
        if max_bitrate > self.SUPPORTED_BITRATES[-1]:
            raise ValueError(
                "max_bitrate must be less than or equal to"
                f" {self.SUPPORTED_BITRATES[-1]}"
            )
        if max_bitrate < min_bitrate:
            raise ValueError("max_bitrate must be >= min_bitrate")

        is_any_supported_bitrate_in_range = any(
            min_bitrate <= bitrate <= max_bitrate for bitrate in self.SUPPORTED_BITRATES
        )
        if not is_any_supported_bitrate_in_range:
            raise ValueError(
                "There is no supported bitrate in the range between the specified"
                " min_bitrate and max_bitrate. The supported bitrates are:"
                f" {self.SUPPORTED_BITRATES}"
            )

        if backend == "pydub":
            warnings.warn(
                'The "pydub" backend is deprecated, because pydub seems to be have been unmaintained for'
                ' several years, and depends on audioop, which was deprecated in Python 3.11 and removed in 3.13.'
                ' Recommendation: Use backend="fast-mp3-augment" instead. It is faster.',
                DeprecationWarning,
            )
            if preserve_delay:
                raise ValueError(
                    'The "pydub" backend does not support preserve_delay=True. Switch to the'
                    ' "fast-mp3-augment" backend (recommended) or pass preserve_delay=False.'
                )
        elif backend == "lameenc":
            warnings.warn(
                'The "lameenc" backend is deprecated. Use backend="fast-mp3-augment" instead. It also uses'
                " the LAME encoder under the hood, but is faster.",
                DeprecationWarning,
            )
            if not preserve_delay:
                raise ValueError(
                    'The "lameenc" backend does not support preserve_delay=False. Switch to the'
                    ' "fast-mp3-augment" backend (recommended) or pass preserve_delay=True.'
                )
        self.preserve_delay = preserve_delay
        self.quality = quality
        self.min_bitrate = min_bitrate
        self.max_bitrate = max_bitrate
        if backend not in ("fast-mp3-augment", "pydub", "lameenc"):
            raise ValueError(
                'backend must be set to either "fast-mp3-augment", "pydub" or "lameenc"'
            )
        self.backend = backend
        self.post_gain_factor = None

    def randomize_parameters(self, samples: NDArray[np.float32], sample_rate: int):
        pass

    def apply(
        self, samples: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        pass

    def maybe_pre_gain(self, samples):
        """
        If the audio is too loud, gain it down to avoid distortion in the audio file to
        be encoded.
        """
        pass

    def maybe_post_gain(self, samples):
        """If the audio was pre-gained down earlier, post-gain it up to compensate here."""
        pass

    def apply_lameenc(
        self, samples: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        pass

    def apply_fast_mp3_augment(
        self, samples: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        pass

    def apply_pydub(
        self, samples: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        pass
