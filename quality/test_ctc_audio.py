"""Check the CTC client's PCM input against its FFmpeg decoder."""

import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from clients.ctc_transcribe import ffmpeg_samples, load_audio


class CtcAudioTest(unittest.TestCase):
    def test_pcm_and_resampled_audio(self):
        values = np.array([-32768, -1, 0, 1, 32767] * 128, dtype="<i2")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audio.wav"
            for rate in (16000, 8000):
                with self.subTest(sample_rate=rate):
                    with wave.open(str(path), "wb") as wav:
                        wav.setparams((1, 2, rate, len(values), "NONE", ""))
                        wav.writeframes(values.tobytes())
                    np.testing.assert_array_equal(load_audio(path), ffmpeg_samples(path))
