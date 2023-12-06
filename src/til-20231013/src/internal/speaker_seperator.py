from pathlib import Path

import torch
from pyannote.audio import Pipeline

from internal.speaker_segment import SpeakerSegment


class SpeakerSeperator:
    """音声から話者分離を行う."""

    def __init__(self, config_path: Path, device_name: str) -> None:
        self._config_path = config_path
        self._device_name = device_name

    def diarization(self, wav_filepath: Path):
        pipeline = Pipeline.from_pretrained(self._config_path)
        pipeline.to(torch.device(self._device_name))
        diarization = pipeline(wav_filepath)

        for segment, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segment = SpeakerSegment(
                start_time=segment.start, end_time=segment.end, speaker_name=speaker
            )
            yield speaker_segment
