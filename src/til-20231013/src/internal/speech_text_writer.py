import os
from pathlib import Path

from internal.speaker_text import SpeakerText


class SpeechTextWriter:
    """書き起こした文字をテキストで保存する."""

    def __init__(self, filepath: Path) -> None:
        self._filepath = filepath

    def save(self, speaker_text: list[SpeakerText]) -> None:
        if len(speaker_text) < 1:
            self._filepath.write_text("")
            return

        # segmentごとの文字列を生成
        segments = [
            os.linesep.join(
                [
                    f"[{text.start_time:03.1f} --> {text.end_time:03.1f}] {text.speaker_name}",
                    text.text,
                ]
            )
            for text in speaker_text
        ]

        self._filepath.write_text(f"{(os.linesep)*2}".join(segments))

    def clean(self) -> None:
        if not self._filepath.exists():
            return

        self._filepath.unlink()
