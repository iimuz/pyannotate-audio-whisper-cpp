from pathlib import Path

from pydantic import BaseModel, RootModel

from internal.speaker_text import SpeakerText


class SpeechTextFile:
    """話者ごとのテキストをファイル保存する."""

    class SpeechTextList(RootModel[list[SpeakerText]]):
        pass

    def __init__(self, filepath: Path) -> None:
        self._filepath = filepath

    def save(self, segments: list[SpeakerText]) -> None:
        segment_list = self.SpeechTextList.model_validate(segments)
        self._filepath.write_text(segment_list.json())

    def get_segment_list(self) -> list[SpeakerText]:
        if not self._filepath.exists():
            return list()

        segment_list = self.SpeechTextList.parse_file(self._filepath)
        return segment_list.root

    def clean(self) -> None:
        if not self._filepath.exists():
            return

        self._filepath.unlink()
