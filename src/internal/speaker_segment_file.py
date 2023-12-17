from pathlib import Path

from pydantic import RootModel

from internal.speaker_segment import SpeakerSegment


class SpeakerSegmentFile:
    """話者分離情報をファイルに保存する."""

    class SpeakerSegmentList(RootModel[list[SpeakerSegment]]):
        pass

    def __init__(self, filepath: Path) -> None:
        self._filepath = filepath

    def save(self, segments: list[SpeakerSegment]) -> None:
        segment_list = self.SpeakerSegmentList.model_validate(segments)
        self._filepath.write_text(segment_list.json())

    def get_segment_list(self) -> list[SpeakerSegment]:
        if not self._filepath.exists():
            return list()

        segment_list = self.SpeakerSegmentList.parse_file(self._filepath)
        return segment_list.root

    def clean(self) -> None:
        if not self._filepath.exists():
            return

        self._filepath.unlink()
