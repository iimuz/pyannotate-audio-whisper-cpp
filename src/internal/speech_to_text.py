import logging
import os
import re
import subprocess
from pathlib import Path

from pydub import AudioSegment

from internal.speaker_segment import SpeakerSegment
from internal.speaker_text import SpeakerText

_logger = logging.getLogger(__name__)


class SpeechToText:
    def __init__(
        self, wav_dirpath: Path, whisper_cpp_path: Path, model_name: str
    ) -> None:
        self._wav_dirpath = wav_dirpath
        self._whisper_cpp_path = whisper_cpp_path.resolve()

        self._re_query = re.compile(
            r"(\[\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}])\s+(.+)"
        )
        self._whisper_cpp_venv = self._whisper_cpp_path / ".venv/bin"
        self._whisper_command = [
            str(whisper_cpp_path / "main"),
            "-m",
            str(whisper_cpp_path / f"models/ggml-{model_name}.bin"),
            "-l",
            "ja",
        ]

    def to_text(
        self, sound: AudioSegment, segments: list[SpeakerSegment]
    ) -> list[SpeakerText]:
        """指定したwavファイルをテキスト化する."""
        speaker_text_list: list[SpeakerText] = list()
        for segment in segments:
            start_time = int(segment.start_time * 1000)  # s -> ms
            end_time = int(segment.end_time * 1000)  # s -> ms
            sound_segment = sound[start_time:end_time]
            try:
                text = self._sound_segment_to_text(sound_segment)
            except Exception:
                _logger.exception(
                    "Unhandled exception in speech to text. continue..."
                    f" [{segment.start_time:03.1f}s - {segment.end_time:03.1f}s]"
                    f" {segment.speaker_name}"
                )
                continue
            speaker_text = SpeakerText(
                start_time=segment.start_time,
                end_time=segment.end_time,
                speaker_name=segment.speaker_name,
                text=text,
            )
            speaker_text_list.append(speaker_text)
            _logger.debug(
                f"[{segment.start_time:03.1f}s - {segment.end_time:03.1f}s]"
                f" {segment.speaker_name} : {text}"
            )

        return speaker_text_list

    def _sound_segment_to_text(self, sound: AudioSegment) -> str:
        """1つ分のオーディオデータをテキスト化する."""
        wav_filepath = self._wav_dirpath / "cut_export.wav"
        sound.export(wav_filepath, format="wav")

        old_path = os.environ["PATH"]
        os.environ[
            "PATH"
        ] = f"{self._whisper_cpp_path}/.venv/bin:{os.environ.get('PATH')}"
        command_args = [
            *self._whisper_command,
            "-f",
            str(wav_filepath),
        ]
        proc = subprocess.Popen(
            command_args,
            encoding="utf-8",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            result, error = proc.communicate(timeout=1800)
            if proc.returncode != 0:
                _logger.error(f"command failed with exit status {proc.returncode}")
                _logger.error(error)
                raise ValueError("speech to text error.")
        except Exception:
            proc.kill()
            raise
        os.environ["PATH"] = old_path
        wav_filepath.unlink()

        result_lines = result.splitlines()
        match_list: list[str] = list()
        for line in result_lines:
            line_str = self._re_query.search(line)
            if line_str is None:
                continue
            if len(line_str.groups()) != 2:
                _logger.warning(f"skip match string: num={len(line_str.groups())}")
                continue
            match_list.append(line_str.group(2))
        match_str = " ".join(match_list)

        return match_str
