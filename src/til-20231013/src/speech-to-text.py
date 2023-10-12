"""pyannote-audio v3を利用して話者分離を実施する."""
import json
import logging
import os
import re
import subprocess
import sys
from argparse import ArgumentParser
from enum import Enum
from logging import Formatter, StreamHandler
from logging.handlers import RotatingFileHandler
from pathlib import Path

import torch
from pyannote.audio import Pipeline
from pydantic import BaseModel
from pydub import AudioSegment

_logger = logging.getLogger(__name__)


class _DeviceType(Enum):
    """pytorchを利用するデバイス設定."""

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class _RunConfig(BaseModel):
    """スクリプト実行のためのオプション."""

    filepath: Path  # 処理対象の音源
    device: str  # デバイス

    force: bool  # 保存済みのファイルを無視して実行するかどうか
    verbose: int  # ログレベル


class _SpeakerSegment(BaseModel):
    """話者分離の一つ分の結果."""

    start_time: float
    end_time: float
    speaker_name: str


class _SpeakerText(BaseModel):
    """文字起こししたテキスト."""

    start_time: float
    end_time: float
    speaker_name: str
    text: str


class _SpeakerSegmentFile:
    """話者分離情報をファイルに保存する."""

    class SpeakerSegmentList(BaseModel):
        __root__: list[_SpeakerSegment]

    def __init__(self, filepath: Path) -> None:
        self._filepath = filepath

    def save(self, segments: list[_SpeakerSegment]) -> None:
        segment_list = self.SpeakerSegmentList(__root__=segments)
        self._filepath.write_text(segment_list.json())

    def get_segment_list(self) -> list[_SpeakerSegment]:
        if not self._filepath.exists():
            return list()

        segment_list = self.SpeakerSegmentList.parse_file(self._filepath)
        return segment_list.__root__

    def clean(self) -> None:
        if not self._filepath.exists():
            return

        self._filepath.unlink()


class _SpeakerSeperator:
    """音声から話者分離を行う."""

    def __init__(self, config_path: Path, device_name: str) -> None:
        self._config_path = config_path
        self._device_name = device_name

    def diarization(self, wav_filepath: Path):
        pipeline = Pipeline.from_pretrained(self._config_path)
        pipeline.to(torch.device(self._device_name))
        diarization = pipeline(wav_filepath)

        for segment, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segment = _SpeakerSegment(
                start_time=segment.start, end_time=segment.end, speaker_name=speaker
            )
            yield speaker_segment


class _SpeakerIntegrator:
    """話者分離情報をテキスト化しやすいように統合する."""

    def __init__(
        self,
        segment_duration_threshold: float = 60.0,
        split_segment_duration: float = 1.0,
        max_segment_duration: float = 120.0,
    ) -> None:
        self._segment_duration_threshold = segment_duration_threshold
        self._split_segment_duration = split_segment_duration
        self._max_segment_duration = max_segment_duration

    def integrate(self, segments: list[_SpeakerSegment]) -> list[_SpeakerSegment]:
        """連続する話者区間を統合する."""
        if len(segments) < 1:
            return list()

        new_segments: list[_SpeakerSegment] = [segments[0]]
        is_force_split = False
        for segment in segments[1:]:
            current_segment = new_segments[-1]

            # 強制的に分割する場合
            if is_force_split:
                new_segments.append(segment)
                is_force_split = False
                continue

            # 話者が変わった場合は分割
            if current_segment.speaker_name != segment.speaker_name:
                new_segments.append(segment)
                is_force_split = False
                continue

            # 閾値以下であればsegmentを統合
            total_duration = segment.end_time - current_segment.start_time
            if total_duration < self._segment_duration_threshold:
                current_segment.end_time = segment.end_time
                is_force_split = False
                continue

            # しきい値を超えている場合に短い区間だったら統合して次で分割
            segment_duration = segment.end_time - segment.start_time
            if segment_duration < self._split_segment_duration:
                current_segment.end_time = segment.end_time
                is_force_split = True
                continue

            # maxより小さければ統合
            if total_duration < self._max_segment_duration:
                current_segment.end_time = segment.end_time
                is_force_split = False
                continue

            # maxを超えていて統合できるポイントがなかったので強制分割
            new_segments.append(segment)
            is_force_split = False

        return new_segments


class _SpeechToText:
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
        self, sound: AudioSegment, segments: list[_SpeakerSegment]
    ) -> list[_SpeakerText]:
        """指定したwavファイルをテキスト化する."""
        speaker_text_list: list[_SpeakerText] = list()
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
            speaker_text = _SpeakerText(
                start_time=segment.start_time,
                end_time=segment.end_time,
                speaker_name=segment.speaker_name,
                text=text,
            )
            speaker_text_list.append(speaker_text)
            _logger.info(
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
            result, error = proc.communicate(timeout=300)
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


class _SpeechIntegrator:
    """書き起こした文字情報の整形を行う."""

    def __init__(self) -> None:
        pass

    def integrate(self, speaker_text: list[_SpeakerText]) -> list[_SpeakerText]:
        # テキストがない区間は削除する
        remove_redundant_text: list[_SpeakerText] = [
            segment for segment in speaker_text if segment.text != ""
        ]

        # 同一話者区間は一つのブロックにまとめる
        integrated_speaker_text = [remove_redundant_text[0]]
        for segment in remove_redundant_text[1:]:
            current_text = integrated_speaker_text[-1]
            if current_text.speaker_name != segment.speaker_name:
                integrated_speaker_text.append(segment)
                continue

            current_text.end_time = segment.end_time
            current_text.text += os.linesep + segment.text

        return integrated_speaker_text


class _SpeechTextFile:
    """話者ごとのテキストをファイル保存する."""

    class SpeechTextList(BaseModel):
        __root__: list[_SpeakerText]

    def __init__(self, filepath: Path) -> None:
        self._filepath = filepath

    def save(self, segments: list[_SpeakerText]) -> None:
        segment_list = self.SpeechTextList(__root__=segments)
        self._filepath.write_text(segment_list.json())

    def get_segment_list(self) -> list[_SpeakerText]:
        if not self._filepath.exists():
            return list()

        segment_list = self.SpeechTextList.parse_file(self._filepath)
        return segment_list.__root__

    def clean(self) -> None:
        if not self._filepath.exists():
            return

        self._filepath.unlink()


class _SpeechTextMarkdown:
    """書き起こした文字をテキストで保存する."""

    def __init__(self, filepath: Path) -> None:
        self._filepath = filepath

    def save(self, speaker_text: list[_SpeakerText]) -> None:
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


def _main() -> None:
    """スクリプトのエントリポイント."""
    # 実行時引数の読み込み
    config = _parse_args()

    # ログ設定
    loglevel = {
        0: logging.ERROR,
        1: logging.WARNING,
        2: logging.INFO,
        3: logging.DEBUG,
    }.get(config.verbose, logging.DEBUG)
    script_filepath = Path(__file__)
    log_filepath = Path("data/interim") / f"{script_filepath.stem}.log"
    log_filepath.parent.mkdir(exist_ok=True)
    _setup_logger(log_filepath, loglevel=loglevel)
    _logger.info(config)

    # データフォルダ
    raw_dir = Path("data/raw")
    interim_dir = Path("data/interim") / config.filepath.stem
    interim_dir.mkdir(exist_ok=True)
    processed_dir = Path("data/processed") / config.filepath.stem
    processed_dir.mkdir(exist_ok=True)
    model_config_filepath = raw_dir / "config.yaml"

    # 話者分離情報の取得
    _logger.info("calc speaker segment ...")
    speaker_segment_file = _SpeakerSegmentFile(
        filepath=(interim_dir / "speaker_segment.json")
    )
    if config.force:
        speaker_segment_file.clean()
    speaker_segments = speaker_segment_file.get_segment_list()
    if len(speaker_segments) < 1:
        speaker_seperator = _SpeakerSeperator(
            config_path=model_config_filepath, device_name=config.device
        )
        for segment in speaker_seperator.diarization(wav_filepath=config.filepath):
            _logger.info(
                f"[{segment.start_time:03.1f}s - {segment.end_time:03.1f}s]"
                f" {segment.speaker_name}"
            )
            speaker_segments.append(segment)
        speaker_segment_file.save(speaker_segments)

    # 話者区間の統合
    _logger.info("calc integrated speaker segment ...")
    speaker_integrate_file = _SpeakerSegmentFile(
        filepath=(interim_dir / "speaker_segment_integrate.json")
    )
    if config.force:
        speaker_integrate_file.clean()
    integrated_segments = speaker_integrate_file.get_segment_list()
    if len(integrated_segments) < 1:
        speaker_integrator = _SpeakerIntegrator(
            segment_duration_threshold=60.0,
            split_segment_duration=1.0,
            max_segment_duration=120.0,
        )
        integrated_segments = speaker_integrator.integrate(speaker_segments)
        speaker_integrate_file.save(integrated_segments)

    # Speech to Text
    _logger.info("speech to text ...")
    speech_text_file = _SpeechTextFile(filepath=(interim_dir / "speech_text.json"))
    if config.force:
        speech_text_file.clean()
    speaker_text_list = speech_text_file.get_segment_list()
    if len(speaker_text_list) < 1:
        sound: AudioSegment = AudioSegment.from_wav(config.filepath)
        speech_to_text = _SpeechToText(
            wav_dirpath=interim_dir,
            whisper_cpp_path=Path("whisper.cpp"),
            model_name="large",
        )
        speaker_text_list = speech_to_text.to_text(
            sound=sound, segments=integrated_segments
        )
        speech_text_file.save(speaker_text_list)

    # 冗長なテキストなどの除去
    _logger.info("integrate text ...")
    speech_integrate_file = _SpeechTextFile(
        filepath=(interim_dir / "speech_integrate_text.json")
    )
    if config.force:
        speech_integrate_file.clean()
    integrated_text = speech_integrate_file.get_segment_list()
    if len(integrated_text) < 1:
        speech_integrator = _SpeechIntegrator()
        integrated_text = speech_integrator.integrate(speaker_text_list)
        speech_integrate_file.save(integrated_text)

    # ファイル出力
    speech_md_file = _SpeechTextMarkdown(filepath=(processed_dir / "speech.md"))
    if config.force:
        speech_md_file.clean()
    speech_md_file.save(integrated_text)


def _parse_args() -> _RunConfig:
    """スクリプト実行のための引数を読み込む."""
    parser = ArgumentParser(description="pyannote-audio v3を利用して話者分離を実施する.")

    parser.add_argument("filepath", help="文字起こしする音源のファイルパス.")
    parser.add_argument(
        "-d",
        "--device",
        default=_DeviceType.CPU.value,
        choices=[v.value for v in _DeviceType],
        help="話者分離に利用するデバイス.",
    )

    parser.add_argument(
        "-f", "--force", action="store_true", help="算出済みの結果を無視して実行するかどうか."
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="詳細メッセージのレベルを設定."
    )

    args = parser.parse_args()
    config = _RunConfig(**vars(args))

    return config


def _setup_logger(
    filepath: Path | None,  # ログ出力するファイルパス. Noneの場合はファイル出力しない.
    loglevel: int,  # 出力するログレベル
) -> None:
    """ログ出力設定

    Notes
    -----
    ファイル出力とコンソール出力を行うように設定する。
    """
    script_path = Path(__file__)
    lib_logger = logging.getLogger(f"src.{script_path.stem}")

    _logger.setLevel(loglevel)
    lib_logger.setLevel(loglevel)

    # consoleログ
    console_handler = StreamHandler()
    console_handler.setLevel(loglevel)
    console_handler.setFormatter(
        Formatter("[%(levelname)7s] %(asctime)s (%(name)s) %(message)s")
    )
    _logger.addHandler(console_handler)
    lib_logger.addHandler(console_handler)

    # ファイル出力するログ
    # 基本的に大量に利用することを想定していないので、ログファイルは多くは残さない。
    if filepath is not None:
        file_handler = RotatingFileHandler(
            filepath,
            encoding="utf-8",
            mode="a",
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=1,
        )
        file_handler.setLevel(loglevel)
        file_handler.setFormatter(
            Formatter("[%(levelname)7s] %(asctime)s (%(name)s) %(message)s")
        )
        _logger.addHandler(file_handler)
        lib_logger.addHandler(file_handler)


if __name__ == "__main__":
    try:
        _main()
    except Exception:
        _logger.exception("Exception")
        sys.exit(1)
