"""pyannote-audio v3を利用して話者分離を実施する."""
import logging
import sys
from argparse import ArgumentParser
from enum import Enum
from logging import Formatter, StreamHandler
from logging.handlers import RotatingFileHandler
from pathlib import Path

from pydantic import BaseModel
from pydub import AudioSegment

from internal import (
    ConvertToWavFile,
    SpeakerIntegrator,
    SpeakerSegmentFile,
    SpeakerSeperator,
    SpeechIntegrator,
    SpeechTextFile,
    SpeechTextWriter,
    SpeechToText,
)

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

    # wavファイルへの変更
    target_filepath = config.filepath
    if target_filepath.suffix != ".wav":
        _logger.info(f"convert to wav file: {target_filepath.name}")
        convert_to_wavfile = ConvertToWavFile(output_dir=interim_dir)
        target_filepath = convert_to_wavfile.convert(target_filepath)

    # 話者分離情報の取得
    _logger.info("calc speaker segment ...")
    speaker_segment_file = SpeakerSegmentFile(
        filepath=(interim_dir / "speaker_segment.json")
    )
    if config.force:
        speaker_segment_file.clean()
    speaker_segments = speaker_segment_file.get_segment_list()
    if len(speaker_segments) < 1:
        speaker_seperator = SpeakerSeperator(
            config_path=model_config_filepath, device_name=config.device
        )
        for segment in speaker_seperator.diarization(wav_filepath=target_filepath):
            _logger.info(
                f"[{segment.start_time:03.1f}s - {segment.end_time:03.1f}s]"
                f" {segment.speaker_name}"
            )
            speaker_segments.append(segment)
        speaker_segment_file.save(speaker_segments)

    # 話者区間の統合
    _logger.info("calc integrated speaker segment ...")
    speaker_integrate_file = SpeakerSegmentFile(
        filepath=(interim_dir / "speaker_segment_integrate.json")
    )
    if config.force:
        speaker_integrate_file.clean()
    integrated_segments = speaker_integrate_file.get_segment_list()
    if len(integrated_segments) < 1:
        speaker_integrator = SpeakerIntegrator(
            segment_duration_threshold=60.0,
            split_segment_duration=1.0,
            max_segment_duration=120.0,
        )
        integrated_segments = speaker_integrator.integrate(speaker_segments)
        speaker_integrate_file.save(integrated_segments)

    # Speech to Text
    _logger.info("speech to text ...")
    speech_text_file = SpeechTextFile(filepath=(interim_dir / "speech_text.json"))
    if config.force:
        speech_text_file.clean()
    speaker_text_list = speech_text_file.get_segment_list()
    if len(speaker_text_list) < 1:
        sound: AudioSegment = AudioSegment.from_wav(target_filepath)
        speech_to_text = SpeechToText(
            wav_dirpath=interim_dir,
            whisper_cpp_path=Path("whisper.cpp"),
            model_name="large-v3",
        )
        speaker_text_list = speech_to_text.to_text(
            sound=sound, segments=integrated_segments
        )
        speech_text_file.save(speaker_text_list)

    # 冗長なテキストなどの除去
    _logger.info("integrate text ...")
    speech_integrate_file = SpeechTextFile(
        filepath=(interim_dir / "speech_integrate_text.json")
    )
    if config.force:
        speech_integrate_file.clean()
    integrated_text = speech_integrate_file.get_segment_list()
    if len(integrated_text) < 1:
        speech_integrator = SpeechIntegrator()
        integrated_text = speech_integrator.integrate(speaker_text_list)
        speech_integrate_file.save(integrated_text)

    # ファイル出力
    speech_md_file = SpeechTextWriter(
        filepath=(processed_dir / f"{target_filepath.stem}.txt")
    )
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
    lib_logger = logging.getLogger("internal")

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
