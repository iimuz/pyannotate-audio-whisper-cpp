"""フォルダ内を探索して音声ファイルに対して文字起こしを実施する."""
import logging
import shutil
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
    SpeakerSegment,
    SpeakerSegmentFile,
    SpeakerSeperator,
    SpeakerText,
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

    root_dir: Path  # 処理対象を探索するルートフォルダ
    device: str  # デバイス

    force: bool  # 保存済みのファイルを無視して実行するかどうか
    verbose: int  # ログレベル


def _calc_speaker_segment(
    wav_filepath: Path,
    model_config_filepath: Path,
    device: str,
    segment_filepath: Path,
    force: bool = False,
) -> list[SpeakerSegment]:
    speaker_segment_file = SpeakerSegmentFile(filepath=segment_filepath)
    speaker_segments = speaker_segment_file.get_segment_list()
    if (not force) and (len(speaker_segments) > 0):
        # 既存データがあり、強制再計算でなければ算出済みの結果を返す
        return speaker_segments

    speaker_seperator = SpeakerSeperator(
        config_path=model_config_filepath, device_name=device
    )
    for segment in speaker_seperator.diarization(wav_filepath=wav_filepath):
        _logger.debug(
            f"[{segment.start_time:03.1f}s - {segment.end_time:03.1f}s]"
            f" {segment.speaker_name}"
        )
        speaker_segments.append(segment)

    # 計算結果を保存
    speaker_segment_file.save(speaker_segments)

    return speaker_segments


def _convert_to_wav_file(source_filepath: Path, output_dir: Path) -> Path:
    """wavファイルへ変換し、変換後のファイルパスを返す."""
    if source_filepath.suffix == ".wav":
        return source_filepath

    convert_to_wavfile = ConvertToWavFile(output_dir=output_dir)
    target_filepath = convert_to_wavfile.convert(source_filepath)

    return target_filepath


def _integrate_speaker(
    speaker_segments: list[SpeakerSegment],
    speaker_segment_filepath: Path,
    force: bool = False,
) -> list[SpeakerSegment]:
    speaker_integrate_file = SpeakerSegmentFile(speaker_segment_filepath)
    integrated_segments = speaker_integrate_file.get_segment_list()
    if (not force) and (len(integrated_segments) > 0):
        # 強制再計算ではなく、計算済みの結果があれば算出済みの結果を返す
        return integrated_segments

    speaker_integrator = SpeakerIntegrator(
        segment_duration_threshold=60.0,
        split_segment_duration=1.0,
        max_segment_duration=120.0,
    )
    integrated_segments = speaker_integrator.integrate(speaker_segments)

    # 計算結果を保存
    speaker_integrate_file.save(integrated_segments)

    return integrated_segments


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

    filepath_list = config.root_dir.glob("**/*.mp4")
    for filepath in filepath_list:
        _logger.info(f"target file: {filepath}")

        # データフォルダ
        raw_dir = Path("data/raw")
        interim_dir = Path("data/interim") / filepath.stem
        interim_dir.mkdir(exist_ok=True)
        processed_dir = Path("data/processed") / filepath.stem
        processed_dir.mkdir(exist_ok=True)

        # 入力用の固定値
        model_config_filepath = raw_dir / "config.yaml"

        # 出力ファイル情報
        segment_filepath = interim_dir / "speaker_segment.json"
        speaker_segment_filepath = interim_dir / "speaker_segment_integrate.json"
        speach_text_filepath = interim_dir / "speech_text.json"
        integrated_text_filepath = interim_dir / "speech_integrate_text.json"
        result_filepath = processed_dir / f"{filepath.stem}.txt"
        dst_filepath = filepath.parent / f"{filepath.stem}.txt"

        # 既に出力ファイルがある場合は処理を行わない
        if dst_filepath.exists():
            _logger.info("speech text is already exist. skip.")
            continue

        # wavファイルへの変更
        _logger.info(f"convert to wav file: {filepath.name}")
        target_filepath = _convert_to_wav_file(filepath, interim_dir)
        # 話者分離情報の取得
        _logger.info("calc speaker segment ...")
        speaker_segments = _calc_speaker_segment(
            wav_filepath=target_filepath,
            model_config_filepath=model_config_filepath,
            device=config.device,
            segment_filepath=segment_filepath,
            force=config.force,
        )
        # 話者区間の統合
        _logger.info("calc integrated speaker segment ...")
        integrated_segements = _integrate_speaker(
            speaker_segments=speaker_segments,
            speaker_segment_filepath=speaker_segment_filepath,
            force=config.force,
        )
        # Speech to Text
        _logger.info("speech to text ...")
        speech_text_list = _speach_to_text(
            wav_fileapth=target_filepath,
            speaker_segments=integrated_segements,
            speech_text_filepath=speach_text_filepath,
            temp_dir=interim_dir,
            force=config.force,
        )
        # 冗長なテキストなどの除去
        _logger.info("integrate text ...")
        integrated_text = _remove_redundant_text(
            speaker_texts=speech_text_list,
            save_filepath=integrated_text_filepath,
            force=config.force,
        )
        # ファイル出力
        speech_md_file = SpeechTextWriter(filepath=result_filepath)
        if config.force:
            speech_md_file.clean()
        speech_md_file.save(integrated_text)
        # 入力ファイルのフォルダに出力ファイルを生成
        _logger.info(f"save speech text: {dst_filepath}")
        shutil.copy(result_filepath, dst_filepath)
        # 中間ファイルを削除
        shutil.rmtree(interim_dir)
        result_filepath.unlink()
        shutil.rmtree(processed_dir)


def _parse_args() -> _RunConfig:
    """スクリプト実行のための引数を読み込む."""
    parser = ArgumentParser(description="pyannote-audio v3を利用して話者分離を実施する.")

    parser.add_argument("root_dir", help="文字起こしする音源を探索するルートフォルダ.")
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


def _remove_redundant_text(
    speaker_texts: list[SpeakerText], save_filepath: Path, force: bool = False
) -> list[SpeakerText]:
    speech_integrate_file = SpeechTextFile(save_filepath)
    integrated_text = speech_integrate_file.get_segment_list()
    if (not force) and (len(integrated_text) > 0):
        # 計算済みの結果があれば再計算しない
        return integrated_text

    speech_integrator = SpeechIntegrator()
    integrated_text = speech_integrator.integrate(speaker_text=speaker_texts)

    speech_integrate_file.save(integrated_text)

    return integrated_text


def _setup_logger(
    filepath: Path | None,  # ログ出力するファイルパス. Noneの場合はファイル出力しない.
    loglevel: int,  # 出力するログレベル
) -> None:
    """ログ出力設定

    Notes
    -----
    ファイル出力とコンソール出力を行うように設定する。
    """
    lib_logger = logging.getLogger("interim")

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


def _speach_to_text(
    wav_fileapth: Path,
    speaker_segments: list[SpeakerSegment],
    speech_text_filepath: Path,
    temp_dir: Path,
    force: bool = False,
) -> list[SpeakerText]:
    speech_text_file = SpeechTextFile(speech_text_filepath)
    speaker_text_list = speech_text_file.get_segment_list()
    if (not force) and (len(speaker_text_list) > 0):
        # 強制再計算ではなく、既存の結果を読み込める場合は、計算済みの結果を返す
        return speaker_text_list

    sound: AudioSegment = AudioSegment.from_wav(wav_fileapth)
    speech_to_text = SpeechToText(
        wav_dirpath=temp_dir,
        whisper_cpp_path=Path("whisper.cpp"),
        model_name="large",
    )
    speaker_text_list = speech_to_text.to_text(sound=sound, segments=speaker_segments)

    speech_text_file.save(speaker_text_list)

    return speaker_text_list


if __name__ == "__main__":
    try:
        _main()
    except Exception:
        _logger.exception("Exception")
        sys.exit(1)
