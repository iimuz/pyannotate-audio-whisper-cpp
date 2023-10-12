"""pyannote-audio v3で利用するモデルをhuggingfaceからダウンロードする."""
import logging
import os
import shutil
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from logging import Formatter, StreamHandler
from logging.handlers import RotatingFileHandler
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download, snapshot_download

_logger = logging.getLogger(__name__)


@dataclass
class _RunConfig:
    """スクリプト実行のためのオプション."""

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
    log_filepath = (
        Path("data/interim")
        / script_filepath.parent.name
        / f"{script_filepath.stem}.log"
    )
    log_filepath.parent.mkdir(exist_ok=True)
    _setup_logger(log_filepath, loglevel=loglevel)
    _logger.info(config)

    # hugging faceのtoken取得
    HF_TOKEN = os.environ.get("HUGGINGFACE_ACCESS_TOKEN", None)
    if HF_TOKEN is None:
        _logger.error("Could not get huggingface token.")
        raise ValueError("Could not get huggingface token.")

    # ファイルのダウンロード
    src_config_path = hf_hub_download(
        repo_id="pyannote/speaker-diarization-3.0",
        filename="config.yaml",
        use_auth_token=HF_TOKEN,
    )
    dst_config_path = Path("data/raw/config.yaml")
    shutil.copy(src_config_path, dst_config_path)
    _logger.info(f"copy config.yaml to {dst_config_path}")

    src_segmentation_model_path = hf_hub_download(
        repo_id="pyannote/segmentation-3.0",
        filename="pytorch_model.bin",
        use_auth_token=HF_TOKEN,
    )
    dst_segmentation_model_path = Path("data/raw/pytorch_model.bin")
    shutil.copy(src_segmentation_model_path, dst_segmentation_model_path)
    _logger.info(f"copy pytorch_model.bin to {dst_segmentation_model_path}")

    snapshot_download("hbredin/wespeaker-voxceleb-resnet34-LM")

    # config.yamlの書き換え
    with dst_config_path.open("r") as f:
        config_yaml = yaml.safe_load(f)
    config_yaml["pipeline"]["params"]["segmentation"] = str(dst_segmentation_model_path)
    with dst_config_path.open("w") as f:
        yaml.dump(config_yaml, f)
    _logger.info("convert config.yaml")


def _parse_args() -> _RunConfig:
    """スクリプト実行のための引数を読み込む."""
    parser = ArgumentParser(description="pyannote-audio v3を利用するために必要なモデルをダウンロードする.")

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
