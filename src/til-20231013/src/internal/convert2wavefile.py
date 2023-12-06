import logging
import subprocess
from pathlib import Path

_logger = logging.getLogger(__name__)


class ConvertToWavFile:
    """指定したファイルをwav 16bitに変換する."""

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir

    def convert(self, filepath: Path) -> Path:
        output_filepath = self._output_dir / f"{filepath.stem}.wav"
        command_args = [
            "ffmpeg",
            "-y",  # 既存ファイルが存在する場合などで入力が必要となる部分を全てYesで進める
            "-i",
            f"{str(filepath.resolve())}",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output_filepath.resolve()),
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
                raise ValueError("error converting to wav file.")
        except Exception:
            proc.kill()
            raise
        _logger.info(f"stdout: {result}")
        _logger.warning(f"stderr: {result}")

        return output_filepath
