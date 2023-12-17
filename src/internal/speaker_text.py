from pydantic import BaseModel


class SpeakerText(BaseModel):
    """文字起こししたテキスト."""

    start_time: float
    end_time: float
    speaker_name: str
    text: str
