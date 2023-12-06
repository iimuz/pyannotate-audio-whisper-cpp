from pydantic import BaseModel


class SpeakerSegment(BaseModel):
    """話者分離の一つ分の結果."""

    start_time: float
    end_time: float
    speaker_name: str
