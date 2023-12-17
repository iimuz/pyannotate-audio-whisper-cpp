import os

from internal.speaker_text import SpeakerText


class SpeechIntegrator:
    """書き起こした文字情報の整形を行う."""

    def __init__(self) -> None:
        pass

    def integrate(self, speaker_text: list[SpeakerText]) -> list[SpeakerText]:
        # テキストがない区間は削除する
        remove_redundant_text: list[SpeakerText] = [
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
