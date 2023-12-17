from internal.speaker_segment import SpeakerSegment


class SpeakerIntegrator:
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

    def integrate(self, segments: list[SpeakerSegment]) -> list[SpeakerSegment]:
        """連続する話者区間を統合する."""
        if len(segments) < 1:
            return list()

        new_segments: list[SpeakerSegment] = [segments[0]]
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
