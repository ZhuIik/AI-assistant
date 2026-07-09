"""Shared building block for chunking strategies: split lecture segments into
sentences, each carrying its own interpolated timecode.

Segments only carry start/end for the whole segment, not per-sentence. When a
segment contains multiple sentences, each sentence's timecode is estimated by
linearly interpolating across the segment's duration by character position -
an approximation, but good enough for citation purposes (nobody is jumping to
the exact millisecond).

Every chunking strategy (by-sentence, semantic, recursive, ...) should build
on top of `split_all_into_timed_sentences` and only vary how the resulting
timed sentences get grouped into blocks - the timecode logic itself doesn't
change per strategy.
"""

import re
from typing import Any

from scripts.build_annotation_drafts import normalize_text

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")


def format_timecode(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    hours, rem_ms = divmod(total_ms, 3_600_000)
    minutes, rem_ms = divmod(rem_ms, 60_000)
    secs, ms = divmod(rem_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def split_into_timed_sentences(segment: dict[str, Any]) -> list[dict[str, Any]]:
    """Split one Whisper segment into sentences with interpolated timecodes."""
    text = normalize_text(segment.get("text", ""))
    if not text:
        return []

    start_sec = float(segment["start_sec"])
    end_sec = float(segment["end_sec"])
    duration = max(0.0, end_sec - start_sec)

    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if not sentences:
        return []

    total_chars = sum(len(s) for s in sentences)
    results = []
    cursor = 0
    for sentence in sentences:
        frac_start = cursor / total_chars if total_chars else 0.0
        cursor += len(sentence)
        frac_end = cursor / total_chars if total_chars else 1.0
        results.append(
            {
                "text": sentence,
                "segment_id": segment["segment_id"],
                "speaker": segment.get("speaker"),
                "start_sec": start_sec + frac_start * duration,
                "end_sec": start_sec + frac_end * duration,
            }
        )
    return results


def split_all_into_timed_sentences(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten a lecture's segments into one timed-sentence stream, in order."""
    timed_sentences: list[dict[str, Any]] = []
    for segment in segments:
        timed_sentences.extend(split_into_timed_sentences(segment))
    return timed_sentences
