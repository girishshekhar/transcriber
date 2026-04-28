#!/usr/bin/env python3
"""
Local Audio Transcriber
Transcribes audio files using OpenAI Whisper (runs locally, no cloud services)
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

import whisper

_SILENCE_NOISE_THRESHOLD = "-35dB"
_MIN_KEPT_CHUNK_SECONDS = 5.0
_MODEL_CACHE: dict[str, Any] = {}


def _load_whisper_model(model_name: str) -> Any:
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = whisper.load_model(model_name)
    return _MODEL_CACHE[model_name]


def get_audio_duration(audio_file: str) -> float | None:
    """Get audio duration using ffprobe, returns None if unavailable."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", audio_file,
            ],
            capture_output=True, text=True, check=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def show_progress_with_stop(stop_event: threading.Event) -> None:
    chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    i = 0
    while not stop_event.is_set():
        print(f"\r{chars[i % len(chars)]} Processing audio...", end="", flush=True)
        time.sleep(0.1)
        i += 1


def detect_repetition(
    text: str, min_length: int = 10, threshold: float = 0.7
) -> tuple[bool, str, int]:
    """
    Detect repetitive n-gram patterns in text that indicate Whisper hallucination.

    Returns (is_repetitive, clean_text, cutoff_position).
    clean_text is truncated at the sentence boundary before repetition starts.
    """
    if len(text) < 100:
        return False, text, -1

    sentences = re.split(r"[.!?]+", text)
    if len(sentences) < 5:
        return False, text, -1

    last_quarter = text[-len(text) // 4 :]
    words = last_quarter.split()

    if len(words) < 20:
        return False, text, -1

    for n in range(3, min(10, len(words) // 3)):
        ngrams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
        ngram_counts = Counter(ngrams)
        most_common = ngram_counts.most_common(1)

        if most_common and most_common[0][1] >= 3:
            repeated_phrase = most_common[0][0]
            repetition_ratio = (most_common[0][1] * n) / len(words)

            if repetition_ratio > threshold:
                matches = list(re.finditer(re.escape(repeated_phrase), text, re.IGNORECASE))

                if len(matches) >= 3:
                    repetition_start = matches[0].start()
                    cutoff_pos = repetition_start
                    for i in range(repetition_start - 1, max(0, repetition_start - 500), -1):
                        if text[i] in ".!?":
                            cutoff_pos = i + 1
                            break
                    return True, text[:cutoff_pos].strip(), cutoff_pos

    return False, text, -1


def post_process_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Remove duplicate and near-duplicate segments that indicate Whisper looping.

    Stops processing when it finds 2+ consecutive segments with >80% word overlap
    or exact text matches.
    """
    if not segments:
        return segments

    cleaned: list[dict[str, Any]] = []
    seen_texts: set[str] = set()
    consecutive_similar = 0

    for i, segment in enumerate(segments):
        text = segment.get("text", "").strip()
        if not text:
            continue

        if text in seen_texts:
            consecutive_similar += 1
            if consecutive_similar >= 2:
                print(f"  Detected repeated segment, stopping at segment {i}")
                break
            continue
        else:
            consecutive_similar = 0

        is_similar = False
        for seen_text in list(seen_texts)[-5:]:
            if len(text) > 10 and len(seen_text) > 10:
                words1 = set(text.lower().split())
                words2 = set(seen_text.lower().split())
                if words1 and words2:
                    similarity = len(words1 & words2) / len(words1 | words2)
                    if similarity > 0.8:
                        is_similar = True
                        break

        if is_similar:
            consecutive_similar += 1
            if consecutive_similar >= 2:
                print(f"  Detected similar content repetition, stopping at segment {i}")
                break
            continue
        else:
            consecutive_similar = 0

        seen_texts.add(text)
        cleaned.append(segment)

        if len(seen_texts) > 50:
            seen_texts = set(list(seen_texts)[-25:])

    return cleaned


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"


def _detect_silence_spans(
    audio_path: Path, threshold_seconds: float
) -> list[tuple[float, float]]:
    result = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-i", str(audio_path),
            "-af", f"silencedetect=noise={_SILENCE_NOISE_THRESHOLD}:d={threshold_seconds}",
            "-f", "null", "-",
        ],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "ffmpeg silencedetect failed"
        raise RuntimeError(f"Unable to detect silence for {audio_path.name}: {message}")

    silence_spans: list[tuple[float, float]] = []
    silence_start: float | None = None
    start_pattern = re.compile(r"silence_start: ([0-9.]+)")
    end_pattern = re.compile(r"silence_end: ([0-9.]+) \| silence_duration: ([0-9.]+)")

    for line in result.stderr.splitlines():
        start_match = start_pattern.search(line)
        if start_match:
            silence_start = float(start_match.group(1))
            continue
        end_match = end_pattern.search(line)
        if end_match and silence_start is not None:
            silence_spans.append((silence_start, float(end_match.group(1))))
            silence_start = None

    return silence_spans


def _derive_keep_segments(
    duration_seconds: float,
    silence_spans: list[tuple[float, float]],
    min_chunk_seconds: float = _MIN_KEPT_CHUNK_SECONDS,
) -> list[tuple[float, float]]:
    keep_segments: list[tuple[float, float]] = []
    cursor = 0.0

    for silence_start, silence_end in sorted(silence_spans):
        if silence_start - cursor >= min_chunk_seconds:
            keep_segments.append((cursor, silence_start))
        cursor = max(cursor, silence_end)

    if duration_seconds - cursor >= min_chunk_seconds:
        keep_segments.append((cursor, duration_seconds))

    return keep_segments


def _is_full_file_segment(segment: tuple[float, float], duration_seconds: float) -> bool:
    return abs(segment[0]) < 0.001 and abs(segment[1] - duration_seconds) < 0.001


def _preprocess_audio_chunks(
    audio_path: Path, threshold_seconds: float
) -> list[tuple[float, float]] | None:
    """Return non-silent (start, end) segments, or None if chunking isn't needed."""
    try:
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path),
            ],
            capture_output=True, text=True, check=True,
        )
        duration_seconds = float(probe.stdout.strip())
    except Exception as exc:
        raise RuntimeError(f"Unable to inspect audio duration for {audio_path.name}") from exc

    silence_spans = _detect_silence_spans(audio_path, threshold_seconds)
    keep_segments = _derive_keep_segments(duration_seconds, silence_spans)

    if not keep_segments:
        return None
    if len(keep_segments) == 1 and _is_full_file_segment(keep_segments[0], duration_seconds):
        return None
    return keep_segments


def _extract_audio_chunk(
    source_path: Path,
    chunk_path: Path,
    start_seconds: float,
    duration_seconds: float,
) -> None:
    result = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{start_seconds:.6f}",
            "-i", str(source_path),
            "-t", f"{duration_seconds:.6f}",
            "-c", "copy",
            str(chunk_path),
        ],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "ffmpeg chunk extraction failed"
        raise RuntimeError(f"Unable to extract audio chunk from {source_path.name}: {message}")


def _merge_chunk_result(
    combined_segments: list[dict[str, Any]],
    chunk_result: dict[str, Any],
    start_offset_seconds: float,
) -> None:
    for segment in chunk_result.get("segments") or []:
        merged = dict(segment)
        merged["id"] = len(combined_segments)

        if isinstance(merged.get("start"), (int, float)):
            merged["start"] = float(merged["start"]) + start_offset_seconds
        if isinstance(merged.get("end"), (int, float)):
            merged["end"] = float(merged["end"]) + start_offset_seconds
        if isinstance(merged.get("seek"), (int, float)):
            merged["seek"] = int(round(float(merged["seek"]) + (start_offset_seconds * 100)))

        combined_segments.append(merged)


def _transcribe_with_silence_split(
    audio_path: Path,
    whisper_model: Any,
    options: dict[str, Any],
    threshold_seconds: float,
) -> dict[str, Any]:
    """Transcribe by splitting on silence, merging results with corrected timestamps."""
    keep_segments = _preprocess_audio_chunks(audio_path, threshold_seconds)
    if not keep_segments:
        return whisper_model.transcribe(str(audio_path), **options)

    print(f"  Silence split: {len(keep_segments)} non-silent chunks detected")

    combined_segments: list[dict[str, Any]] = []
    combined_text_parts: list[str] = []
    detected_language: str | None = None

    with tempfile.TemporaryDirectory(prefix="transcriber-") as temp_dir:
        temp_root = Path(temp_dir)

        for index, (start_seconds, end_seconds) in enumerate(keep_segments, 1):
            chunk_path = temp_root / f"chunk{index:02d}{audio_path.suffix.lower()}"
            _extract_audio_chunk(
                audio_path, chunk_path,
                start_seconds=start_seconds,
                duration_seconds=end_seconds - start_seconds,
            )

            chunk_result = whisper_model.transcribe(str(chunk_path), **options)

            chunk_text = (chunk_result.get("text") or "").strip()
            if not chunk_text:
                chunk_text = "\n".join(
                    s.get("text", "").strip()
                    for s in (chunk_result.get("segments") or [])
                    if s.get("text")
                ).strip()

            if chunk_text:
                combined_text_parts.append(chunk_text)
            if detected_language is None and chunk_result.get("language"):
                detected_language = str(chunk_result["language"])

            _merge_chunk_result(combined_segments, chunk_result, start_seconds)

    return {
        "language": detected_language,
        "segments": combined_segments,
        "text": " ".join(part for part in combined_text_parts if part).strip(),
    }


def transcribe_audio(
    audio_file: str,
    model_size: str = "base",
    output_format: str = "txt",
    fix_hallucinations: bool = False,
    silence_split_threshold: float | None = None,
) -> bool:
    """
    Transcribe audio file using Whisper.

    Args:
        audio_file: Path to audio file
        model_size: Whisper model size (tiny, base, small, medium, large)
        output_format: Output format (txt, srt, json)
        fix_hallucinations: Enable post-processing to detect/remove looping content
        silence_split_threshold: Split on silence longer than this many seconds
    """
    if not os.path.exists(audio_file):
        print(f"Error: Audio file '{audio_file}' not found")
        return False

    print(f"Loading audio file: {audio_file}")

    duration = get_audio_duration(audio_file)
    if duration:
        print(f"Audio duration: {duration / 60:.1f} minutes")
        time_multipliers = {"tiny": 0.1, "base": 0.2, "small": 0.4, "medium": 0.8, "large": 1.2}
        estimated_time = duration * time_multipliers.get(model_size, 0.2)
        print(f"Estimated transcription time: {estimated_time / 60:.1f} minutes")

    print(f"Using Whisper model: {model_size}")

    try:
        print(f"Loading Whisper model '{model_size}' (downloads on first use)...")
        model = _load_whisper_model(model_size)

        transcribe_kwargs: dict[str, Any] = dict(
            verbose=True,
            temperature=0,
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6,
        )

        device = str(getattr(model, "device", "cpu"))
        if device.startswith("cpu"):
            transcribe_kwargs["fp16"] = False

        if fix_hallucinations:
            transcribe_kwargs["condition_on_previous_text"] = False
            transcribe_kwargs["word_timestamps"] = True
            transcribe_kwargs["initial_prompt"] = None

        print("Starting transcription...")
        start_time = time.time()

        progress_stop = threading.Event()
        progress_thread = threading.Thread(
            target=show_progress_with_stop, args=(progress_stop,), daemon=True
        )
        progress_thread.start()

        if silence_split_threshold is not None:
            result = _transcribe_with_silence_split(
                Path(audio_file), model, transcribe_kwargs, silence_split_threshold
            )
        else:
            result = model.transcribe(audio_file, **transcribe_kwargs)

        progress_stop.set()
        print("\r" + " " * 50 + "\r", end="")

        elapsed = time.time() - start_time
        print(f"Transcription completed in {elapsed / 60:.1f} minutes")
        if duration:
            print(f"Processing speed: {duration / elapsed:.1f}x realtime")

        if fix_hallucinations:
            print("Running hallucination cleanup...")
            original_count = len(result.get("segments", []))
            cleaned_segments = post_process_segments(result.get("segments", []))
            removed = original_count - len(cleaned_segments)
            if removed:
                print(f"  Removed {removed} potentially hallucinated segments")

            cleaned_text = " ".join(s.get("text", "").strip() for s in cleaned_segments)
            is_repetitive, final_text, cutoff_pos = detect_repetition(cleaned_text)
            if is_repetitive:
                print(f"  Removed repetitive tail content (cutoff at char {cutoff_pos})")
                cleaned_text = final_text

            result["text"] = cleaned_text
            result["segments"] = cleaned_segments

        audio_path = Path(audio_file)
        output_file = audio_path.with_suffix(f".{output_format}")

        if output_format == "txt":
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result["text"])
        elif output_format == "srt":
            with open(output_file, "w", encoding="utf-8") as f:
                for i, segment in enumerate(result["segments"], 1):
                    start = format_timestamp(segment["start"])
                    end = format_timestamp(segment["end"])
                    text = segment["text"].strip()
                    f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
        elif output_format == "json":
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"Saved to: {output_file}")
        print(f"Detected language: {result.get('language', 'unknown')}")

        if output_format == "txt":
            word_count = len(result["text"].split())
            print(f"Word count: {word_count:,} words")

        return True

    except Exception as e:
        print(f"Error during transcription: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe audio files locally using Whisper (no cloud, no API key)"
    )
    parser.add_argument("audio_file", help="Path to audio file (m4a, mp3, wav, flac, etc.)")
    parser.add_argument(
        "-m", "--model", default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "-f", "--format", default="txt",
        choices=["txt", "srt", "json"],
        help="Output format (default: txt)",
    )
    parser.add_argument(
        "--fix-hallucinations", action="store_true",
        help=(
            "Enable post-processing to detect and remove Whisper looping/hallucination. "
            "Useful when output contains repeated phrases, but may occasionally drop "
            "valid content. Off by default."
        ),
    )
    parser.add_argument(
        "--silence-split", type=float, metavar="SECONDS",
        help=(
            "Split audio on silence longer than SECONDS before transcribing. "
            "Skips silent sections, reducing hallucinations on recordings with pauses. "
            "Recommended values: 1.0–3.0 seconds."
        ),
    )

    args = parser.parse_args()

    print("Local Audio Transcriber")
    print("=" * 40)

    model_info = {
        "tiny":   "~39MB  | fastest  | basic accuracy",
        "base":   "~74MB  | fast     | good accuracy  [default]",
        "small":  "~244MB | moderate | better accuracy",
        "medium": "~769MB | slow     | high accuracy",
        "large":  "~1.5GB | slowest  | best accuracy",
    }
    print(f"Model: {model_info.get(args.model, args.model)}")
    if args.fix_hallucinations:
        print("Hallucination cleanup: enabled")
    if args.silence_split is not None:
        print(f"Silence split: >{args.silence_split}s silence gaps will be skipped")
    print()

    success = transcribe_audio(
        args.audio_file,
        args.model,
        args.format,
        args.fix_hallucinations,
        args.silence_split,
    )

    if success:
        print("\nDone.")
        print("\nTips:")
        print("  Faster processing:    -m tiny")
        print("  Better accuracy:      -m medium or -m large")
        print("  Subtitle output:      -f srt")
        print("  Fix looping output:   --fix-hallucinations")
        print("  Skip silence gaps:    --silence-split 1.5")
    else:
        print("\nTranscription failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
