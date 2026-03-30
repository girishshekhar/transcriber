#!/usr/bin/env python3
"""
Local Audio Transcriber
Transcribes audio files using OpenAI Whisper (runs locally, no cloud services)
"""

import whisper
import argparse
import os
import sys
import json
import re
from pathlib import Path
from collections import Counter
import time
import warnings
import threading

def get_audio_duration(audio_file):
    """Get audio duration using ffprobe if available, otherwise return None"""
    try:
        import subprocess
        result = subprocess.run([
            'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', audio_file
        ], capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except:
        return None

def show_progress_with_stop(stop_event):
    """Show a simple progress indicator that can be stopped"""
    chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    i = 0
    while not stop_event.is_set():
        print(f"\r{chars[i % len(chars)]} Processing audio...", end="", flush=True)
        time.sleep(0.1)
        i += 1

def detect_repetition(text, min_length=10, threshold=0.7):
    """
    Detect repetitive n-gram patterns in text that indicate Whisper hallucination.

    Returns (is_repetitive, clean_text, cutoff_position).
    clean_text is truncated at the sentence boundary before repetition starts.
    """
    if len(text) < 100:
        return False, text, -1

    sentences = re.split(r'[.!?]+', text)
    if len(sentences) < 5:
        return False, text, -1

    last_quarter = text[-len(text)//4:]
    words = last_quarter.split()

    if len(words) < 20:
        return False, text, -1

    for n in range(3, min(10, len(words)//3)):
        ngrams = [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]
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
                        if text[i] in '.!?':
                            cutoff_pos = i + 1
                            break
                    return True, text[:cutoff_pos].strip(), cutoff_pos

    return False, text, -1

def post_process_segments(segments):
    """
    Remove duplicate and near-duplicate segments that indicate Whisper looping.

    Stops processing when it finds 2+ consecutive segments with >80% word overlap
    or exact text matches.
    """
    if not segments:
        return segments

    cleaned = []
    seen_texts = set()
    consecutive_similar = 0

    for i, segment in enumerate(segments):
        text = segment.get('text', '').strip()
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

def format_timestamp(seconds):
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

def transcribe_audio(audio_file, model_size="base", output_format="txt", fix_hallucinations=False):
    """
    Transcribe audio file using Whisper.

    Args:
        audio_file: Path to audio file
        model_size: Whisper model size (tiny, base, small, medium, large)
        output_format: Output format (txt, srt, json)
        fix_hallucinations: Enable post-processing to detect and remove looping/hallucinated content
    """
    if not os.path.exists(audio_file):
        print(f"Error: Audio file '{audio_file}' not found")
        return False

    warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

    print(f"Loading audio file: {audio_file}")

    duration = get_audio_duration(audio_file)
    if duration:
        print(f"Audio duration: {duration/60:.1f} minutes")
        time_multipliers = {"tiny": 0.1, "base": 0.2, "small": 0.4, "medium": 0.8, "large": 1.2}
        estimated_time = duration * time_multipliers.get(model_size, 0.2)
        print(f"Estimated transcription time: {estimated_time/60:.1f} minutes")

    print(f"Using Whisper model: {model_size}")

    try:
        print(f"Loading Whisper model '{model_size}' (downloads on first use)...")
        model = whisper.load_model(model_size)

        print("Starting transcription...")
        start_time = time.time()

        progress_stop = threading.Event()
        progress_thread = threading.Thread(
            target=show_progress_with_stop, args=(progress_stop,), daemon=True
        )
        progress_thread.start()

        transcribe_kwargs = dict(
            verbose=True,
            temperature=0,
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6,
        )

        if fix_hallucinations:
            # These params reduce hallucination risk but can occasionally affect output quality
            transcribe_kwargs["condition_on_previous_text"] = False
            transcribe_kwargs["word_timestamps"] = True
            transcribe_kwargs["initial_prompt"] = None

        result = model.transcribe(audio_file, **transcribe_kwargs)

        progress_stop.set()
        print("\r" + " " * 50 + "\r", end="")

        elapsed = time.time() - start_time
        print(f"Transcription completed in {elapsed/60:.1f} minutes")
        if duration:
            print(f"Processing speed: {duration/elapsed:.1f}x realtime")

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
        output_file = audio_path.with_suffix(f'.{output_format}')

        if output_format == "txt":
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result["text"])
        elif output_format == "srt":
            with open(output_file, 'w', encoding='utf-8') as f:
                for i, segment in enumerate(result["segments"], 1):
                    start = format_timestamp(segment["start"])
                    end = format_timestamp(segment["end"])
                    text = segment["text"].strip()
                    f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
        elif output_format == "json":
            with open(output_file, 'w', encoding='utf-8') as f:
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

def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio files locally using Whisper (no cloud, no API key)"
    )
    parser.add_argument("audio_file", help="Path to audio file (m4a, mp3, wav, flac, etc.)")
    parser.add_argument(
        "-m", "--model", default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)"
    )
    parser.add_argument(
        "-f", "--format", default="txt",
        choices=["txt", "srt", "json"],
        help="Output format (default: txt)"
    )
    parser.add_argument(
        "--fix-hallucinations", action="store_true",
        help=(
            "Enable post-processing to detect and remove Whisper looping/hallucination. "
            "Useful when output contains repeated phrases, but may occasionally drop "
            "valid content. Off by default."
        )
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
    print()

    success = transcribe_audio(args.audio_file, args.model, args.format, args.fix_hallucinations)

    if success:
        print("\nDone.")
        print("\nTips:")
        print("  Faster processing:    -m tiny")
        print("  Better accuracy:      -m medium or -m large")
        print("  Subtitle output:      -f srt")
        print("  Fix looping output:   --fix-hallucinations")
    else:
        print("\nTranscription failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
