import ffmpeg
from datetime import timedelta
import os

from constants import TOKEN_END_KEY, TOKEN_KEY, TOKEN_ONSET_KEY


def format_timestamp(seconds):
    """Format timestamp for subtitles."""
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = td.microseconds // 1000
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def json_to_srt(asr_json):
    """Convert timed ASR results to SRT subtitle format."""
    srt_entries = []
    for i, entry in enumerate(asr_json, start=1):
        start_time = format_timestamp(entry[TOKEN_ONSET_KEY])
        end_time = format_timestamp(entry[TOKEN_END_KEY])
        text = entry[TOKEN_KEY]
        srt_entries.append(f"{i}\n{start_time} --> {end_time}\n{text}\n")
    
    return "\n".join(srt_entries)


def write_srt(str_string, outfile):
    """Write subtitles SRT string to outfile."""
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, "w") as f:
        f.write(str_string)


def add_subtitles_to_video(input_video: str,
                           subtitle_file: str,
                           subtitle_language: str="",
                           soft_subtitle: bool=True):
    """Add subtitles to a video."""
    video_dirname = os.path.dirname(input_video)
    input_video_name, _ = os.path.splitext(os.path.basename(input_video))
    subtitle_name, _ = os.path.splitext(os.path.basename(subtitle_file))
    video_input_stream = ffmpeg.input(input_video)
    subtitle_input_stream = ffmpeg.input(subtitle_file)
    output_video = os.path.join(video_dirname, f"{input_video_name}_subtitled-{subtitle_name}.mp4")
    subtitle_track_title = subtitle_file.replace(".srt", "")

    if soft_subtitle:
        stream = ffmpeg.output(
            video_input_stream, subtitle_input_stream, output_video, **{"c": "copy", "c:s": "mov_text"},
            **{"metadata:s:s:0": f"language={subtitle_language}",
            "metadata:s:s:0": f"title={subtitle_track_title}"}
        )
        ffmpeg.run(stream, overwrite_output=True)
    else:
        stream = ffmpeg.output(video_input_stream, output_video,
                               vf=f"subtitles={subtitle_file}")
        ffmpeg.run(stream, overwrite_output=True)
