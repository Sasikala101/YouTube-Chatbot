"""Validate video links and retrieve timed English captions."""

import re
from urllib.parse import parse_qs, urlsplit

from requests import Session
from youtube_transcript_api import YouTubeTranscriptApi


def extract_video_id(url: str) -> str:
    parsed = urlsplit(url.strip())
    if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password:
        raise ValueError("Paste a full YouTube video link starting with https://.")
    parts = parsed.path.strip("/").split("/")
    host = (parsed.hostname or "").lower()
    video_id = ""
    if host == "youtu.be" and len(parts) == 1:
        video_id = parts[0]
    elif host in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        elif len(parts) == 2 and parts[0] in {"shorts", "embed", "live"}:
            video_id = parts[1]
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise ValueError("Use a valid YouTube watch, share, Shorts, or live video link.")
    return video_id


class TimeoutSession(Session):
    def request(self, method, url, **kwargs):
        kwargs.setdefault("timeout", (5, 15))
        return super().request(method, url, **kwargs)


def fetch_transcript(video_id: str) -> dict:
    with TimeoutSession() as session:
        transcript = YouTubeTranscriptApi(http_client=session).fetch(
            video_id, languages=["en", "en-US", "en-GB"]
        )
    segments = [
        {"text": item.text, "start": item.start, "duration": item.duration}
        for item in transcript if item.text.strip()
    ]
    return {
        "video_id": video_id,
        "video_url": f"https://www.youtube.com/watch?v={video_id}",
        "language": transcript.language,
        "is_generated": transcript.is_generated,
        "segments": segments,
    }
