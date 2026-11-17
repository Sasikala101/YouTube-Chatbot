"""Small, local TF-IDF retrieval baseline; no model or external API needed."""

import math
import re
import time
from collections import Counter, OrderedDict
from threading import Lock
from uuid import uuid4

STOP_WORDS = set("a an and are as at be been but by can do does for from had has have how i in is it its me of on or s that the their there these they this to was we were what when where which who why will with you your about tell explain please".split())


def tokenize(text):
    return [word for word in re.findall(r"[^\W_]+", text.casefold()) if word not in STOP_WORDS]


def chunk_segments(segments, size=140, overlap=30):
    if not 0 <= overlap < size:
        raise ValueError("Overlap must be smaller than chunk size.")
    words = []
    for segment in segments:
        for word in segment["text"].split():
            words.append((word, segment["start"], segment["start"] + segment["duration"]))
    chunks = []
    for start in range(0, len(words), size - overlap):
        window = words[start:start + size]
        chunks.append({
            "text": " ".join(word[0] for word in window),
            "start": window[0][1], "end": max(word[2] for word in window),
        })
        if start + size >= len(words):
            break
    return chunks


class TranscriptIndex:
    def __init__(self, transcript):
        self.video_url = transcript["video_url"]
        self.full_text = "\n".join(segment["text"] for segment in transcript["segments"])
        self.summary = None
        self.summary_status = "idle"
        self.summary_error = None
        self.summary_started = None
        self.summary_progress = ""
        self.summary_deadline = None
        self.summary_timed_out = False
        self.summary_lock = Lock()
        self.chunks = chunk_segments(transcript["segments"])
        counts = [Counter(tokenize(chunk["text"])) for chunk in self.chunks]
        frequency = Counter(term for count in counts for term in count)
        self.idf = {term: math.log((1 + len(counts)) / (1 + number)) + 1 for term, number in frequency.items()}
        self.vectors = [self.vector(count) for count in counts]

    def vector(self, counts):
        vector = {term: (1 + math.log(count)) * self.idf[term] for term, count in counts.items() if term in self.idf}
        norm = math.sqrt(sum(value * value for value in vector.values()))
        return {term: value / norm for term, value in vector.items()} if norm else {}

    def search(self, question, limit=3):
        query = self.vector(Counter(tokenize(question)))
        scored = []
        for chunk, vector in zip(self.chunks, self.vectors):
            score = sum(value * vector.get(term, 0) for term, value in query.items())
            if score > 0:
                scored.append({**chunk, "score": round(score, 4),
                               "url": f"{self.video_url}&t={int(chunk['start'])}s"})
        return sorted(scored, key=lambda item: item["score"], reverse=True)[:limit]


class IndexStore:
    """Bound memory and expire local video indexes after an hour."""

    def __init__(self, capacity=8, ttl=3600):
        self.capacity, self.ttl = capacity, ttl
        self.items = OrderedDict()
        self.lock = Lock()

    def add(self, index):
        key = uuid4().hex
        with self.lock:
            now = time.monotonic()
            self.items = OrderedDict((key, value) for key, value in self.items.items() if value[0] > now)
            while len(self.items) >= self.capacity:
                self.items.popitem(last=False)
            self.items[key] = (now + self.ttl, index)
        return key

    def get(self, key):
        with self.lock:
            item = self.items.get(key)
            if not item:
                return None
            if item[0] <= time.monotonic():
                del self.items[key]
                return None
            self.items.move_to_end(key)
            return item[1]


indexes = IndexStore()
