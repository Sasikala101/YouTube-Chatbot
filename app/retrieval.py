"""Hybrid semantic and keyword retrieval with a local TF-IDF fallback."""

import math
import re
import time
from collections import Counter, OrderedDict
from threading import Lock
from uuid import uuid4

from app.embeddings import EmbeddingError, embed_texts, model_name

STOP_WORDS = set("a an and are as at be been but by can do does for from had has have how i in is it its me of on or s that the their there these they this to was we were what when where which who why will with you your about tell explain please did her she him he his they them their more most want would could should tell give get".split())


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
        self.semantic_vectors = None
        self.semantic_status = "idle"
        self.semantic_error = None
        self.embedding_model = None
        self.semantic_lock = Lock()
        self.query_cache = OrderedDict()
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

    def prepare_semantic(self):
        try:
            model = model_name()
            vectors = embed_texts([chunk["text"] for chunk in self.chunks], "RETRIEVAL_DOCUMENT", model)
            with self.semantic_lock:
                self.semantic_vectors = vectors
                self.embedding_model = model
                self.semantic_status = "ready"
                self.semantic_error = None
        except EmbeddingError as exc:
            with self.semantic_lock:
                self.semantic_status = "fallback"
                self.semantic_error = str(exc)
        except Exception:
            with self.semantic_lock:
                self.semantic_status = "fallback"
                self.semantic_error = "Semantic indexing failed; using keyword search."

    def retrieval_state(self):
        with self.semantic_lock:
            return {"status": self.semantic_status, "error": self.semantic_error,
                    "method": "hybrid" if self.semantic_status == "ready" else "tf-idf"}

    def retrieve(self, question, limit=5):
        lexical = self.search(question, limit=max(limit, 10))
        with self.semantic_lock:
            vectors = self.semantic_vectors
            model = self.embedding_model
            query_vector = self.query_cache.get(question)
        if vectors is None:
            return lexical[:limit], "tf-idf", self.retrieval_state().get("error") or "Semantic search is not ready; using keyword search."
        try:
            if query_vector is None:
                query_vector = embed_texts([question], "RETRIEVAL_QUERY", model)[0]
                with self.semantic_lock:
                    self.query_cache[question] = query_vector
                    while len(self.query_cache) > 64:
                        self.query_cache.popitem(last=False)
            if len(query_vector) != len(vectors[0]):
                raise EmbeddingError("Query embedding dimensions do not match; using keyword search.")
            similarities = [sum(a * b for a, b in zip(query_vector, vector)) for vector in vectors]
            dense = sorted((i for i, score in enumerate(similarities) if score >= 0.35),
                           key=lambda i: similarities[i], reverse=True)[:max(limit, 10)]
            # Reciprocal-rank fusion keeps exact names useful while adding
            # meaning-based matches. Scores are rankings, not confidence.
            by_text = {(chunk["text"], chunk["start"], chunk["end"]): i for i, chunk in enumerate(self.chunks)}
            scores = Counter()
            for rank, i in enumerate(dense):
                scores[i] += 1 / (60 + rank + 1)
            for rank, passage in enumerate(lexical):
                scores[by_text[(passage["text"], passage["start"], passage["end"]) ]] += 1 / (60 + rank + 1)
            passages = [{**self.chunks[i], "score": round(score, 6),
                         "url": f"{self.video_url}&t={int(self.chunks[i]['start'])}s"}
                        for i, score in scores.most_common(limit)]
            return passages, "hybrid", None
        except EmbeddingError as exc:
            return lexical[:limit], "tf-idf", str(exc)


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
