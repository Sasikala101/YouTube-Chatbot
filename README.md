# YouTube Chatbot

Ask questions about a YouTube video, read an AI summary, and explore the timestamped transcript behind each answer.

![YouTube Chatbot interface](interface-preview.png)

## Features

- Load public YouTube videos with available English captions.
- Generate a transcript-based overview and key takeaways automatically.
- Ask follow-up questions and inspect timestamped source passages.
- Play the video alongside the summary and chat.
- Switch between light and dark themes and revisit recent video links.
- See background summary progress and actionable provider errors.

## How it works

```text
YouTube link → timed captions → overlapping chunks → hybrid semantic + keyword retrieval
                                                        ↓
Question + recent conversation + source passages → Gemini → answer + sources

Full transcript → section summaries when needed → video summary
```

Retrieval combines Gemini `gemini-embedding-001` semantic embeddings with TF-IDF keyword matches. Captions are split into 140-word chunks with a 30-word overlap. Each chunk receives a normalized 768-dimensional document vector through the embedding API; questions use the corresponding query task. Cosine similarity ranks semantic candidates, and reciprocal-rank fusion combines semantic and keyword rankings. Chat uses up to five retrieved chunks and the most recent eight conversation messages.

Document embeddings are prepared in background batches of up to 32 chunks. The app uses keyword retrieval while indexing is pending, or if indexing or query embedding fails. The UI identifies the retrieval mode instead of silently claiming semantic search succeeded. Vectors and up to 64 question embeddings are cached in each in-memory video session. No local neural model download or extra dependency is required.

Gemini is instructed to answer only from the supplied evidence. Generated content can still contain errors; inspect the linked transcript sources. Visual-only information is not analyzed. Transcript-provider fallbacks and durable storage are planned improvements. Semantic rankings and the minimum similarity cutoff are heuristics, not guarantees of relevance; the answer model must still judge whether the passages support an answer.

## Stack

Python 3.12 · FastAPI · RAG · Gemini generation and embeddings APIs · TF-IDF · HTML · CSS · JavaScript

The frontend and API share one server. API keys remain on the backend.

## Run locally

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
cp .env.example .env
```

If you already have a configured `.env`, keep it instead of copying the example over it. Set `GEMINI_API_KEY` in `.env` to your own key. `GEMINI_MODEL` can be changed to a model supported by your account and the Interactions API.

```sh
python -m uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000. Interactive API docs are at `/docs`. Restart after changing `.env`.

## Deploy to Render

1. Publish this source to your GitHub repository, excluding `.env` and `.venv`.
2. Create a Render Blueprint from that repository. The included `render.yaml` defines a Python web service.
3. Supply `GEMINI_API_KEY` as a secret environment variable in Render. Never put the value in the YAML or repository.
4. Deploy and use the `https://…onrender.com` URL returned by Render.
5. Verify loading a video, summary completion, and a timestamped chat answer on the deployed URL.

The service uses one worker because indexes and jobs are stored in process memory. They expire after an hour, can be evicted after eight loaded sessions, and are lost on restart. Free hosting can sleep when idle. YouTube can block caption requests from cloud-hosted IPs, even for videos that work locally; provider fallbacks are not implemented yet.

This is a portfolio demo. Gemini quotas and usage costs depend on the configured account. Public visitors share the backend's provider quota. Authentication, per-user usage controls, shared job storage, and durable caching are needed before a broader production rollout.

## Privacy

Transcripts and questions are sent to Google for generation. Interactions use `store: false`, subject to Google's applicable service policies. The last 12 loaded video links and titles, plus theme preference, are saved in this browser's local storage. Chat is kept in browser memory. Transcript indexes and summaries stay temporarily in server memory. Only process videos you are permitted to use.

## Validation

```sh
python -m unittest discover -s tests -v
node --check app/static/app.js
```

Tests cover URL validation, chunk overlap, timestamps, ranking, no-match behavior, session expiry, background summary jobs, provider error handling, retries, and grounding payloads. Provider responses are mocked in automated tests. Local live caption, summary, and chat flows have also been exercised; verify them separately after deployment.

## Project structure

- `app/main.py`: HTTP routes and background summary jobs.
- `app/transcripts.py`: YouTube URL parsing and timed caption retrieval.
- `app/retrieval.py`: chunking, hybrid ranking, and temporary index storage.
- `app/embeddings.py`: batched document/query embeddings, normalization, and safe provider errors.
- `app/ai.py`: backend Gemini requests and provider error handling.
- `app/static/`: responsive video, summary, and chat interface.
- `tests/`: backend unit tests, including embedding validation and fallback behavior.
- `requirements.lock`: exact installed dependencies used for deployment.
- `render.yaml`: Render service configuration, without credentials.

### AI response timing

Generation requests use low reasoning effort and an output cap to suit interactive video summaries and chat. Each provider attempt has a 5-second connect timeout and 30-second read-idle timeout; transient HTTP server errors receive at most one retry. These are network idle timeouts, not guaranteed total wall-clock deadlines. Chat shows elapsed time while waiting. Provider availability and quotas can still prevent a response.


### Semantic retrieval validation

The unit suite includes a paraphrase with no meaningful keyword overlap (`pay rise` versus `Compensation doubled after joining Amazon`) and verifies semantic retrieval, timestamp preservation, query caching, malformed vectors, dimension mismatches, and provider failure fallback. A live check embedded the 15 chunks of the example career interview and ran three paraphrased questions through both retrieval methods. This is a smoke comparison, not a general accuracy benchmark; hybrid search does not improve every question.

`GEMINI_EMBEDDING_MODEL` defaults to `gemini-embedding-001`, whose document/query task types are supported by this implementation. Changing embedding models requires reloading videos so document and query vectors stay in the same vector space. Embedding calls consume the provider's separate embedding quota and may incur costs under its plan. The `/api/retrieval-status/{transcript_id}` endpoint reports readiness; chat and `/api/search` report the method used for each result.
