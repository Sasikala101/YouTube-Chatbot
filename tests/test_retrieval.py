import unittest
from unittest.mock import patch

from fastapi import HTTPException

from app.main import SearchRequest, TranscriptRequest, search, transcript
from app.retrieval import IndexStore, TranscriptIndex, chunk_segments


class RetrievalTests(unittest.TestCase):
    def test_chunk_boundaries_overlap_and_timestamps(self):
        source = [{'text': 'one two three four', 'start': 0, 'duration': 3},
                  {'text': 'five six seven eight', 'start': 5, 'duration': 4}]
        chunks = chunk_segments(source, size=5, overlap=2)
        self.assertEqual([c['text'] for c in chunks], ['one two three four five', 'four five six seven eight'])
        self.assertEqual(chunks[0]['start'], 0)
        self.assertEqual(chunks[1]['end'], 9)
        self.assertEqual(chunk_segments([]), [])

    def test_ranks_topic_and_returns_no_irrelevant_passages(self):
        source = {'video_url': 'https://www.youtube.com/watch?v=dQw4w9WgXcQ', 'segments': [
            {'text': ' '.join(['oranges fruit orchard'] * 50), 'start': 0, 'duration': 30},
            {'text': ' '.join(['python functions arguments'] * 50), 'start': 30, 'duration': 30},
        ]}
        index = TranscriptIndex(source)
        result = index.search('How do python functions work?')
        self.assertIn('python', result[0]['text'])
        self.assertGreater(result[0]['start'], 0)
        self.assertTrue(result[0]['url'].endswith('&t=30s'))
        self.assertEqual(index.search('astronomy planets'), [])
        self.assertEqual(index.search('what is the'), [])

    def test_cache_eviction_and_expiry(self):
        store = IndexStore(capacity=1, ttl=10)
        with patch('app.retrieval.time.monotonic', return_value=0):
            first = store.add('first')
            second = store.add('second')
            self.assertIsNone(store.get(first))
            self.assertEqual(store.get(second), 'second')
        with patch('app.retrieval.time.monotonic', return_value=11):
            self.assertIsNone(store.get(second))

    def test_load_and_search_flow(self):
        source = {'video_id': 'dQw4w9WgXcQ', 'video_url': 'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
                  'segments': [{'text': 'Embeddings represent text as numeric vectors.', 'start': 12, 'duration': 5}]}
        with patch('app.main.fetch_transcript', return_value=source):
            loaded = transcript(TranscriptRequest(url='https://youtu.be/dQw4w9WgXcQ'))
        result = search(SearchRequest(transcript_id=loaded['transcript_id'], question='What are embeddings?'))
        self.assertEqual(loaded['chunk_count'], 1)
        self.assertEqual(result['passages'][0]['start'], 12)
        self.assertEqual(result['method'], 'tf-idf')

    def test_missing_index_requests_reload(self):
        with self.assertRaises(HTTPException) as caught:
            search(SearchRequest(transcript_id='0' * 32, question='embeddings'))
        self.assertEqual(caught.exception.status_code, 404)


if __name__ == '__main__':
    unittest.main()
