import os
import unittest
from unittest.mock import Mock, patch

from app.embeddings import EmbeddingError, embed_texts, normalize
from app.retrieval import TranscriptIndex


class EmbeddingTests(unittest.TestCase):
    def make_index(self):
        return TranscriptIndex({'video_url': 'https://www.youtube.com/watch?v=l5alL4UmLoA',
            'segments': [{'text': 'Compensation doubled after joining Amazon.', 'start': 15, 'duration': 8}]})

    def test_vectors_are_normalized_and_invalid_values_rejected(self):
        self.assertEqual(normalize([3, 4]), (0.6, 0.8))
        for values in ([], [0, 0], [float('nan')], [float('inf')], ['text'], [True]):
            with self.subTest(values=values), self.assertRaises(EmbeddingError):
                normalize(values)

    def test_batch_payload_keeps_document_and_query_tasks_distinct(self):
        response = Mock(ok=True, status_code=200)
        response.json.return_value = {'embeddings': [{'values': [3, 4]}]}
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test-key'}), patch('app.embeddings.requests.post', return_value=response) as post:
            result = embed_texts(['a document'], 'RETRIEVAL_DOCUMENT')
        self.assertEqual(result, [(0.6, 0.8)])
        request = post.call_args.kwargs['json']['requests'][0]
        self.assertEqual(request['taskType'], 'RETRIEVAL_DOCUMENT')
        self.assertEqual(request['outputDimensionality'], 768)

    def test_incomplete_batch_is_rejected(self):
        response = Mock(ok=True, status_code=200)
        response.json.return_value = {'embeddings': []}
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test-key'}), patch('app.embeddings.requests.post', return_value=response):
            with self.assertRaises(EmbeddingError):
                embed_texts(['document'], 'RETRIEVAL_DOCUMENT')

    def test_paraphrase_retrieves_without_keyword_overlap_and_caches_query(self):
        index = self.make_index()
        with patch('app.retrieval.embed_texts', return_value=[(1., 0.)]) as embed:
            index.prepare_semantic()
            self.assertEqual(index.search('pay rise'), [])
            results, method, notice = index.retrieve('pay rise')
            index.retrieve('pay rise')
        self.assertEqual(method, 'hybrid')
        self.assertIsNone(notice)
        self.assertEqual(results[0]['start'], 15)
        self.assertEqual(embed.call_count, 2)  # one document batch, one query
        self.assertEqual(embed.call_args.args[1], 'RETRIEVAL_QUERY')

    def test_query_failure_falls_back_without_losing_index(self):
        index = self.make_index()
        with patch('app.retrieval.embed_texts', return_value=[(1., 0.)]):
            index.prepare_semantic()
        with patch('app.retrieval.embed_texts', side_effect=EmbeddingError('Quota reached')):
            results, method, notice = index.retrieve('compensation')
        self.assertEqual(method, 'tf-idf')
        self.assertEqual(notice, 'Quota reached')
        self.assertTrue(results)
        self.assertEqual(index.semantic_status, 'ready')

    def test_document_failure_leaves_keyword_search_available(self):
        index = self.make_index()
        with patch('app.retrieval.embed_texts', side_effect=EmbeddingError('Unavailable')):
            index.prepare_semantic()
        self.assertEqual(index.retrieval_state()['status'], 'fallback')
        self.assertTrue(index.retrieve('compensation')[0])

    def test_dimension_mismatch_falls_back(self):
        index = self.make_index()
        with patch('app.retrieval.embed_texts', return_value=[(1., 0.)]):
            index.prepare_semantic()
        with patch('app.retrieval.embed_texts', return_value=[(1., 0., 0.)]):
            self.assertEqual(index.retrieve('compensation')[1], 'tf-idf')


if __name__ == '__main__':
    unittest.main()
