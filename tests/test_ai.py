import asyncio
import json
import os
import unittest
from unittest.mock import Mock, patch

from fastapi import BackgroundTasks, HTTPException

from app import ai
from app.main import ChatRequest, VideoSessionRequest, chat, summary, summary_status
from app.retrieval import TranscriptIndex, indexes


class AITests(unittest.TestCase):
    def setUp(self):
        self.index = TranscriptIndex({'video_url': 'https://www.youtube.com/watch?v=aircAruvnKk',
            'segments': [{'text': 'The sigmoid function maps values between zero and one.', 'start': 32, 'duration': 8}]})
        self.key = indexes.add(self.index)

    def test_missing_key_is_actionable(self):
        with patch.dict(os.environ, {'GEMINI_API_KEY': ''}):
            with self.assertRaises(HTTPException) as caught:
                ai.generate('instructions', 'text')
        self.assertEqual(caught.exception.status_code, 503)

    def test_provider_request_and_output(self):
        response = Mock(ok=True, status_code=200)
        response.json.return_value = {'status': 'completed', 'steps': [
            {'type': 'thought', 'content': [{'type': 'text', 'text': 'private reasoning'}]},
            {'type': 'model_output', 'content': [{'type': 'text', 'text': 'Answer [1].'}]},
        ]}
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test-key'}), patch('app.ai.requests.post', return_value=response) as post:
            self.assertEqual(ai.generate('rules', 'question'), 'Answer [1].')
            payload = post.call_args.kwargs['json']
            self.assertFalse(payload['store'])
            self.assertEqual(payload['system_instruction'], 'rules')
            self.assertEqual(payload['generation_config']['thinking_level'], 'low')
            self.assertLessEqual(post.call_args.kwargs['timeout'][1], 30)

    def test_provider_timeout_returns_clear_failure_without_retry(self):
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test-key'}), patch(
            'app.ai.requests.post', side_effect=ai.requests.Timeout
        ) as post:
            with self.assertRaises(HTTPException) as caught:
                ai.generate('rules', 'question')
            self.assertEqual(caught.exception.status_code, 504)
            self.assertEqual(post.call_count, 1)

    def test_provider_quota_error(self):
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test-key'}), patch('app.ai.requests.post', return_value=Mock(status_code=429)):
            with self.assertRaises(HTTPException) as caught:
                ai.generate('rules', 'question')
        self.assertEqual(caught.exception.status_code, 429)

    def test_temporary_provider_error_retries_once(self):
        success = Mock(ok=True, status_code=200)
        success.json.return_value = {'status': 'completed', 'steps': [
            {'type': 'model_output', 'content': [{'type': 'text', 'text': 'Recovered'}]}]}
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test-key'}), patch('app.ai.time.sleep'), patch(
            'app.ai.requests.post', side_effect=[Mock(status_code=503), success]
        ) as post:
            self.assertEqual(ai.generate('rules', 'question'), 'Recovered')
            self.assertEqual(post.call_count, 2)

    def test_stalled_job_requires_new_session(self):
        self.index.summary_status = 'pending'
        self.index.summary_started = 0
        self.index.summary_deadline = 180
        with patch('app.main.time.monotonic', return_value=181):
            state = summary_status(self.key)
            self.assertEqual(state['status'], 'failed')
            with self.assertRaises(HTTPException) as caught:
                summary(VideoSessionRequest(transcript_id=self.key), BackgroundTasks())
        self.assertEqual(caught.exception.status_code, 409)

    def test_background_summary_cached_and_not_duplicated(self):
        tasks = BackgroundTasks()
        request = VideoSessionRequest(transcript_id=self.key)
        with patch('app.ai.configured', return_value=True), patch('app.ai.generate', return_value='A video about sigmoid functions.') as generate:
            self.assertEqual(summary(request, tasks)['status'], 'pending')
            self.assertEqual(summary(request, tasks)['status'], 'pending')
            self.assertEqual(len(tasks.tasks), 1)
            asyncio.run(tasks())
            self.assertEqual(summary_status(self.key)['status'], 'complete')
            self.assertEqual(summary(request, BackgroundTasks())['summary'], 'A video about sigmoid functions.')
            generate.assert_called_once()

    def test_summary_failure_can_retry(self):
        request = VideoSessionRequest(transcript_id=self.key)
        with patch('app.ai.configured', return_value=True), patch('app.ai.generate', side_effect=HTTPException(429, 'Quota reached')):
            tasks = BackgroundTasks()
            summary(request, tasks)
            asyncio.run(tasks())
            self.assertEqual(summary_status(self.key)['status'], 'failed')
            self.assertEqual(summary_status(self.key)['error'], 'Quota reached')
            self.assertEqual(summary(request, BackgroundTasks())['status'], 'pending')

    def test_chat_uses_evidence_and_returns_timed_sources(self):
        with patch('app.ai.generate', return_value='It maps values between zero and one [1].') as generate:
            result = chat(ChatRequest(transcript_id=self.key, question='What is the sigmoid function?'))
            evidence = json.loads(generate.call_args.args[1])['video_passages']
        self.assertIn('sigmoid', evidence[0]['text'])
        self.assertEqual(result['sources'][0]['id'], 1)
        self.assertEqual(result['sources'][0]['start'], 32)

    def test_unmatched_question_skips_generation(self):
        with patch('app.ai.generate') as generate:
            result = chat(ChatRequest(transcript_id=self.key, question='astronomy planets'))
            generate.assert_not_called()
        self.assertEqual(result['sources'], [])
        self.assertIn("couldn't find", result['answer'])


if __name__ == '__main__':
    unittest.main()
