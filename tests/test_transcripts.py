import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException
from youtube_transcript_api import NoTranscriptFound, RequestBlocked

from app.main import TranscriptRequest, transcript
from app.transcripts import extract_video_id, fetch_transcript


class TranscriptTests(unittest.TestCase):
    def test_supported_links(self):
        for url in (
            'https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=10',
            'https://youtu.be/dQw4w9WgXcQ?si=share',
            'https://m.youtube.com/shorts/dQw4w9WgXcQ',
            'https://youtube.com/embed/dQw4w9WgXcQ',
            'https://youtube.com/live/dQw4w9WgXcQ',
        ):
            with self.subTest(url=url):
                self.assertEqual(extract_video_id(url), 'dQw4w9WgXcQ')

    def test_rejects_other_hosts_and_invalid_ids(self):
        for url in (
            'https://youtube.com.evil.test/watch?v=dQw4w9WgXcQ',
            'https://youtube.com@evil.test/watch?v=dQw4w9WgXcQ',
            'file:///etc/passwd', 'https://youtube.com/watch?v=short',
            'https://youtube.com/playlist?list=123', 'not a url',
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                extract_video_id(url)

    def test_preserves_caption_timing_and_metadata(self):
        class Captions(list):
            language = 'English'
            is_generated = True
        captions = Captions([SimpleNamespace(text='<b>hello</b>', start=12.5, duration=3.1)])
        with patch('app.transcripts.YouTubeTranscriptApi') as api:
            api.return_value.fetch.return_value = captions
            result = fetch_transcript('dQw4w9WgXcQ')
        self.assertEqual(result['segments'][0], {'text': '<b>hello</b>', 'start': 12.5, 'duration': 3.1})
        self.assertTrue(result['is_generated'])

    def test_invalid_link_never_fetches(self):
        with patch('app.main.fetch_transcript') as fetch:
            with self.assertRaises(HTTPException) as caught:
                transcript(TranscriptRequest(url='https://example.com'))
            self.assertEqual(caught.exception.status_code, 400)
            fetch.assert_not_called()

    def test_upstream_errors_are_explained(self):
        errors = [
            (NoTranscriptFound('dQw4w9WgXcQ', ['en'], []), 404),
            (RequestBlocked('dQw4w9WgXcQ'), 503),
        ]
        for error, status in errors:
            with self.subTest(status=status), patch('app.main.fetch_transcript', side_effect=error):
                with self.assertRaises(HTTPException) as caught:
                    transcript(TranscriptRequest(url='https://youtu.be/dQw4w9WgXcQ'))
                self.assertEqual(caught.exception.status_code, status)


if __name__ == '__main__':
    unittest.main()
