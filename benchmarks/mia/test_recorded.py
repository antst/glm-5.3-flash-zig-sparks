"""The recorder preserves urllib inputs and raw SSE; incomplete streams are rejected without a network."""
import json
import unittest
import urllib.request
from run_recorded import Capture, evidence


class RecordingTests(unittest.TestCase):
    def test_original_request_and_stream_are_unchanged(self):
        body = dict(stream=True, max_tokens=400, temperature=0, messages=[dict(role='user', content='x')])
        request = urllib.request.Request('http://unused/v1/chat/completions', json.dumps(body).encode())
        lines = [b'data: {"choices":[{"delta":{"content":"x"}}]}\n',
                 b'data: {"usage":{"completion_tokens":1},"tensorfold":{"token_sha256":"abc"}}\n',
                 b'data: [DONE]\n']
        calls = []

        def original(req, *args, **kwargs):
            calls.append((req, args, kwargs))
            return iter(lines)

        recorder = Capture(original)
        self.assertEqual(lines, list(recorder.open(request, timeout=600)))
        self.assertIs(calls[0][0], request)
        self.assertEqual(((), dict(timeout=600)), calls[0][1:])
        self.assertEqual(body, recorder.records[0]['body'])
        proof = evidence(recorder.records[0])
        self.assertTrue(proof['valid'])
        self.assertEqual(b''.join(lines), proof['raw'])

    def test_incomplete_or_malformed_stream_is_retained_and_rejected(self):
        for lines in ([b'data: {"usage":{"completion_tokens":1}}\n'],
                      [b'data: broken\n', b'data: [DONE]\n']):
            recorder = Capture(lambda *a, **k: iter(lines))
            request = urllib.request.Request('http://unused/v1/chat/completions', b'{"stream":true}')
            self.assertEqual(lines, list(recorder.open(request)))
            proof = evidence(recorder.records[0])
            self.assertFalse(proof['valid'])
            self.assertEqual(b''.join(lines), proof['raw'])


if __name__ == '__main__':
    unittest.main()
