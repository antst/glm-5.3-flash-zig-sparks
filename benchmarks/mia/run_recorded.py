"""Run byte-pinned Mia clients without changing their requests; retain SSE evidence after timing ends."""
import argparse
import hashlib
import json
import os
import pathlib
import runpy
import sys
import threading
import time
import urllib.request

P = pathlib.Path(__file__).resolve().parent
EXPECTED = {'acc.py': 13, 'acc8.py': 28, 'bench2.py': 22}


class Capture:
    def __init__(self, open_request):
        self.open_request = open_request
        self.lock = threading.Lock()
        self.records = []

    def open(self, request, *args, **kwargs):
        # The exact original Request and timeout reach urllib; no prompt, decoding or server option is added.
        body = bytes(request.data)
        record = dict(url=request.full_url, body_sha256=hashlib.sha256(body).hexdigest(),
                      body=json.loads(body), started_ns=time.perf_counter_ns(), chunks=[], complete=False)
        with self.lock:
            self.records.append(record)
        try:
            response = self.open_request(request, *args, **kwargs)
        except BaseException as error:
            record['error'] = repr(error)
            raise
        return RecordedResponse(response, record)


class RecordedResponse:
    def __init__(self, response, record):
        self.response = response
        self.record = record

    def __iter__(self):
        return self

    def __next__(self):
        try:
            line = next(self.response)
        except StopIteration:
            self.record['ended_ns'] = time.perf_counter_ns()
            self.record['complete'] = True
            raise
        self.record['chunks'].append((time.perf_counter_ns(), line))
        return line

    def __enter__(self):
        self.response.__enter__()
        return self

    def __exit__(self, *args):
        return self.response.__exit__(*args)

    def __getattr__(self, name):
        return getattr(self.response, name)


def evidence(record):
    usage, native, done, malformed = {}, {}, False, False
    raw = b''.join(line for _, line in record['chunks'])
    for _, line in record['chunks']:
        if not line.startswith(b'data: '):
            continue
        if line[6:].strip() == b'[DONE]':
            done = True
            continue
        try:
            value = json.loads(line[6:])
        except ValueError:
            malformed = True
            continue
        usage.update(value.get('usage') or {})
        native.update(value.get('tensorfold') or {})
    valid = record['complete'] and done and not malformed and usage.get('completion_tokens', 0) > 0
    return dict(valid=bool(valid), usage=usage, native=native, sse_sha256=hashlib.sha256(raw).hexdigest(),
                raw=raw, done=done)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('client', choices=EXPECTED)
    parser.add_argument('base')
    parser.add_argument('model')
    parser.add_argument('tag')
    parser.add_argument('--output', required=True, type=pathlib.Path)
    args = parser.parse_args()
    expected = json.loads((P/'SHA256.json').read_text())[args.client]
    script = P/args.client
    assert hashlib.sha256(script.read_bytes()).hexdigest() == expected
    assert args.output.is_absolute() and not args.output.exists()
    args.output.mkdir(parents=True)
    original = urllib.request.urlopen
    capture = Capture(original)
    passed = False
    error = None
    try:
        os.chdir(args.output)
        sys.argv = [str(script), args.base, args.model, args.tag]
        if args.client == 'acc.py':
            sys.argv.append('2')
        urllib.request.urlopen = capture.open
        runpy.run_path(str(script), run_name='__main__')
        assert len(capture.records) == EXPECTED[args.client], 'A threaded client request is missing'
        assert all(evidence(row)['valid'] for row in capture.records), 'Incomplete SSE or missing usage'
        passed = True
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        urllib.request.urlopen = original
        # No evidence-file writes occur while the original client measures a request.
        rows = []
        for i, row in enumerate(capture.records):
            proof = evidence(row)
            target = args.output/f'http-{i:03d}.sse'
            target.write_bytes(proof.pop('raw'))
            chunks = row.pop('chunks')
            rows.append(dict(row, **proof, raw_path=str(target),
                             chunk_times_ns=[t for t, _ in chunks]))
        report = dict(passed=passed, error=error, client=args.client, source_sha256=expected,
                      requested=EXPECTED[args.client], recorded=len(rows), requests=rows,
                      request_options_unchanged=True, artifact_writes_after_client_timing=True,
                      limits='The recording adds an in-memory timestamp/append per SSE line; no output or '
                             'sampling fields are changed. Original client metrics retain their definitions.')
        (args.output/'EVIDENCE.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
