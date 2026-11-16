"""Run the fixed 48-request matrix with the byte-preserved client, or reduce its saved results."""
import argparse
import hashlib
import json
import math
import pathlib
import statistics
import subprocess
import sys

P = pathlib.Path(__file__).resolve().parent / 'balanced'


def read(path):
    return json.loads(path.read_text())


def matrix():
    value = read(P / 'MATRIX.json')
    assert value['tokens'] == 400 and value['reps'] == 2 and len(value['cases']) == 11
    for case in value['cases']:
        path = P / case['prompts']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == case['sha256']
        assert case['concurrency'] in (1, 4, 8)
    return value


def reduce(directory, definition):
    cells, requests = [], 0
    for case in definition['cases']:
        cell, = read(directory / (case['name'] + '.json'))['cells']
        assert cell['timing_valid'] and cell['failed'] == cell['unmeasured'] == 0
        assert cell['streams'] == case['concurrency'] and len(cell['reps']) == 2
        rates, elapsed_rates = [], []
        for rep in cell['reps']:
            assert rep['timing_valid'] and rep['failed'] == rep['unmeasured'] == 0
            spans, evidence = rep['client_intervals'], rep['evidence']
            assert len(spans) == len(evidence) == case['concurrency']
            for row in evidence:
                assert row['complete'] and row['tokens'] == 400 and row['cached_tokens'] == 0
                assert len(row['token_sha256']) == len(row['output_sha256']) == 64
            seconds = max(row['last'] for row in spans) - min(row['first'] for row in spans)
            elapsed = max(row['end'] for row in spans) - min(row['start'] for row in spans)
            rate = case['concurrency'] * 399 / seconds
            end_to_end = case['concurrency'] * 400 / elapsed
            assert abs(rate - rep['aggregate_tps']) < 1e-6
            assert abs(end_to_end - rep['end_to_end_tps']) < 1e-6
            rates.append(rate)
            elapsed_rates.append(end_to_end)
            requests += len(evidence)
        cells.append(dict(case=case['name'], concurrency=case['concurrency'],
            common_window_decode_tps=statistics.median(rates), repetitions=rates,
            end_to_end_tps=statistics.median(elapsed_rates)))
    assert requests == 48
    levels = {str(c): math.exp(statistics.mean(math.log(row['common_window_decode_tps'])
        for row in cells if row['concurrency'] == c)) for c in (1, 4, 8)}
    return dict(passed=True, timed_requests=requests, common_window_decode_tps=levels, cells=cells,
        cached_tokens_zero_in_client_evidence=True, server_configuration_independently_attested=False,
        scope='Original 400-token client intervals, median of two repetitions per case, then geometric '
            'mean across cases at each concurrency. No cross-model output or quality claim.')


def main(args):
    definition = matrix()
    output = args.output.resolve()
    if args.reduce_only:
        assert output.is_dir()
    else:
        assert args.url and not output.exists(), 'Specify a server URL and a fresh output directory'
        output.mkdir(parents=True)
        for case in definition['cases']:
            target = output / (case['name'] + '.json')
            command = [sys.executable, str(P / 'tools/bench_concurrent.py'), args.url, args.model,
                '--levels', str(case['concurrency']), '--tokens', '400', '--reps', '2',
                '--temperatures', '0', '--prompts', str(P / case['prompts']), '--output', str(target)]
            if args.api_key_env:
                command += ['--api-key-env', args.api_key_env]
            with target.with_suffix('.stdout').open('xb') as stdout, target.with_suffix('.stderr').open('xb') as stderr:
                child = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=args.case_timeout)
            assert child.returncode == 0, case['name'] + ' failed; inspect its retained stderr'
            assert not target.with_suffix('.stderr').read_text().strip(), case['name'] + ' reported an error'
    report = reduce(output, definition)
    target = output / 'BALANCED-SUMMARY.json'
    assert not target.exists(), 'Preserve the previous summary; choose a fresh output directory'
    target.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(passed=True, timed_requests=48, rates=report['common_window_decode_tps'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', help='HTTP API root, for example http://node-a:18888')
    parser.add_argument('--model', default='GLM-5.3-Flash')
    parser.add_argument('--output', type=pathlib.Path, required=True)
    parser.add_argument('--reduce-only', action='store_true')
    parser.add_argument('--api-key-env', help='Name of the environment variable holding the bearer key')
    parser.add_argument('--case-timeout', type=int, default=1800)
    main(parser.parse_args())
