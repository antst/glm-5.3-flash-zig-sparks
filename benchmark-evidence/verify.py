"""Verify exported original Mia HTTP evidence and recompute both metrics, without an endpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def request(root, row):
    assert hashlib.sha256(json.dumps(row['body']).encode()).hexdigest() == row['body_sha256']
    path = root / row['raw_path']
    assert Path(row['raw_path']).name == row['raw_path'] and path.suffix == '.sse'
    wire = path.read_bytes()
    assert hashlib.sha256(wire).hexdigest() == row['sse_sha256']
    lines = wire.splitlines(keepends=True)
    assert len(lines) == len(row['chunk_times_ns'])
    assert all(a <= b for a, b in zip(row['chunk_times_ns'], row['chunk_times_ns'][1:]))
    assert row['complete'] and row['valid'] and row['done']
    arrivals, usage, done = [], {}, False
    for line, at in zip(lines, row['chunk_times_ns'], strict=True):
        if not line.startswith(b'data: '):
            continue
        if line[6:].strip() == b'[DONE]':
            done = True
            continue
        event = json.loads(line[6:])
        usage.update(event.get('usage') or {})
        if any(c.get('delta', {}).get('content') for c in event.get('choices', [])):
            arrivals.append(at)
    assert done and usage == row['usage'] and arrivals
    assert row['started_ns'] <= arrivals[0] <= arrivals[-1] <= row['ended_ns']
    return dict(arrivals=arrivals, tokens=usage['completion_tokens'] - 1)


def main(root):
    manifest = read(root / 'MANIFEST.json')['files']
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    assert actual <= set(manifest) | {'MANIFEST.json', 'INDEPENDENT-VERIFIED.json'}
    assert set(manifest) <= actual
    for name, expected in manifest.items():
        path = root / name
        assert not path.is_symlink() and sha(path) == expected, name
    index = read(root / 'INDEX.json')
    assert len(index['profiles']) == 5
    profiles, total = [], 0
    for profile in index['profiles']:
        rows, groups = {}, []
        for client, count in [('acc', 13), ('acc8', 28), ('bench2', 22)]:
            folder = root / profile['id'] / client
            evidence = read(folder / 'EVIDENCE.json')
            original = read(folder / 'ORIGINAL-EVIDENCE.json')
            assert evidence['passed'] and evidence['request_options_unchanged']
            assert len(evidence['requests']) == count
            assert evidence['source_sha256'] == sha(root / 'clients' / (client + '.py'))
            for saved, raw in zip(evidence['requests'], original['requests'], strict=True):
                changed = dict(raw, raw_path=Path(raw['raw_path']).name)
                assert saved == changed
            rows[client] = [request(folder, row) for row in evidence['requests']]
            total += count
        levels = {}
        for client, ranges in [('acc', [(4, 5, 9), (4, 9, 13)]),
                               ('acc8', [(6, 0, 6), (6, 6, 12), (8, 12, 20), (8, 20, 28)])]:
            for concurrency, begin, end in ranges:
                selected = rows[client][begin:end]
                assert len(selected) == concurrency
                assert all(len(x['arrivals']) > 1 and x['tokens'] > 0 for x in selected)
                legacy = sum(x['tokens'] / ((x['arrivals'][-1] - x['arrivals'][0]) / 1e9)
                             for x in selected)
                common = sum(x['tokens'] for x in selected) / ((
                    max(x['arrivals'][-1] for x in selected) -
                    min(x['arrivals'][0] for x in selected)) / 1e9)
                groups.append(dict(client=client, concurrency=concurrency,
                    index_range=[begin, end], legacy=legacy, common=common))
        for c in ('4', '6', '8'):
            selected = [x for x in groups if x['concurrency'] == int(c)]
            legacy, common = [statistics.mean(x[k] for x in selected) for k in ('legacy', 'common')]
            expected = profile['levels'][c]
            assert abs(legacy - expected['legacy_sum_mean']) < 1e-6
            assert abs(common - expected['same_requests_common_window_mean']) < 1e-6
            levels[c] = dict(legacy=legacy, common=common)
        profiles.append(dict(id=profile['id'], requests=63, levels=levels))
    assert total == 315
    report = dict(passed=True, profiles=profiles, requests=total,
        original_body_and_SSE_hashes_verified=True, source_client_hashes_verified=True,
        relative_path_translation_only=True, both_metrics_reconstructed_from_raw_SSE=True,
        no_endpoint_build_or_model_operations=True,
        all_four_main_rows_current_coherent_build=index['all_four_main_rows_current_coherent_build'])
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root.resolve())
