"""Archive verified runtime assets, preserving execute permission for every profile's native binary."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def executables(recipe):
    names = set()
    assets = {x['name']: x for x in recipe['runtime_assets']}
    for label, profile in recipe['profiles'].items():
        runtime = profile.get('runtime')
        name = runtime['native_binary'] if runtime else 'tensorfold-native'
        expected = runtime['native_sha256'] if runtime else recipe['native_sha256']
        assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', name)
        assert name in assets and assets[name]['sha256'] == expected, label
        assert 'profiles' not in assets[name] or label in assets[name]['profiles'], label
        names.add(name)
    return names


def main(root, recipe_path, output):
    recipe = json.loads(recipe_path.read_text())
    assets = recipe['runtime_assets']
    assert len(assets) == len({x['name'] for x in assets})
    native = executables(recipe)
    assert native and not output.exists() and not output.with_name(output.name + '.json').exists()
    for row in assets:
        assert Path(row['name']).name == row['name']
        path = root / row['name']
        assert path.is_file() and not path.is_symlink(), row['name']
        assert path.stat().st_size == row['bytes'] and digest(path) == row['sha256'], row['name']
        if row['name'] in native:
            assert path.stat().st_mode & 0o111, 'Native executable lacks execute permission: ' + row['name']
    with output.open('xb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode='w|', format=tarfile.PAX_FORMAT) as archive:
            for row in assets:
                body = (root / row['name']).read_bytes()
                assert hashlib.sha256(body).hexdigest() == row['sha256']
                member = tarfile.TarInfo('runtime/' + row['name'])
                member.size = len(body)
                member.mode = 0o755 if row['name'] in native else 0o644
                member.uid = member.gid = member.mtime = 0
                archive.addfile(member, io.BytesIO(body))
    with tarfile.open(output) as archive:
        for row, member in zip(assets, archive, strict=True):
            assert member.isfile() and member.name == 'runtime/' + row['name']
            assert hashlib.sha256(archive.extractfile(member).read()).hexdigest() == row['sha256']
            assert member.mode == (0o755 if row['name'] in native else 0o644)
    proof = dict(passed=True, source_commit=recipe['source_commit'], native_sha256=recipe['native_sha256'],
        archive=output.name, bytes=output.stat().st_size, sha256=digest(output), files=assets,
        native_executables=sorted(native), all_profile_executable_modes='0755', other_asset_modes='0644',
        no_model_weights_or_benchmark_wrappers=True, no_dynamic_loading_or_GPU_execution=True)
    output.with_name(output.name + '.json').write_text(json.dumps(proof, indent=2) + '\n')
    print(json.dumps({k: v for k, v in proof.items() if k != 'files'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('recipe', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    main(args.root.resolve(), args.recipe.resolve(), args.output.resolve())
