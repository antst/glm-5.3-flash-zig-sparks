"""Build one reproducible TP2 runtime archive from the pinned files."""

import argparse
import gzip
import io
import json
from pathlib import Path
import tarfile

from .config import digest, verify_runtime

HERE = Path(__file__).resolve().parent


def package(bundle, source, root, output):
    verify_runtime(root, bundle['files'])
    if output.name != bundle['archive']:
        raise ValueError(f'Expected archive name {bundle["archive"]}')
    root = root.resolve()
    if output.resolve().is_relative_to(root):
        raise ValueError('Choose an archive path outside the input directory')
    manifest = json.dumps({'source_commit':source['source_commit'], 'native_sha256':source['native_sha256'],
                           'files':bundle['files']}, indent=2, sort_keys=True).encode() + b'\n'
    with output.open('xb') as binary:
        try:
            with gzip.GzipFile(fileobj=binary, mode='wb', filename='', mtime=0, compresslevel=1) as zipped:
                with tarfile.open(fileobj=zipped, mode='w|') as archive:
                    record = tarfile.TarInfo('MANIFEST.json')
                    record.mode, record.size = 0o644, len(manifest)
                    archive.addfile(record,io.BytesIO(manifest))
                    for row in sorted(bundle['files'], key=lambda item:item['path']):
                        name = row['path']
                        item = tarfile.TarInfo(name)
                        item.size = row['bytes']
                        item.mode = 0o755 if name == 'tensorfold-native' else 0o644
                        with (root/name).open('rb') as file:
                            archive.addfile(item,file)
        except BaseException:
            output.unlink()
            raise
    return {'name':output.name,'bytes':output.stat().st_size,'sha256':digest(output),
            'source_commit':source['source_commit'],'native_sha256':source['native_sha256']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', choices=['bf16','q4','q8'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    bundles = json.loads((HERE/'recipe.json').read_text())['bundles']
    sources = json.loads((HERE/'SOURCE.json').read_text())
    print(json.dumps(package(bundles[args.bundle],sources[args.bundle],args.root,args.output)))


if __name__ == '__main__':
    main()
