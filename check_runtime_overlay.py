"""Check every overlaid public runtime file against the tested-image receipt."""
import hashlib
import json
import os
import pathlib
import sys


def main(manifest):
    assert not list(pathlib.Path('/dev').glob('nvidia*')), 'This is the CPU build check'
    result = json.loads(manifest.read_text())
    for row in result['members']:
        path = pathlib.Path('/') / row['path']
        if 'link' in row:
            assert path.is_symlink() and os.readlink(path) == row['link'], row['path']
        else:
            assert path.is_file() and path.stat().st_size == row['bytes'], row['path']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'], row['path']
    print(json.dumps(dict(all_runtime_package_members_equal_tested_image=True,
                          members=len(result['members']), GPU_access=False)))


if __name__ == '__main__':
    main(pathlib.Path(sys.argv[1]))
