"""Render the pinned two-rank EXL3 launch profiles."""

import copy
import hashlib
from ipaddress import IPv4Address, IPv4Network
import json
import math
from pathlib import Path, PurePosixPath
import re
import shlex


HEADS = {'bf16': ('16', '16', '0'), 'q4': ('4', '16', '0'),
         'q4-fp8': ('4', '8', '1'), 'q8': ('8', '16', '0')}
MAX_CONTEXT = 1_048_576


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def set_arg(command, name, value):
    require(command.count(name) == 1, f'Expected one {name} argument')
    command[command.index(name) + 1] = str(value)


def validate(args, hosts, ports):
    require(args.expert_checkpoint == 'exl3', 'TP2 has EXL3 profiles')
    require(args.prefill == 'stepped' and args.verify_rows == 32 and not args.warm_turns,
            'TP2 uses stepped prefill, 32 verification rows and warm turns off')
    require(1 <= args.parallel <= 8 and 4096 <= args.context <= MAX_CONTEXT,
            f'TP2 requires 1 to 8 lanes and a context from 4096 to {MAX_CONTEXT}')
    require(math.isfinite(args.prompt_cache_gib) and 0 <= args.prompt_cache_gib <= 32,
            'Invalid RAM cache size')
    require(math.isfinite(args.learn_gib) and 1 <= args.learn_gib <= 4096,
            'Invalid NVMe cache size')
    require(0 <= args.require_cache_tokens <= MAX_CONTEXT, 'Invalid cached-prefix threshold')
    require(len(hosts) == 2 and len(set(hosts)) == 2, 'Choose two distinct rank hosts for TP2')
    require(all(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,252}', host) for host in hosts),
            'Invalid rank host label')
    require(len(set(ports)) == 4 and all(1024 <= port <= 65535 for port in ports),
            'Reserve distinct HTTP, NCCL, rank-control (+1) and gather-bootstrap ports')
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,40}', args.instance),
            'Invalid instance name')
    master = IPv4Address(args.master)
    require(master in IPv4Network(args.roce_address_range, strict=True),
            'Use a rank-0 direct-link IPv4 address in the supplied RoCE network')
    roots = {key: getattr(args, key + '_root') for key in ('runtime', 'model', 'drafter', 'cache')}
    for key, root in roots.items():
        require(root.is_absolute() and '..' not in root.parts and
                not any(c in str(root) for c in (',', '\n', '\r', '\x00')), f'Invalid {key} root')
    distinct = [root.resolve() for root in roots.values()]
    require(len(set(distinct)) == 4, 'Use distinct runtime, model, drafter and cache roots')
    writable = distinct[-1]
    require(all(writable not in root.parents and root not in writable.parents for root in distinct[:-1]),
            'Keep writable cache separate from the three read-only roots')
    return roots


def verify_runtime(root, inventory):
    for item in inventory:
        name = PurePosixPath(item['path'])
        require(not name.is_absolute() and '..' not in name.parts and name.parts,
                'Invalid runtime inventory path')
        file = root / name
        require(file.resolve().is_relative_to(root.resolve()) and file.is_file(),
                f'Missing runtime asset {name}')
        require(file.stat().st_size == item['bytes'] and digest(file) == item['sha256'],
                f'Runtime digest mismatch for {name}')
        if str(name) == 'tensorfold-native':
            require(file.stat().st_mode & 0o111, 'Restore the executable permissions from the archive')


def construct(args, recipe, sources, create_command):
    hosts = args.hosts or recipe['rank_hosts']
    ports = [args.port, args.nccl_port, args.nccl_port + 1, args.plan_port]
    roots = validate(args, hosts, ports)
    selected = recipe['profiles'][args.precision]
    bundle = recipe['bundles'][selected['bundle']]
    source = sources[selected['bundle']]
    tiered = args.tier == 'nvme'
    if tiered:
        require(selected['qualified_nvme'], 'TP2 NVMe was qualified on the Q8/BF16 profile')
        require(args.prompt_cache_gib > 0 and args.require_cache_tokens <= args.context,
                'NVMe needs a RAM prompt cache and a cache threshold within the context')
    if args.verify_runtime:
        verify_runtime(roots['runtime'], bundle['files'])
    assets = {item['path']: item['sha256'] for item in bundle['files']}
    require(len(assets) == len(bundle['files']) and assets.get('tensorfold-native') == source['native_sha256'],
            'Runtime and engine source identities disagree')
    env = dict(recipe['common_environment'], **selected['environment'])
    require(tuple(env.get(k, '0') for k in ('TF_GLM_DENSE_BITS', 'TF_GLM_HEAD_BITS', 'TF_GLM_Q4_HEAD_FP8'))
            == HEADS[args.precision], 'Projection and head settings disagree')
    require(env['TF_GLM_WORLD'] == '2' and env['NCCL_IB_TC'] == '104' and
            env['NCCL_SOCKET_IFNAME'] == '=enp1s0f1np1' and
            env['NCCL_IB_HCA'] == '=rocep1s0f1:1,roceP2p1s0f1:1',
            'TP2 requires the direct-link second-port configuration')
    for key, value in env.items():
        if value.startswith('/opt/glm/'):
            require(value.removeprefix('/opt/glm/') in assets, f'Missing module for {key}')
    for mount in bundle['mounts']:
        key = {'/opt/calibration':'TF_GLM_CALIBRATION_SHA256',
               '/opt/correlation':'TF_GLM_Q8_CORRELATION_SHA256'}[mount['target']]
        require(assets.get(mount['path'] + '/manifest.json') == env[key], f'{key} disagrees with the archive')
    env.update(NCCL_COMM_ID=f'{args.master}:{args.nccl_port}', TF_TP_MASTER=args.master,
               TF_TP_PLAN_PORT=str(args.plan_port), NCCL_IB_ADDR_RANGE=args.roce_address_range,
               TF_GLM_NVME_TIER='native-tiered' if tiered else '0',
               TF_GLM_NVME_MIN_CACHE_TOKENS=str(args.require_cache_tokens))
    if tiered:
        env['TF_GLM_NVME_STAGING_MIB'] = '256'
    config = dict(tensor_parallel=2, topology='direct', precision=args.precision,
                  source_commit=source['source_commit'], native_sha256=source['native_sha256'],
                  model=recipe['checkpoints']['exl3'], drafter=recipe['checkpoints']['drafter'],
                  model_root=str(roots['model']), drafter_root=str(roots['drafter']),
                  context=args.context, parallel=args.parallel, tier=args.tier, verify_rows=32,
                  context_and_parallel_match_measured_scope=(args.context == 196608 and args.parallel == 8),
                  prompt_cache_gib=args.prompt_cache_gib,
                  calibration_sha256=env.get('TF_GLM_CALIBRATION_SHA256'),
                  correlation_sha256=env.get('TF_GLM_Q8_CORRELATION_SHA256'))
    if tiered:
        config.update(learn_gib=args.learn_gib, require_cache_tokens=args.require_cache_tokens,
                      nvme_staging_mib=int(env['TF_GLM_NVME_STAGING_MIB']),
                      nvme_index_mib=int(env['TF_GLM_NVME_INDEX_MIB']))
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    namespace = f'tp2-{args.precision}-{source["source_commit"][:12]}-{fingerprint}'
    nodes, commands = [], []
    for rank, host in enumerate(hosts):
        payload = copy.deepcopy(recipe['payload_template'])
        rank_root = roots['cache'] / namespace / f'rank-{rank}'
        cache = rank_root / 'prefix'
        mounts = payload['HostConfig']['Mounts']
        for mount in mounts:
            key = mount['Source'].strip('@').lower()
            mount['Source'] = str(cache if key == 'cache' else roots[key])
        for mount in bundle['mounts']:
            mounts.append(dict(Type='bind', Source=str(roots['runtime'] / mount['path']),
                               Target=mount['target'], ReadOnly=True))
        prepare = ['mkdir', '-p', str(cache)]
        if bundle['mounts']:
            digests = rank_root / 'weight-digests'
            mounts.append(dict(Type='bind', Source=str(digests), ReadOnly=False,
                               Target='/root/.cache/tensorfold/file-digests/v1'))
            prepare.append(str(digests))
        rank_env = dict(env, TF_GLM_RANK=str(rank), TF_TP_RANK=str(rank))
        payload['Env'] = [f'{key}={value}' for key, value in sorted(rank_env.items())]
        for key, value in [('--port', args.port), ('--context', args.context), ('--parallel', args.parallel),
                           ('--prompt-cache-gib', format(args.prompt_cache_gib, 'g'))]:
            set_arg(payload['Cmd'], key, value)
        if tiered:
            payload['Cmd'] += ['--learn-conversations', '--learn-dir', '/cache', '--learn-gib', str(args.learn_gib)]
        payload['Labels'].update({'glm.recipe.source':source['source_commit'], 'glm.recipe.native':source['native_sha256'],
                                  'glm.recipe.rank':str(rank), 'glm.recipe.world':'2',
                                  'glm.recipe.checkpoint':'exl3', 'glm.recipe.precision':args.precision})
        name = f'{args.instance}-tp2-exl3-{args.precision}-r{rank}'
        nodes.append(dict(host=host, rank=rank, name=name, cache=str(cache), payload=payload))
        commands.append(dict(host=host, rank=rank, prepare_cache=shlex.join(prepare),
                             create=shlex.join(create_command(name, payload)), start=shlex.join(['docker','start',name])))
    return nodes, dict(**config, commands=commands, namespace=namespace, rank_control_port=args.nccl_port + 1,
                       source_branch=source['branch'], source_tag=source.get('tag'),
                       source_manifest_sha256=source['source_manifest_sha256'],
                       runtime_archive=bundle['archive'], runtime_assets_required=sorted(assets),
                       runtime_file_hashes_checked=args.verify_runtime, rendered_only=True, deployed=False)
