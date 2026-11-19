"""Render TP4 switch or TP2 direct-link serving payloads and commands locally."""
import argparse
import copy
import hashlib
import ipaddress
import json
import math
import pathlib
import re
import shlex

P = pathlib.Path(__file__).resolve().parent
MAX_CONTEXT = 1_048_576


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def set_arg(command, name, value):
    require(command.count(name) == 1, 'Recipe must have one ' + name)
    command[command.index(name) + 1] = str(value)


def profile_runtime(recipe, selected):
    keys = {'source_commit', 'source_manifest_sha256', 'native_sha256', 'native_binary'}
    runtime = selected.get('runtime')
    if runtime is None:
        runtime = {key: recipe[key] for key in keys - {'native_binary'}}
        runtime['native_binary'] = 'tensorfold-native'
    else:
        require(set(runtime) == keys, 'A profile runtime must pin its complete source and executable identity')
    for key, width in [('source_commit', 40), ('source_manifest_sha256', 64), ('native_sha256', 64)]:
        require(isinstance(runtime[key], str) and
            re.fullmatch('[0-9a-f]{' + str(width) + '}', runtime[key]) and set(runtime[key]) != {'0'},
            'Invalid profile runtime identity: ' + key)
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', runtime['native_binary']),
        'Profile executable must be a runtime asset basename')
    candidates = [item for item in recipe['runtime_assets'] if item['name'] == runtime['native_binary']]
    require(len(candidates) == 1 and candidates[0]['sha256'] == runtime['native_sha256'],
        'Profile executable is missing or disagrees with the runtime manifest')
    return runtime


def required_assets(recipe, profile):
    require(profile in recipe['profiles'], 'Unknown runtime profile')
    result = []
    for item in recipe['runtime_assets']:
        profiles = item.get('profiles')
        if profiles is not None:
            require(isinstance(profiles, list) and profiles and all(isinstance(x, str) for x in profiles),
                'Runtime asset profile scope must be a nonempty list')
            require(len(set(profiles)) == len(profiles) and set(profiles) <= set(recipe['profiles']),
                'Runtime asset profile scope has duplicate or unknown profiles')
        if profiles is None or profile in profiles:
            result.append(item)
    return result


def create_command(name, payload):
    config = payload['HostConfig']
    command = ['docker', 'create', '--name', name, '--network', config['NetworkMode'],
        '--ipc', config['IpcMode'], '--gpus', 'all', '--workdir', payload['WorkingDir'],
        '--shm-size', str(config['ShmSize']), '--memory', str(config['Memory']),
        '--memory-swap', str(config['MemorySwap']), '--cpuset-cpus', config['CpusetCpus']]
    for capability in config['CapAdd']:
        command += ['--cap-add', capability]
    for value in config['Ulimits']:
        command += ['--ulimit', value['Name'] + '=' + str(value['Soft']) + ':' + str(value['Hard'])]
    for device in config['Devices']:
        command += ['--device', device['PathOnHost'] + ':' + device['PathInContainer'] + ':' + device['CgroupPermissions']]
    for mount in config['Mounts']:
        command += ['--mount', 'type=bind,src=' + mount['Source'] + ',dst=' + mount['Target'] +
            (',readonly' if mount['ReadOnly'] else '')]
    for key, value in sorted(payload['Labels'].items()):
        command += ['--label', key + '=' + value]
    for value in payload['Env']:
        command += ['--env', value]
    command += ['--entrypoint', payload['Entrypoint'][0], payload['Image'], *payload['Cmd']]
    return command


def construct(args, recipe):
    require(args.expert_checkpoint in ('exl3', 'nvfp4'), 'Choose exl3 or nvfp4 experts')
    require(args.precision in ('bf16', 'q4', 'q4-fp8', 'q8'), 'Unknown projection/head precision')
    require(args.verify_rows in (16, 32), 'Verification rows must be16 or32')
    require(1 <= args.parallel <= 8 and 4096 <= args.context <= MAX_CONTEXT,
        f'TP4 requires 1 to 8 lanes and a context from 4096 to {MAX_CONTEXT}')
    require(math.isfinite(args.prompt_cache_gib) and 0 <= args.prompt_cache_gib <= 32, 'Invalid RAM cache size')
    require(math.isfinite(args.learn_gib) and 1 <= args.learn_gib <= 4096, 'Invalid NVMe cache size')
    require(args.tier in ('off', 'nvme') and args.prefill in ('whole', 'stepped'), 'Unknown cache/prefill mode')
    require(not args.warm_turns, 'Background warm turns have not passed activation qualification; keep them off')
    require(args.tier != 'nvme' or args.prompt_cache_gib > 0, 'Persistent snapshots require useful RAM cache room')
    require(0 <= args.require_cache_tokens <= MAX_CONTEXT, 'Invalid required cached-prefix size')
    require(args.tier != 'nvme' or args.require_cache_tokens <= args.context,
        'Required cached prefix exceeds the context; choose --require-cache-tokens explicitly')
    ipaddress.IPv4Address(args.master)
    ipaddress.IPv4Network(args.roce_address_range, strict=True)
    hosts = args.hosts or recipe['rank_hosts']
    require(len(hosts) == 4 and len(set(hosts)) == 4, 'Choose four distinct rank host labels')
    require(all(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,252}', h) for h in hosts),
        'Invalid rank host label')
    require(1024 <= args.nccl_port <= 65534, 'NCCL bootstrap port must leave room for its rank-control port +1')
    ports = [args.port, args.nccl_port, args.nccl_port + 1, args.plan_port]
    require(len(set(ports)) == 4 and all(1024 <= p <= 65535 for p in ports),
        'Reserve distinct HTTP, NCCL bootstrap, rank-control (+1) and gather-bootstrap ports')
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,40}', args.instance) is not None, 'Invalid instance name')
    roots = {key: getattr(args, key + '_root') for key in ('runtime', 'model', 'drafter', 'cache')}
    require(all(p.is_absolute() for p in roots.values()), 'All host roots must be absolute')
    require(len({str(p) for p in roots.values()}) == 4, 'Use distinct runtime/model/drafter/cache roots')
    require(all('..' not in p.parts for p in roots.values()), 'Use normalized host paths without parent traversal')
    for label in ('runtime', 'model', 'drafter'):
        require(roots['cache'] not in roots[label].parents and roots[label] not in roots['cache'].parents,
            'Writable cache and read-only ' + label + ' roots must not contain each other')
    require(all(not any(c in str(p) for c in (',', '\n', '\r', '\x00')) for p in roots.values()), 'Invalid mount path')
    profile = args.expert_checkpoint + '-' + args.precision
    require(profile in recipe['profiles'], 'This checkpoint/precision combination is not a selected serving option')
    selected = recipe['profiles'][profile]
    require(selected['qualification'].startswith('CLOSED_'),
        'This executable/profile has not completed its quality and performance gates')
    require(args.tier != 'nvme' or selected.get('NVMe_tier_qualification', True),
        'The retained NVFP4 executable has no qualified current NVMe worker; use --tier off')
    runtime = profile_runtime(recipe, selected)
    assets = required_assets(recipe, profile)
    names = {item['name'] for item in assets}
    require(runtime['native_binary'] in names, 'Profile executable is outside its runtime asset scope')
    environment = dict(recipe['common_environment'], **selected['environment'])
    environment.update(TF_GLM_VERIFY_ROWS=str(args.verify_rows),
        TF_GLM_FIRST_COPY_TOKENS=str(args.verify_rows - 1),
        TF_GLM_PROMPT_STEPS='1' if args.prefill == 'stepped' else '0',
        TF_GLM_WARM_TURNS='1' if args.warm_turns else '0',
        TF_GLM_NVME_TIER='native-tiered' if args.tier == 'nvme' else '0',
        TF_GLM_NVME_MIN_CACHE_TOKENS=str(args.require_cache_tokens),
        TF_TP_MASTER=args.master, TF_TP_PLAN_PORT=str(args.plan_port),
        NCCL_IB_ADDR_RANGE=args.roce_address_range,
        NCCL_COMM_ID=args.master + ':' + str(args.nccl_port))
    require(environment['NCCL_IB_TC'] == '104' and environment['TF_GLM_COMM_1107'] == '1',
        'This recipe requires the qualified switched borrowed-native transport')
    expected = ('16', '16', '0') if args.precision == 'bf16' else (
        ('8', '16', '0') if args.precision == 'q8' else (
            ('4', '8', '1') if args.precision == 'q4-fp8' else ('4', '16', '0')))
    require(tuple(environment[k] for k in ('TF_GLM_DENSE_BITS', 'TF_GLM_HEAD_BITS', 'TF_GLM_Q4_HEAD_FP8')) == expected,
        'Projection/head settings disagree')
    require(environment['TF_GLM_NVFP4_FFN_GRAPHS'] == ('1' if args.expert_checkpoint == 'nvfp4' else '0'),
        'Expert checkpoint and graph backend disagree')
    for key, value in environment.items():
        if value.startswith('/opt/glm/'):
            name = value.removeprefix('/opt/glm/')
            require('/' not in name and len([item for item in recipe['runtime_assets'] if item['name'] == name]) == 1,
                'Runtime module must name exactly one declared asset: ' + key)
            require(name in names, 'Runtime module is outside its profile asset scope: ' + key)
    settings = dict(source=runtime['source_commit'], binary_sha256=runtime['native_sha256'], expert_checkpoint=args.expert_checkpoint,
        precision=args.precision, verify_rows=args.verify_rows, context=args.context, parallel=args.parallel,
        prefill=args.prefill, warm_turns=args.warm_turns, tier=args.tier,
        model_repository=recipe['checkpoints'][args.expert_checkpoint]['repository'],
        model_revision=recipe['checkpoints'][args.expert_checkpoint]['revision'],
        drafter_repository=recipe['checkpoints']['drafter']['repository'],
        drafter_revision=recipe['checkpoints']['drafter']['revision'],
        model_root=str(roots['model']), drafter_root=str(roots['drafter']))
    key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]
    namespace = runtime['source_commit'][:12] + '-' + args.expert_checkpoint + '-' + args.precision + '-' + key
    nodes, commands = [], []
    for rank, host in enumerate(hosts):
        payload = copy.deepcopy(recipe['payload_template'])
        require(payload['Entrypoint'] == ['/opt/glm/tensorfold-native'], 'Unexpected native entrypoint template')
        payload['Entrypoint'] = ['/opt/glm/' + runtime['native_binary']]
        cache = roots['cache'] / namespace / ('rank-' + str(rank))
        for mount in payload['HostConfig']['Mounts']:
            target = mount['Source'].strip('@').lower()
            require(target in roots, 'Unknown recipe mount placeholder')
            mount['Source'] = str(cache if target == 'cache' else roots[target])
        env = dict(environment, TF_GLM_RANK=str(rank), TF_TP_RANK=str(rank))
        require(not any(k.startswith('GLM_') or k == 'TF_LANE_TRACE' for k in env),
            'Finite experiment controls must not enter serving')
        payload['Env'] = [key + '=' + value for key, value in sorted(env.items())]
        for flag, value in [('--port', args.port), ('--context', args.context), ('--parallel', args.parallel),
            ('--prompt-cache-gib', args.prompt_cache_gib)]:
            set_arg(payload['Cmd'], flag, value)
        if args.tier == 'nvme':
            payload['Cmd'] += ['--learn-conversations', '--learn-dir', '/cache', '--learn-gib', str(args.learn_gib)]
        payload['Labels'].update({'glm.recipe.source': runtime['source_commit'],
            'glm.recipe.native': runtime['native_sha256'], 'glm.recipe.rank': str(rank),
            'glm.recipe.checkpoint': args.expert_checkpoint, 'glm.recipe.precision': args.precision})
        name = args.instance + '-' + args.expert_checkpoint + '-' + args.precision + '-r' + str(rank)
        nodes.append(dict(host=host, rank=rank, name=name, cache=str(cache), payload=payload))
        commands.append(dict(host=host, rank=rank, prepare_cache=shlex.join(['mkdir', '-p', str(cache)]),
            create=shlex.join(create_command(name, payload)), start=shlex.join(['docker', 'start', name])))
    return nodes, dict(source_commit=runtime['source_commit'], native_sha256=runtime['native_sha256'],
        source_manifest_sha256=runtime['source_manifest_sha256'], native_binary=runtime['native_binary'],
        rank_control_port=args.nccl_port + 1,
        **settings, namespace=namespace, commands=commands,
        release_status=recipe['release_status'], profile_qualification=selected['qualification'],
        context_and_parallel_match_measured_scope=(args.context == 196608 and args.parallel == 8),
        rendered_only=True, deployed=False, proxy_registration_changed=False)


def main(args):
    require(args.output.is_absolute() and not args.output.exists(), 'Use a fresh absolute output directory')
    for root in (args.runtime_root, args.model_root, args.drafter_root, args.cache_root):
        require(args.output.resolve() != root.resolve() and root.resolve() not in args.output.resolve().parents,
            'Do not write rendered files inside runtime, weights or cache roots')
    if args.tp == 2:
        from tp2 import config
        path = P / 'tp2/recipe.json'
        source = json.loads((P / 'tp2/SOURCE.json').read_text())
        nodes, report = config.construct(args, json.loads(path.read_text()), source, create_command)
        report['recipe_sha256'] = sha(path)
        report['release_status'] = 'TP2_PROFILE_ASSEMBLY'
        write_configuration(args, nodes, report)
        return
    path = P / 'recipe.json'
    recipe = json.loads(path.read_text())
    assets = required_assets(recipe, args.expert_checkpoint + '-' + args.precision)
    if args.verify_runtime:
        executables = {profile_runtime(recipe, value)['native_binary'] for value in recipe['profiles'].values()}
        for item in assets:
            file = args.runtime_root / item['name']
            require(file.is_file(), 'Missing runtime asset: ' + item['name'])
            require(sha(file) == item['sha256'], 'Runtime digest mismatch: ' + item['name'])
            if item['name'] in executables:
                require((args.runtime_root / item['name']).stat().st_mode & 0o111,
                    'Native executable has no execute permission; restore the release archive modes')
    nodes, report = construct(args, recipe)
    report.update(recipe_sha256=sha(path), runtime_file_hashes_checked=args.verify_runtime,
        runtime_assets_required=[item['name'] for item in assets])
    write_configuration(args, nodes, report)


def write_configuration(args, nodes, report):
    args.output.mkdir(parents=True, exist_ok=False)
    for node in nodes:
        (args.output / ('rank-' + str(node['rank']) + '.json')).write_text(json.dumps(node, indent=2) + '\n')
    (args.output / 'CONFIGURATION.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(rendered=True, nodes=len(nodes), output=str(args.output), deployed=False,
        release_status=report['release_status'], runtime_verified=args.verify_runtime)))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tp', type=int, choices=[2, 4], default=4)
    p.add_argument('--expert-checkpoint', choices=['exl3', 'nvfp4'], default='exl3')
    p.add_argument('--precision', choices=['bf16', 'q4', 'q4-fp8', 'q8'], default='bf16')
    p.add_argument('--verify-rows', type=int, choices=[16, 32], default=32)
    p.add_argument('--tier', choices=['off', 'nvme'], default='off')
    p.add_argument('--prefill', choices=['whole', 'stepped'], default='stepped')
    p.add_argument('--warm-turns', action='store_true')
    for name in ('runtime', 'model', 'drafter', 'cache'):
        p.add_argument('--' + name + '-root', type=pathlib.Path, required=True)
    p.add_argument('--output', type=pathlib.Path, required=True)
    p.add_argument('--verify-runtime', action='store_true')
    p.add_argument('--instance', default='glm53')
    p.add_argument('--master', required=True, help='IPv4 bootstrap address on rank0')
    p.add_argument('--roce-address-range', required=True, help='IPv4 network containing the RoCE data addresses')
    p.add_argument('--hosts', nargs='+', metavar='HOST',
        help='Two or four host labels in tensor-parallel rank order')
    p.add_argument('--port', type=int, default=18888)
    p.add_argument('--nccl-port', type=int, default=28664)
    p.add_argument('--plan-port', type=int, default=28651)
    p.add_argument('--context', type=int, default=196608)
    p.add_argument('--parallel', type=int, default=8)
    p.add_argument('--prompt-cache-gib', type=float, default=1.75)
    p.add_argument('--learn-gib', type=float, default=32)
    p.add_argument('--require-cache-tokens', type=int, default=137216)
    return p


if __name__ == '__main__':
    main(parser().parse_args())
