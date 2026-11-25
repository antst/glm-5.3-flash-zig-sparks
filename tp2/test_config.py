"""Offline regression checks for the unified rank renderer."""
import json
from pathlib import Path
import tempfile
import unittest

import render_config
from tp2 import config

ROOT = Path(__file__).resolve().parent


class DirectRecipeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recipe = json.loads((ROOT/'recipe.json').read_text())
        cls.sources = json.loads((ROOT/'SOURCE.json').read_text())

    def arguments(self, *extra):
        return render_config.parser().parse_args([
            '--tp','2','--runtime-root','/srv/runtime','--model-root','/srv/model',
            '--drafter-root','/srv/drafter','--cache-root','/var/cache/glm53','--output','/tmp/rendered',
            '--master','10.79.12.1','--roce-address-range','10.79.0.0/16','--no-verify-runtime',*extra])

    def render(self,*extra):
        return config.construct(self.arguments(*extra),self.recipe,self.sources,render_config.create_command,render_config.cache_budget)

    def test_profiles_preserve_ranks_and_separate_cache_names(self):
        namespaces = set()
        for profile, expected in config.HEADS.items():
            nodes, report = self.render('--precision',profile)
            self.assertEqual([node['host'] for node in nodes],['spark1','spark2'])
            namespaces.add(report['namespace'])
            for rank,node in enumerate(nodes):
                env = dict(value.split('=',1) for value in node['payload']['Env'])
                self.assertEqual(env['TF_GLM_RANK'],str(rank))
                self.assertEqual(env['TF_TP_RANK'],str(rank))
                self.assertEqual((env['TF_GLM_WORLD'],env['NCCL_IB_TC']),('2','104'))
                self.assertEqual(tuple(env.get(k,'0') for k in ('TF_GLM_DENSE_BITS','TF_GLM_HEAD_BITS',
                                                                'TF_GLM_Q4_HEAD_FP8')),expected)
                mounts = {value['Target']:value for value in node['payload']['HostConfig']['Mounts']}
                self.assertEqual('/opt/calibration' in mounts,profile!='bf16')
                self.assertEqual('/opt/correlation' in mounts,profile=='q8')
                for target in ('/opt/calibration','/opt/correlation'):
                    if target in mounts:
                        self.assertTrue(mounts[target]['ReadOnly'])
        self.assertEqual(len(namespaces),4)

    def test_nvme_defaults_and_explicit_off(self):
        nodes, report = self.render('--precision','q8')
        for node in nodes:
            env = dict(value.split('=',1) for value in node['payload']['Env'])
            self.assertEqual(env['TF_GLM_NVME_TIER'],'native-tiered')
            self.assertEqual(env['TF_GLM_NVME_MODE'],'serial')
            self.assertEqual(env['TF_GLM_NVME_STAGING_MIB'],'256')
            self.assertEqual(env['TF_GLM_NVME_INDEX_MIB'],'64')
            self.assertIn('--learn-conversations',node['payload']['Cmd'])
        _, other = self.render('--precision','q8','--tier','nvme','--learn-gib','48')
        self.assertEqual(report['namespace'],other['namespace'])
        self.assertEqual((report['learn_gib'], other['learn_gib']), (256, 48))
        off, _ = self.render('--precision','q8','--tier','off')
        self.assertIn('TF_GLM_NVME_STAGING_MIB=192',off[0]['payload']['Env'])

    def test_refuses_mismatched_topology_and_ports(self):
        with self.assertRaisesRegex(ValueError,'two distinct'):
            self.render('--hosts','spark1','spark2','spark3')
        with self.assertRaisesRegex(ValueError,'distinct HTTP'):
            self.render('--port','28665')
        with self.assertRaisesRegex(ValueError,'direct-link IPv4'):
            self.render('--master','192.168.2.201')

    def test_million_token_q8_profile(self):
        flags = ('--precision','q8','--context','1048576','--parallel','1','--prompt-cache-gib','0','--tier','off')
        nodes, report = self.render(*flags)
        self.assertEqual((report['context'], report['parallel'], report['tier']), (1048576, 1, 'off'))
        for node in nodes:
            argv = node['payload']['Cmd']
            self.assertEqual(argv[argv.index('--context')+1], '1048576')
        _, other = self.render(*flags,'--parallel','4','--precision','q4')
        self.assertEqual(other['context'],1048576)
        self.assertFalse(other['context_and_parallel_match_measured_scope'])
        with self.assertRaisesRegex(ValueError,'1048576'):
            self.render(*flags,'--context','1048577')

    def test_switch_recipe_accepts_model_context(self):
        recipe = json.loads((ROOT.parent/'recipe.json').read_text())
        args = self.arguments('--tp','4','--master','192.168.2.201','--roce-address-range','192.168.2.0/24',
                              '--context','1048576','--parallel','1','--prompt-cache-gib','0','--tier','off')
        nodes, report = render_config.construct(args,recipe)
        self.assertEqual((len(nodes),report['context']), (4,1048576))
        self.assertFalse(report['context_and_parallel_match_measured_scope'])

    def test_exl3_profiles_default_to_the_model_context(self):
        switch = json.loads((ROOT.parent/'recipe.json').read_text())
        for precision in ('bf16', 'q4', 'q4-fp8', 'q8'):
            with self.subTest(topology='tp2', precision=precision):
                nodes, report = self.render('--precision', precision)
                self.assert_full_context(nodes, report, 4)
            with self.subTest(topology='tp4', precision=precision):
                args = self.arguments('--tp', '4', '--master', '192.168.2.201',
                                      '--roce-address-range', '10.78.0.0/15',
                                      '--precision', precision)
                nodes, report = render_config.construct(args, switch)
                self.assert_full_context(nodes, report, 6)
        for recipe, lanes in ((self.recipe, 4), (switch, 6)):
            self.assert_full_context_args(recipe['payload_template']['Cmd'], lanes)

    def test_default_precision_is_q4_fp8_on_both_topologies(self):
        switch = json.loads((ROOT.parent/'recipe.json').read_text())
        switch_args = ('--tp', '4', '--master', '192.168.2.201',
                       '--roce-address-range', '10.78.0.0/15')
        tp4 = render_config.construct(self.arguments(*switch_args), switch)
        for topology, (nodes, report), lanes in (('tp2', self.render(), 4), ('tp4', tp4, 6)):
            with self.subTest(topology=topology):
                self.assertEqual(report['precision'], 'q4-fp8')
                self.assert_full_context(nodes, report, lanes)
                for node in nodes:
                    env = dict(value.split('=', 1) for value in node['payload']['Env'])
                    self.assertEqual((env['TF_GLM_DENSE_BITS'], env['TF_GLM_HEAD_BITS'],
                                      env['TF_GLM_Q4_HEAD_FP8']), ('4', '8', '1'))
        self.assertEqual(tuple(switch['defaults'][field] for field in
                               ('precision', 'context', 'parallel')), ('q4-fp8', 1048576, 6))

    def test_explicit_lane_overrides(self):
        _, serving = self.render('--parallel', '1')
        _, bench = self.render('--context', '196608', '--parallel', '8',
                               '--prompt-cache-gib', '0', '--tier', 'off')
        self.assertEqual(serving['parallel'], 1)
        self.assertEqual(bench['parallel'], 8)
        self.assertTrue(bench['context_and_parallel_match_measured_scope'])

        switch = json.loads((ROOT.parent/'recipe.json').read_text())
        for lanes in (4, 6, 8):
            args = self.arguments('--tp', '4', '--master', '192.168.2.201',
                                  '--roce-address-range', '10.78.0.0/15', '--parallel', str(lanes))
            nodes, report = render_config.construct(args, switch)
            self.assert_full_context(nodes, report, lanes)

    def assert_full_context(self, nodes, report, lanes):
        self.assertEqual((report['context'], report['parallel']), (1048576, lanes))
        self.assertEqual(report['tier'], 'nvme')
        self.assertEqual(report['prompt_cache_gib'], 8)
        self.assertEqual(report['learn_gib'], 256)
        for node in nodes:
            self.assert_full_context_args(node['payload']['Cmd'], lanes)
            env = dict(item.split('=', 1) for item in node['payload']['Env'])
            self.assertEqual(env['TF_GLM_PREFILL_OPT'], '1')
            self.assertEqual(env['TF_KV_PAGES'], 'off')
            self.assertEqual(env['TF_GLM_NVME_TIER'], 'native-tiered')
            self.assertEqual(env['TF_GLM_NVME_MIN_CACHE_TOKENS'], '1048576')
            self.assertIn('--learn-conversations', node['payload']['Cmd'])
            cmd = node['payload']['Cmd']
            self.assertEqual(cmd[cmd.index('--prompt-cache-gib')+1], '8')
            self.assertEqual(cmd[cmd.index('--learn-gib')+1], '256')

    def assert_full_context_args(self, argv, lanes):
        self.assertEqual(argv[argv.index('--context') + 1], '1048576')
        self.assertEqual(argv[argv.index('--parallel') + 1], str(lanes))
        self.assertEqual(argv[argv.index('--max-tokens') + 1], '131072')

    def test_rejects_bad_runtime_bytes_and_escape(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'asset').write_text('correct')
            item = dict(path='asset',bytes=7,sha256=config.digest(root/'asset'))
            config.verify_runtime(root,[item])
            (root/'asset').write_text('changed')
            with self.assertRaisesRegex(ValueError,'digest mismatch'):
                config.verify_runtime(root,[item])
            with self.assertRaisesRegex(ValueError,'inventory path'):
                config.verify_runtime(root,[dict(item,path='../asset')])


if __name__ == '__main__':
    unittest.main()
