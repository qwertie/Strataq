"""Checks the shared opt-in path without starting an inference engine."""
import contextlib
import io
import unittest

from strata_common import engine_client, paged_launch
from strata_cli.launch.runner import derive_config
from strata_cli.launch.settings import LaunchSettings


class PagedLaunchTests(unittest.TestCase):
    def test_saved_token_limits_reach_both_engines(self):
        args = ['--max-context', '204800', '--conversation-tokens', '180000', '--tokens-max', '401000',
                '--vram-conversations', '3', '--conversation-idle-seconds', '66', '--decode-share-percent', '0',
                '--prompt-turn-ms', '20000']
        for exe in ('strata-q38', 'strata-ds41'):
            with self.subTest(exe=exe):
                self.assertEqual(engine_client.engine_args({'exe': exe, 'args': args}), args)

    def test_both_families_translate_generated_options(self):
        for exe in ('strata-q38.exe', 'strata-ds41.exe'):
            with self.subTest(exe=exe), contextlib.redirect_stdout(io.StringIO()) as log:
                args = engine_client.engine_args({
                    'exe': exe, 'parallel': 3, 'args': ['--pack', 'pack', '--max-context', '8192',
                    '--slots', '2', '--tp', '2', '--batch-groups', 'auto', '--suffix-draft', '1',
                    '--vision', '--gpu-mib', '4096', '--ram-mib', '2048']})
                self.assertEqual(args, ['--pack', 'pack', '--max-context', '8192', '--batch', '2',
                                        '--batch-groups', 'auto', '--suffix-draft', '1', '--vision',
                                        *(['on'] if exe == 'strata-ds41.exe' else []),
                                        '--gpu-mib', '4096', '--ram-mib', '2048'])
                self.assertIn('unsupported settings', log.getvalue())
                self.assertIn('--tp 2', log.getvalue())

    def test_group_options_reach_both_engines(self):
        for exe in ('strata-q38', 'strata-ds41'):
            for groups in ('1', '2', 'auto'):
                args = ['--batch-groups', groups, '--batch', '2']
                self.assertEqual(engine_client.engine_args({'exe': exe, 'args': args}), args)

    def test_deepseek_groups_survive_disabled_drafting(self):
        for groups in ('1', '2', 'auto'):
            cfg = {'exe': 'strata-ds41.exe', 'drafting': 'off', 'parallel': 4,
                   'args': ['--batch-groups', groups]}
            args = engine_client.engine_args(cfg)
            self.assertEqual(args[:2], ['--batch-groups', groups])
            self.assertEqual(args[args.index('--batch') + 1], '4')
            self.assertEqual(args[args.index('--dspark') + 1], 'none')

    def test_lookup_options_reach_both_engines(self):
        args = ['--suffix-draft', '3', '--lookup-chain', '2', '--lookup-chain-min', '6', '--mtp-max-t', '4']
        for exe in ('strata-q38', 'strata-ds41'):
            self.assertEqual(engine_client.engine_args({'exe': exe, 'args': args}), args)

    def test_model_off_keeps_explicit_lookup(self):
        for exe in ('strata-q38', 'strata-ds41'):
            cfg = {'exe': exe, 'mtp': False,
                   'args': ['--mtp', 'weights', '--suffix-draft', '3', '--lookup-chain', '2']}
            args = engine_client.engine_args(cfg)
            self.assertNotIn('--mtp', args)
            self.assertNotIn('--lookahead', args)
            self.assertEqual(args[:4], ['--suffix-draft', '3', '--lookup-chain', '2'])
            cfg['drafting'] = 'off'
            self.assertIn('--lookahead', engine_client.engine_args(cfg))

    def test_parallel_and_explicit_scheduling(self):
        args = engine_client.engine_args({'exe': 'strata-ds41', 'args': ['--concurrent', '2'], 'parallel': 3})
        self.assertEqual(args, ['--concurrent', '2', '--batch', '3'])

    def test_terminal_uses_existing_set_mechanism(self):
        s = LaunchSettings(extra={'--exe': 'C:/engine/strata-ds41.exe', '--gpu-mib': '4096'})
        cfg = derive_config({'exe': 'strata-dsv41.exe', 'args': []}, s, bool_flags=frozenset())
        self.assertEqual(cfg['exe'], s.extra['--exe'])
        self.assertEqual(cfg['args'], ['--gpu-mib', '4096'])

    def test_legacy_path_is_not_selected(self):
        for exe in ('strata.exe', 'strata-dsv41.exe', 'strata-q36.exe'):
            self.assertFalse(paged_launch.is_paged_engine({'exe': exe}))

    def test_windows_path_and_disabled_drafting(self):
        cfg = {'exe': r'C:\engine\strata-ds41.exe', 'args': ['--dspark', 'on'], 'mtp': False}
        self.assertTrue(paged_launch.is_paged_engine(cfg))
        self.assertEqual(engine_client.engine_args(cfg)[-2:], ['--dspark', 'none'])

    def test_gpu_names_and_explicit_pool_budget(self):
        self.assertEqual(engine_client.engine_args({'exe': 'strata-ds41', 'args': ['--gpu', '1,0',
                                                                             '--vram-reserve-mib', '700']}),
                         ['--gpus', '1,0', '--reserve-mib', '700'])
        cfg = {'exe': 'strata-q38', 'args': ['--gpus', '1', '--resident-budget-gib', '2.5']}
        self.assertEqual(engine_client.engine_args(cfg), ['--gpu', '1', '--ram-mib', '2560'])
        cfg['args'] += ['--ram-mib', '1024']
        self.assertEqual(engine_client.engine_args(cfg), ['--gpu', '1', '--ram-mib', '1024'])

    def test_terminal_keeps_the_paged_deepseek_choice(self):
        from strata_cli.launch.runner import Runner
        from unittest.mock import Mock
        runner = Mock()
        runner.s = LaunchSettings()
        cfg = {'exe': 'strata-ds41.exe', 'arch': 'deepseek41'}
        Runner.use_own_dsv41_engine(runner, cfg)
        runner.dsv41_engine_found.assert_not_called()
        self.assertEqual(cfg['exe'], 'strata-ds41.exe')

    def test_an_explicit_program_needs_no_default_package(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import Mock
        from strata_cli.launch.runner import Runner
        with tempfile.TemporaryDirectory() as directory:
            exe = Path(directory) / 'strata-q38.exe'
            exe.touch()
            runner = Mock()
            runner.s = LaunchSettings(extra={'--exe': str(exe)})
            Runner.ensure_engine(runner)
            runner.multi_file.assert_not_called()
            runner.emit.assert_called_once()
