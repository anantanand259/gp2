import runpy
import shlex
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'rag_backend'))
from deployment_config import allowed_origins, server_port, storage_base


class DeploymentTests(unittest.TestCase):
    def test_render_startup_resolves_config_before_gunicorn_runs(self):
        # Gunicorn resolves --config from its launch directory, before --chdir.
        # Read the actual Blueprint command to catch the deployment regression.
        line = next(line for line in (ROOT / 'render.yaml').read_text().splitlines()
                    if line.strip().startswith('startCommand:'))
        command = shlex.split(line.split(':', 1)[1].strip())
        self.assertEqual(command[:3], ['cd', 'rag_backend', '&&'])
        self.assertEqual(command[3], 'gunicorn')
        launch_dir = ROOT / command[1]
        config_path = launch_dir / command[command.index('--config') + 1]
        self.assertTrue(config_path.is_file(), str(config_path))
        self.assertTrue((launch_dir / 'server.py').is_file())
        self.assertEqual(command[-1], 'server:app')

    def test_render_port_precedes_local_port(self):
        self.assertEqual(server_port({'PORT': '10000', 'RAG_PORT': '5000'}), 10000)
        self.assertEqual(server_port({'RAG_PORT': '5050'}), 5050)
        self.assertEqual(server_port({}), 5000)

    def test_storage_preserves_local_and_drive_setup(self):
        self.assertEqual(storage_base({}, '/local'), Path('/local'))
        self.assertEqual(storage_base({'GOOGLE_DRIVE_PATH': '/drive'}, '/local'), Path('/drive'))
        self.assertEqual(storage_base({'RAG_STORAGE_DIR': '/data', 'GOOGLE_DRIVE_PATH': '/drive'}, '/local'), Path('/data'))

    def test_hosted_cors_does_not_include_development_wildcard(self):
        self.assertEqual(allowed_origins({'RAG_ALLOWED_ORIGINS': 'https://example.edu, https://example.org'}, ['*']),
                         ['https://example.edu', 'https://example.org'])
        self.assertEqual(allowed_origins({}, ['*']), ['*'])

    def test_single_worker_for_shared_disk_and_provider_locks(self):
        config = runpy.run_path(str(ROOT / 'rag_backend/gunicorn.conf.py'))
        self.assertEqual(config['workers'], 1)
        self.assertEqual(config['worker_class'], 'gthread')
        self.assertGreater(config['threads'], 1)


if __name__ == '__main__':
    unittest.main()
