"""#4675: a child process that writes non-UTF-8 bytes (a GBK display adapter name) must not kill Step 1.

    python -m unittest tools.test_setup_gbk_adapters
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402
from strata_common import engine_component as EC  # noqa: E402

GBK_LINE = b"Microsoft \xbb\xf9\xb1\xbe\xcf\xd4\xca\xbe\xca\xca\xc5\xe4\xc6\xf7|PCI/VEN_10DE&DEV_0A60&SUBSYS|0\r\n"


class GbkOutput(unittest.TestCase):
    def test_out_survives_bytes_that_are_not_utf8(self):
        code = "import sys; sys.stdout.buffer.write(%r)" % (GBK_LINE,)
        text = setup.out([sys.executable, "-X", "utf8", "-c", code])
        self.assertIsInstance(text, str)
        self.assertIn("Microsoft", text)
        self.assertIn("VEN_10DE", text)

    @unittest.skipUnless(os.name == "nt", "Win32_VideoController is read on Windows only")
    def test_adapter_list_never_gets_none(self):
        with mock.patch.object(EC.subprocess, "run", return_value=mock.Mock(stdout=None)):
            self.assertEqual(EC.windows_present_adapters(), [])
        name, pnp, ram = GBK_LINE.decode("utf-8", "replace").split("|")
        with mock.patch.object(EC.subprocess, "run", return_value=mock.Mock(stdout="|".join((pnp, name, ram)))):
            found = EC.windows_present_adapters()
        self.assertEqual(len(found), 1)
        self.assertEqual((found[0]["vendor"], found[0]["device"], found[0]["name"][:9]), ("10de", 0x0A60, "Microsoft"))

    def test_a_mangled_name_still_matches_its_registry_entry_by_pci_id(self):
        present = [{"vendor": "1002", "device": 0x744C, "name": "AMD Radeon(TM) Graphics �", "vram_gb": 0.0}]
        registry = [{"vendor": "1002", "device": 0x744C, "name": "AMD Radeon RX 7900 XTX", "driver": "32.0", "vram_gb": 24.0}]
        g = setup.amd_gpus_windows(present, registry)
        self.assertEqual(len(g), 1)
        self.assertEqual(g[0]["vram_gb"], 24.0)

if __name__ == "__main__":
    unittest.main()
