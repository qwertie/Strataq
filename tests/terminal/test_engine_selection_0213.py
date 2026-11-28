"""Round 0213 TUICFG, step 4: choosing the engine on a PC with cards of several makes (#4971, #5111).

Every case is a temporary /sys/class/drm tree (a real two-vendor sysfs cannot be made on a test box), in both card orders."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from strata_common import engine_component as ec

AMD, INTEL, NVIDIA = "0x1002", "0x8086", "0x10de"


def build(root: Path, *cards) -> None:
    """cards: (name, vendor, device id or None, own memory in MiB or None); a name like card0 / card10 / card1-DP-1 is the directory."""
    for name, vendor, did, mib in cards:
        dev = root / name / "device"
        dev.mkdir(parents=True)
        if vendor:
            (dev / "vendor").write_text(vendor + "\n", encoding="utf-8")
        if did:
            (dev / "device").write_text(did + "\n", encoding="utf-8")
        if mib is not None:
            (dev / "mem_info_vram_total").write_text(str(mib * 1024 * 1024) + "\n", encoding="utf-8")


class Tree(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def backend(self, *cards):
        build(self.root, *cards)
        return ec._drm_vendor_backend(str(self.root))


APU = ("card0", AMD, "0x164e", 512)
ARC = ("card1", INTEL, "0xe211", None)


class Backend(Tree):
    def test_an_apu_beside_an_arc_is_not_an_amd_pc_in_either_order(self):
        self.assertEqual(self.backend(APU, ARC), "intel")
        self.tearDown(); self.setUp()
        self.assertEqual(self.backend(("card0", INTEL, "0xe211", None), ("card1", AMD, "0x164e", 512)), "intel")

    def test_an_arc_of_unknown_size_beside_an_apu(self):
        self.assertEqual(self.backend(APU, ("card1", INTEL, "0xe2f0", None)), "intel")

    def test_a_sole_apu_stays_amd(self):
        self.assertEqual(self.backend(APU), "hip")

    def test_a_sole_arc_is_intel(self):
        self.assertEqual(self.backend(ARC), "intel")

    def test_a_discrete_amd_card_and_nvidia_keep_their_order(self):
        self.assertEqual(self.backend(("card0", AMD, "0x7550", 16384), ARC), "hip")
        self.tearDown(); self.setUp()
        self.assertEqual(self.backend(APU, ("card1", NVIDIA, "0x2204", None)), "cuda")

    def test_a_large_unified_memory_apu_is_a_card(self):
        self.assertEqual(self.backend(("card0", AMD, "0x1586", 32768)), "hip")
        self.tearDown(); self.setUp()
        self.assertEqual(self.backend(("card0", AMD, "0x1586", 32768), ARC), "hip", "the older order; an explicit backend decides otherwise")

    def test_connectors_and_unreadable_entries_are_ignored(self):
        self.assertEqual(self.backend(("card0-DP-1", None, None, None), APU, ARC), "intel")

    def test_no_card_is_none(self):
        self.assertIsNone(ec._drm_vendor_backend(str(self.root / "missing")))


if __name__ == "__main__":
    unittest.main()
