"""FIX3 (11-04): two launcher and planning bugs found in round 0211.

1. The draft-layer lookup walked up from the model file to a folder named "models" and looked only beside it. With the model folder outside the data
   folder the plan asked for the 5.2 GB layer although <data-dir>/mtp/rt was there (a second download).
2. The plan's card line gave no way to tell the --gpu number (GPU 2 is card 1) and listed cards CUDA_VISIBLE_DEVICES hides. It counts from 1 like the screens and now adds the number --gpu takes:
   "GPU 2 (--gpu 1)".
All of it with fake cards and fake sysfs; nothing is started, probed or downloaded.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_cli.launch import cmd as launch_cmd
from strata_cli.launch import modelstate
from strata_cli.launch import settings as S
from strata_cli.launch.catalog import Catalog
from strata_cli.launch.runner import Runner
from strata_control import discovery
from strata_common import gpu_roles
from tests.terminal.launch_support import HW_12, Home, catalog, run_cli
from tests.terminal.test_own_files import IQ3XXS, gguf_bytes


class Base(unittest.TestCase):
    one_card_pc = True                                  # the plan tests run on a fake one-card PC; the others hand their own hardware to catalog()

    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)
        if self.one_card_pc:
            p = mock.patch.object(Catalog, "hardware", lambda self, fresh=False: dict(HW_12, disk_free_gb=500.0))
            p.start()
            self.addCleanup(p.stop)
        self.tmp = tempfile.TemporaryDirectory(prefix="strata-fix3-")
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)

    def model_files(self, folder: Path) -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        for n in IQ3XXS:
            (folder / n).write_bytes(gguf_bytes(["IQ3_XXS"]))
        return folder

    def layer_in(self, data: Path) -> Path:
        rt = data / "mtp" / "rt"
        rt.mkdir(parents=True, exist_ok=True)
        (rt / "experts.bin").write_bytes(b"x")
        return rt

    def plan_text(self, *args):
        code, out, err = run_cli(["launch", "--plan", *args])
        self.assertEqual(code, 0, err or out)
        return out


# ----------------------------------------------------------------------------------------------------------------------- 1. the draft layer

class DraftLayerInTheDataDir(Base):
    def test_prepared_beside_looks_in_the_data_dir_when_the_model_folder_is_elsewhere(self):
        data = self.base / "data"
        rt = self.layer_in(data)
        first = self.model_files(self.base / "ext" / "models" / "iq3_xxs") / IQ3XXS[0]
        self.assertIsNone(discovery.prepared_beside(first)["mtp"], "the old lookup: only beside the models folder")
        got = discovery.prepared_beside(first, [data])
        self.assertEqual(got["mtp"], rt)
        self.assertEqual(got["data_dir"], self.base / "ext", "the folder the model sits in is still what data_dir says")

    def test_a_model_folder_with_no_models_ancestor_finds_it_too(self):
        data = self.base / "data"
        rt = self.layer_in(data)
        first = self.model_files(self.base / "somewhere" / "IQ3_XXS") / IQ3XXS[0]
        got = discovery.prepared_beside(first, [None, self.base / "nope", data])
        self.assertEqual((got["mtp"], got["data_dir"]), (rt, None))

    def test_the_layer_beside_the_models_folder_wins_and_nothing_is_invented(self):
        beside, data = self.base / "ext", self.base / "data"
        mine = self.layer_in(beside)
        self.layer_in(data)
        first = self.model_files(beside / "models" / "iq3_xxs") / IQ3XXS[0]
        self.assertEqual(discovery.prepared_beside(first, [data])["mtp"], mine)
        none = discovery.prepared_beside(self.model_files(self.base / "lone" / "models" / "x") / IQ3XXS[0], [self.base / "empty"])
        self.assertEqual((none["mtp"], none["pack"]), (None, None))

    def test_a_pack_in_the_data_dir_is_found_by_the_models_tag(self):
        data = self.base / "data"
        (data / "packs" / "iq3_xxs").mkdir(parents=True)
        first = self.model_files(self.base / "ext" / "models" / "iq3_xxs") / IQ3XXS[0]
        self.assertIsNone(discovery.prepared_beside(first)["pack"])
        self.assertEqual(discovery.prepared_beside(first, [data])["pack"], data / "packs" / "iq3_xxs")

    def test_assess_counts_no_draft_layer_when_data_dir_has_it(self):
        data = self.base / "data"
        (data / "packs" / "iq3_xxs").mkdir(parents=True)
        self.layer_in(data)
        folder = self.model_files(self.base / "ext" / "models" / "iq3_xxs")
        c = catalog(HW_12)
        with mock.patch.object(Catalog, "data_dir", return_value=self.base / "elsewhere"):
            st = modelstate.assess(c, "qwen", "IQ3_XXS", vision="off", model_path=str(folder), data_dir=data)
            self.assertEqual([p.kind for p in st.parts], [])
            st = modelstate.assess(c, "qwen", "IQ3_XXS", vision="off", model_path=str(folder), data_dir=self.base / "empty")
            self.assertEqual([p.kind for p in st.parts], [], "no pack anywhere: the draft layer is not counted either (as before)")
            (self.base / "empty" / "packs" / "iq3_xxs").mkdir(parents=True)
            st = modelstate.assess(c, "qwen", "IQ3_XXS", vision="off", model_path=str(folder), data_dir=self.base / "empty")
        self.assertEqual([p.kind for p in st.parts], ["mtp"], "a pack but no layer anywhere: it is still asked for")

    def test_assess_finds_the_picture_reader_in_the_data_dir_too(self):
        data = self.base / "data"
        (data / "packs" / "iq3_xxs").mkdir(parents=True)
        self.layer_in(data)
        folder = self.model_files(self.base / "ext" / "models" / "iq3_xxs")
        with mock.patch.object(Catalog, "data_dir", return_value=self.base / "elsewhere"):
            st = modelstate.assess(catalog(HW_12), "qwen", "IQ3_XXS", vision="gpu", model_path=str(folder), data_dir=data)
            self.assertEqual([p.kind for p in st.parts], ["mmproj"], "a pack in the data dir but no reader anywhere: the reader is asked for")
            (data / "models").mkdir()
            (data / "models" / "mmproj-Qwen3.8-Flash-Next-BF16.gguf").write_bytes(b"x")
            st = modelstate.assess(catalog(HW_12), "qwen", "IQ3_XXS", vision="gpu", model_path=str(folder), data_dir=data)
        self.assertEqual([p.kind for p in st.parts], [])

    def test_the_remembered_data_dir_counts_when_none_is_given(self):
        remembered = self.base / "remembered"
        (remembered / "packs" / "iq3_xxs").mkdir(parents=True)
        self.layer_in(remembered)
        folder = self.model_files(self.base / "ext" / "models" / "iq3_xxs")
        with mock.patch.object(Catalog, "data_dir", return_value=remembered):
            st = modelstate.assess(catalog(HW_12), "qwen", "IQ3_XXS", vision="off", model_path=str(folder))
        self.assertEqual([p.kind for p in st.parts], [])

    # the plan as `strata launch --plan` prints it (check (a))
    def test_plan_says_nothing_to_download_for_a_model_folder_outside_the_data_dir(self):
        data = self.base / "data"
        self.layer_in(data)
        folder = self.model_files(self.base / "ext" / "IQ3_XXS")
        with mock.patch.object(Runner, "existing_data_dir", return_value=None), mock.patch.object(Catalog, "data_dir", return_value=self.base / "emptydata"):
            out = self.plan_text("--model-path", str(folder), "--data-dir", str(data), "--vision", "off")
        self.assertIn("download  Nothing to download: your model files are used where they are.", out)

    def test_plan_with_a_models_ancestor_outside_the_data_dir_is_the_same(self):
        data = self.base / "data"
        self.layer_in(data)
        folder = self.model_files(self.base / "ext" / "models" / "iq3_xxs")
        with mock.patch.object(Catalog, "data_dir", return_value=self.base / "emptydata"):
            out = self.plan_text("--model-path", str(folder), "--data-dir", str(data), "--vision", "off")
        self.assertIn("download  Nothing to download: your model files are used where they are.", out)

    def test_plan_still_asks_for_the_layer_when_no_data_dir_has_it(self):
        folder = self.model_files(self.base / "ext" / "models" / "iq3_xxs")
        with mock.patch.object(Catalog, "data_dir", return_value=self.base / "emptydata"):
            out = self.plan_text("--model-path", str(folder), "--data-dir", str(self.base / "data"), "--vision", "off")
        self.assertIn("About 5.2 GB to download: the draft layer (5.2 GB).", out)

    def test_the_picture_reader_in_the_data_dir_is_found_too(self):
        data = self.base / "data"
        self.layer_in(data)
        (data / "models").mkdir()
        (data / "models" / "mmproj-Qwen3.8-Flash-Next-BF16.gguf").write_bytes(b"x")
        folder = self.model_files(self.base / "ext" / "IQ3_XXS")
        with mock.patch.object(Runner, "existing_data_dir", return_value=None), mock.patch.object(Catalog, "data_dir", return_value=self.base / "emptydata"):
            out = self.plan_text("--model-path", str(folder), "--data-dir", str(data), "--vision", "on")
        self.assertIn("download  Nothing to download", out)

    def test_the_import_preview_sees_the_layer_in_the_data_folder_in_use(self):
        from strata_control import models
        data = self.base / "inuse"
        rt = self.layer_in(data)
        first = self.model_files(self.base / "ext" / "models" / "iq3_xxs") / IQ3XXS[0]
        with mock.patch.object(discovery, "remembered_data_dirs", return_value=[data]):
            got = models._already_set_up([first], {"family": "qwen", "model": "IQ3_XXS"})
        self.assertEqual(got["mtp"], str(rt))

    def test_remembered_data_dirs_reads_only_the_folders_that_exist(self):
        with mock.patch.object(discovery, "data_dir_choices", return_value=[(self.base, "settings"), (self.base / "gone", "settings"), (self.base, "documents")]):
            self.assertEqual(discovery.remembered_data_dirs(), [self.base])


def four(vendor="nvidia"):
    return [{"index": i, "name": "NVIDIA RTX A6000", "vram_gb": 48.0, "usable": True, "vendor": vendor} for i in range(4)]


class CardNumbers(Base):
    one_card_pc = False

    def line(self, env=None, vendor="nvidia", cards=None, **kw):
        s = S.LaunchSettings(**{"family": "qwen", "model": "UD-Q4_K_XL", **kw})
        c = catalog({**HW_12, "engine_vendor": vendor, "gpus": cards or four(vendor=vendor)})
        with mock.patch.dict(os.environ, env or {}):
            return launch_cmd.roles_line(s, c)

    def test_describe_keeps_the_screens_count_and_the_plan_adds_the_gpu_flag_number(self):
        cs = [{"index": 0, "name": "RTX 3060", "vram_gb": 12}, {"index": 1, "name": "RTX 4070 Ti", "vram_gb": 12}]
        self.assertEqual(gpu_roles.describe({0: "helper", 1: "main"}, cs), "Main: GPU 2 RTX 4070 Ti. Helper: GPU 1 RTX 3060.")
        self.assertEqual(gpu_roles.describe({0: "helper", 1: "main"}, cs, card_numbers=True), "Main: GPU 2 (--gpu 1) RTX 4070 Ti. Helper: GPU 1 (--gpu 0) RTX 3060.")

    def test_gpu_0_1_on_four_cards_with_cuda_visible_devices_0_1(self):
        got = self.line({"CUDA_VISIBLE_DEVICES": "0,1"}, gpus=[0, 1])
        self.assertEqual(got, "Main: GPU 2 (--gpu 1) RTX A6000. Helper: GPU 1 (--gpu 0) RTX A6000.", "no Off cards for the ones the environment hides")

    def test_without_the_variable_the_other_cards_are_off_with_their_own_numbers(self):
        got = self.line({}, gpus=[0, 1])
        self.assertEqual(got, "Main: GPU 2 (--gpu 1) RTX A6000. Helper: GPU 1 (--gpu 0) RTX A6000. Off: GPU 3 (--gpu 2) RTX A6000, GPU 4 (--gpu 3) RTX A6000.")

    def test_one_visible_card_named_by_number(self):
        self.assertEqual(self.line({"CUDA_VISIBLE_DEVICES": "1"}, gpus=[1]), "Main: GPU 2 (--gpu 1) RTX A6000.")

    def test_a_named_card_the_environment_hides_stays_because_the_server_sets_visibility_from_it(self):
        self.assertEqual(self.line({"CUDA_VISIBLE_DEVICES": "1"}, gpus=[0]), "Main: GPU 1 (--gpu 0) RTX A6000. Off: GPU 2 (--gpu 1) RTX A6000.")

    def test_roles_by_number(self):
        got = self.line({"CUDA_VISIBLE_DEVICES": "2,3"}, gpu_roles={"2": "helper", "3": "main"})
        self.assertEqual(got, "Main: GPU 4 (--gpu 3) RTX A6000. Helper: GPU 3 (--gpu 2) RTX A6000.")

    def test_amd_honours_hip_and_rocr(self):
        amd = four(vendor="amd")
        self.assertEqual(self.line({"HIP_VISIBLE_DEVICES": "0,1"}, vendor="amd", cards=amd, gpus=[0, 1]), "Main: GPU 2 (--gpu 1) RTX A6000. Helper: GPU 1 (--gpu 0) RTX A6000.")
        # ROCR first (cards 1 and 3 of the box), then HIP numbers those: HIP 0 is box card 1
        self.assertEqual(self.line({"ROCR_VISIBLE_DEVICES": "1,3", "HIP_VISIBLE_DEVICES": "0"}, vendor="amd", cards=amd, gpus=[1]), "Main: GPU 2 (--gpu 1) RTX A6000.")
        self.assertEqual(self.line({"ROCR_VISIBLE_DEVICES": "1,3"}, vendor="amd", cards=amd, gpus=[1, 3]), "Main: GPU 4 (--gpu 3) RTX A6000. Helper: GPU 2 (--gpu 1) RTX A6000.")

    def test_cuda_visible_devices_does_not_hide_amd_cards(self):
        self.assertEqual(self.line({"CUDA_VISIBLE_DEVICES": "0"}, vendor="amd", cards=four(vendor="amd"), gpus=[0, 1]),
                         "Main: GPU 2 (--gpu 1) RTX A6000. Helper: GPU 1 (--gpu 0) RTX A6000. Off: GPU 3 (--gpu 2) RTX A6000, GPU 4 (--gpu 3) RTX A6000.")


class EnvVisible(unittest.TestCase):
    def test_numbers_and_order(self):
        ev = lambda v, env, **kw: gpu_roles.env_visible([0, 1, 2, 3], v, env, **kw)             # noqa: E731
        self.assertIsNone(ev("nvidia", {}))
        self.assertEqual(ev("nvidia", {"CUDA_VISIBLE_DEVICES": "1,0"}, alike=True), [1, 0])
        self.assertEqual(ev("nvidia", {"CUDA_VISIBLE_DEVICES": ""}), [])
        self.assertEqual(ev("nvidia", {"CUDA_VISIBLE_DEVICES": "0,9,1"}, alike=True), [0], "an entry that is no card ends the list, as CUDA reads it")

    def test_a_uuid_or_an_ambiguous_order_is_not_guessed(self):
        self.assertIsNone(gpu_roles.env_visible([0, 1], "nvidia", {"CUDA_VISIBLE_DEVICES": "GPU-1234"}, alike=True))
        self.assertIsNone(gpu_roles.env_visible([0, 1], "nvidia", {"CUDA_VISIBLE_DEVICES": "1"}), "different cards, CUDA's own order: not the nvidia-smi numbers")
        self.assertEqual(gpu_roles.env_visible([0, 1], "nvidia", {"CUDA_VISIBLE_DEVICES": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}), [1])
        self.assertIsNone(gpu_roles.env_visible([0, 1], "amd", {"ROCR_VISIBLE_DEVICES": "GPU-1234"}))
        self.assertIsNone(gpu_roles.env_visible([0, 1], "intel", {"CUDA_VISIBLE_DEVICES": "0"}))


if __name__ == "__main__":
    unittest.main()
