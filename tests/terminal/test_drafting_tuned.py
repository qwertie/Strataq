"""The tuned DFlash2 drafter (round 0216, DF-SHIP): where it applies on CUDA and HIP, how it is fetched from the signed feed, and its pins.

The pinned pack is replaced by a small synthetic file whose size and SHA-256 are patched in, so nothing here needs 340 MB.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import sys

from strata_common import acquisition, drafting
from strata_control import drafting_fetch
from strata_base import transfer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packaging"))
import feed_tool as ft  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
Q4 = "qwen3.8-flash-next-unsloth-ud-q4_k_xl"
V31 = "qwen3.8-flash-next-unsloth-ud-iq4_xs-v31"
IQ3S = "qwen3.8-flash-next-iq3_s"
Q5 = "qwen3.8-flash-next-unsloth-ud-q5_k_xl"
TUNED_BYTES = b"tuned-pack-" * 200_000                      # 2.2 MB, stands in for the 340 MB file
STOCK_BYTES = b"s" * (2 << 20)
#: The engine family and card of Strix Halo, where SET_DEFAULTS gives the 4-bit sets the tuned pack whatever the card's memory.
STRIX_HALO = {"backend": "hip", "archs": ["gfx1151"]}


def flag(args, name):
    return args[args.index(name) + 1] if name in args and args.index(name) + 1 < len(args) else None


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / "data"
        for sub in ("dflash2", "mtp/rt", "packs/tag"):
            (self.data / sub).mkdir(parents=True)
        (self.data / "dflash2" / drafting.VOCAB_NAME).write_bytes(b"v" * 100)
        (self.data / "mtp" / "rt" / "experts.bin").write_bytes(b"e" * 100)
        (self.data / "mtp" / "rt" / drafting.VOCAB_NAME).write_bytes(b"v" * 100)
        self.exe = self.root / "strata"
        self.exe.write_bytes(b"--mtp-draft-vocab FILE (or --mtp DIR ... --draft-fallback ... selector codebooks: Q4_0 rows")   # an engine of this tree
        self.args = ["--native", "m.gguf", "--pack", str(self.data / "packs" / "tag"), "--mtp", str(self.data / "mtp" / "rt"), "--spec", "2", "--max-context", "65536"]
        self.pins = mock.patch.multiple(drafting, TUNED_BYTES=len(TUNED_BYTES), TUNED_SHA256=hashlib.sha256(TUNED_BYTES).hexdigest())
        self.pins.start()
        self.addCleanup(self.pins.stop)
        patcher = mock.patch.object(drafting, "ENGINE_SERVES_DFLASH2", True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def put_tuned(self, content=TUNED_BYTES):
        (self.data / "dflash2" / drafting.TUNED_NAME).write_bytes(content)

    def put_stock(self):
        (self.data / "dflash2" / drafting.PACK_NAME).write_bytes(STOCK_BYTES)

    def cfg(self, name=Q4, **extra):
        return {"model_name": name, "args": list(self.args), **extra}


class TunedFile(Base):
    def test_the_marker_is_not_trusted_for_a_changed_file(self):
        self.put_tuned()
        self.assertTrue(drafting.tuned_ok(self.data))
        self.assertTrue((self.data / "dflash2" / (drafting.TUNED_NAME + ".ok")).is_file())
        self.put_tuned(b"y" * len(TUNED_BYTES))
        self.assertFalse(drafting.tuned_ok(self.data))


class Fetch(Base):
    def setUp(self):
        super().setUp()
        self.host = self.root / "host"
        self.host.mkdir()
        (self.host / drafting.TUNED_NAME).write_bytes(TUNED_BYTES)
        self.url = (self.host / drafting.TUNED_NAME).as_uri()
        env = {k: v for k, v in __import__("os").environ.items() if k not in (drafting.TUNED_URL_ENV, "STRATA_UPDATE_FEED", "STRATA_UPDATE_TRUSTED_KEYS")}
        env["STRATA_UPDATE_FEED"] = str(self.root / "no-feed-here")       # nothing in these tests reaches the network
        patcher = mock.patch.dict("os.environ", env, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.keys = self.root / "keys"
        self.pub = ft.keygen(self.keys, allow_git=True)
        (self.root / "trust.json").write_text((self.keys / "feed-key.pub.json").read_text(encoding="utf-8"), encoding="utf-8")
        __import__("os").environ["STRATA_UPDATE_TRUSTED_KEYS"] = str(self.root / "trust.json")

    def make_feed(self, sub="feed", listed=True, backends=("cuda", "hip"), pin_sha=None):
        """A local, test-signed release folder: the feed and the drafter beside it as a flat asset (what the GitHub release would hold)."""
        out = self.root / sub
        entries = self.root / (sub + "-entries")
        if listed:
            entry = ft.pack_drafter(self.host / drafting.TUNED_NAME, entries, version="1", backends=list(backends))
            if pin_sha:
                entry["sha256"] = pin_sha
                (entries / (drafting.TUNED_NAME + ".entry.json")).write_text(json.dumps(entry), encoding="utf-8")
        out.mkdir(parents=True, exist_ok=True)
        if listed:
            (out / drafting.TUNED_NAME).write_bytes(TUNED_BYTES)
        ft.build_feed(out, key=ft._load_key(self.keys / "feed-key.json"), sequence=1, drafters_dir=entries if listed else None)
        return out

    def test_a_fetch_from_a_file_address_checks_places_and_marks_the_file_and_copies_the_licence(self):
        seen = []
        out = drafting_fetch.ensure_tuned(self.data, url=self.url, progress=lambda *a: seen.append(a))
        self.assertEqual(out, drafting.tuned_path(self.data))
        self.assertTrue(drafting.tuned_ok(self.data))
        self.assertEqual(seen[-1][1:], (len(TUNED_BYTES), len(TUNED_BYTES)))
        self.assertFalse((self.data / "dflash2" / (drafting.TUNED_NAME + ".part")).exists())
        for name in ("LICENSE-dflash2-tuned", "NOTICE-dflash2-tuned.md"):
            self.assertTrue((self.data / "dflash2" / name).is_file(), name)
        licence = (self.data / "dflash2" / "LICENSE-dflash2-tuned").read_text(encoding="utf-8")
        notice = (self.data / "dflash2" / "NOTICE-dflash2-tuned.md").read_text(encoding="utf-8")
        self.assertIn("Apache License", licence)
        self.assertIn("Version 2.0", licence)
        for word in ("modified copy", "a22a1744e85c17364dcc54a6a420af7135d4c7f6", "Inco AI", "Strata changed"):
            self.assertIn(word, notice)
        again = drafting_fetch.ensure_tuned(self.data, url="file:///no/such/file")     # already here: no download
        self.assertEqual(again, out)

    def test_the_normal_way_is_the_signed_feed(self):
        feed = self.make_feed()
        seen = []
        out = drafting_fetch.ensure_tuned(self.data, feed=str(feed), progress=lambda *a: seen.append(a))
        self.assertEqual(out, drafting.tuned_path(self.data))
        self.assertTrue(drafting.tuned_ok(self.data))
        self.assertEqual(out.read_bytes(), TUNED_BYTES)
        self.assertTrue((self.data / "dflash2" / "LICENSE-dflash2-tuned").is_file())
        # the environment names the feed too, and ensure_for takes the tuned pack from it without touching the stock download
        out.unlink()
        out.with_name(out.name + ".ok").unlink()
        with mock.patch.dict("os.environ", {"STRATA_UPDATE_FEED": str(feed)}), mock.patch.object(drafting_fetch, "ensure", side_effect=AssertionError("stock fetched")):
            self.assertEqual(drafting_fetch.ensure_for(self.data, model_name=Q4, **STRIX_HALO), out)

    def test_the_feed_entry_is_a_drafters_member_and_older_clients_ignore_it(self):
        feed = self.make_feed()
        doc = json.loads((feed / "strata-feed.json").read_text(encoding="utf-8"))
        entry = doc["drafters"]["packages"][0]
        self.assertEqual((entry["name"], entry["backends"], entry["size"], entry["sha256"]),
                         (drafting.TUNED_NAME, ["cuda", "hip"], len(TUNED_BYTES), hashlib.sha256(TUNED_BYTES).hexdigest()))
        self.assertEqual(entry["url"], drafting.TUNED_NAME, "a flat release asset")
        parsed = drafting_fetch.feedmod.parse((feed / "strata-feed.json").read_bytes(), source=str(feed / "strata-feed.json"), key_id="k")
        self.assertEqual([d.name for d in parsed.drafters], [drafting.TUNED_NAME])
        doc.pop("drafters")
        self.assertEqual(drafting_fetch.feedmod.parse(json.dumps(doc).encode(), source="x", key_id="k").drafters, ())

    def test_a_later_per_platform_switch_is_a_table_entry_and_a_feed_backend(self):
        feed = self.make_feed(sub="both", backends=("hip", "cuda"))
        card = dict(archs=["sm_120"], args=["--max-context", "65536"], vram_mib=73728)
        with mock.patch.dict(drafting.SET_DEFAULTS, {("cuda", "unsloth-ud-q4_k_xl"): drafting.SetDefault("dflash2", True, "resident")}):
            self.assertEqual(drafting_fetch.ensure_for(self.data, model_name=Q4, backend="cuda", feed=str(feed), **card), drafting.tuned_path(self.data))
        self.put_stock()
        self.assertFalse(drafting.tuned_applies("cuda", Q4, **card), "without the entry CUDA is back to the stock pack")

    def test_a_feed_that_does_not_list_it_or_lists_other_bytes_gives_nothing(self):
        empty = self.make_feed(listed=False)
        self.assertIsNone(drafting_fetch.ensure_tuned(self.data, feed=str(empty)))
        other = self.make_feed(sub="other", pin_sha="0" * 64)
        self.assertIsNone(drafting_fetch.ensure_tuned(self.data, feed=str(other)), "other bytes than the ones this Strata measured")
        elsewhere = self.make_feed(sub="hip-only", backends=["hip"])
        self.assertIsNone(drafting_fetch.ensure_tuned(self.data, feed=str(elsewhere)))
        self.assertFalse(drafting.tuned_path(self.data).exists())

    def test_a_feed_with_a_bad_signature_is_refused_in_a_line(self):
        feed = self.make_feed()
        text = (feed / "strata-feed.json").read_text(encoding="utf-8")
        (feed / "strata-feed.json").write_text(text.replace('"sequence": 1', '"sequence": 2'), encoding="utf-8")
        with self.assertRaises(drafting_fetch.DraftingError) as caught:
            drafting_fetch.ensure_tuned(self.data, feed=str(feed))
        self.assertIn("could not be trusted", str(caught.exception))
        self.assertFalse(drafting.tuned_path(self.data).exists())
        with mock.patch.object(drafting_fetch, "ensure", return_value=Path("/stock")):
            self.assertEqual(drafting_fetch.ensure_for(self.data, model_name=Q4, **STRIX_HALO, feed=str(feed)), Path("/stock"), "the stock drafter takes over")

    def test_a_feed_that_cannot_be_read_or_a_build_with_no_key_is_silent(self):
        self.assertIsNone(drafting_fetch.ensure_tuned(self.data, feed=str(self.root / "no-such-feed")))
        feed = self.make_feed()
        with mock.patch.dict("os.environ", {"STRATA_UPDATE_TRUSTED_KEYS": ""}), mock.patch("strata_common.signing.load_trusted_keys", return_value={}):
            self.assertIsNone(drafting_fetch.ensure_tuned(self.data, feed=str(feed)))

    def test_a_launch_that_got_nothing_does_not_ask_again_for_twelve_hours_but_setup_always_asks(self):
        nofeed = str(self.root / "no-feed-here")
        self.assertIsNone(drafting_fetch.ensure_tuned(self.data, feed=nofeed, throttle=True))
        self.assertTrue((self.data / "dflash2" / drafting_fetch.MISS_MARK).is_file())
        with mock.patch.object(drafting_fetch, "tuned_from_feed", side_effect=AssertionError("asked again")):
            self.assertIsNone(drafting_fetch.ensure_tuned(self.data, feed=nofeed, throttle=True), "a launch a moment later: no wait")
        feed = self.make_feed()
        self.assertEqual(drafting_fetch.ensure_tuned(self.data, feed=str(feed)), drafting.tuned_path(self.data), "setup (no throttle) asks and gets it")
        self.assertFalse((self.data / "dflash2" / drafting_fetch.MISS_MARK).exists(), "and the mark is gone")
        old = self.data / "dflash2" / drafting_fetch.MISS_MARK
        old.write_text("x")
        import os
        os.utime(old, (1, 1))
        self.assertFalse(drafting_fetch._recent_miss(self.data), "a mark older than the interval is ignored")

    def test_with_no_feed_named_the_projects_release_feed_is_read(self):
        from strata_control import updater
        asked = []

        def fetch(source, trusted, **kw):
            asked.append(source)
            raise drafting_fetch.feedmod.FeedError("the update feed could not be read: offline", 6)
        env = {k: v for k, v in __import__("os").environ.items() if k != updater.ENV_FEED}
        with mock.patch.dict("os.environ", env, clear=True), mock.patch("strata_control.feed.fetch", fetch):
            self.assertIsNone(drafting_fetch.ensure_tuned(self.data))
        self.assertEqual(asked, [updater.default_feed()])
        self.assertTrue(updater.default_feed().startswith("https://github.com/qwertie/Strataq/releases/"))

    def test_a_download_that_fails_the_safety_check_is_deleted_and_said(self):
        (self.host / drafting.TUNED_NAME).write_bytes(b"z" * len(TUNED_BYTES))
        with self.assertRaises(drafting_fetch.DraftingError) as caught:
            drafting_fetch.ensure_tuned(self.data, url=self.url)
        self.assertIn("safety check", str(caught.exception))
        self.assertFalse(drafting.tuned_path(self.data).exists())
        self.assertFalse((self.data / "dflash2" / (drafting.TUNED_NAME + ".part")).exists())

    def test_an_interrupted_download_resumes(self):
        partial = self.data / "dflash2" / (drafting.TUNED_NAME + ".part")
        partial.write_bytes(TUNED_BYTES[:123_456])
        out = drafting_fetch.ensure_tuned(self.data, url=self.url)
        self.assertTrue(drafting.tuned_ok(self.data))
        self.assertEqual(out.read_bytes(), TUNED_BYTES)
        partial.write_bytes(TUNED_BYTES[:99])
        out.unlink()
        calls = []
        real = transfer._transfer

        def spy(location, part, have, size, progress, cancel, timeout):
            calls.append(have)
            return real(location, part, have, size, progress, cancel, timeout)
        with mock.patch.object(transfer, "_transfer", spy):
            drafting_fetch.ensure_tuned(self.data, url=self.url)
        self.assertEqual(calls, [99], "it continued at byte 99, not from the start")

    def test_no_internet_is_silent(self):
        said = []
        with mock.patch.object(drafting_fetch, "online", return_value=False):
            self.assertIsNone(drafting_fetch.ensure_tuned(self.data, url="https://example.invalid/x.pack", progress=lambda *a: said.append(a)))
            self.assertIsNone(drafting_fetch.ensure_for(self.data, model_name=Q4, **STRIX_HALO, progress=lambda *a: said.append(a)))
        self.assertEqual(said, [], "nothing is reported and nothing is written")
        self.assertFalse((self.data / "dflash2" / drafting.TUNED_NAME).exists())
        self.assertFalse((self.data / "dflash2" / (drafting.TUNED_NAME + ".part")).exists())

    def test_a_host_that_answers_with_an_error_page_is_silent_too(self):
        def not_found(*_a, **_k):
            raise transfer.TransferError("the update server answered 404 for x")
        with mock.patch.object(drafting_fetch, "online", return_value=True), mock.patch.object(transfer, "download", not_found):
            self.assertIsNone(drafting_fetch.ensure_tuned(self.data, url="https://example.invalid/x.pack"))

    def test_a_start_never_downloads(self):
        with acquisition.forbidden():
            with self.assertRaises(acquisition.AcquisitionRefused):
                drafting_fetch.ensure_tuned(self.data, url=self.url)

    def test_the_set_gets_the_tuned_pack_and_not_the_stock_download(self):
        with mock.patch.dict("os.environ", {drafting.TUNED_URL_ENV: self.url}), mock.patch.object(drafting_fetch, "ensure", side_effect=AssertionError("stock fetched")):
            got = drafting_fetch.ensure_for(self.data, model_name=V31, **STRIX_HALO)
        self.assertEqual(got, drafting.tuned_path(self.data))

    def test_when_the_tuned_pack_cannot_be_had_the_stock_drafter_is_fetched(self):
        with mock.patch.dict("os.environ", {drafting.TUNED_URL_ENV: "file:///no/such/host/file.pack"}), \
                mock.patch.object(drafting_fetch, "ensure", return_value=Path("/stock")) as stock:
            got = drafting_fetch.ensure_for(self.data, model_name=Q4, **STRIX_HALO)
        self.assertEqual(got, Path("/stock"))
        stock.assert_called_once()

    def test_a_failed_tuned_check_falls_back_and_only_a_double_failure_is_said(self):
        (self.host / drafting.TUNED_NAME).write_bytes(b"z" * len(TUNED_BYTES))
        with mock.patch.dict("os.environ", {drafting.TUNED_URL_ENV: self.url}), mock.patch.object(drafting_fetch, "ensure", return_value=Path("/stock")):
            self.assertEqual(drafting_fetch.ensure_for(self.data, model_name=Q4, **STRIX_HALO), Path("/stock"))
        (self.host / drafting.TUNED_NAME).write_bytes(b"z" * len(TUNED_BYTES))
        with mock.patch.dict("os.environ", {drafting.TUNED_URL_ENV: self.url}), \
                mock.patch.object(drafting_fetch, "ensure", side_effect=drafting_fetch.OfflineError("no internet")):
            with self.assertRaises(drafting_fetch.DraftingError) as caught:
                drafting_fetch.ensure_for(self.data, model_name=Q4, **STRIX_HALO)
        self.assertIn("safety check", str(caught.exception))

    def test_a_stock_pack_already_here_is_kept_when_the_tuned_one_is_offline(self):
        self.put_stock()
        with mock.patch.object(drafting_fetch, "online", return_value=False), mock.patch.object(drafting_fetch, "ensure", side_effect=AssertionError("fetched")):
            got = drafting_fetch.ensure_for(self.data, model_name=Q4, **STRIX_HALO)
        self.assertEqual(got, drafting.pack_path(self.data))

    def test_a_set_without_a_tuned_entry_is_fetched_as_before(self):
        with mock.patch.object(drafting_fetch, "ensure", return_value=Path("/stock")) as stock:
            self.assertEqual(drafting_fetch.ensure_for(self.data, model_name=Q5, **STRIX_HALO), Path("/stock"))
            self.assertEqual(drafting_fetch.ensure_for(self.data, model_name=Q4, backend="cuda", archs=["sm_120"], vram_mib=73728), Path("/stock"))
        self.assertEqual(stock.call_count, 2)


OLD_XS = "qwen3.8-flash-next-unsloth-ud-iq4_xs"
GYRO = "qwen3.8-flash-next-gyro-gyro-s"
#: (name, backend, archs, the one card's memory in MiB or None) of the boxes DF-VAL measured and of the ones it must not reach
L620 = ("l620 RTX PRO 5000 72 GB", "cuda", ["sm_120"], 73728)
SPARK = ("DGX Spark GB10", "cuda", ["sm_121"], None)
STRIX = ("Strix Halo", "hip", ["gfx1151"], None)
P2003 = ("RTX 3060 12 GB", "cuda", ["sm_86"], 12288)
CARD24 = ("a 24 GB card", "cuda", ["sm_89"], 24576)
R9700 = ("R9700 32 GB", "hip", ["gfx1201"], 32768)
RX7900 = ("RX 7900 XTX", "hip", ["gfx1100"], 24576)
TWO_CARDS = ("two RTX PRO cards", "cuda", ["sm_120", "sm_120"], None)


class CudaAndHip(Fetch):
    """Owner 11-06 ~10:00: the tuned pack is the DFlash2 pack at each platform's CURRENT confidence, (a) on CUDA when every expert is resident (the CONF_STEPS first step, 0.35: l620, the Spark),
    (b) on Strix Halo (gfx1151); for the sets DF-VAL measured there. Not a 12 GB or streaming card, not gfx1201, not several cards."""

    def apply_on(self, hardware, name, **extra):
        _label, backend, archs, vram = hardware
        cfg = self.cfg(name, **extra)
        decision = drafting.apply(cfg, data_dir=self.data, backend=backend, archs=archs, engine=self.exe, vram_mib=vram)
        return decision, cfg["args"]

    def drop_tuned(self):
        (self.data / "dflash2" / drafting.TUNED_NAME).unlink(missing_ok=True)
        (self.data / "dflash2" / (drafting.TUNED_NAME + ".ok")).unlink(missing_ok=True)

    def test_the_measured_cells_get_the_tuned_pack_at_the_conf_the_backend_already_had(self):
        self.put_tuned()
        self.put_stock()
        cells = ((L620, IQ3S), (SPARK, IQ3S), (SPARK, V31), (STRIX, IQ3S), (STRIX, Q4), (STRIX, OLD_XS), (STRIX, V31))
        for hardware, name in cells:
            self.put_tuned()
            d, args = self.apply_on(hardware, name)
            where = (hardware[0], name)
            self.assertEqual((d.drafter, d.note), ("dflash2", ""), where)
            self.assertEqual(flag(args, "--draft-model"), str(drafting.tuned_path(self.data)), where)
            self.drop_tuned()                                  # the same start without the tuned file: the stock pack, everything else the same
            d2, args2 = self.apply_on(hardware, name)
            self.assertEqual(flag(args2, "--draft-model"), str(drafting.pack_path(self.data, self.exe)), where)
            self.assertNotEqual(flag(args2, "--draft-model"), flag(args, "--draft-model"), where)
            self.assertEqual(flag(args2, "--draft-conf"), "0.35", where)
            self.assertEqual(flag(args, "--draft-conf"), "0.55" if (hardware is L620 and name == IQ3S) else "0.35", where)      # IQ3_S on l620's class with the tuned pack: 0.55 (11-08); the stock pack keeps 0.35
            self.assertEqual(flag(args, "--spec"), flag(args2, "--spec"), where)
            strip = lambda a: [x for i, x in enumerate(a) if not (i and a[i - 1] in ("--draft-model", "--draft-conf"))]      # noqa: E731  (the conf is compared above)
            self.assertEqual(strip(args), strip(args2), where)

    def test_the_cells_nobody_measured_are_not_enabled(self):
        self.put_tuned()
        self.put_stock()
        for hardware, name in ((L620, Q4), (L620, OLD_XS), (SPARK, Q4), (L620, Q5), (STRIX, Q5), (L620, GYRO), (STRIX, GYRO)):
            d, args = self.apply_on(hardware, name)
            self.assertNotEqual(flag(args, "--draft-model"), str(drafting.tuned_path(self.data)), (hardware[0], name))
            self.assertFalse(drafting.tuned_applies(hardware[1], name, hardware[2], args=[], vram_mib=hardware[3]), (hardware[0], name))

    def test_a_12_gb_card_an_r9700_several_cards_and_other_amd_cards_never_get_it(self):
        self.put_tuned()
        self.put_stock()
        for hardware in (P2003, CARD24, R9700, RX7900, TWO_CARDS):
            for name in (IQ3S, Q4, V31, OLD_XS):
                if hardware is P2003 and name == V31:
                    continue                                    # the 12 GB class has its own entry for v31 (test_drafting_streaming_class.py)
                d, args = self.apply_on(hardware, name)
                where = (hardware[0], name)
                self.assertNotEqual(flag(args, "--draft-model"), str(drafting.tuned_path(self.data)), where)
                self.assertFalse(drafting.tuned_applies(hardware[1], name, hardware[2], args=self.args, vram_mib=hardware[3]), where)
                self.assertFalse(drafting.needs_tuned("auto", model_name=name, backend=hardware[1], archs=hardware[2], data_dir=self.data, args=self.args, vram_mib=hardware[3]), where)

    def test_their_confidences_are_exactly_what_they_were(self):
        """The conf of every box class, with the tuned file here and without it, against the numbers CONF_STEPS has always given."""
        self.assertEqual(drafting.CONF_STEPS, ((12000, 0.35), (7500, 0.5), (3000, 0.7), (0, 0.8)))
        self.assertEqual((drafting.DRAFT_CONF, drafting.SPEC_WINDOW), (0.35, 6))
        self.put_stock()
        want = {L620[0]: "0.55", SPARK[0]: "0.35", STRIX[0]: "0.35", P2003[0]: "0.8"}
        for hardware in (L620, SPARK, STRIX, P2003):
            name = V31 if hardware is P2003 else IQ3S           # (IQ3S on the 12 GB class is the MTP now: no DFlash2 conf to compare)
            before = flag(self.apply_on(hardware, name)[1], "--draft-conf")
            self.put_tuned()
            after = flag(self.apply_on(hardware, name)[1], "--draft-conf")
            self.drop_tuned()
            self.assertEqual(before, "0.35" if hardware is L620 else after, hardware[0])      # (l620's IQ3_S: 0.55 only with the tuned pack)
            self.assertEqual(after, want[hardware[0]], hardware[0])
        for hardware in (L620, SPARK, STRIX):
            self.assertEqual(drafting.conf_for_slots(drafting.slots_of(self.args, hardware[3])), 0.35, hardware[0])
        cuda_hip = {k: v for k, v in drafting.SET_DEFAULTS.items() if k[0] in ("cuda", "hip")}
        self.assertEqual(set(cuda_hip), {("cuda", "iq3_s"), ("cuda", "unsloth-ud-iq4_xs-v31"), ("hip", "iq3_s"), ("hip", "unsloth-ud-q4_k_xl"), ("hip", "unsloth-ud-iq4_xs"), ("hip", "unsloth-ud-iq4_xs-v31")})
        self.assertEqual({k: v.where for k, v in cuda_hip.items() if k[0] == "cuda"}, {("cuda", "iq3_s"): "resident", ("cuda", "unsloth-ud-iq4_xs-v31"): "resident"})

    def test_the_resident_rule_reads_the_signals_the_confidence_reads(self):
        for given, qualifies in (("10000", True), ("9400", True), ("9000", False), ("3000", False), ("64", False)):       # an --expert-cache N is 1.3 slots per N; the first step is 12,000 slots
            args = ["--max-context", "65536", "--expert-cache", given]
            self.assertEqual(drafting.tuned_applies("cuda", IQ3S, ["sm_120"], args=args, vram_mib=12288), qualifies, given)
            self.assertEqual(drafting.conf_for_slots(drafting.slots_of(args, 12288)) == drafting.DRAFT_CONF, qualifies, "the same boundary as the 0.35 step")
        for vram in (73728, 49152, 32768, 24576, 16384, 12288, 8192):
            slots = drafting.slots_of(list(self.args), vram)
            self.assertEqual(drafting.tuned_applies("cuda", IQ3S, ["sm_120"], args=list(self.args), vram_mib=vram), slots >= 12000, vram)
            self.assertEqual(drafting.conf_for_slots(slots) == 0.35, drafting.tuned_applies("cuda", IQ3S, ["sm_120"], args=list(self.args), vram_mib=vram), vram)
        self.assertFalse(drafting.tuned_applies("cuda", IQ3S, ["sm_120"], args=list(self.args), vram_mib=None), "a card whose memory is not known is not resident")
        self.assertTrue(drafting.tuned_applies("cuda", IQ3S, ["sm_121"], args=[], vram_mib=None), "the Spark: one pool of memory")
        self.assertTrue(drafting.tuned_applies("hip", IQ3S, ["gfx1151"]))
        self.assertFalse(drafting.tuned_applies("hip", IQ3S, ["gfx1150"]))
        self.assertFalse(drafting.tuned_applies("hip", IQ3S, []), "an unknown AMD card")
        self.assertFalse(drafting.tuned_applies("cuda", IQ3S, []), "an unknown NVIDIA card")

    def test_fetch_follows_the_rule_and_reads_the_feed_only_when_it_applies(self):
        feed = self.make_feed(sub="all", backends=("cuda", "hip"))
        for hardware, name, fetched in ((L620, IQ3S, True), (SPARK, V31, True), (STRIX, Q4, True), (STRIX, V31, True), (P2003, IQ3S, False), (R9700, IQ3S, False),
                                        (TWO_CARDS, IQ3S, False), (L620, Q4, False)):
            (self.data / "dflash2").mkdir(parents=True, exist_ok=True)
            for f in (self.data / "dflash2").glob("*strata1*"):
                f.unlink()
            _label, backend, archs, vram = hardware
            where = (hardware[0], name)
            self.assertEqual(drafting.needs_tuned("auto", model_name=name, backend=backend, archs=archs, data_dir=self.data, args=self.args, vram_mib=vram), fetched, where)
            with mock.patch.object(drafting_fetch, "ensure", return_value=Path("/stock")) as stock:
                got = drafting_fetch.ensure_for(self.data, model_name=name, backend=backend, feed=str(feed), archs=archs, args=self.args, vram_mib=vram)
            if fetched:
                self.assertEqual(got, drafting.tuned_path(self.data), where)
                stock.assert_not_called()
            else:
                self.assertEqual(got, Path("/stock"), where)
                self.assertFalse(drafting.tuned_path(self.data).exists(), where)
        for name in (IQ3S, Q4):                                    # the feed was never read for the ones that do not qualify
            with mock.patch.object(drafting_fetch, "tuned_from_feed", side_effect=AssertionError("feed read")), mock.patch.object(drafting_fetch, "ensure", return_value=Path("/stock")):
                for hardware in (P2003, R9700, TWO_CARDS):
                    drafting_fetch.ensure_for(self.data, model_name=name, backend=hardware[1], archs=hardware[2], args=self.args, vram_mib=hardware[3])

    def test_a_feed_that_lists_only_metal_gives_cuda_and_hip_the_stock_drafter(self):
        feed = self.make_feed(sub="mac-only", backends=("metal",))
        for hardware, name in ((L620, IQ3S), (STRIX, Q4)):
            with mock.patch.object(drafting_fetch, "ensure", return_value=Path("/stock")) as stock:
                got = drafting_fetch.ensure_for(self.data, model_name=name, backend=hardware[1], feed=str(feed), archs=hardware[2], args=self.args, vram_mib=hardware[3])
            self.assertEqual(got, Path("/stock"), hardware[0])
            stock.assert_called_once()

    def test_the_fallback_order_is_tuned_then_stock_then_the_mtp_and_offline_is_silent(self):
        said = []
        with mock.patch.object(drafting_fetch, "ensure", side_effect=drafting_fetch.OfflineError("no internet")):
            for hardware, name in ((L620, IQ3S), (SPARK, IQ3S), (STRIX, Q4)):
                got = drafting_fetch.ensure_for(self.data, model_name=name, backend=hardware[1], archs=hardware[2], args=self.args, vram_mib=hardware[3],
                                                feed=str(self.root / "no-feed-here"), progress=lambda *a: said.append(a))
                self.assertIsNone(got, hardware[0])
        self.assertEqual(said, [])
        self.assertEqual(self.apply_on(L620, IQ3S)[0].drafter, "mtp", "nothing fetched: the MTP, exactly as before")
        self.put_stock()
        d, args = self.apply_on(L620, IQ3S)
        self.assertEqual((d.drafter, flag(args, "--draft-model")), ("dflash2", str(drafting.pack_path(self.data, self.exe))), "the stock pack")

    def test_the_plan_says_the_small_download_only_where_it_applies(self):
        self.assertEqual(drafting_fetch.plan(self.data, model_name=IQ3S, backend="cuda", archs=["sm_121"], args=self.args)["gb"], drafting.TUNED_GB)
        self.assertEqual(drafting_fetch.plan(self.data, model_name=Q4, backend="hip", archs=["gfx1151"], args=self.args)["gb"], drafting.TUNED_GB)
        self.assertEqual(drafting_fetch.plan(self.data, model_name=IQ3S, backend="cuda", archs=["sm_86"], args=self.args, vram_mib=12288)["gb"], drafting.DOWNLOAD_GB)
        self.assertEqual(drafting_fetch.plan(self.data, model_name=IQ3S, backend="hip", archs=["gfx1201"], args=self.args, vram_mib=32768)["gb"], drafting.DOWNLOAD_GB)

    def test_the_card_memory_read(self):
        class Reader:
            def __init__(self, total):
                self.total = total

            def ok(self):
                return self.total is not None

            def read(self):
                return {"mem_total": self.total}
        gib = 1 << 30
        with mock.patch("strata_common.telemetry.gpu_reader", lambda index, amd=False: Reader(72 * gib)):
            self.assertEqual(drafting.card_vram_mib({"gpu": 0}, "cuda", ["sm_120"]), 72 * 1024)
            self.assertEqual(drafting.card_vram_mib({}, "cuda", ["sm_120"]), 72 * 1024, "no card named: card 0, as the server asks")
            self.assertIsNone(drafting.card_vram_mib({"gpu": [0, 1]}, "cuda", ["sm_120"]), "several cards")
            self.assertIsNone(drafting.card_vram_mib({"gpu": "0,1"}, "cuda", ["sm_120"]))
            self.assertIsNone(drafting.card_vram_mib({"gpu": 0}, "cuda", ["sm_121"]), "the Spark: one pool")
            self.assertIsNone(drafting.card_vram_mib({"gpu": 0}, "hip", ["gfx1151"]), "Strix Halo")
        with mock.patch("strata_common.telemetry.gpu_reader", lambda index, amd=False: Reader(None)):
            self.assertIsNone(drafting.card_vram_mib({"gpu": 0}, "cuda", ["sm_120"]), "a driver that does not answer")


class Pins(unittest.TestCase):
    def test_the_pinned_numbers_are_the_final_pack(self):
        self.assertEqual(drafting.TUNED_BYTES, 339694592)
        self.assertEqual(drafting.TUNED_BYTES, drafting.CONVERTED_BYTES, "the same format and size as the stock Q4_0-selector pack")
        self.assertEqual(drafting.TUNED_SHA256, "4df2e5ab2b47c064afbe662e16e4f06569877125c2d0e2123a4ac755266c664c")
        self.assertNotEqual(drafting.TUNED_SHA256, drafting.CONVERTED_SHA256)
        self.assertNotEqual(drafting.TUNED_NAME, drafting.CONVERTED_NAME)

    def test_the_licence_files_ship_in_data(self):
        folder = ROOT / drafting.TUNED_LICENCE_FOLDER
        self.assertIn("Apache License", (folder / "LICENSE").read_text(encoding="utf-8"))
        notice = (folder / "NOTICE.md").read_text(encoding="utf-8")
        self.assertIn(drafting.TUNED_SHA256, notice)
        self.assertIn("a22a1744e85c17364dcc54a6a420af7135d4c7f6", notice)

    def test_the_base_revision_is_the_one_the_stock_drafter_is_pinned_to(self):
        self.assertIn(drafting.REVISION, drafting.TUNED_BASE)

    def test_set_names(self):
        self.assertEqual(drafting.set_of(Q4), "unsloth-ud-q4_k_xl")
        self.assertEqual(drafting.set_of(V31), "unsloth-ud-iq4_xs-v31")
        self.assertEqual(drafting.set_of("Qwen3.8-Flash-Next-IQ3_S"), "iq3_s")
        self.assertEqual(drafting.set_of(None), "")
        self.assertTrue(drafting.supported_model(V31), "the v31 name passes the supported-model pattern")


if __name__ == "__main__":
    unittest.main()
