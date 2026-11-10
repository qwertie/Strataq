"""A test feed (a loopback address or a folder) used once on a PC is not the feed every later update check goes to."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_control import updater


def remembered(root, source):
    state = updater.State(root)
    state.data["feed"] = {"source": source}
    state.save()


class FeedMemoryTest(unittest.TestCase):
    def source(self, stored, **args):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop(updater.ENV_FEED, None)
            remembered(Path(tmp), stored)
            return updater.feed_source(args.get("feed"), Path(tmp))

    def test_a_loopback_test_feed_is_forgotten(self):
        self.assertEqual(self.source("http://127.0.0.1:18831"), updater.default_feed())

    def test_a_local_folder_is_forgotten(self):
        self.assertEqual(self.source(r"C:\rel\FEED-interim"), updater.default_feed())
        self.assertEqual(self.source("/tmp/feed"), updater.default_feed())

    def test_an_https_mirror_is_kept(self):
        self.assertEqual(self.source("https://mirror.example.com/strata-feed.json"), "https://mirror.example.com/strata-feed.json")

    def test_nothing_remembered_is_the_project_feed(self):
        self.assertEqual(self.source(None), updater.default_feed())

    def test_a_named_feed_still_wins(self):
        self.assertEqual(self.source("https://a.example/f.json", feed="http://127.0.0.1:1"), "http://127.0.0.1:1")


if __name__ == "__main__":
    unittest.main()
