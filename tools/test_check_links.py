"""The local link check (tools/check_links.py)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_links as cl  # noqa: E402
import doc_pages  # noqa: E402


class LinkCheck(unittest.TestCase):
    def test_current_pages_have_no_broken_local_links(self):
        problems = []
        for page in doc_pages.current_pages():
            problems += cl.check_page(page, want_anchors=True)
        self.assertEqual(problems, [], "\n".join("%s:%d %s (%s)" % (p, n, t, w) for p, n, t, w in problems))

    def test_links_in_code_and_comments_are_not_checked(self):
        text = "[a](real.md) `[b](code.md)`\n```\n[c](fence.md)\n```\n<!-- [d](comment.md) -->\n[e](https://example.com/x.md) [f](#top) [g](other.md#h)\n"
        self.assertEqual([t for _, t in cl.targets(text)], ["real.md", "https://example.com/x.md", "#top", "other.md#h"])

    def test_a_missing_target_is_reported_and_urls_are_not_local(self):
        self.assertIsNone(cl.resolve("docs/README.md", "https://github.com/Niko1221/Strata"))
        self.assertIsNone(cl.resolve("docs/README.md", "mailto:a@b.c"))
        self.assertTrue(cl.resolve("docs/README.md", "INSTALL.md").endswith(os.path.join("docs", "INSTALL.md")))
        self.assertTrue(cl.resolve("docs/README.md", "/LICENSE").endswith("LICENSE"))


if __name__ == "__main__":
    unittest.main()
