"""The release manifest (docs/data/release.json) and the pages that quote it."""
import copy
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import release_manifest as rm  # noqa: E402


class ReleaseManifest(unittest.TestCase):
    def test_current_pages_agree_with_the_manifest(self):
        problems = rm.check(rm.load())
        self.assertEqual(problems, [], "\n".join(problems))

    def test_version_lines_are_rewritten_and_a_second_pass_changes_nothing(self):
        m = rm.load()
        page, regex, keys = [r for r in rm.LINES if r[0] == "AGENTS.md"][0]
        text = "Version 0.2.5, engine 0.1.50. The public model families"
        once = rm.rewrite(text, regex, keys, m)
        self.assertEqual(once, "Version %s, engine %s. The public model families" % (m["product_version"], m["engine_version"]))
        self.assertEqual(rm.rewrite(once, regex, keys, m), once)

    def test_a_drifting_manifest_is_reported(self):
        m = copy.deepcopy(rm.load())
        m["product_version"] = "0.2.99"
        m["tag"] = "v0.2.99"
        m.pop("terminal_version", None)                             # the Terminal then has to say the product version
        problems = "\n".join(rm.check(m))
        self.assertIn("strata_cli/__init__.py says", problems)

    def test_separate_control_and_engine_versions_are_checked(self):
        """Checks independent Terminal, Control and engine release numbers."""
        m = rm.load()
        self.assertEqual(rm.check(m), [])
        wrong = copy.deepcopy(m)
        wrong["control_version"] = m["engine_version"]
        problems = "\n".join(rm.check(wrong))
        self.assertIn("VERSION (Control) says", problems)
        wrong = copy.deepcopy(m)
        wrong["engine_version"] = "0.1.58"
        self.assertIn("CMakeLists.txt (engine) says", "\n".join(rm.check(wrong)))
        wrong = copy.deepcopy(m)
        del wrong["terminal_version"]                               # no terminal_version: the Terminal must say the product version
        self.assertIn("strata_cli/__init__.py says 0.2.50", "\n".join(rm.check(wrong)))


if __name__ == "__main__":
    unittest.main()
