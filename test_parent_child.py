import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from experiments import parent_child

ROOT = Path(__file__).resolve().parent


class ParentChildMigrationTests(unittest.TestCase):
    def test_cross_minute_fragment_is_one_target_but_shared_context(self):
        segments = [
            {"id": "S1", "start": 59, "end": 61, "text": "Cancel that date."},
            {"id": "S2", "start": 60, "end": 63, "text": "Use Monday instead."},
        ]
        parents = parent_child.build_parents(segments)
        parent_child.verify_layout(segments, parents)
        self.assertEqual([p["target_ids"] for p in parents], [["S1"], ["S2"]])
        self.assertTrue(all(p["context_ids"] == ["S1", "S2"] for p in parents))

    def test_prepare_works_from_another_directory_without_a_key(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "prepared"
            command = [sys.executable, str(ROOT / "experiments/parent_child.py"),
                       str(ROOT / "examples/synthetic-transcript.json"), "--out", str(out)]
            result = subprocess.run(command, cwd=directory, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(list((out / "responses").iterdir()), [])
            source = json.loads((ROOT / "examples/synthetic-transcript.json").read_text())
            copied = json.loads((out / "source-transcript.json").read_text())
            self.assertEqual(source, copied)
            manifest = json.loads((out / "experiment.json").read_text())
            targets = [sid for p in manifest["parents"] for sid in p["target_ids"]]
            self.assertEqual(targets, [s["id"] for s in source["segments"]])

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "existing"
            out.mkdir()
            sentinel = out / "keep.txt"
            sentinel.write_text("prior experiment")
            result = subprocess.run([sys.executable, str(ROOT / "experiments/parent_child.py"),
                str(ROOT / "examples/synthetic-transcript.json"), "--out", str(out)],
                cwd=directory, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(sentinel.read_text(), "prior experiment")
            self.assertEqual(list(out.iterdir()), [sentinel])


if __name__ == "__main__":
    unittest.main()
