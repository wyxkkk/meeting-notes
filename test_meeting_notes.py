import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import meeting_notes as app


class PreservationTests(unittest.TestCase):
    def test_whisper_milliseconds_and_negation_are_preserved(self):
        raw = {"transcription": [{"offsets": {"from": 1250, "to": 3800}, "text": " Do not ship on Friday. "}]}
        segments = app.validate_segments(app.parse_whisper(raw, "远端"))
        self.assertEqual(segments[0]["start"], 1.25)
        self.assertEqual(segments[0]["end"], 3.8)
        self.assertEqual(segments[0]["text"], "Do not ship on Friday.")
        self.assertEqual(segments[0]["source"], "远端")

    def test_overlapping_tracks_keep_both_sources(self):
        segments = app.validate_segments([
            {"start": 2, "end": 4, "text": "I will do it.", "source": "我"},
            {"start": 1, "end": 3, "text": "Can you own it?", "source": "远端"},
        ])
        self.assertEqual([s["source"] for s in segments], ["远端", "我"])
        self.assertEqual([s["id"] for s in segments], ["S00001", "S00002"])

    def test_ambiguous_and_business_content_are_never_dropped_as_chatter(self):
        scores = {"requirement": .02, "action": .02, "decision": .02, "unresolved": .02, "chatter": .89}
        self.assertFalse(app.should_drop(scores))
        scores["chatter"] = .99
        scores["decision"] = .21
        self.assertFalse(app.should_drop(scores))

    def test_api_failure_keeps_all_original_evidence(self):
        segments = app.validate_segments([
            {"start": 0, "end": 10, "text": "Deliver on Friday."},
            {"start": 61, "end": 65, "text": "Actually cancel that deadline."},
        ])
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-only"}), patch.object(app, "call_jev", side_effect=app.MeetingError("Jev HTTP 529")):
            selected, records = app.screen(segments, Path(directory), "test")
        self.assertEqual(selected, segments)
        self.assertTrue(all(r["reason"] == "service_error_keep_for_review" for r in records))

    def test_context_neighbors_are_restored(self):
        segments = app.validate_segments([{"start": i * 61, "end": i * 61 + 5, "text": f"turn {i}"} for i in range(5)])

        def mock_call(payload, key):
            important = payload["state"]["target"][0]["text"] == "turn 2"
            scores = {"requirement": .01, "action": .99 if important else .01, "decision": .01, "unresolved": .01, "chatter": .01 if important else .99}
            return {"response": {"answers": {k: {"type": "noul", "noul": v} for k, v in scores.items()}, "usage": {"input_tokens": 100}}, "elapsed_ms": 1}

        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-only"}), patch.object(app, "call_jev", side_effect=mock_call):
            selected, _ = app.screen(segments, Path(directory), "test")
        self.assertEqual([s["text"] for s in selected], ["turn 1", "turn 2", "turn 3"])

    def test_explicit_asr_uncertainty_skips_remote_filtering(self):
        segments = app.validate_segments([{"start": 0, "end": 4, "text": "fifteen or fifty", "uncertain": True}])
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-only"}), patch.object(app, "call_jev") as call:
            selected, _ = app.screen(segments, Path(directory), "test")
        call.assert_not_called()
        self.assertEqual(selected, segments)

    def test_full_and_selected_exports_have_citations_and_no_key(self):
        segments = app.validate_segments([{"start": 15, "end": 19, "text": "The prototype is a demo, not a launch."}])
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"TYPESAFE_API_KEY": "secret-test-sentinel"}), contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            app.prepare(segments, root, "none", "test")
            full = (root / "codex-full.md").read_text()
            selected = (root / "codex-selected.md").read_text()
            self.assertIn("S00001 00:00:15", full)
            self.assertIn("The prototype is a demo, not a launch.", selected)
            self.assertIn("说话人未识别", selected)
            self.assertNotIn("secret-test-sentinel", "".join(p.read_text() for p in root.iterdir()))
            self.assertEqual(json.loads((root / "stats.json").read_text())["jev_requests_this_run"], 0)

    def test_empty_or_invalid_transcript_is_rejected(self):
        for data in ([], [{"start": -1, "end": 2, "text": "x"}], [{"start": 0, "end": float("nan"), "text": "x"}]):
            with self.assertRaises(app.MeetingError):
                app.validate_segments(data)


if __name__ == "__main__":
    unittest.main()
