import unittest

from actions.decision_engine import DecisionEngine
from detector.base import Detection


class DecisionEngineTests(unittest.TestCase):
    def test_selects_highest_priority_eligible_label(self):
        detections = [
            Detection("continue", 0.99, 10, 10),
            Detection("skip", 0.81, 20, 20),
            Detection("close_button", 0.79, 30, 30),
        ]
        self.assertEqual(DecisionEngine().choose(detections).label, "skip")

    def test_confidence_breaks_priority_ties(self):
        detections = [
            Detection("confirm", 0.82, 1, 1),
            Detection("confirm", 0.95, 2, 2),
        ]
        self.assertEqual(DecisionEngine().choose(detections).center_x, 2)

    def test_selects_highest_confidence_close_button(self):
        detections = [
            Detection("close_button", 0.82, 1, 1),
            Detection("close_button", 0.95, 2, 2),
        ]
        self.assertEqual(DecisionEngine().choose(detections).center_x, 2)

    def test_selects_confirm_detection(self):
        detection = Detection("confirm", 0.95, 2, 2)
        self.assertEqual(DecisionEngine().choose([detection]), detection)

    def test_selects_watch_detection(self):
        detection = Detection("watch", 0.95, 2, 2)
        self.assertEqual(DecisionEngine().choose([detection]), detection)

    def test_returns_none_when_no_detection_meets_threshold(self):
        self.assertIsNone(
            DecisionEngine().choose([Detection("claim", 0.79, 1, 1)])
        )

    def test_rejects_invalid_confidence_threshold(self):
        with self.assertRaises(ValueError):
            DecisionEngine(1.1)
