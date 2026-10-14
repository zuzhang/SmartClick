import unittest

from detector.labels import label_for_text


class LabelTests(unittest.TestCase):
    def test_maps_supported_text_to_business_labels(self):
        cases = {
            "OK": "confirm",
            "Yes": "confirm",
            "Accept": "confirm",
            "No Thanks": "close_button",
            "Cancel": "close_button",
            "Dismiss": "close_button",
            "Skip": "skip",
            "Collect": "collect",
            "Done": "continue",
            "Tap to Claim": "claim",
            "Watch": "watch",
            "Play": "play",
            "GO!": "play",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(label_for_text(text), expected)

    def test_does_not_match_partial_words(self):
        self.assertIsNone(label_for_text("skipper"))
