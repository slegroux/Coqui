import os
import unittest

from tests import get_tests_input_path
from TTS.tts.datasets.formatters import common_voice, spgi, spgi_vca


class TestTTSFormatters(unittest.TestCase):
    def test_common_voice_preprocessor(self):  # pylint: disable=no-self-use
        root_path = get_tests_input_path()
        meta_file = "common_voice.tsv"
        items = common_voice(root_path, meta_file)
        assert items[0]["text"] == "The applicants are invited for coffee and visa is given immediately."
        assert items[0]["audio_file"] == os.path.join(get_tests_input_path(), "clips", "common_voice_en_20005954.wav")

        assert items[-1]["text"] == "Competition for limited resources has also resulted in some local conflicts."
        assert items[-1]["audio_file"] == os.path.join(get_tests_input_path(), "clips", "common_voice_en_19737074.wav")

    def test_spgi(self):  # pylint: disable=no-self-use
        items = spgi(split='test')
        assert len(items) == 39341
        assert items[0]["text"] == "This product continues to be very well received by users. Daily active user is approaching 10 million"
        assert items[-1]["text"] == "This is one of our largest opportunities to increase overall profitability."
        items = spgi(split='validation')
        assert len(items) == 39304

    def test_spgi_vca(self):
        root_path = "/home/syl20/data/spgi"
        meta_file = "spgi.txt"
        items = spgi_vca(root_path, meta_file)
        assert items[0]["text"] == "to structure an entry into new countries where Natura is not present, not considering Aesop, I'm talking about the actual Natura brand."
