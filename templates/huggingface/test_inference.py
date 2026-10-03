"""Dataset-free checks for the downloadable inference release."""
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from batch_plan import plan_batch
from diagnostics import repetition_statistics
from generation import generate_tokens
from guidance import PROFILE_NAMES, PROFILE_SPECS
from inference import generate_records, load_local_model, select_device
from rtttl import encode_rtttl, parse_rtttl


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model, cls.tokenizer = load_local_model(Path(__file__).resolve().parent, "cpu")

    def test_architecture_and_tied_weights(self):
        self.assertEqual(self.model.parameter_count(), 2009472)
        self.assertEqual(self.tokenizer.vocab_size, 940)
        self.assertIs(self.model.lm_head.weight, self.model.token_embedding.weight)
        self.assertFalse(self.model.training)

    def test_forward(self):
        ids = torch.tensor([[self.tokenizer.bos_id, self.tokenizer.bpm_ids[95]]])
        with torch.inference_mode():
            logits = self.model(ids)
        self.assertEqual(tuple(logits.shape), (1, 2, 940))
        self.assertTrue(bool(torch.isfinite(logits).all()))

    def test_cpu_generation_and_roundtrip(self):
        result = generate_tokens(self.model, self.tokenizer, min_events=8, max_events=32,
                                 bpm=120, seed=123, device="cpu")
        song = self.tokenizer.decode(result["token_ids"])
        parsed = parse_rtttl(encode_rtttl(song))
        self.assertEqual(parsed.events, song.events)
        self.assertEqual(parsed.bpm, 120)
        self.assertGreaterEqual(len(song.events), 8)
        self.assertLessEqual(len(song.events), 32)
        stats = repetition_statistics(song)
        self.assertLessEqual(stats["longest_pitch_run"], 4)
        self.assertLessEqual(stats["max_pitch_motif_repeats"], 3)

    def test_same_seed_repeats_on_cpu(self):
        kwargs = dict(min_events=4, max_events=12, seed=17, device="cpu")
        self.assertEqual(generate_tokens(self.model, self.tokenizer, **kwargs),
                         generate_tokens(self.model, self.tokenizer, **kwargs))

    def test_invalid_generation_arguments(self):
        for kwargs in [dict(temperature=0), dict(top_p=0), dict(min_events=5,max_events=4)]:
            with self.assertRaises(ValueError):
                generate_tokens(self.model, self.tokenizer, **kwargs)

    def test_mixed_records_match_decoded_tempos(self):
        kwargs = dict(num_songs=5, seed=4000, profile="mixed", tonic=None, mode=None,
                      bpm=None, bpm_range=None, min_events=4, max_events=12)
        records, songs = generate_records(self.model, self.tokenizer, device="cpu", **kwargs)
        self.assertEqual({r["generation_settings"]["profile"] for r in records}, set(PROFILE_NAMES))
        self.assertGreater(len({song.bpm for song in songs}), 1)
        for row, song in zip(records, songs):
            settings = row["generation_settings"]
            minimum, maximum = PROFILE_SPECS[settings["profile"]].bpm_range
            self.assertTrue(minimum <= song.bpm <= maximum)
            self.assertEqual(song.bpm, settings["bpm"])
            self.assertEqual(parse_rtttl(row["rtttl"]).events, song.events)

    def test_explicit_tempo_and_key_override(self):
        plan = plan_batch(5, seed=19, profile="mixed", bpm=137, tonic="Bb", mode="natural-minor")
        self.assertTrue(all(p["bpm"]==137 and p["tonic"]=="A#" and p["mode"]=="natural-minor" for p in plan))
        with self.assertRaises(ValueError):
            plan_batch(5, bpm=120, bpm_range=(80,140))

    def test_accelerator_selection_and_explicit_failure(self):
        with patch("torch.cuda.is_available", return_value=False), patch("torch.backends.mps.is_available", return_value=True):
            self.assertEqual(select_device("auto"), "mps")
        with patch("torch.cuda.is_available", return_value=False), patch("torch.backends.mps.is_available", return_value=False):
            self.assertEqual(select_device("auto"), "cpu")
            with self.assertRaises(RuntimeError):
                select_device("mps")


if __name__ == "__main__":
    unittest.main()
