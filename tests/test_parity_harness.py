"""Tests for the greedy-parity harness in `jax_neuron/parity.py`.

The harness is the gate that decides whether the Inf2 port is correct, so its
own logic has to be right before it runs anywhere expensive. Everything tested
here is deliberately pure: no torch, no checkpoint, no accelerator. The parts
that need those are the two decode functions, which are thin wrappers around
`transformers` and `JaxGemmaEngine` respectively.

The property that matters most is that a wrong answer is never reported as a
pass — the repo has already been burned once by a green test run against the
wrong oracle.
"""

import json
import os
import sys
import tempfile
import unittest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from jax_neuron.parity import (  # noqa: E402
    compare,
    effective_prompt_ids,
    resolve_eos_ids,
    tokenizer_sanity,
)


class _Encoding:
    def __init__(self, ids):
        self.input_ids = ids


class FakeTokenizer:
    """Minimal stand-in with the surface the harness actually touches."""

    def __init__(self, ids=None, decoded="hello world", bos=2, eos=1, unk=3,
                 special=None):
        self._ids = [9, 8] if ids is None else ids
        self._decoded = decoded
        self.bos_token_id = bos
        self.eos_token_id = eos
        self.unk_token_id = unk
        self.vocab_size = 262144
        self._special = special or {}

    def __call__(self, text, add_special_tokens=True):
        return _Encoding(list(self._ids))

    def decode(self, ids):
        return self._decoded

    def convert_tokens_to_ids(self, token):
        return self._special.get(token, -1)

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        return [105, 2364] + list(self._ids) + [106]


class TestTokenizerSanity(unittest.TestCase):

    def test_healthy_tokenizer_has_no_problems(self):
        check = tokenizer_sanity(FakeTokenizer(), "m")
        self.assertEqual(check["problems"], [])
        self.assertEqual(check["input_ids"], [9, 8])
        self.assertEqual(check["bos_token_id"], 2)

    def test_all_unk_is_caught(self):
        """The single most common cause of garbage output in the sibling port."""
        tok = FakeTokenizer(ids=[3, 3, 3], decoded="<unk><unk><unk>", unk=3)
        problems = tokenizer_sanity(tok, "m")["problems"]
        self.assertTrue(any("<unk>" in p for p in problems), problems)

    def test_empty_tokenization_is_caught(self):
        problems = tokenizer_sanity(FakeTokenizer(ids=[], decoded=""), "m")["problems"]
        self.assertTrue(any("no ids" in p for p in problems), problems)

    def test_broken_round_trip_is_caught(self):
        problems = tokenizer_sanity(FakeTokenizer(decoded="⁇⁇"), "m")["problems"]
        self.assertTrue(any("round-trip" in p for p in problems), problems)

    def test_missing_bos_is_caught(self):
        """Gemma without <bos> echoes the prompt instead of answering."""
        problems = tokenizer_sanity(FakeTokenizer(bos=None), "m")["problems"]
        self.assertTrue(any("bos" in p for p in problems), problems)


class TestEffectivePromptIds(unittest.TestCase):

    def test_bos_is_prepended(self):
        ids = effective_prompt_ids(FakeTokenizer(ids=[9, 8], bos=2), "x", 2, False)
        self.assertEqual(ids, [2, 9, 8])

    def test_existing_bos_is_not_doubled(self):
        """Mirrors the guard in JaxGemmaEngine.generate_stream exactly.

        If these two rules disagree the engine appends a second <bos> that the
        reference never saw, and the harness reports a divergence that is its
        own fault.
        """
        ids = effective_prompt_ids(FakeTokenizer(ids=[2, 9, 8], bos=2), "x", 2, False)
        self.assertEqual(ids, [2, 9, 8])

    def test_no_bos_configured_leaves_ids_alone(self):
        ids = effective_prompt_ids(FakeTokenizer(ids=[9, 8]), "x", None, False)
        self.assertEqual(ids, [9, 8])

    def test_chat_template_path_also_gets_bos(self):
        ids = effective_prompt_ids(FakeTokenizer(ids=[9, 8], bos=2), "x", 2, True)
        self.assertEqual(ids[0], 2)
        self.assertIn(105, ids)

    def test_engine_prepend_is_a_noop_on_harness_output(self):
        """The whole point: after the harness, the engine must not change the ids."""
        tok = FakeTokenizer(ids=[9, 8], bos=2)
        ids = effective_prompt_ids(tok, "x", 2, False)

        # Replicated from jax_engine.JaxGemmaEngine.generate_stream.
        bos_token_id, out = 2, list(ids)
        if bos_token_id is not None and (not out or out[0] != bos_token_id):
            out = [bos_token_id] + out

        self.assertEqual(out, ids)


class TestResolveEosIds(unittest.TestCase):

    def test_scalar_eos(self):
        self.assertEqual(resolve_eos_ids(FakeTokenizer(eos=1)), {1})

    def test_list_eos(self):
        self.assertEqual(resolve_eos_ids(FakeTokenizer(eos=[1, 106])), {1, 106})

    def test_end_of_turn_is_included(self):
        """Instruction-tuned Gemma ends a turn with <end_of_turn>, not <eos>."""
        tok = FakeTokenizer(eos=1, special={"<end_of_turn>": 106})
        self.assertEqual(resolve_eos_ids(tok), {1, 106})

    def test_unknown_special_tokens_are_not_added(self):
        tok = FakeTokenizer(eos=1, unk=3, special={"<end_of_turn>": 3})
        self.assertEqual(resolve_eos_ids(tok), {1})


class TestCompare(unittest.TestCase):

    def test_identical_sequences_match(self):
        v = compare([1, 2, 3], [1, 2, 3])
        self.assertTrue(v["matched"])
        self.assertIsNone(v["first_divergence"])
        self.assertEqual(v["agreement"], 1.0)

    def test_divergence_index_is_reported(self):
        v = compare([1, 2, 3, 4], [1, 2, 9, 4])
        self.assertFalse(v["matched"])
        self.assertEqual(v["first_divergence"], 2)
        self.assertAlmostEqual(v["agreement"], 0.75)

    def test_first_token_divergence(self):
        v = compare([5, 2], [6, 2])
        self.assertEqual(v["first_divergence"], 0)

    def test_prefix_match_but_different_lengths_is_a_divergence(self):
        """A short subject that agrees everywhere it exists is still not parity.

        Early stopping is exactly how a broken EOS or an off-by-one decode budget
        presents, so reporting this as a pass would hide a real bug.
        """
        v = compare([1, 2, 3], [1, 2])
        self.assertFalse(v["matched"])
        self.assertEqual(v["first_divergence"], 2)
        self.assertEqual(v["agreement"], 1.0)

    def test_longer_subject_is_also_a_divergence(self):
        v = compare([1, 2], [1, 2, 3])
        self.assertFalse(v["matched"])
        self.assertEqual(v["first_divergence"], 2)

    def test_two_empty_sequences_match(self):
        v = compare([], [])
        self.assertTrue(v["matched"])
        self.assertEqual(v["agreement"], 0.0)

    def test_margin_is_taken_from_the_divergence_index(self):
        """The margin decides 'numerical tie' vs 'real bug', so it must be the
        margin at the token that broke, not at some other step."""
        v = compare([1, 2, 3], [1, 2, 9], margins=[5.0, 4.0, 0.001])
        self.assertEqual(v["first_divergence"], 2)
        self.assertAlmostEqual(v["divergence_margin"], 0.001)

    def test_missing_margins_are_tolerated(self):
        v = compare([1, 2], [1, 9], margins=[])
        self.assertIsNone(v["divergence_margin"])
        self.assertEqual(v["first_divergence"], 1)

    def test_margin_shorter_than_divergence_index(self):
        v = compare([1, 2, 3], [1, 2, 9], margins=[5.0])
        self.assertIsNone(v["divergence_margin"])


class TestSavedReferenceContract(unittest.TestCase):
    """The saved-reference file is what lets the device run without torch.

    It is only sound if the tokenization on both sides is identical, so the
    harness refuses to compare when the recorded prompt ids differ. These pin
    the shape that check depends on.
    """

    def test_report_roundtrips_through_json(self):
        from dataclasses import asdict
        from jax_neuron.parity import ParityReport, PromptResult

        report = ParityReport(model_id="m", max_new_tokens=4)
        report.results.append(PromptResult(
            prompt="hi", prompt_token_ids=[2, 9],
            reference_tokens=[10, 11], reference_margins=[1.0, 2.0],
            reference_text="ok",
        ))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ref.json")
            with open(path, "w") as fh:
                json.dump(asdict(report), fh)
            with open(path) as fh:
                loaded = json.load(fh)

        entry = loaded["results"][0]
        self.assertEqual(entry["prompt"], "hi")
        self.assertEqual(entry["prompt_token_ids"], [2, 9])
        self.assertEqual(entry["reference_tokens"], [10, 11])
        self.assertEqual(loaded["model_id"], "m")


if __name__ == "__main__":
    unittest.main()
