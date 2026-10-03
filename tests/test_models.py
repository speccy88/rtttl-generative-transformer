import io

import pytest
import torch

from rtttl_gen.generation import generate_tokens, neural_context
from rtttl_gen.ngram import NGramLM
from rtttl_gen.recurrent import GRULM
from rtttl_gen.rtttl import encode_rtttl, parse_rtttl
from rtttl_gen.tokenizer import EventTokenizer
from rtttl_gen.transformer import TransformerLM, count_parameters


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def small_transformer():
    return TransformerLM(EventTokenizer().vocab_size, context_length=32,
                         d_model=24, n_layers=2, n_heads=4, dropout=0.0)


@pytest.mark.parametrize("kind", ["transformer", "gru"])
def test_forward_gradient_and_checkpoint(kind):
    torch.manual_seed(7)
    tok = EventTokenizer()
    model = small_transformer() if kind == "transformer" else GRULM(tok.vocab_size, 32, 24, 1, 0.0)
    ids = torch.tensor([tok.encode(parse_rtttl("X:d=4,o=5,b=120:c,e,g"))])
    logits = model(ids[:, :-1])
    assert logits.shape == (1, ids.shape[1] - 1, tok.vocab_size)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, tok.vocab_size), ids[:, 1:].reshape(-1))
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert count_parameters(model) == model.parameter_count()
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    clone = type(model)(**model.config)
    clone.load_state_dict(torch.load(buffer, weights_only=True))
    model.eval()
    clone.eval()
    torch.testing.assert_close(model(ids), clone(ids))


@pytest.mark.parametrize("kind", ["transformer", "gru"])
def test_future_tokens_do_not_change_past_logits(kind):
    torch.manual_seed(11)
    model = small_transformer() if kind == "transformer" else GRULM(940, 32, 24, 2, 0.0)
    model.eval()
    first = torch.randint(0, 940, (2, 10))
    changed = first.clone()
    changed[:, 5:] = (changed[:, 5:] + 123) % 940
    a, b = model(first), model(changed)
    torch.testing.assert_close(a[:, :5], b[:, :5], rtol=1e-5, atol=1e-6)
    assert not torch.allclose(a[:, 5:], b[:, 5:])


def test_ngram_probabilities_backoff_and_reload():
    model = NGramLM(order=3).fit([[1, 3, 4, 2], [1, 3, 5, 2]], vocab_size=8)
    for context in ([], [1], [1, 3], [7, 7, 7]):
        probabilities = model.next_logits(context).exp()
        assert torch.all(probabilities > 0)
        torch.testing.assert_close(probabilities.sum(), torch.tensor(1.0))
        for target in range(8):
            assert model.log_probability(context, target) == pytest.approx(
                float(model.next_logits(context)[target]), abs=1e-6)
    clone = NGramLM()
    clone.load_state_dict(model.state_dict())
    torch.testing.assert_close(model.next_logits([1, 3]), clone.next_logits([1, 3]))
    assert model.next_logits([1, 3])[4] > model.next_logits([1, 3])[7]


@pytest.mark.parametrize("kind", ["transformer", "gru", "ngram"])
def test_generation_terminates_and_output_reparses(kind):
    tok = EventTokenizer()
    if kind == "transformer":
        model = small_transformer()
    elif kind == "gru":
        model = GRULM(tok.vocab_size, 32, 24, 1, 0.0)
    else:
        model = NGramLM().fit([tok.encode(parse_rtttl("X:d=4,o=5,b=120:c,e,g,p"))], tok.vocab_size)
    generated = generate_tokens(model, tok, min_events=4, max_events=6, bpm=123, seed=22)
    ids = generated["token_ids"]
    assert ids[-1] == tok.eos_id
    song = tok.decode(ids)
    assert 4 <= len(song.events) <= 6
    assert song.bpm == 123
    restored = parse_rtttl(encode_rtttl(song))
    assert restored.events == song.events
    assert restored.bpm == song.bpm
    repeated = generate_tokens(model, tok, min_events=4, max_events=6, bpm=123, seed=22)
    assert repeated == generated


def test_generation_forced_eos_and_model_state_restoration():
    tok = EventTokenizer()
    model = small_transformer()
    model.train()
    generated = generate_tokens(model, tok, min_events=2, max_events=2, seed=3)
    assert generated["forced_eos"]
    assert len(tok.decode(generated["token_ids"]).events) == 2
    assert model.training


def test_long_generation_retains_tempo_and_event_aligned_context():
    tok = EventTokenizer()

    class RecordingTransformer(TransformerLM):
        def __init__(self):
            super().__init__(tok.vocab_size, context_length=8, d_model=16, n_layers=1, n_heads=2, dropout=0)
            self.contexts = []

        def forward(self, ids):
            self.contexts.append(ids[0].tolist())
            return super().forward(ids)

    model = RecordingTransformer()
    result = generate_tokens(model, tok, min_events=20, max_events=20, bpm=123, seed=1)
    assert len(tok.decode(result["token_ids"]).events) == 20
    assert len(model.contexts) > 8
    for context in model.contexts:
        assert len(context) <= model.context_length
        assert context[:2] == [tok.bos_id, tok.token_to_id["<BPM_123>"]]
        if len(context) > 2:
            assert context[2] in tok.event_start_ids
            for index, token in enumerate(context[2:]):
                assert token in (tok.event_start_ids if index % 2 == 0 else tok.duration_ids)
    for length in range(3, 20):
        prefix = result["token_ids"][:length]
        for context_size in (4, 5, 6, 7):
            context = neural_context(prefix, context_size)
            assert len(context) <= context_size
            assert context[-1] == prefix[-1]


@pytest.mark.parametrize("kwargs", [{"temperature": 0}, {"top_p": 0}, {"top_k": -1},
                                     {"min_events": 0}, {"bpm": 901}])
def test_invalid_sampling_arguments(kwargs):
    with pytest.raises(ValueError):
        generate_tokens(small_transformer(), EventTokenizer(), **kwargs)
