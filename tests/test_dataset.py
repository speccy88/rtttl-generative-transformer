from collections import Counter
import torch
from rtttl_gen.dataset import SequenceDataset
from rtttl_gen.rtttl import Event, Song
from rtttl_gen.tokenizer import EventTokenizer

def test_every_target_counted_once_with_no_false_eos():
    song = Song('long', 120, tuple(Event(60+i%12, 8, bool(i%2)) for i in range(43)))
    t = EventTokenizer()
    ds = SequenceDataset([song],t,context_length=16)
    observed = Counter()
    for window in ds:
        assert window['input_ids'].shape == (16,)
        observed.update(x for x in window['targets'].tolist() if x != t.pad_id)
    assert observed == Counter(t.encode(song)[1:])
    assert observed[t.eos_id] == 1

def test_augmentation_is_deterministic_per_epoch_and_valid():
    song = Song('safe',120,(Event(60,8),Event(107,4),Event(None,8)))
    t = EventTokenizer()
    ds = SequenceDataset([song],t,32,augment_semitones=5)
    for epoch in range(3):
        ds.set_epoch(epoch)
        assert torch.equal(ds[0]['targets'],ds[0]['targets'])
        assert torch.equal(ds[0]['targets'],SequenceDataset([song],t,32)[0]['targets'])
