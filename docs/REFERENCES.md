# Primary references

These references support the implementation concepts. Actual project results come from the recorded experiment rather than these sources.

- [PyTorch 2.6 scaled dot-product attention](https://docs.pytorch.org/docs/2.6/generated/torch.nn.functional.scaled_dot_product_attention.html): causal attention and dropout behavior
- [PyTorch 2.6 mixed-precision examples](https://docs.pytorch.org/docs/2.6/notes/amp_examples.html): autocast, scaling, clipping and accumulation
- [Official PyTorch installation commands](https://pytorch.org/get-started/previous-versions/): version-specific CPU/CUDA wheels
- [NCL RTTTL documentation](https://docs.ncl.ie/NCL/doc/api/ie/ncl/media/music/RTTTL.html): historical syntax and extensions
- [PICAXE tune command](https://picaxe.com/basic-commands/digital-inputoutput/tune/): a distinct device-specific representation, excluded from this importer

The project uses scratch training. Symbolic-music pretrained models would require a separate representation/architecture experiment and are not drop-in RTTTL checkpoints.
