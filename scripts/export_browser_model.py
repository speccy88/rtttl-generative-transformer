#!/usr/bin/env python3
"""Export the published melody weights to a portable, last-token ONNX graph.

Manual attention uses ordinary matrix/softmax operators to support browser
WebGPU and WASM. Export parity is measured against the original PyTorch forward
pass at multiple sequence lengths. No training, optimizer or corpus is needed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import torch
from safetensors.torch import load_model
from rtttl_gen.transformer import TransformerLM


class BrowserTransformer(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        m = self.model
        length = input_ids.shape[1]
        positions = torch.arange(length, device=input_ids.device)
        x = m.token_embedding(input_ids) + m.position_embedding(positions)
        causal = positions.unsqueeze(0) <= positions.unsqueeze(1)
        for block in m.blocks:
            residual = x
            normalized = block.ln1(x)
            attention = block.attention
            qkv = attention.qkv(normalized).reshape(1, length, 3, attention.n_heads, attention.head_dim)
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            scores = torch.matmul(q, k.transpose(-2, -1)) * (attention.head_dim ** -0.5)
            scores = torch.where(causal, scores, torch.full_like(scores, -10000))
            attended = torch.matmul(torch.softmax(scores, dim=-1), v)
            attended = attended.transpose(1, 2).reshape(1, length, m.config['d_model'])
            x = residual + attention.projection(attended)
            x = x + block.mlp(block.ln2(x))
        return m.lm_head(m.final_norm(x[:, -1, :]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    model = TransformerLM(**json.loads((args.model_dir / 'config.json').read_text())).eval()
    load_model(model, str(args.model_dir / 'model.safetensors'), strict=True, device='cpu')
    wrapper = BrowserTransformer(model).eval()
    sample = torch.tensor([[1, 123]], dtype=torch.int64)
    path = args.output / 'model.onnx'
    torch.onnx.export(wrapper, sample, path, input_names=['input_ids'], output_names=['logits'],
                      dynamic_axes={'input_ids': {1: 'sequence'}}, opset_version=17, dynamo=False)
    import onnx
    import onnxruntime as ort
    onnx.checker.check_model(onnx.load(path))
    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    checks = []
    rng = torch.Generator().manual_seed(2026)
    for length in (1, 2, 17, 97, 256):
        ids = torch.randint(0, 940, (1, length), generator=rng)
        with torch.inference_mode():
            expected = model(ids)[:, -1, :].numpy()
            manual = wrapper(ids).numpy()
        actual = session.run(['logits'], {'input_ids': ids.numpy()})[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(manual, expected, rtol=1e-4, atol=1e-4)
        checks.append({'sequence_length':length,'max_absolute_error':float(np.max(np.abs(actual-expected))),
                       'top_token_equal':int(actual.argmax()) == int(expected.argmax())})
        if length == 17:
            (args.output / 'parity_fixture.json').write_text(json.dumps({'input_ids':ids.tolist(), 'logits':expected.tolist()}))
    (args.output / 'tokenizer.json').write_bytes((args.model_dir / 'tokenizer.json').read_bytes())
    validation = {'status':'PASS','weights_sha256':hashlib.sha256((args.model_dir/'model.safetensors').read_bytes()).hexdigest(),
                  'onnx_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'onnx_bytes':path.stat().st_size,
                  'opset':17,'input':'input_ids int64 [1, sequence]','output':'logits float32 [1, 940]',
                  'torch_version':torch.__version__,'onnxruntime_version':ort.__version__,'checks':checks,
                  'scope':'Local CPU export parity; browser WebGPU is validated separately.'}
    (args.output / 'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    print(json.dumps(validation,indent=2))


if __name__ == '__main__':
    main()
