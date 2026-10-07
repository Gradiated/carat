#!/usr/bin/env python3
"""Generate deterministic Gemma 4 decoder-layer parity fixtures from pinned weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4TextDecoderLayer,
    Gemma4TextRotaryEmbedding,
)


def load_layer_weights(model_directory: Path, layer: int) -> dict[str, torch.Tensor]:
    index = json.loads((model_directory / "model.safetensors.index.json").read_text())
    prefix = f"model.language_model.layers.{layer}."
    selected = {name: shard for name, shard in index["weight_map"].items() if name.startswith(prefix)}
    shards: dict[str, list[str]] = {}
    for name, shard in selected.items():
        shards.setdefault(shard, []).append(name)
    state: dict[str, torch.Tensor] = {}
    for shard, names in shards.items():
        with safe_open(model_directory / shard, framework="pt", device="cpu") as weights:
            for name in names:
                state[name.removeprefix(prefix)] = weights.get_tensor(name)
    return state


def write_tensor(directory: Path, name: str, tensor: torch.Tensor) -> dict[str, object]:
    value = tensor.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    raw = value.view(torch.uint8).numpy().tobytes()
    path = directory / f"{name}.bf16"
    path.write_bytes(raw)
    return {
        "file": path.name,
        "shape": list(value.shape),
        "dtype": "BF16",
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root_config = AutoConfig.from_pretrained(args.model_directory, local_files_only=True)
    config = root_config.text_config
    config._attn_implementation = "eager"
    if not 0 <= args.layer < config.num_hidden_layers:
        raise ValueError("layer index is outside the model")
    torch.manual_seed(0x47524144 + args.layer)
    layer = Gemma4TextDecoderLayer(config, args.layer).eval()
    layer.load_state_dict(load_layer_weights(args.model_directory, args.layer), strict=True)
    layer = layer.to(device="cuda", dtype=torch.bfloat16)
    rotary = Gemma4TextRotaryEmbedding(config, device="cuda").to(device="cuda")
    hidden = torch.randn((1, args.tokens, config.hidden_size), device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(args.tokens, device="cuda", dtype=torch.long).unsqueeze(0)
    layer_type = config.layer_types[args.layer]
    position_embeddings = rotary(hidden, positions, layer_type)
    mask = torch.full((args.tokens, args.tokens), float("-inf"), device="cuda", dtype=torch.float32)
    mask = torch.triu(mask, diagonal=1)
    if layer_type == "sliding_attention":
        too_old = torch.arange(args.tokens, device="cuda")[:, None] - torch.arange(
            args.tokens, device="cuda"
        )[None, :] >= config.sliding_window
        mask = mask.masked_fill(too_old, float("-inf"))
    mask = mask[None, None, :, :]
    with torch.inference_mode():
        output = layer(
            hidden,
            shared_kv_states={},
            position_embeddings=position_embeddings,
            attention_mask=mask,
            position_ids=positions,
        )
    args.output.mkdir(parents=True, exist_ok=True)
    tensors = {
        "input": write_tensor(args.output, "input", hidden),
        "last_output": write_tensor(args.output, "last_output", output[:, -1, :]),
    }
    manifest = {
        "schema_version": 1,
        "model_revision": "b9ea41a2887d8607f594846523f94c6cc75ac8a4",
        "layer": args.layer,
        "layer_type": layer_type,
        "tokens": args.tokens,
        "seed": 0x47524144 + args.layer,
        "tensors": tensors,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
