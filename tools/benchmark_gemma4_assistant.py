#!/usr/bin/env python3
"""Measure Gemma 4 MTP drafter cost independently of target verification."""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("feature", type=Path)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    feature = torch.load(args.feature, map_location="cpu", weights_only=True)
    anchor = int(feature["anchors"][0])
    position_to_index = {
        int(position): index for index, position in enumerate(feature["needed_positions"])
    }
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16, local_files_only=True
    ).cuda().eval()
    token_embeddings = feature["token_embeddings"].cuda()
    target_hidden = feature["target_hidden_states"].cuda()
    shared = {
        name: (
            key[:, :, : anchor + 1].cuda().expand(args.batch, -1, -1, -1),
            value[:, :, : anchor + 1].cuda().expand(args.batch, -1, -1, -1),
        )
        for name, (key, value) in feature["shared_kv_states"].items()
    }
    attention_mask = torch.ones((args.batch, anchor + 1), dtype=torch.long, device="cuda")

    @torch.inference_mode()
    def draft() -> None:
        previous_hidden = target_hidden[position_to_index[anchor]].view(1, 1, -1)
        previous_hidden = previous_hidden.expand(args.batch, -1, -1)
        for depth in range(args.depth):
            index = position_to_index[anchor + depth]
            embedding = token_embeddings[index].view(1, 1, -1).expand(args.batch, -1, -1)
            output = model(
                inputs_embeds=torch.cat((embedding, previous_hidden), dim=-1),
                attention_mask=attention_mask,
                position_ids=torch.full(
                    (args.batch, 1), anchor + depth, dtype=torch.long, device="cuda"
                ),
                shared_kv_states=shared,
                use_cache=False,
            )
            previous_hidden = output.last_hidden_state

    for _ in range(args.warmup):
        draft()
    torch.cuda.synchronize()
    begin = time.perf_counter()
    for _ in range(args.iterations):
        draft()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - begin
    cycle_seconds = elapsed / args.iterations
    result = {
                "batch": args.batch,
                "depth": args.depth,
                "context_tokens": anchor + 1,
                "iterations": args.iterations,
                "eager_cycle_ms": cycle_seconds * 1000,
                "eager_draft_candidates_per_second": args.batch * args.depth / cycle_seconds,
            }

    # The production engine will use a fixed-shape CUDA graph. Capturing the unmodified
    # Transformers implementation separates launch overhead from its attention/data-layout cost.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        draft()
    torch.cuda.synchronize()
    begin = time.perf_counter()
    for _ in range(args.iterations):
        graph.replay()
    torch.cuda.synchronize()
    graph_seconds = (time.perf_counter() - begin) / args.iterations
    result["graph_cycle_ms"] = graph_seconds * 1000
    result["graph_draft_candidates_per_second"] = args.batch * args.depth / graph_seconds
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
