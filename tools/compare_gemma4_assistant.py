#!/usr/bin/env python3
"""Emit a one-prompt PyTorch oracle for the native Gemma 4 assistant."""

import argparse
import json

import torch
from transformers import AutoModelForCausalLM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target")
    parser.add_argument("assistant")
    parser.add_argument("fixture")
    args = parser.parse_args()
    fixture = json.load(open(args.fixture, encoding="utf-8"))
    ids = torch.tensor([fixture["input_ids"]], device="cuda")
    target = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=torch.bfloat16, local_files_only=True
    ).cuda().eval()
    assistant = AutoModelForCausalLM.from_pretrained(
        args.assistant, dtype=torch.bfloat16, local_files_only=True
    ).cuda().eval()
    with torch.inference_mode():
        output = target(
            input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True,
            return_shared_kv_states=True, use_cache=False
        )
        previous_hidden = output.hidden_states[-1][:, -1:]
        target_first = target.get_output_embeddings()(previous_hidden).argmax(-1).item()
        current = ids[:, -1:]
        proposals = []
        anchor = ids.shape[1] - 1
        mask = torch.ones((1, ids.shape[1]), dtype=torch.long, device="cuda")
        for depth in range(4):
            token_embedding = target.get_input_embeddings()(current)
            drafted = assistant(
                inputs_embeds=torch.cat((token_embedding, previous_hidden), -1),
                attention_mask=mask,
                position_ids=torch.tensor([[anchor + depth]], device="cuda"),
                shared_kv_states=output.shared_kv_states,
                use_cache=False,
            )
            current = drafted.logits[:, -1:].argmax(-1)
            proposals.append(current.item())
            previous_hidden = drafted.last_hidden_state
    print(json.dumps({"target_first": target_first, "proposals": proposals}))


if __name__ == "__main__":
    main()
