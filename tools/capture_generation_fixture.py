#!/usr/bin/env python3
"""Capture token-level greedy output from the an SGLang reference server for parity checks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import urllib.request

from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:30000/generate")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_directory, local_files_only=True)
    messages = [
        {"role": "user", "content": "Write a Python function that returns the first duplicate integer in a list."}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    payload = {
        "input_ids": input_ids,
        "sampling_params": {"temperature": 0, "max_new_tokens": 8},
        "return_logprob": True,
        "top_logprobs_num": 0,
        "return_prompt_token_ids": True,
    }
    request = urllib.request.Request(
        args.url,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {os.environ.get('CARAT_REFERENCE_API_KEY', '')}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    response = json.load(urllib.request.urlopen(request, timeout=120))
    token_logprobs = response["meta_info"]["output_token_logprobs"]
    output_ids = [int(item[1]) for item in token_logprobs]
    fixture = {
        "schema_version": 1,
        "model_revision": "b9ea41a2887d8607f594846523f94c6cc75ac8a4",
        "input_ids": input_ids,
        "expected_output_ids": output_ids,
        "prompt_tokens": len(input_ids),
        "control_finish_reason": response["meta_info"].get("finish_reason"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n")
    print(json.dumps(fixture, sort_keys=True))


if __name__ == "__main__":
    main()
