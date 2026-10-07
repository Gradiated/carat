#!/usr/bin/env python3

import argparse
import json
import os
import urllib.error
import urllib.request
from collections.abc import Mapping


def main() -> None:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description="Send a text prompt to a running Carat server.")
    parser.add_argument("model_directory")
    parser.add_argument("prompt")
    parser.add_argument("--url", default="http://127.0.0.1:30000/v1/token-completions")
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")

    tokenizer = AutoTokenizer.from_pretrained(args.model_directory, local_files_only=True)
    tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}], tokenize=True, add_generation_prompt=True
    )
    input_ids = tokens["input_ids"] if isinstance(tokens, Mapping) else tokens
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]

    headers = {"Content-Type": "application/json"}
    if api_key := os.environ.get("CARAT_RUNTIME_API_KEY"):
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        args.url,
        data=json.dumps({"input_ids": input_ids, "max_tokens": args.max_tokens}).encode(),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        parser.exit(1, f"Carat returned HTTP {error.code}: {error.read().decode()}\n")
    except urllib.error.URLError as error:
        parser.exit(1, f"Cannot reach Carat: {error.reason}\n")

    print(tokenizer.decode(result["output_ids"], skip_special_tokens=True))


if __name__ == "__main__":
    main()
