#!/usr/bin/env python3
"""Render a deterministic HumanEval partition for native-engine evaluation."""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from collections.abc import Mapping
from pathlib import Path

from transformers import AutoTokenizer


DATASET = "openai/openai_humaneval"
ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"


def fetch_rows(offset: int, length: int) -> list[dict[str, str]]:
    query = urllib.parse.urlencode(
        {
            "dataset": DATASET,
            "config": "openai_humaneval",
            "split": "test",
            "offset": offset,
            "length": length,
        }
    )
    with urllib.request.urlopen(f"{ROWS_ENDPOINT}?{query}", timeout=60) as response:
        payload = json.load(response)
    return [item["row"] for item in payload["rows"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--tasks", type=int, default=32)
    args = parser.parse_args()
    if args.offset < 0 or args.tasks <= 0 or args.offset + args.tasks > 164:
        parser.error("the requested partition must be inside the 164 HumanEval tasks")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prompts: list[dict[str, object]] = []
    for row in fetch_rows(args.offset, args.tasks):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert Python programmer. Complete the requested "
                    "function. Return only valid Python source code containing the "
                    "complete function and any required imports. Do not use Markdown fences."
                ),
            },
            {"role": "user", "content": row["prompt"]},
        ]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        input_ids = rendered["input_ids"] if isinstance(rendered, Mapping) else rendered
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        prompts.append(
            {
                "id": row["task_id"],
                "input_ids": input_ids,
                "prompt": row["prompt"],
                "test": row["test"],
                "entry_point": row["entry_point"],
            }
        )

    args.output.write_text(
        json.dumps(
            {
                "schema": "carat.humaneval-prompts.v1",
                "dataset": DATASET,
                "offset": args.offset,
                "tasks": args.tasks,
                "model": args.model,
                "prompts": prompts,
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
