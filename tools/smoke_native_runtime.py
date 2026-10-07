#!/usr/bin/env python3
"""Exercise the authenticated native token runtime with concurrent long prompts."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.error
import urllib.request


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise RuntimeError("cannot compute a percentile of an empty sample")
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def request_json(url: str, api_key: str, payload: dict[str, object]) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        if response.status != 200:
            raise RuntimeError(f"native runtime returned HTTP {response.status}")
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompts_json")
    parser.add_argument("--url", default="http://127.0.0.1:30000/v1/token-completions")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--conversations",
        type=int,
        default=0,
        help="unique conversations to queue continuously; defaults to concurrency",
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--turns", type=int, default=1)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--suffix-tokens", type=int, default=256)
    parser.add_argument("--asynchronous-turns", action="store_true")
    args = parser.parse_args()

    for name in ("concurrency", "max_tokens", "turns", "cycles"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.conversations < 0 or args.suffix_tokens < 0:
        parser.error("--conversations and --suffix-tokens must not be negative")
    if args.asynchronous_turns and args.turns < 2:
        parser.error("--asynchronous-turns requires at least two turns")

    conversation_count = args.conversations or args.concurrency
    if conversation_count < args.concurrency:
        raise RuntimeError("conversations must be at least concurrency")
    with open(args.prompts_json, encoding="utf-8") as source:
        prompts = json.load(source)["prompts"][:conversation_count]
    if len(prompts) != conversation_count:
        raise RuntimeError("fixture has fewer prompts than requested conversations")

    unauthorized = urllib.request.Request(
        args.url,
        data=b'{"input_ids":[2],"max_tokens":1}',
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(unauthorized, timeout=10)
    except urllib.error.HTTPError as error:
        if error.code != 401:
            raise
    else:
        raise RuntimeError("native runtime accepted an unauthenticated request")

    def complete(prompt: dict[str, object]) -> dict[str, object]:
        started = time.monotonic()
        response = request_json(
            args.url,
            args.api_key,
            {"input_ids": prompt["input_ids"], "max_tokens": args.max_tokens},
        )
        response["client_seconds"] = time.monotonic() - started
        response["prompt_id"] = prompt["id"]
        return response

    if args.asynchronous_turns:
        def run_conversation(prompt: dict[str, object]) -> list[dict[str, object]]:
            full_input = prompt["input_ids"]
            reserved = (args.suffix_tokens + args.max_tokens) * max(0, args.turns - 1)
            initial_tokens = max(1, len(full_input) - reserved)
            state = {
                "id": prompt["id"],
                "input_ids": list(full_input[:initial_tokens]),
                "remaining_ids": list(full_input[initial_tokens:]),
            }
            results = []
            for turn in range(args.turns):
                result = complete(state)
                result["turn"] = turn + 1
                results.append(result)
                if turn + 1 < args.turns:
                    suffix = state["remaining_ids"][: args.suffix_tokens]
                    state["remaining_ids"] = state["remaining_ids"][args.suffix_tokens :]
                    state["input_ids"].extend(result["output_ids"])
                    state["input_ids"].extend(suffix)
            return results

        benchmark_started = time.monotonic()
        work = prompts * args.cycles
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            conversations = list(executor.map(run_conversation, work))
        all_results = [
            result for conversation in conversations for result in conversation
        ]
        benchmark_seconds = time.monotonic() - benchmark_started
        cached = [result for result in all_results if result["turn"] > 1]
        initial = [result for result in all_results if result["turn"] == 1]
        total_output_tokens = sum(len(result["output_ids"]) for result in all_results)
        cached_output_tokens = sum(len(result["output_ids"]) for result in cached)
        total_input_tokens = sum(
            result["usage"]["input_tokens"] for result in all_results
        )
        total_cached_input_tokens = sum(
            result["usage"]["cached_input_tokens"] for result in all_results
        )
        cached_ttft = [result["timing"]["ttft_ms"] for result in cached]
        cached_tpot = [result["timing"]["mean_tpot_ms"] for result in cached]
        cached_max_tpot = [result["timing"]["max_tpot_ms"] for result in cached]
        cached_request = [result["timing"]["total_ms"] for result in cached]
        print(json.dumps({
            "asynchronous_turns": True,
            "concurrency": args.concurrency,
            "conversations": conversation_count,
            "turns": args.turns,
            "cycles": args.cycles,
            "benchmark_seconds": benchmark_seconds,
            "output_tokens": total_output_tokens,
            "aggregate_output_tokens_per_second": total_output_tokens / benchmark_seconds,
            "input_tokens": total_input_tokens,
            "cached_input_tokens": total_cached_input_tokens,
            "uncached_input_tokens": total_input_tokens - total_cached_input_tokens,
            "cached_output_tokens": cached_output_tokens,
            "cached_mean_ttft_ms": statistics.mean(cached_ttft),
            "cached_p50_ttft_ms": percentile(cached_ttft, 0.50),
            "cached_p95_ttft_ms": percentile(cached_ttft, 0.95),
            "cached_mean_tpot_ms": statistics.mean(cached_tpot),
            "cached_p50_tpot_ms": percentile(cached_tpot, 0.50),
            "cached_p95_tpot_ms": percentile(cached_tpot, 0.95),
            "cached_p50_max_tpot_ms": percentile(cached_max_tpot, 0.50),
            "cached_p95_max_tpot_ms": percentile(cached_max_tpot, 0.95),
            "cached_p50_request_ms": percentile(cached_request, 0.50),
            "cached_p95_request_ms": percentile(cached_request, 0.95),
            "cached_max_request_ms": max(cached_request),
            "initial_p50_ttft_ms": percentile(
                [result["timing"]["ttft_ms"] for result in initial], 0.50
            ),
            "initial_p95_ttft_ms": percentile(
                [result["timing"]["ttft_ms"] for result in initial], 0.95
            ),
            "initial_p50_request_ms": percentile(
                [result["timing"]["total_ms"] for result in initial], 0.50
            ),
            "initial_p95_request_ms": percentile(
                [result["timing"]["total_ms"] for result in initial], 0.95
            ),
        }, indent=2, sort_keys=True))
        return

    turn_summaries = []
    benchmark_started = time.monotonic()
    for cycle in range(args.cycles):
        prompt_states: list[dict[str, object]] = []
        for prompt in prompts:
            full_input = prompt["input_ids"]
            # Generated outputs become part of the next turn too. Reserve both components so the
            # final turn remains within the source trace's tested context length.
            reserved = (args.suffix_tokens + args.max_tokens) * max(0, args.turns - 1)
            initial_tokens = max(1, len(full_input) - reserved)
            prompt_states.append(
                {
                    "id": prompt["id"],
                    "input_ids": list(full_input[:initial_tokens]),
                    "remaining_ids": list(full_input[initial_tokens:]),
                }
            )

        for turn in range(args.turns):
            wall_started = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                results = list(executor.map(complete, prompt_states))
            wall_seconds = time.monotonic() - wall_started

            for state, result in zip(prompt_states, results, strict=True):
                output_ids = result["output_ids"]
                usage = result["usage"]
                if not 1 <= len(output_ids) <= args.max_tokens or usage["output_tokens"] != len(output_ids):
                    raise RuntimeError(f"invalid completion response for {result['prompt_id']}")
                if turn + 1 < args.turns:
                    remaining = state["remaining_ids"]
                    suffix = remaining[: args.suffix_tokens]
                    state["remaining_ids"] = remaining[args.suffix_tokens :]
                    state["input_ids"].extend(output_ids)
                    state["input_ids"].extend(suffix)
            timing = [result["timing"] for result in results]
            output_tokens = sum(len(result["output_ids"]) for result in results)
            turn_summaries.append(
                {
                    "cycle": cycle + 1,
                    "turn": turn + 1,
                    "prompt_tokens": [result["usage"]["input_tokens"] for result in results],
                    "output_tokens": output_tokens,
                    "wall_seconds": wall_seconds,
                    "aggregate_output_tokens_per_second": output_tokens / wall_seconds,
                    "mean_ttft_ms": statistics.mean(item["ttft_ms"] for item in timing),
                    "maximum_ttft_ms": max(item["ttft_ms"] for item in timing),
                    "mean_tpot_ms": statistics.mean(item["mean_tpot_ms"] for item in timing),
                    "maximum_tpot_ms": max(item["max_tpot_ms"] for item in timing),
                    "maximum_request_ms": max(item["total_ms"] for item in timing),
                }
            )
    benchmark_seconds = time.monotonic() - benchmark_started
    total_output_tokens = sum(item["output_tokens"] for item in turn_summaries)
    print(
        json.dumps(
            {
                "concurrency": args.concurrency,
                "cycles": args.cycles,
                "benchmark_seconds": benchmark_seconds,
                "output_tokens": total_output_tokens,
                "aggregate_output_tokens_per_second": total_output_tokens / benchmark_seconds,
                "turns": turn_summaries,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
