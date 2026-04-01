"""Send a batch of prompts to the local API and save the responses.

Example:
    poetry run python scripts/run_prompt_batch.py \
        artifacts/travel_test_prompts.json \
        --output artifacts/travel_test_results.json
"""

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send a batch of prompts to the local chat API and save the responses.",
    )
    parser.add_argument(
        "prompts_file",
        help="Path to a JSON file containing a list of prompt strings.",
    )
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8000/chat",
        help="Chat endpoint URL.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/prompt_batch_results.json",
        help="Output path for the JSON results.",
    )
    return parser.parse_args()


def load_prompts(path: Path) -> list[str]:
    prompts = json.loads(path.read_text())
    if not isinstance(prompts, list) or not all(isinstance(item, str) for item in prompts):
        raise ValueError("Prompts file must contain a JSON array of strings.")
    return prompts


def post_prompt(api_url: str, prompt: str) -> dict:
    payload = json.dumps({"message": prompt}).encode("utf-8")
    request = urllib.request.Request(
        api_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    args = parse_args()
    prompts_path = Path(args.prompts_file)
    output_path = Path(args.output)

    try:
        prompts = load_prompts(prompts_path)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to load prompts: {exc}", file=sys.stderr)
        return 1

    results = []
    for index, prompt in enumerate(prompts, start=1):
        try:
            response = post_prompt(args.api_url, prompt)
            results.append(
                {
                    "index": index,
                    "prompt": prompt,
                    "response": response.get("response", ""),
                    "status": "ok",
                }
            )
            print(f"[{index}/{len(prompts)}] completed", flush=True)
        except urllib.error.URLError as exc:
            # Keep failed requests in the output artifact so the batch remains auditable.
            results.append(
                {
                    "index": index,
                    "prompt": prompt,
                    "response": "",
                    "status": "error",
                    "error": str(exc),
                }
            )
            print(f"[{index}/{len(prompts)}] failed: {exc}", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"Saved results to {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
