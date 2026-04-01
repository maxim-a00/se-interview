"""Send one prompt to the API and print recent Phoenix spans.

Example:
    poetry run python scripts/run_traced_prompt.py \
        "Find hotel options in Barcelona for two travelers." \
        --project se-interview
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

from phoenix.client import Client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send a prompt to the local API and print recent Phoenix spans.",
    )
    parser.add_argument("prompt", help="The user prompt to send to the /chat endpoint.")
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8000/chat",
        help="Chat endpoint URL.",
    )
    parser.add_argument(
        "--phoenix-url",
        default="http://127.0.0.1:6006",
        help="Phoenix base URL.",
    )
    parser.add_argument(
        "--project",
        default="se-interview",
        help="Phoenix project name.",
    )
    parser.add_argument(
        "--span-limit",
        type=int,
        default=12,
        help="Maximum number of recent spans to print.",
    )
    parser.add_argument(
        "--wait-seconds",
        type=float,
        default=2.0,
        help="Seconds to wait before querying Phoenix for spans.",
    )
    return parser.parse_args()


def post_prompt(api_url: str, prompt: str) -> dict:
    payload = json.dumps({"message": prompt}).encode("utf-8")
    request = urllib.request.Request(
        api_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def print_spans(phoenix_url: str, project: str, span_limit: int) -> None:
    client = Client(base_url=phoenix_url)
    spans_df = client.spans.get_spans_dataframe(
        project_identifier=project,
        limit=span_limit,
    )
    if spans_df.empty:
        print("\nNo spans found in Phoenix yet.")
        return

    preferred_columns = [
        "name",
        "span_kind",
        "status_code",
        "parent_id",
    ]
    available_columns = [column for column in preferred_columns if column in spans_df.columns]
    print("\nRecent Phoenix spans:")
    print(spans_df[available_columns].head(span_limit).to_string(index=False))


def main() -> int:
    args = parse_args()

    try:
        response = post_prompt(args.api_url, args.prompt)
    except urllib.error.URLError as exc:
        print(f"Failed to call API at {args.api_url}: {exc}", file=sys.stderr)
        return 1

    print("Model response:\n")
    print(response.get("response", "<missing response field>"))

    # Give Phoenix a brief moment to ingest the trace before querying it.
    time.sleep(args.wait_seconds)

    try:
        print_spans(args.phoenix_url, args.project, args.span_limit)
    except Exception as exc:  # noqa: BLE001
        print(f"\nFailed to query Phoenix at {args.phoenix_url}: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
