"""Export spans from a Phoenix project to CSV.

Example:
    poetry run python scripts/export_phoenix_spans.py \
        --project se-interview \
        --limit 500 \
        --output artifacts/phoenix_spans.csv
"""

import argparse
from pathlib import Path

from phoenix.client import Client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export spans from a local Phoenix project.",
    )
    parser.add_argument(
        "--phoenix-url",
        default="http://127.0.0.1:6006",
        help="Phoenix base URL.",
    )
    parser.add_argument(
        "--project",
        default="se-interview",
        help="Phoenix project name or id.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Maximum number of spans to export.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/phoenix_spans.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    client = Client(base_url=args.phoenix_url)
    spans_df = client.spans.get_spans_dataframe(
        project_identifier=args.project,
        limit=args.limit,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    spans_df.to_csv(output_path, index=False)

    print(f"Exported {len(spans_df)} spans to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
