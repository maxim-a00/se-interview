"""Create a Phoenix dataset from reviewed frustrated interactions.

Example:
    poetry run python scripts/create_frustrated_dataset.py \
        --project se-interview \
        --dataset-name frustrated-interactions \
        --output artifacts/frustrated_interactions.json
"""

import argparse
import json
from pathlib import Path

from phoenix.client import Client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a Phoenix dataset from interactions labeled as frustrated.",
    )
    parser.add_argument("--phoenix-url", default="http://127.0.0.1:6006", help="Phoenix base URL.")
    parser.add_argument("--project", default="se-interview", help="Phoenix project name or id.")
    parser.add_argument(
        "--dataset-name",
        default="frustrated-interactions",
        help="Dataset name to create in Phoenix.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=300,
        help="Maximum number of spans to inspect.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/frustrated_interactions.json",
        help="Local JSON artifact for the filtered dataset rows.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    client = Client(base_url=args.phoenix_url)
    spans_df = client.spans.get_spans_dataframe(
        project_identifier=args.project,
        limit=args.limit,
        root_spans_only=True,
    )
    spans_df = spans_df[spans_df["name"] == "LangGraph"].copy()
    annotations_df = client.spans.get_span_annotations_dataframe(
        spans_dataframe=spans_df,
        project_identifier=args.project,
        include_annotation_names=["user_feedback"],
        exclude_annotation_names=[],
    )
    if annotations_df.empty:
        raise ValueError("No user_feedback annotations found on spans.")

    feedback_df = annotations_df[
        [
            "result.label",
            "result.score",
            "result.explanation",
        ]
    ].copy()
    feedback_df = feedback_df.rename(
        columns={
            "result.label": "feedback_label",
            "result.score": "feedback_score",
            "result.explanation": "feedback_explanation",
        }
    )

    spans_df["span_id"] = spans_df["context.span_id"]
    dataset_df = spans_df.merge(
        feedback_df,
        left_on="span_id",
        right_index=True,
        how="inner",
    )
    dataset_df = dataset_df[dataset_df["feedback_label"] == "frustrated"].copy()
    if dataset_df.empty:
        raise ValueError("No frustrated interactions matched the filter.")

    # Normalize the traced interaction into dataset-friendly input/output fields.
    dataset_df["prompt"] = dataset_df["attributes.input.value"]
    dataset_df["response"] = dataset_df["attributes.output.value"]
    dataset_df["frustration_label"] = dataset_df["feedback_label"]
    dataset_df["frustration_score"] = dataset_df["feedback_score"]
    dataset_df["frustration_explanation"] = dataset_df["feedback_explanation"]
    dataset_df["trace_id"] = dataset_df["context.trace_id"]

    dataset_payload = dataset_df[
        [
            "prompt",
            "response",
            "frustration_label",
            "frustration_score",
            "frustration_explanation",
            "span_id",
            "trace_id",
        ]
    ].copy()

    client.datasets.create_dataset(
        name=args.dataset_name,
        dataframe=dataset_payload,
        input_keys=["prompt"],
        output_keys=["response"],
        metadata_keys=[
            "frustration_label",
            "frustration_score",
            "frustration_explanation",
            "trace_id",
        ],
        span_id_key="span_id",
        dataset_description="Interactions filtered to likely-frustrated travel assistant responses.",
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(dataset_payload.to_dict(orient="records"), indent=2)
    )

    print(
        f"Created Phoenix dataset '{args.dataset_name}' with {len(dataset_payload)} frustrated interactions."
    )
    print(f"Saved filtered examples to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
