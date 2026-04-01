"""Run an LLM-based frustration evaluation on Phoenix traces.

Example:
    poetry run python scripts/evaluate_user_frustration.py \
        --project se-interview \
        --output artifacts/user_frustration_eval.csv
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from phoenix.client import Client
from phoenix.evals import LLM, create_classifier, evaluate_dataframe
from phoenix.evals.utils import to_annotation_dataframe

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Phoenix user-frustration evaluation and attach results to spans.",
    )
    parser.add_argument("--phoenix-url", default="http://127.0.0.1:6006", help="Phoenix base URL.")
    parser.add_argument("--project", default="se-interview", help="Phoenix project name or id.")
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Maximum number of LangGraph root spans to evaluate.",
    )
    parser.add_argument(
        "--annotation-name",
        default="user_frustration",
        help="Annotation name to write back to Phoenix.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/user_frustration_eval.csv",
        help="Local CSV artifact for the raw evaluation results.",
    )
    return parser.parse_args()


def _extract_human_prompt(payload: str) -> str:
    data = json.loads(payload)
    messages = data.get("messages", [])
    for message in messages:
        if message.get("type") == "human":
            return message.get("content", "")
    return ""


def _extract_final_response(payload: str) -> str:
    data = json.loads(payload)
    messages = data.get("messages", [])
    for message in reversed(messages):
        if message.get("type") == "ai" and message.get("content"):
            return message.get("content", "")
    return ""


def load_interactions(client: Client, project: str, limit: int) -> pd.DataFrame:
    spans_df = client.spans.get_spans_dataframe(
        project_identifier=project,
        limit=limit,
        root_spans_only=True,
    )
    spans_df = spans_df[spans_df["name"] == "LangGraph"].copy()
    spans_df["user_prompt"] = spans_df["attributes.input.value"].map(_extract_human_prompt)
    spans_df["assistant_response"] = spans_df["attributes.output.value"].map(_extract_final_response)
    spans_df["span_id"] = spans_df["context.span_id"]
    # Keep just the fields the evaluator needs to judge likely user frustration.
    spans_df = spans_df[
        [
            "span_id",
            "context.trace_id",
            "start_time",
            "user_prompt",
            "assistant_response",
        ]
    ].copy()
    spans_df = spans_df[spans_df["assistant_response"].str.len() > 0]
    spans_df = spans_df.set_index("span_id")
    return spans_df


def build_evaluator():
    model = LLM(provider="openai", model="gpt-4o-mini")
    return create_classifier(
        name="user_frustration",
        llm=model,
        prompt_template=(
            "You are evaluating whether a user is likely to feel frustrated by an assistant response "
            "in a travel assistant application.\n\n"
            "Classify the interaction as:\n"
            "- frustrated: The answer is likely to leave the user dissatisfied because it is outdated, "
            "generic, incomplete, ignores part of the request, refuses incorrectly, gives poor travel guidance, "
            "uses stale time references, or fails to provide concrete travel planning value.\n"
            "- not_frustrated: The answer is likely helpful, relevant, and responsive to the user's travel request.\n\n"
            "Mark the interaction as frustrated if any of these are true:\n"
            "- it includes stale references such as old years or obviously outdated framing\n"
            "- it gives a generic destination list instead of specific travel planning help\n"
            "- it omits key requested categories like hotels, flights, attractions, neighborhoods, or food spots\n"
            "- it gives shallow or low-confidence recommendations without practical details\n"
            "- it does not adapt to the user's stated destination, duration, or trip style\n\n"
            "Input:\n"
            "User prompt: {user_prompt}\n\n"
            "Assistant response: {assistant_response}\n"
        ),
        choices={"frustrated": 1.0, "not_frustrated": 0.0},
    )


def main() -> int:
    args = parse_args()
    client = Client(base_url=args.phoenix_url)
    interactions_df = load_interactions(client, args.project, args.limit)
    evaluator = build_evaluator()

    results_df = evaluate_dataframe(
        dataframe=interactions_df,
        evaluators=[evaluator],
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path)

    # Attach the eval back to Phoenix so the traces can be filtered in the UI.
    annotations_df = to_annotation_dataframe(dataframe=results_df, score_names=["user_frustration"])
    client.spans.log_span_annotations_dataframe(
        dataframe=annotations_df,
        annotation_name=args.annotation_name,
        annotator_kind="LLM",
        sync=True,
    )

    frustrated_count = 0
    if "label" in annotations_df.columns:
        frustrated_count = int((annotations_df["label"] == "frustrated").sum())

    print(
        f"Evaluated {len(results_df)} interactions, attached annotations to Phoenix, "
        f"and flagged {frustrated_count} as frustrated."
    )
    print(f"Saved raw eval results to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
