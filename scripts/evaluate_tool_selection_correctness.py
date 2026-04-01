"""Evaluate agent tool selection with Phoenix's built-in evaluator.

Example:
    poetry run python scripts/evaluate_tool_selection_correctness.py \
        --project se-interview \
        --output artifacts/tool_selection_correctness_eval.csv
"""

import argparse
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from phoenix.client import Client
from phoenix.evals import LLM, evaluate_dataframe
from phoenix.evals.metrics import ToolSelectionEvaluator
from phoenix.evals.utils import to_annotation_dataframe

from app.agent import get_tools

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Phoenix tool-selection evaluation and attach results to spans.",
    )
    parser.add_argument("--phoenix-url", default="http://127.0.0.1:6006", help="Phoenix base URL.")
    parser.add_argument("--project", default="se-interview", help="Phoenix project name or id.")
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Maximum number of spans to inspect while building the evaluation set.",
    )
    parser.add_argument(
        "--annotation-name",
        default="tool_selection_correctness",
        help="Annotation name to write back to Phoenix.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/tool_selection_correctness_eval.csv",
        help="Local CSV artifact for the evaluation results.",
    )
    return parser.parse_args()


def build_available_tools_text() -> str:
    tool_lines = []
    for tool in get_tools():
        description = getattr(tool, "description", "") or ""
        tool_lines.append(f"{tool.name}: {description}".strip())
    return "\n".join(tool_lines)


def build_eval_dataframe(client: Client, project: str, limit: int) -> pd.DataFrame:
    spans_df = client.spans.get_spans_dataframe(project_identifier=project, limit=limit)
    root_df = spans_df[(spans_df["name"] == "LangGraph") & (spans_df["parent_id"].isna())].copy()
    if root_df.empty:
        return pd.DataFrame()

    tool_df = spans_df[spans_df["span_kind"] == "TOOL"].copy()
    tools_by_trace = tool_df.groupby("context.trace_id")["name"].apply(list).to_dict()

    # Phoenix evaluates tool choice per root interaction, so we flatten the
    # available tools plus the selected child tool spans into one row.
    available_tools = build_available_tools_text()
    root_df["span_id"] = root_df["context.span_id"]
    root_df["input"] = root_df["attributes.input.value"].fillna("")
    root_df["available_tools"] = available_tools
    root_df["tool_selection"] = root_df["context.trace_id"].map(
        lambda trace_id: ", ".join(tools_by_trace.get(trace_id, [])) or "none"
    )

    eval_df = root_df[
        [
            "span_id",
            "context.trace_id",
            "input",
            "available_tools",
            "tool_selection",
        ]
    ].copy()
    eval_df = eval_df.set_index("span_id")
    return eval_df


def main() -> int:
    args = parse_args()
    client = Client(base_url=args.phoenix_url)
    eval_df = build_eval_dataframe(client, args.project, args.limit)
    if eval_df.empty:
        raise ValueError("No root LangGraph spans found to evaluate.")

    llm = LLM(provider="openai", model="gpt-4o-mini")
    evaluator = ToolSelectionEvaluator(llm=llm)
    results_df = evaluate_dataframe(dataframe=eval_df, evaluators=[evaluator])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path)

    annotations_df = to_annotation_dataframe(
        dataframe=results_df,
        score_names=["tool_selection"],
    )
    client.spans.log_span_annotations_dataframe(
        dataframe=annotations_df,
        annotation_name=args.annotation_name,
        annotator_kind="LLM",
        sync=True,
    )

    score_col = "tool_selection_score"
    correct = incorrect = 0
    if score_col in results_df.columns:
        labels = results_df[score_col].astype(str)
        correct = int(labels.str.contains("'label': 'correct'").sum())
        incorrect = int(labels.str.contains("'label': 'incorrect'").sum())

    print(
        f"Evaluated {len(results_df)} interactions and attached tool-selection annotations. "
        f"correct={correct}, incorrect={incorrect}"
    )
    print(f"Saved raw eval results to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
