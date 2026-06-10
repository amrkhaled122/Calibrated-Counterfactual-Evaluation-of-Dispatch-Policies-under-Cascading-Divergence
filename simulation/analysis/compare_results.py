"""Compare consolidated simulation metrics for baseline, BC, and DDQN."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable


def load_metrics(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def metric_at(report: Dict[str, Any], dotted_path: str, default: float = 0.0) -> Any:
    value: Any = report
    for key in dotted_path.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def summarize(report: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "mode": metric_at(report, "metadata.mode", "unknown"),
        "total_events": metric_at(report, "input_data.total_events"),
        "unique_orders": metric_at(report, "input_data.unique_orders"),
        "delivery_rate": metric_at(report, "order_kpis.delivery_rate"),
        "acceptance_rate": metric_at(report, "order_kpis.acceptance_rate"),
        "lost_rate": metric_at(report, "order_kpis.lost_rate"),
        "lateness_rate": metric_at(report, "order_kpis.lateness_rate"),
        "divergence_rate": metric_at(report, "order_kpis.divergence_rate"),
        "total_income_delta": metric_at(report, "income_metrics.total_income_delta"),
        "overall_utilization": metric_at(
            report, "courier_kpis.utilization_metrics.overall_utilization"
        ),
    }


def print_table(rows: Iterable[Dict[str, Any]]) -> None:
    metrics = [
        ("delivery_rate", "Delivery rate", ".2%"),
        ("acceptance_rate", "Acceptance rate", ".2%"),
        ("lost_rate", "Lost rate", ".2%"),
        ("lateness_rate", "Late rate", ".2%"),
        ("divergence_rate", "Divergence rate", ".2%"),
        ("overall_utilization", "Utilization", ".2%"),
        ("total_income_delta", "Income delta", ".2f"),
    ]
    rows = list(rows)
    modes = [str(r["mode"]) for r in rows]

    print("Simulation comparison")
    print("=" * 80)
    print(f"{'Metric':<22}" + "".join(f"{mode:>22}" for mode in modes))
    print("-" * 80)
    for key, label, fmt in metrics:
        values = []
        for row in rows:
            value = row[key]
            values.append(f"{value:{fmt}}" if isinstance(value, (int, float)) else str(value))
        print(f"{label:<22}" + "".join(f"{value:>22}" for value in values))


def compare_results(results_dir: Path, output_path: Path | None = None) -> Dict[str, Any]:
    paths = [
        results_dir / "system_simulation_metrics_baseline.json",
        results_dir / "system_simulation_metrics_bc_agent.json",
        results_dir / "system_simulation_metrics_ddqn_agent.json",
    ]
    reports = [load_metrics(path) for path in paths]
    summaries = [summarize(report) for report in reports]
    print_table(summaries)

    comparison = {"summaries": summaries, "source_files": [str(path) for path in paths]}
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2)
        print(f"\nWrote {output_path}")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare consolidated simulation metrics")
    parser.add_argument("--results_dir", default="results", help="Directory containing result JSONs")
    parser.add_argument(
        "--output",
        default="results/comparison_summary.json",
        help="Path to write comparison summary JSON",
    )
    args = parser.parse_args()
    compare_results(Path(args.results_dir), Path(args.output) if args.output else None)


if __name__ == "__main__":
    main()
