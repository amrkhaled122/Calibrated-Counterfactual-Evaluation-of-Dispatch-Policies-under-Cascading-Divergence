"""
Build a paper-ready divergence case-study figure and metrics table.

Uses:
- chosen courier case output (from select_case_courier)
- DDQN simulation metrics JSON
- baseline simulation metrics JSON
- idle-block logs (optional, for extra context)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_pct(num: float, den: float) -> float:
    return (num / den * 100.0) if den else 0.0


def _extract_idle_block_stats(log_path: Path, courier_id: int) -> dict[str, float]:
    if not log_path.exists():
        return {"blocks": 0, "avg_idle_util_pct": 0.0}

    header_pat = re.compile(rf"^COURIER:\s+{courier_id}\s+\|\s+Block\s+\d+\s+of\s+\d+")
    util_pat = re.compile(r"UTILIZATION:\s+([0-9]+\.?[0-9]*)%")

    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    in_block = False
    util_values = []
    for line in lines:
        if header_pat.search(line):
            in_block = True
            continue
        if in_block and line.startswith("COURIER:"):
            in_block = False
        if in_block and "UTILIZATION:" in line:
            m = util_pat.search(line)
            if m:
                util_values.append(float(m.group(1)))
                in_block = False

    if not util_values:
        return {"blocks": 0, "avg_idle_util_pct": 0.0}
    return {
        "blocks": float(len(util_values)),
        "avg_idle_util_pct": float(sum(util_values) / len(util_values)),
    }


def build_case_figure(
    chosen_case_path: Path,
    ddqn_metrics_path: Path,
    baseline_metrics_path: Path,
    ddqn_idle_log_path: Path,
    baseline_idle_log_path: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    chosen_data = _load_json(chosen_case_path)
    ddqn_metrics = _load_json(ddqn_metrics_path)
    baseline_metrics = _load_json(baseline_metrics_path)

    case = chosen_data["chosen_courier"]
    cid = int(case["courier_id"])

    ddqn_idle = _extract_idle_block_stats(ddqn_idle_log_path, cid)
    baseline_idle = _extract_idle_block_stats(baseline_idle_log_path, cid)

    hist_offers = float(case["offers"])
    hist_accepts = float(case["historical_accepts"])
    hist_rejects = float(case["historical_rejects"])

    sim_offers = float(case["sim_offers_seen"])
    sim_accepts = float(case["sim_accepts"])
    sim_rejects = float(case["sim_rejects"])
    sim_delivered = float(case["sim_delivered"])
    sim_late = float(case["sim_late_deliveries"])

    div_origin = float(case["divergence_originated"])
    div_assigned = float(case["divergence_assigned"])
    reassigned_out = float(case["reassigned_out"])
    lost_from_origin = float(case["lost_from_origin"])
    local_cascade = reassigned_out + lost_from_origin
    local_amp = local_cascade / max(1.0, div_origin)

    ddqn_order = ddqn_metrics["order_kpis"]
    base_order = baseline_metrics["order_kpis"]

    # ---------- Figure ----------
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(15, 9))

    # A) System-level baseline vs DDQN
    ax1 = plt.subplot(2, 2, 1)
    sys_labels = ["Delivery Rate", "Late Rate", "Lost Rate", "Divergence Rate"]
    base_vals = [
        base_order["delivery_rate"] * 100.0,
        base_order["lateness_rate"] * 100.0,
        base_order["lost_rate"] * 100.0,
        base_order["divergence_rate"] * 100.0,
    ]
    ddqn_vals = [
        ddqn_order["delivery_rate"] * 100.0,
        ddqn_order["lateness_rate"] * 100.0,
        ddqn_order["lost_rate"] * 100.0,
        ddqn_order["divergence_rate"] * 100.0,
    ]
    x = np.arange(len(sys_labels))
    w = 0.36
    ax1.bar(x - w / 2, base_vals, w, label="History Replay", color="#4C78A8")
    ax1.bar(x + w / 2, ddqn_vals, w, label="DDQN Integrated", color="#F58518")
    ax1.set_xticks(x)
    ax1.set_xticklabels(sys_labels, rotation=0)
    ax1.set_ylabel("Percent (%)")
    ax1.set_title("System-Level Shift (24h Window)")
    ax1.legend()

    # B) Courier 82: history vs simulation behavior
    ax2 = plt.subplot(2, 2, 2)
    courier_labels = ["Accept Rate", "Reject Rate", "Late Among Delivered"]
    hist_accept_rate = _safe_pct(hist_accepts, hist_offers)
    hist_reject_rate = _safe_pct(hist_rejects, hist_offers)
    sim_accept_rate = _safe_pct(sim_accepts, sim_offers)
    sim_reject_rate = _safe_pct(sim_rejects, sim_offers)
    sim_late_rate = _safe_pct(sim_late, sim_delivered)
    # Historical late rate is not directly tracked in chosen file; set to NaN for visual gap
    hist_vals = [hist_accept_rate, hist_reject_rate, np.nan]
    sim_vals = [sim_accept_rate, sim_reject_rate, sim_late_rate]
    x2 = np.arange(len(courier_labels))
    ax2.bar(x2 - w / 2, hist_vals, w, label="Historical Decisions", color="#54A24B")
    ax2.bar(x2 + w / 2, sim_vals, w, label="Simulated Outcomes", color="#E45756")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(courier_labels)
    ax2.set_ylabel("Percent (%)")
    ax2.set_title(f"Courier {cid}: History vs Simulation")
    ax2.legend()

    # C) Local cascade decomposition (mechanism panel)
    ax3 = plt.subplot(2, 2, 3)
    stay_local = max(0.0, div_origin - local_cascade)
    cascade_parts = [reassigned_out, lost_from_origin, stay_local]
    cascade_labels = ["Reassigned Out", "Lost", "Other Origin Divergences"]
    cascade_colors = ["#72B7B2", "#B279A2", "#BAB0AC"]
    left = 0.0
    for val, label, col in zip(cascade_parts, cascade_labels, cascade_colors):
        ax3.barh([0], [val], left=left, color=col, label=f"{label}: {int(val)}")
        left += val
    ax3.set_yticks([0])
    ax3.set_yticklabels([f"Courier {cid} Originated Divergences ({int(div_origin)})"])
    ax3.set_xlabel("Count")
    ax3.set_title("Cascading Split of Local Mismatches")
    ax3.legend(loc="upper right", fontsize=9)
    ax3.text(
        0.02,
        -0.55,
        f"Local amplification proxy = (reassigned + lost) / originated = {local_cascade:.0f}/{div_origin:.0f} = {local_amp:.2f}",
        transform=ax3.transAxes,
        fontsize=10,
    )

    # D) Divergence timeline sample for selected courier
    ax4 = plt.subplot(2, 2, 4)
    timeline = case.get("divergence_timeline_sample", [])
    if timeline:
        base_ts = min(int(t["timestamp"]) for t in timeline)
        xs = [((int(t["timestamp"]) - base_ts) / 60.0) for t in timeline]
        ys = []
        colors = []
        for t in timeline:
            role = t.get("role", "original")
            dtype = t.get("divergence_type", "")
            y = 1 if role == "original" else 0
            ys.append(y)
            if dtype == "agent_prune_hist_keep":
                colors.append("#EECA3B")
            else:
                colors.append("#4C78A8")
        ax4.scatter(
            xs, ys, c=colors, s=50, alpha=0.85, edgecolors="black", linewidths=0.3
        )
        ax4.set_yticks([0, 1])
        ax4.set_yticklabels(["Assigned-to-Courier", "Originated-by-Courier"])
        ax4.set_xlabel("Minutes from first shown divergence")
        ax4.set_title("Divergence Timeline Sample")
    else:
        ax4.text(0.5, 0.5, "No timeline sample", ha="center", va="center")
        ax4.set_axis_off()

    fig.suptitle(
        f"Case Study: Courier {cid} Divergence Cascade (Hours 96-120)",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])

    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / f"courier_{cid}_divergence_case.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    # ---------- Metrics markdown ----------
    md_path = output_dir / f"courier_{cid}_divergence_case_metrics.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Courier {cid} Divergence Case Study\n\n")
        f.write("## Courier-Level History vs Simulation\n")
        f.write(f"- Historical offers: {int(hist_offers)}\n")
        f.write(
            f"- Historical accepts / rejects: {int(hist_accepts)} / {int(hist_rejects)}\n"
        )
        f.write(f"- Historical accept rate: {hist_accept_rate:.2f}%\n")
        f.write(f"- Sim offers seen: {int(sim_offers)}\n")
        f.write(f"- Sim accepts / rejects: {int(sim_accepts)} / {int(sim_rejects)}\n")
        f.write(f"- Sim accept rate: {sim_accept_rate:.2f}%\n")
        f.write(f"- Sim delivered: {int(sim_delivered)}\n")
        f.write(
            f"- Sim late deliveries: {int(sim_late)} ({sim_late_rate:.2f}% of delivered)\n\n"
        )

        f.write("## Divergence Cascade Metrics\n")
        f.write(f"- Originated divergences: {int(div_origin)}\n")
        f.write(f"- Assigned divergences (incoming): {int(div_assigned)}\n")
        f.write(f"- Reassigned out: {int(reassigned_out)}\n")
        f.write(f"- Lost from origin: {int(lost_from_origin)}\n")
        f.write(f"- Local cascade proxy (reassigned + lost): {int(local_cascade)}\n")
        f.write(f"- Local amplification ratio: {local_amp:.2f}\n")
        f.write(f"- Divergence types: {case.get('divergence_type_counts', {})}\n\n")

        f.write("## System-Level Baseline vs DDQN (Same Window)\n")
        f.write(
            f"- Delivery rate: {base_order['delivery_rate'] * 100:.2f}% -> {ddqn_order['delivery_rate'] * 100:.2f}%\n"
        )
        f.write(
            f"- Late rate: {base_order['lateness_rate'] * 100:.2f}% -> {ddqn_order['lateness_rate'] * 100:.2f}%\n"
        )
        f.write(
            f"- Lost rate: {base_order['lost_rate'] * 100:.2f}% -> {ddqn_order['lost_rate'] * 100:.2f}%\n"
        )
        f.write(
            f"- Divergence rate: {base_order['divergence_rate'] * 100:.2f}% -> {ddqn_order['divergence_rate'] * 100:.2f}%\n"
        )
        f.write(
            f"- Agent prune / hist keep: {ddqn_order['divergence_types'].get('agent_prune_hist_keep', 0)}\n"
        )
        f.write(
            f"- Agent keep / hist prune: {ddqn_order['divergence_types'].get('agent_keep_hist_prune', 0)}\n\n"
        )

        f.write("## Idle-Block Log Context\n")
        f.write(
            f"- Baseline idle blocks for courier {cid}: {int(baseline_idle['blocks'])}, avg utilization in those blocks: {baseline_idle['avg_idle_util_pct']:.1f}%\n"
        )
        f.write(
            f"- DDQN idle blocks for courier {cid}: {int(ddqn_idle['blocks'])}, avg utilization in those blocks: {ddqn_idle['avg_idle_util_pct']:.1f}%\n\n"
        )

        f.write("## Caption Draft\n")
        f.write(
            "A single courier-level mismatch stream (Courier "
            f"{cid}) shows how local action deviations compound into downstream reassignment and loss. "
            f"This courier originated {int(div_origin)} divergences, with {int(reassigned_out)} reassigned and "
            f"{int(lost_from_origin)} becoming unassigned/lost in this window. At the system level, the same run shifts "
            f"lost rate from {base_order['lost_rate'] * 100:.2f}% (history replay) to {ddqn_order['lost_rate'] * 100:.2f}% "
            "(DDQN), supporting the cascading-divergence mechanism.\n"
        )

    return fig_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot courier divergence case-study figure"
    )
    parser.add_argument(
        "--chosen_case",
        default="results/ddqn_integrated_case_selection_96_120/chosen_courier_case_study.json",
    )
    parser.add_argument(
        "--ddqn_metrics",
        default="results/ddqn_integrated_case_selection_96_120/system_simulation_metrics_ddqn_agent.json",
    )
    parser.add_argument(
        "--baseline_metrics",
        default="results/baseline_case_selection_96_120/system_simulation_metrics_baseline.json",
    )
    parser.add_argument(
        "--ddqn_idle_log",
        default="results/ddqn_integrated_case_selection_96_120/log_ddqn_agent_idle_blocks.txt",
    )
    parser.add_argument(
        "--baseline_idle_log",
        default="results/baseline_case_selection_96_120/log_baseline_idle_blocks.txt",
    )
    parser.add_argument(
        "--output_dir",
        default="results/ddqn_integrated_case_selection_96_120",
    )
    args = parser.parse_args()

    fig_path, md_path = build_case_figure(
        chosen_case_path=Path(args.chosen_case),
        ddqn_metrics_path=Path(args.ddqn_metrics),
        baseline_metrics_path=Path(args.baseline_metrics),
        ddqn_idle_log_path=Path(args.ddqn_idle_log),
        baseline_idle_log_path=Path(args.baseline_idle_log),
        output_dir=Path(args.output_dir),
    )
    print(f"Saved figure: {fig_path}")
    print(f"Saved metrics/caption: {md_path}")


if __name__ == "__main__":
    main()
