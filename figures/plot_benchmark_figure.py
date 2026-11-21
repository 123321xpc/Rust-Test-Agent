#!/usr/bin/env python3
"""Publication-grade 4-panel benchmark comparison figure (Nature double-column)."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.transforms import ScaledTranslation

SKILL_SCRIPTS_DIR = Path(
    r"C:\Users\vickey\.gemini\config\plugins\nature-skills\skills\nature-figure\scripts"
)
if str(SKILL_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SKILL_SCRIPTS_DIR))

from audit_panel_alignment import require_matplotlib_panel_alignment


# ── MANDATORY editable font & SVG/PDF rules ───────────────────────────────────
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7.0,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 7.0,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "legend.frameon": False,
})

# Unified method palette: neutral grey baseline + progressive blue system family
METHOD_ORDER = ["swe-agent", "normal", "ACI-impr", "graph"]
METHOD_COLORS = {
    "swe-agent": "#B2B2B2",   # neutral external baseline
    "normal":    "#8EB2D9",   # base system
    "ACI-impr":  "#3775BA",   # improved ACI
    "graph":     "#0F4D92",   # full graph system (hero)
}
DELTA_UP = "#1B7837"
DELTA_DOWN = "#B64342"
EDGE_DARK = "#272727"
MUTED_TEXT = "#4D4D4D"


def add_panel_label(
    ax: plt.Axes,
    label: str,
    x_offset_pt: float = -24.0,
    y_offset_pt: float = 5.0,
    fontsize: float = 9.0,
) -> None:
    """Place bold lowercase panel label at a fixed physical point offset."""
    offset = ScaledTranslation(
        x_offset_pt / 72.0,
        y_offset_pt / 72.0,
        ax.figure.dpi_scale_trans,
    )
    ax.text(
        0.0,
        1.0,
        label,
        transform=ax.transAxes + offset,
        fontsize=fontsize,
        fontweight="bold",
        color="black",
        ha="left",
        va="bottom",
    )


def add_step_bracket(
    ax: plt.Axes,
    x1: float,
    x2: float,
    y_base: float,
    height: float,
    text: str,
    color: str,
    fontsize: float = 6.0,
) -> None:
    """Draw a clean horizontal comparison bracket with delta label above it."""
    y_top = y_base + height
    ax.plot(
        [x1, x1, x2, x2],
        [y_base, y_top, y_top, y_base],
        color=color,
        lw=0.85,
        solid_capstyle="butt",
        clip_on=False,
    )
    ax.text(
        0.5 * (x1 + x2),
        y_top + 0.8,
        text,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        fontweight="bold",
        color=color,
    )


def load_benchmark_csv(csv_path: Path) -> dict[str, dict[str, float]]:
    """Load unmodified benchmark metrics keyed by canonical method name."""
    rows: dict[str, dict[str, float]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            method = row["method"]
            rows[method] = {
                key: float(val)
                for key, val in row.items()
                if key != "method"
            }
    missing = [m for m in METHOD_ORDER if m not in rows]
    if missing:
        raise ValueError(f"Missing methods in CSV: {missing}")
    return rows


def build_figure(data: dict[str, dict[str, float]], out_dir: Path) -> None:
    """Assemble and validate the 4-panel figure, then export vector and raster files."""
    fig = plt.figure(figsize=(183.0 / 25.4, 128.0 / 25.4), dpi=300, facecolor="white")

    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.0, 1.32],
        height_ratios=[1.02, 1.0],
        left=0.075,
        right=0.975,
        top=0.885,
        bottom=0.095,
        wspace=0.25,
        hspace=0.36,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    colors = [METHOD_COLORS[m] for m in METHOD_ORDER]
    x_pos = np.arange(len(METHOD_ORDER))

    # ── Shared top legend ─────────────────────────────────────────────────────
    legend_handles = [
        Patch(
            facecolor=METHOD_COLORS[m],
            edgecolor=EDGE_DARK,
            linewidth=0.6,
            label=m,
        )
        for m in METHOD_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.525, 0.985),
        ncol=4,
        handlelength=1.4,
        handleheight=0.85,
        columnspacing=1.8,
        frameon=False,
    )

    # ── Panel a: Overall resolution (n = 100) ─────────────────────────────────
    resolved_pct = [data[m]["resolved_pct"] for m in METHOD_ORDER]
    resolved_cnt = [int(data[m]["resolved_count"]) for m in METHOD_ORDER]
    bars_a = ax_a.bar(
        x_pos,
        resolved_pct,
        width=0.58,
        color=colors,
        edgecolor=EDGE_DARK,
        linewidth=0.6,
        zorder=3,
    )
    for bar, pct, cnt in zip(bars_a, resolved_pct, resolved_cnt):
        ax_a.text(
            bar.get_x() + bar.get_width() / 2.0,
            pct + 0.9,
            f"{cnt} ({pct:.0f}%)",
            ha="center",
            va="bottom",
            fontsize=6.0,
            color=EDGE_DARK,
        )

    rel_a1 = (data["ACI-impr"]["resolved_count"] - data["normal"]["resolved_count"]) / data["normal"]["resolved_count"] * 100.0
    rel_a2 = (data["graph"]["resolved_count"] - data["ACI-impr"]["resolved_count"]) / data["ACI-impr"]["resolved_count"] * 100.0
    add_step_bracket(ax_a, 1.0, 1.96, 35.5, 1.6, f"+{rel_a1:.2f}%", DELTA_UP)
    add_step_bracket(ax_a, 2.04, 3.0, 38.8, 1.6, f"+{rel_a2:.2f}%", DELTA_UP)

    ax_a.set_xticks(x_pos)
    ax_a.set_xticklabels(METHOD_ORDER)
    ax_a.set_ylabel("Resolved tasks (%, n = 100)")
    ax_a.set_ylim(0, 46)
    ax_a.set_yticks([0, 10, 20, 30, 40])
    add_panel_label(ax_a, "a")

    # ── Panel b: Stratified resolution by patch size ──────────────────────────
    strata = [
        ("Small\n(n = 40)", "small_pct", "small_solved"),
        ("Medium\n(n = 55)", "medium_pct", "medium_solved"),
        ("Large\n(n = 5)", "large_pct", "large_solved"),
    ]
    n_strata = len(strata)
    n_methods = len(METHOD_ORDER)
    group_x = np.arange(n_strata)
    bar_w = 0.72 / n_methods

    for i, method in enumerate(METHOD_ORDER):
        offset = (i - (n_methods - 1) / 2.0) * bar_w
        vals = [data[method][pct_key] for _, pct_key, _ in strata]
        cnts = [int(data[method][cnt_key]) for _, _, cnt_key in strata]
        bars_b = ax_b.bar(
            group_x + offset,
            vals,
            width=bar_w,
            color=METHOD_COLORS[method],
            edgecolor=EDGE_DARK,
            linewidth=0.55,
            zorder=3,
        )
        # Keep bar annotations sparse and legible: show exact % on Small/Medium peak & transitions
        for s_idx, (bar, val, cnt) in enumerate(zip(bars_b, vals, cnts)):
            if s_idx < 2:
                ax_b.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    val + 1.1,
                    f"{cnt}",
                    ha="center",
                    va="bottom",
                    fontsize=5.8,
                    color=EDGE_DARK,
                )

    # Highlight relative transitions for Small & Medium and the 0/5 Large wall
    small_r1 = (data["ACI-impr"]["small_solved"] - data["normal"]["small_solved"]) / data["normal"]["small_solved"] * 100.0
    small_r2 = (data["graph"]["small_solved"] - data["ACI-impr"]["small_solved"]) / data["ACI-impr"]["small_solved"] * 100.0
    med_r1 = (data["ACI-impr"]["medium_solved"] - data["normal"]["medium_solved"]) / data["normal"]["medium_solved"] * 100.0
    med_r2 = (data["graph"]["medium_solved"] - data["ACI-impr"]["medium_solved"]) / data["ACI-impr"]["medium_solved"] * 100.0

    add_step_bracket(ax_b, 0 - 0.5 * bar_w, 0 + 0.45 * bar_w, 57.5, 2.0, f"+{small_r1:.2f}%", DELTA_UP, fontsize=5.6)
    add_step_bracket(ax_b, 0 + 0.55 * bar_w, 0 + 1.5 * bar_w, 64.0, 2.0, f"+{small_r2:.2f}%", DELTA_UP, fontsize=5.6)
    add_step_bracket(ax_b, 1 - 0.5 * bar_w, 1 + 0.45 * bar_w, 25.5, 2.0, f"{med_r1:.2f}%", DELTA_DOWN, fontsize=5.6)
    add_step_bracket(ax_b, 1 + 0.55 * bar_w, 1 + 1.5 * bar_w, 32.0, 2.0, f"{med_r2:.2f}%", MUTED_TEXT, fontsize=5.6)

    ax_b.text(
        2.0,
        4.0,
        "0 / 5 solved\n(all methods)",
        ha="center",
        va="bottom",
        fontsize=6.0,
        color=MUTED_TEXT,
        style="italic",
    )

    ax_b.set_xticks(group_x)
    ax_b.set_xticklabels([label for label, _, _ in strata])
    ax_b.set_ylabel("Resolution rate within stratum (%)")
    ax_b.set_ylim(0, 72)
    ax_b.set_yticks([0, 20, 40, 60])
    add_panel_label(ax_b, "b")

    # ── Panel c: Mean token cost per task (Input + Output split) ──────────────
    input_k = [data[m]["input_tokens"] / 100.0 / 1000.0 for m in METHOD_ORDER]
    output_k = [data[m]["output_tokens"] / 100.0 / 1000.0 for m in METHOD_ORDER]
    total_k = [data[m]["mean_tokens"] / 1000.0 for m in METHOD_ORDER]

    ax_c.bar(
        x_pos,
        input_k,
        width=0.58,
        color=colors,
        edgecolor=EDGE_DARK,
        linewidth=0.6,
        zorder=3,
    )
    ax_c.bar(
        x_pos,
        output_k,
        bottom=input_k,
        width=0.58,
        color=EDGE_DARK,
        edgecolor=EDGE_DARK,
        linewidth=0.6,
        zorder=3,
    )

    for idx, (tot, out) in enumerate(zip(total_k, output_k)):
        ax_c.text(
            x_pos[idx],
            tot + 18.0,
            f"{tot:.1f}k\n(out {out:.1f}k)",
            ha="center",
            va="bottom",
            fontsize=5.7,
            color=EDGE_DARK,
            linespacing=1.35,
        )

    tok_r1 = (data["ACI-impr"]["total_tokens"] - data["normal"]["total_tokens"]) / data["normal"]["total_tokens"] * 100.0
    tok_r2 = (data["graph"]["total_tokens"] - data["ACI-impr"]["total_tokens"]) / data["ACI-impr"]["total_tokens"] * 100.0
    add_step_bracket(ax_c, 1.0, 2.0, 445.0, 24.0, f"+{tok_r1:.1f}%", DELTA_DOWN, fontsize=5.8)
    add_step_bracket(ax_c, 2.0, 3.0, 870.0, 24.0, f"+{tok_r2:.1f}%", DELTA_DOWN, fontsize=5.8)

    # Compact legend clarifying Input vs Output cap
    io_handles = [
        Patch(facecolor="#D8D8D8", edgecolor=EDGE_DARK, linewidth=0.5, label="Input (>97.9%)"),
        Patch(facecolor=EDGE_DARK, edgecolor=EDGE_DARK, linewidth=0.5, label="Output (<2.2%)"),
    ]
    ax_c.legend(
        handles=io_handles,
        loc="upper left",
        fontsize=5.8,
        handlelength=1.1,
        handleheight=0.8,
        borderaxespad=0.3,
    )

    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels(METHOD_ORDER)
    ax_c.set_ylabel("Mean tokens per task (×10³)")
    ax_c.set_ylim(0, 980)
    ax_c.set_yticks([0, 200, 400, 600, 800])
    add_panel_label(ax_c, "c")

    # ── Panel d: Cost–performance Pareto frontier (Runtime × API calls) ───────
    times = [data[m]["mean_time_s"] for m in METHOD_ORDER]
    res_pct = [data[m]["resolved_pct"] for m in METHOD_ORDER]
    calls = [int(data[m]["api_calls"]) for m in METHOD_ORDER]
    hours = [data[m]["total_time_h"] for m in METHOD_ORDER]

    # Progression trajectory from normal -> ACI-impr -> graph
    ax_d.plot(
        times[1:],
        res_pct[1:],
        color="#767676",
        linestyle="--",
        linewidth=0.95,
        zorder=2,
    )

    # Scale marker area linearly with API calls (clearly readable range)
    sizes = [c * 0.045 for c in calls]
    for idx, method in enumerate(METHOD_ORDER):
        ax_d.scatter(
            times[idx],
            res_pct[idx],
            s=sizes[idx],
            color=METHOD_COLORS[method],
            edgecolor=EDGE_DARK,
            linewidth=0.8,
            zorder=4,
        )

    # Direct labels positioned cleanly away from points and trajectory
    label_specs = {
        "swe-agent": (197.0, 14.0, "left", "center"),
        "normal":    (155.0, 30.2, "left", "bottom"),
        "ACI-impr":  (338.0, 33.8, "center", "bottom"),
        "graph":     (472.0, 26.5, "center", "top"),
    }
    for idx, method in enumerate(METHOD_ORDER):
        lx, ly, ha, va = label_specs[method]
        ax_d.text(
            lx,
            ly,
            f"{method}\n{calls[idx]:,} calls · {hours[idx]:.2f} h",
            ha=ha,
            va=va,
            fontsize=5.8,
            color=EDGE_DARK,
            linespacing=1.35,
        )

    # Annotate transition cost vs gain along the dashed trajectory
    time_r1 = (data["ACI-impr"]["mean_time_s"] - data["normal"]["mean_time_s"]) / data["normal"]["mean_time_s"] * 100.0
    time_r2 = (data["graph"]["mean_time_s"] - data["ACI-impr"]["mean_time_s"]) / data["ACI-impr"]["mean_time_s"] * 100.0
    ax_d.text(
        262.0,
        26.5,
        f"+{time_r1:.1f}% time\n(+14.8% res.)",
        ha="center",
        va="top",
        fontsize=5.6,
        color=MUTED_TEXT,
        linespacing=1.35,
    )
    ax_d.text(
        416.0,
        33.4,
        f"+{time_r2:.1f}% time\n(+3.2% res.)",
        ha="center",
        va="bottom",
        fontsize=5.6,
        color=MUTED_TEXT,
        linespacing=1.35,
    )

    ax_d.set_xlabel("Mean runtime per task (s)")
    ax_d.set_ylabel("Resolved tasks (%)")
    ax_d.set_xlim(140, 530)
    ax_d.set_ylim(10, 39)
    ax_d.set_xticks([150, 250, 350, 450, 500])
    ax_d.set_yticks([10, 20, 30, 35])
    add_panel_label(ax_d, "d")

    # ── Render-time panel alignment gate (blocking) ───────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "rust_swe_bench_comparison"

    require_matplotlib_panel_alignment(
        fig,
        axes=[ax_a, ax_b, ax_c, ax_d],
        panel_ids=["a", "b", "c", "d"],
        row_groups=[["a", "b"], ["c", "d"]],
        column_groups=[["a", "c"], ["b", "d"]],
        json_out=str(stem) + ".alignment.json",
        overlay_svg=str(stem) + ".alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )

    fig.savefig(str(stem) + ".svg", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(str(stem) + ".pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(str(stem) + ".tiff", dpi=600, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(str(stem) + ".png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> None:
    fig_dir = Path(__file__).resolve().parent
    data = load_benchmark_csv(fig_dir / "benchmark_summary.csv")
    build_figure(data, fig_dir)
    print("Exported figure bundle to:", fig_dir)


if __name__ == "__main__":
    main()
