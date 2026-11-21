#!/usr/bin/env python3
"""Publication-grade 4-panel Chinese benchmark comparison figure (Nature double-column)."""

from __future__ import annotations

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
    "font.sans-serif": [
        "Microsoft YaHei",
        "SimHei",
        "Arial",
        "DejaVu Sans",
        "sans-serif",
    ],
    "axes.unicode_minus": False,
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

# Unified method palette: neutral gray baseline + progressive blue system family
METHOD_ORDER = ["swe-agent", "normal", "ACI-impr", "graph"]
METHOD_COLORS = {
    "swe-agent": "#B2B2B2",   # neutral external baseline
    "normal":    "#8EB2D9",   # base system
    "ACI-impr":  "#3775BA",   # improved ACI
    "graph":     "#0F4D92",   # full graph system (hero)
}
DELTA_UP = "#1B7837"
DELTA_DOWN = "#B64342"
DELTA_NEUTRAL = "#4D4D4D"
EDGE_DARK = "#272727"
MUTED_TEXT = "#4D4D4D"

# Exact data from user table (n = 100 tasks: 40 small, 55 medium, 5 large)
DATA = {
    "swe-agent": {
        "resolved": 14, "resolved_pct": 14.0,
        "small": 11, "small_pct": 27.5,
        "medium": 3, "medium_pct": 5.5,
        "large": 0, "large_pct": 0.0,
        "total_tokens": 16_196_675, "input_tokens": 15_857_220, "output_tokens": 339_455,
        "mean_tokens": 161_966.8, "api_calls": 1_788,
        "total_time_s": 18_249.53, "total_time_h": 5.07, "mean_time_s": 182.50,
    },
    "normal": {
        "resolved": 27, "resolved_pct": 27.0,
        "small": 16, "small_pct": 40.0,
        "medium": 11, "medium_pct": 20.0,
        "large": 0, "large_pct": 0.0,
        "total_tokens": 9_388_374, "input_tokens": 9_254_849, "output_tokens": 133_525,
        "mean_tokens": 93_883.7, "api_calls": 1_732,
        "total_time_s": 18_048.77, "total_time_h": 5.01, "mean_time_s": 180.49,
    },
    "ACI-impr": {
        "resolved": 31, "resolved_pct": 31.0,
        "small": 21, "small_pct": 52.5,
        "medium": 10, "medium_pct": 18.2,
        "large": 0, "large_pct": 0.0,
        "total_tokens": 30_374_421, "input_tokens": 30_048_423, "output_tokens": 325_998,
        "mean_tokens": 303_744.2, "api_calls": 3_336,
        "total_time_s": 34_019.04, "total_time_h": 9.45, "mean_time_s": 340.19,
    },
    "graph": {
        "resolved": 32, "resolved_pct": 32.0,
        "small": 22, "small_pct": 55.0,
        "medium": 10, "medium_pct": 18.2,
        "large": 0, "large_pct": 0.0,
        "total_tokens": 72_830_868, "input_tokens": 72_171_500, "output_tokens": 659_368,
        "mean_tokens": 728_308.7, "api_calls": 5_242,
        "total_time_s": 47_926.91, "total_time_h": 13.31, "mean_time_s": 479.27,
    },
}


def add_panel_label(
    ax: plt.Axes,
    label: str,
    x_offset_pt: float = -26.0,
    y_offset_pt: float = 6.0,
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
    text_pad_ratio: float = 0.25,
) -> None:
    """Draw a horizontal step comparison bracket with a label above."""
    y_top = y_base + height
    ax.plot(
        [x1, x1, x2, x2],
        [y_base, y_top, y_top, y_base],
        color=color,
        linewidth=0.85,
        solid_capstyle="butt",
        clip_on=False,
    )
    ax.text(
        0.5 * (x1 + x2),
        y_top + height * text_pad_ratio,
        text,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        fontweight="bold",
        color=color,
    )


def build_figure(out_dir: Path) -> None:
    """Construct, align-audit, and export the 4-panel figure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    width_mm = 183.0
    height_mm = 132.0
    fig = plt.figure(
        figsize=(width_mm / 25.4, height_mm / 25.4),
        facecolor="white",
    )

    gs = fig.add_gridspec(
        2, 2,
        left=0.075,
        right=0.98,
        top=0.895,
        bottom=0.095,
        wspace=0.22,
        hspace=0.33,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # ── Shared top legend ─────────────────────────────────────────────────────
    legend_handles = [
        Patch(
            facecolor=METHOD_COLORS[m],
            edgecolor=EDGE_DARK,
            linewidth=0.7,
            label=m,
        )
        for m in METHOD_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.985),
        ncol=4,
        columnspacing=1.8,
        handlelength=1.4,
        handleheight=0.85,
        fontsize=7.2,
        frameon=False,
    )

    # ── Panel a: Overall resolution rate (n = 100) ────────────────────────────
    x_a = np.arange(len(METHOD_ORDER))
    resolved_pcts = [DATA[m]["resolved_pct"] for m in METHOD_ORDER]
    resolved_cnts = [DATA[m]["resolved"] for m in METHOD_ORDER]
    bar_w_a = 0.56

    bars_a = ax_a.bar(
        x_a,
        resolved_pcts,
        width=bar_w_a,
        color=[METHOD_COLORS[m] for m in METHOD_ORDER],
        edgecolor=EDGE_DARK,
        linewidth=0.75,
        zorder=3,
    )
    for bar, cnt, pct in zip(bars_a, resolved_cnts, resolved_pcts):
        ax_a.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.9,
            f"{cnt} ({pct:.0f}%)",
            ha="center",
            va="bottom",
            fontsize=6.2,
            color=EDGE_DARK,
        )

    add_step_bracket(ax_a, 1.0, 2.0, 34.8, 1.6, "+14.81%", DELTA_UP, fontsize=6.2)
    add_step_bracket(ax_a, 2.0, 3.0, 39.2, 1.6, "+3.23%", DELTA_UP, fontsize=6.2)

    ax_a.set_xticks(x_a)
    ax_a.set_xticklabels(METHOD_ORDER)
    ax_a.set_ylabel("总体解决率 (%, n = 100)")
    ax_a.set_yticks([0, 10, 20, 30, 40])
    ax_a.set_ylim(0, 46.0)
    add_panel_label(ax_a, "a")

    # ── Panel b: Patch-size stratification ────────────────────────────────────
    strata_labels = ["小补丁\n(n = 40)", "中补丁\n(n = 55)", "大补丁\n(n = 5)"]
    strata_keys = [("small_pct", "small"), ("medium_pct", "medium"), ("large_pct", "large")]
    x_b = np.arange(len(strata_labels))
    n_m = len(METHOD_ORDER)
    group_w = 0.74
    sub_w = group_w / n_m

    for i, m in enumerate(METHOD_ORDER):
        offset = (i - (n_m - 1) / 2.0) * sub_w
        pcts = [DATA[m][k_pct] for k_pct, _ in strata_keys]
        cnts = [DATA[m][k_cnt] for _, k_cnt in strata_keys]
        bars = ax_b.bar(
            x_b + offset,
            pcts,
            width=sub_w,
            color=METHOD_COLORS[m],
            edgecolor=EDGE_DARK,
            linewidth=0.65,
            zorder=3,
        )
        for j, (bar, cnt) in enumerate(zip(bars, cnts)):
            if j < 2:  # Small and Medium: print exact solved count above bar
                ax_b.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 1.2,
                    f"{cnt}",
                    ha="center",
                    va="bottom",
                    fontsize=5.8,
                    color=EDGE_DARK,
                )

    # Small stratum brackets (normal idx 1 -> ACI-impr idx 2 -> graph idx 3)
    x_s_norm = 0.0 + (1 - 1.5) * sub_w
    x_s_aci = 0.0 + (2 - 1.5) * sub_w
    x_s_graph = 0.0 + (3 - 1.5) * sub_w
    add_step_bracket(ax_b, x_s_norm, x_s_aci, 57.5, 2.2, "+31.25%", DELTA_UP, fontsize=5.8)
    add_step_bracket(ax_b, x_s_aci, x_s_graph, 63.8, 2.2, "+4.76%", DELTA_UP, fontsize=5.8)

    # Medium stratum brackets
    x_m_norm = 1.0 + (1 - 1.5) * sub_w
    x_m_aci = 1.0 + (2 - 1.5) * sub_w
    x_m_graph = 1.0 + (3 - 1.5) * sub_w
    add_step_bracket(ax_b, x_m_norm, x_m_aci, 25.0, 2.2, "-9.09%", DELTA_DOWN, fontsize=5.8)
    add_step_bracket(ax_b, x_m_aci, x_m_graph, 31.3, 2.2, "0.00%", DELTA_NEUTRAL, fontsize=5.8)

    # Large stratum note (all 0/5)
    ax_b.text(
        2.0,
        4.0,
        "0 / 5\n(均未解决)",
        ha="center",
        va="bottom",
        fontsize=6.0,
        color=MUTED_TEXT,
    )

    ax_b.set_xticks(x_b)
    ax_b.set_xticklabels(strata_labels)
    ax_b.set_ylabel("组内解决率 (%)")
    ax_b.set_yticks([0, 20, 40, 60])
    ax_b.set_ylim(0, 74.0)
    add_panel_label(ax_b, "b")

    # ── Panel c: Mean token cost breakdown (Input + Output) ───────────────────
    x_c = np.arange(len(METHOD_ORDER))
    input_k = np.array([DATA[m]["input_tokens"] / 100_000.0 for m in METHOD_ORDER])
    output_k = np.array([DATA[m]["output_tokens"] / 100_000.0 for m in METHOD_ORDER])
    mean_k = input_k + output_k
    bar_w_c = 0.56

    ax_c.bar(
        x_c,
        input_k,
        width=bar_w_c,
        color=[METHOD_COLORS[m] for m in METHOD_ORDER],
        edgecolor=EDGE_DARK,
        linewidth=0.75,
        zorder=3,
    )
    ax_c.bar(
        x_c,
        output_k,
        bottom=input_k,
        width=bar_w_c,
        color=EDGE_DARK,
        edgecolor=EDGE_DARK,
        linewidth=0.75,
        zorder=4,
    )

    for xi, tot_k, out_k in zip(x_c, mean_k, output_k):
        ax_c.text(
            xi,
            tot_k + 16.0,
            f"{tot_k:.1f}k\n(Out {out_k:.1f}k)",
            ha="center",
            va="bottom",
            fontsize=5.8,
            color=EDGE_DARK,
            linespacing=1.15,
        )

    add_step_bracket(ax_c, 1.0, 2.0, 445.0, 28.0, "+223.53%", DELTA_DOWN, fontsize=6.0)
    add_step_bracket(ax_c, 2.0, 3.0, 865.0, 28.0, "+139.78%", DELTA_DOWN, fontsize=6.0)

    # Compact Input vs Output legend inside Panel c top-left
    io_handles = [
        Patch(facecolor="#D8D8D8", edgecolor=EDGE_DARK, linewidth=0.6, label="Input (≥97.9%)"),
        Patch(facecolor=EDGE_DARK, edgecolor=EDGE_DARK, linewidth=0.6, label="Output (≤2.1%)"),
    ]
    ax_c.legend(
        handles=io_handles,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        fontsize=6.2,
        handlelength=1.1,
        handleheight=0.8,
        labelspacing=0.3,
        frameon=False,
    )

    ax_c.set_xticks(x_c)
    ax_c.set_xticklabels(METHOD_ORDER)
    ax_c.set_ylabel("单题平均 Token (×10³)")
    ax_c.set_yticks([0, 200, 400, 600, 800])
    ax_c.set_ylim(0, 960.0)
    add_panel_label(ax_c, "c")

    # ── Panel d: Efficiency-Performance Trade-off (Runtime, API calls, Resolved) ─
    # Trajectory connecting normal -> ACI-impr -> graph
    traj_methods = ["normal", "ACI-impr", "graph"]
    traj_x = [DATA[m]["mean_time_s"] for m in traj_methods]
    traj_y = [DATA[m]["resolved_pct"] for m in traj_methods]
    ax_d.plot(
        traj_x,
        traj_y,
        linestyle="--",
        linewidth=1.0,
        color="#767676",
        zorder=2,
    )

    # Marker area scaled strictly proportionally to API calls
    call_scale = 0.045
    for m in METHOD_ORDER:
        mx = DATA[m]["mean_time_s"]
        my = DATA[m]["resolved_pct"]
        calls = DATA[m]["api_calls"]
        ax_d.scatter(
            mx,
            my,
            s=calls * call_scale,
            color=METHOD_COLORS[m],
            edgecolor=EDGE_DARK,
            linewidth=0.9,
            zorder=4,
        )

    # Direct method + call + total-time annotations placed cleanly away from markers/path
    ax_d.text(
        DATA["swe-agent"]["mean_time_s"] + 16.0,
        DATA["swe-agent"]["resolved_pct"] - 0.1,
        f"swe-agent\n{DATA['swe-agent']['api_calls']:,} 次 · {DATA['swe-agent']['total_time_h']:.2f} h",
        ha="left",
        va="center",
        fontsize=6.0,
        color=EDGE_DARK,
        linespacing=1.15,
    )
    ax_d.text(
        DATA["normal"]["mean_time_s"] - 18.0,
        DATA["normal"]["resolved_pct"] + 2.4,
        f"normal\n{DATA['normal']['api_calls']:,} 次 · {DATA['normal']['total_time_h']:.2f} h",
        ha="left",
        va="bottom",
        fontsize=6.0,
        color=EDGE_DARK,
        linespacing=1.15,
    )
    ax_d.text(
        DATA["ACI-impr"]["mean_time_s"] - 10.0,
        DATA["ACI-impr"]["resolved_pct"] + 2.5,
        f"ACI-impr\n{DATA['ACI-impr']['api_calls']:,} 次 · {DATA['ACI-impr']['total_time_h']:.2f} h",
        ha="center",
        va="bottom",
        fontsize=6.0,
        color=EDGE_DARK,
        linespacing=1.15,
    )
    ax_d.text(
        DATA["graph"]["mean_time_s"] + 10.0,
        DATA["graph"]["resolved_pct"] - 3.3,
        f"graph\n{DATA['graph']['api_calls']:,} 次 · {DATA['graph']['total_time_h']:.2f} h",
        ha="right",
        va="top",
        fontsize=6.0,
        color=EDGE_DARK,
        linespacing=1.15,
    )

    # Segment marginal delta labels
    ax_d.text(
        262.0,
        26.5,
        "+88.48% 耗时\n(+14.81% 解决率)",
        ha="center",
        va="top",
        fontsize=5.7,
        color=MUTED_TEXT,
        linespacing=1.15,
    )
    ax_d.text(
        412.0,
        33.4,
        "+40.88% 耗时\n(+3.23% 解决率)",
        ha="center",
        va="bottom",
        fontsize=5.7,
        color=MUTED_TEXT,
        linespacing=1.15,
    )

    # Bubble area note in bottom-right corner
    ax_d.text(
        0.98,
        0.04,
        "气泡面积 ∝ API 调用总次数",
        transform=ax_d.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.8,
        color=MUTED_TEXT,
    )

    ax_d.set_xlabel("单题平均运行时间 (s)")
    ax_d.set_ylabel("总体解决率 (%)")
    ax_d.set_xticks([150, 200, 250, 300, 350, 400, 450, 500])
    ax_d.set_xlim(140.0, 515.0)
    ax_d.set_yticks([10, 15, 20, 25, 30, 35])
    ax_d.set_ylim(10.0, 39.0)
    add_panel_label(ax_d, "d")

    # ── Mandatory multi-panel alignment gate before export ────────────────────
    base = out_dir / "agent_benchmark_comparison"
    require_matplotlib_panel_alignment(
        fig,
        axes=[ax_a, ax_b, ax_c, ax_d],
        panel_ids=["a", "b", "c", "d"],
        row_groups=[["a", "b"], ["c", "d"]],
        column_groups=[["a", "c"], ["b", "d"]],
        json_out=str(base) + ".alignment.json",
        overlay_svg=str(base) + ".alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )

    fig.savefig(str(base) + ".svg")
    fig.savefig(str(base) + ".pdf")
    fig.savefig(str(base) + ".png", dpi=300)
    fig.savefig(str(base) + ".tiff", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    build_figure(Path(__file__).resolve().parent)
