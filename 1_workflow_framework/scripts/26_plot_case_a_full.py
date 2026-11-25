"""
Plot the full Case A chain: replay -> execute -> retrieve -> suggest.

Reads only exported artifacts: the full-chain record and tables from
``results/case_a_full``, plus the frozen funnel from the original Case A replay
in ``results/workflow_run_evidence``. Nothing is recomputed here, and nothing
that the chain refused to do -- no gate, no accept/reject verdict -- is drawn as
if it had happened.

Usage
-----
    python scripts/26_plot_case_a_full.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results" / "case_a_full"
REPLAY = ROOT / "results" / "workflow_run_evidence"
FIGDIR = ROOT / "figures"

INK = "#2B2B2B"
MUTED = "#8A8A8A"
FAIL_C = "#B04A1A"
PASS_C = "#1F5F54"
MID_C = "#4E8C7E"
LIGHT_C = "#93BEB4"
PALE_C = "#D7E6E2"
OOD_C = "#C47A2C"
RETRIEVE_C = "#3F6E9A"
VIRTUAL_C = "#8A6FAE"


def read_csv(path: Path):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "svg.fonttype": "none",
    })


def label_panel(ax, letter, subtitle, dx=-0.075):
    ax.text(dx, 1.06, letter, transform=ax.transAxes,
            fontsize=10.5, fontweight="bold", va="bottom", ha="left")
    ax.text(dx + 0.045, 1.06, subtitle, transform=ax.transAxes,
            fontsize=8.5, color=INK, va="bottom", ha="left")


def _luminance(hex_color):
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _box(ax, x, y, w, h, text, color, *, hollow=False):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.008,rounding_size=0.05",
        facecolor="none" if hollow else color,
        edgecolor=color if hollow else "none",
        linestyle=(0, (2.5, 1.6)) if hollow else "solid",
        linewidth=0.9 if hollow else 0,
        transform=ax.transAxes, clip_on=False,
    )
    ax.add_patch(patch)
    text_color = INK if (hollow or _luminance(color) > 0.45) else "#FFFFFF"
    ax.text(x + w / 2, y + h / 2, text, transform=ax.transAxes,
            ha="center", va="center", fontsize=7.5,
            color=text_color, linespacing=1.3)


def panel_chain(ax, record, coverage):
    label_panel(ax, "a",
                "All six module interfaces in one run; feedback is "
                "simulation-grade, not measured")
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    n_deep = len(record["deep_dive"])
    width, step = 0.152, 0.1686
    stages = [
        ("HT Predictor\nlookup\n56 published rows", LIGHT_C),
        ("HT Predictor\nexecuted\nML-SSE model", PASS_C),
        ("Experiment Feedback\nsimulation-grade\npublished AIMD served", VIRTUAL_C),
        ("Knowledge Miner\nretrieved\n723-paper corpus", RETRIEVE_C),
        ("Mechanism Simulator\nlookup\nreleased structures", LIGHT_C),
        ("Materials Designer\nrule-based\ncited next steps", MID_C),
    ]
    for index, (text, color) in enumerate(stages):
        _box(ax, 0.005 + index * step, 0.40, width, 0.34, text, color)

    edge_labels = [
        f"{record['n_shortlist']} handoffs\ncomposition → formula",
        f"{n_deep} handoffs\nformula + flags",
        f"{n_deep} handoffs\nretrieval target",
        f"{n_deep} handoffs\nfamily + citations",
        f"{n_deep} handoffs\nstructure availability",
    ]
    for index, text in enumerate(edge_labels):
        x = 0.005 + width + index * step
        ax.annotate("", xy=(x + 0.0145, 0.57), xytext=(x + 0.0015, 0.57),
                    xycoords=ax.transAxes, textcoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))
        ax.text(x + 0.008, 0.78, text, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=7.5, color=INK, linespacing=1.25)

    ax.text(0.005, 0.02,
            "Process Coordinator: %s field handoffs · %s executed-model steps · "
            "%s retrieval steps · %s simulation-feedback steps"
            % (record["field_handoffs"], record["executed_steps"],
               record["retrieval_steps"], record["feedback_steps"]),
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=7.5, color=INK)
    ax.text(0.005, 0.14,
            f"{record['gate_steps']} gates applied · 0 thresholds invented · "
            "no measurement used: the feedback stage serves a published simulation",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=7.5, color=FAIL_C)
    _ = coverage


def panel_funnel(ax, funnel, record):
    label_panel(ax, "b", "Published funnel (replayed)")
    stages = ["input"] + [row["stage"] for row in funnel]
    counts = [int(funnel[0]["entering"])] + [int(row["survived"]) for row in funnel]
    colors = [PALE_C, LIGHT_C, MID_C, PASS_C]

    ys = list(range(len(counts)))[::-1]
    ax.barh(ys, counts, color=colors, height=0.62, edgecolor="none", zorder=3)
    for y, count, stage in zip(ys, counts, stages):
        ax.text(count + 8, y, str(count), va="center", ha="left",
                fontsize=8.5, color=INK)
        ax.text(-8, y, stage, va="center", ha="right", fontsize=8.5, color=INK)
    ax.set_xlim(0, max(counts) * 1.18)
    ax.set_ylim(-0.7, len(counts) - 0.3)
    ax.set_yticks([])
    ax.set_xlabel("compositions")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=2.5)


def panel_ranks(ax, ranked, convention, n_cards):
    label_panel(ax, "c", "Executed ML-SSE prediction, ranked")
    values = [float(row["log10_sigma"]) for row in ranked]
    carded = [row["literature_card"] == "True" for row in ranked]
    aimd = [row.get("aimd_exact_match") == "True" for row in ranked]
    ood = [bool(row.get("ood_elements")) for row in ranked]
    colors = [
        VIRTUAL_C if matched else (RETRIEVE_C if card else (OOD_C if flag else PASS_C))
        for matched, card, flag in zip(aimd, carded, ood)
    ]
    xs = list(range(1, len(values) + 1))
    ax.bar(xs, values, color=colors, width=0.86, edgecolor="none", zorder=3)
    ax.axhline(convention, color=FAIL_C, lw=0.9, ls="--", zorder=4)
    ax.set_xlim(0.4, len(values) + 0.6)
    ax.set_ylim(min(values) - 0.25, max(values) + 0.25)
    ax.set_xlabel("rank among the %d survivors" % len(values))
    ax.set_ylabel(r"executed $\log_{10}(\sigma/\mathrm{S\,cm^{-1}})$")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=2.5)

    # Bars hang from zero, so the free space is the bottom-left corner. The
    # LiAlS2 mismatch is explained in the SI caption; the legend only names it.
    for offset, (color, text) in enumerate((
        (RETRIEVE_C, "top %d: literature card retrieved" % n_cards),
        (VIRTUAL_C, "AIMD match: LiAlS$_2$, rank 40 (post-hoc)"),
        (OOD_C, "%d with an element outside ML-SSE training" % sum(ood)),
        (PASS_C, "executed only"),
    )):
        y = 0.360 - offset * 0.082
        ax.add_patch(plt.Rectangle((0.04, y), 0.030, 0.040, facecolor=color,
                                   edgecolor="none", transform=ax.transAxes,
                                   clip_on=False, zorder=5))
        ax.text(0.085, y + 0.020, text, transform=ax.transAxes, fontsize=7.5,
                color=INK, va="center", zorder=5)
    ax.plot([0.04, 0.07], [0.050, 0.050], transform=ax.transAxes, color=FAIL_C,
            lw=0.9, ls="--", clip_on=False, zorder=5)
    ax.text(0.085, 0.050, "1e-4 S/cm field convention (never enforced)",
            transform=ax.transAxes, fontsize=7.5, color=FAIL_C, va="center", zorder=5)


def panel_cards(ax, cards, record):
    label_panel(ax, "d", "Retrieved literature per shortlisted composition")
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    n = len(cards)
    top = 0.90
    row_h = min(0.155, 0.76 / max(n, 1))
    max_papers = max((int(c["n_papers"]) for c in cards), default=1)

    ax.text(0.00, top + 0.055, "composition", fontsize=7.5, color=INK)
    ax.text(0.30, top + 0.055, "papers retrieved / family term / cited",
            fontsize=7.5, color=INK)
    ax.text(0.84, top + 0.055, "rules fired", fontsize=7.5, color=INK)

    for index, card in enumerate(cards):
        y = top - index * row_h
        papers = int(card["n_papers"])
        family_term = int(card["n_family_term_papers"])
        cited = int(card["n_cited"])
        ax.text(0.00, y, card["composition"], fontsize=8.5, color=INK, va="center")
        ax.text(0.00, y - 0.058, card["family"].split(" (")[0],
                fontsize=7.5, color=INK, va="center")

        scale = 0.42 / max_papers
        ax.barh(y, papers * scale, left=0.30, height=0.052,
                color=PALE_C, edgecolor="none")
        ax.barh(y, family_term * scale, left=0.30, height=0.052,
                color=LIGHT_C, edgecolor="none")
        ax.barh(y, cited * scale, left=0.30, height=0.052,
                color=RETRIEVE_C, edgecolor="none")
        ax.text(0.30 + papers * scale + 0.012, y,
                "%d / %d / %d" % (papers, family_term, cited),
                fontsize=7.5, color=INK, va="center")
        ax.text(0.84, y, "%s of %d" % (card["n_rules_fired"], len(record["suggestion_rules"])),
                fontsize=8.5, color=INK, va="center")

    legend_y = top - n * row_h - 0.02
    for offset, (color, text) in enumerate((
        (PALE_C, "above relevance floor"),
        (LIGHT_C, "family term in text"),
        (RETRIEVE_C, "cited by a card claim"),
    )):
        ax.add_patch(plt.Rectangle((0.30 + offset * 0.235, legend_y), 0.022, 0.028,
                                   facecolor=color, edgecolor="none",
                                   transform=ax.transAxes, clip_on=False))
        ax.text(0.33 + offset * 0.235, legend_y + 0.014, text,
                fontsize=7.5, color=INK, va="center")


def main():
    style()
    record = read_json(EVIDENCE / "run_record_case_a_full.json")
    ranked = read_csv(EVIDENCE / "table_case_a_ranked.csv")
    ranked.sort(key=lambda row: int(row["rank"]))
    cards = read_csv(EVIDENCE / "table_case_a_cards.csv")
    cards.sort(key=lambda row: int(row["rank"]))
    coverage = read_csv(EVIDENCE / "table_case_a_agent_coverage.csv")
    funnel = read_csv(REPLAY / "table_htvs_funnel.csv")

    fig = plt.figure(figsize=(7.4, 8.0))
    grid = fig.add_gridspec(
        3, 2, height_ratios=[0.80, 1.0, 1.0],
        hspace=0.62, wspace=0.30,
        left=0.13, right=0.975, top=0.925, bottom=0.065,
    )
    panel_chain(fig.add_subplot(grid[0, :]), record, coverage)
    panel_funnel(fig.add_subplot(grid[1, 0]), funnel, record)
    panel_ranks(fig.add_subplot(grid[1, 1]), ranked,
                float(record["convention_log10_S_cm"]), record["n_top"])
    panel_cards(fig.add_subplot(grid[2, :]), cards, record)

    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext, kwargs in (("svg", {}), ("png", {"dpi": 400})):
        out = FIGDIR / ("Fig_case_a_full.%s" % ext)
        fig.savefig(out, format=ext, **kwargs)
        print("wrote %s" % out)
    plt.close(fig)


if __name__ == "__main__":
    main()
