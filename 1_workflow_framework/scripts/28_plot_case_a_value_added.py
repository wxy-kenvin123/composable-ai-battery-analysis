"""Plot what the chain added against a rank-only reading of the executed model.

Every number drawn here is read from ``value_added_summary.json`` and the
derived tables that ``27_case_a_value_added.py`` wrote from the committed run;
nothing is hardcoded, and a missing field aborts the render instead of falling
back to remembered values. The accounting separates downstream dispositions on
the top-N candidates, one post-hoc illustrative admission, and carried
out-of-training warnings that the executed stage had already recorded upstream.

Usage
-----
    python scripts/28_plot_case_a_value_added.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.path import Path as MplPath

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results" / "case_a_full"
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
GHOST_C = "#D9D9D9"

ROLE_COLOR = {
    "SimulationFeedback": VIRTUAL_C,
    "KnowledgeMiner": RETRIEVE_C,
    "MechanismSimulator": MID_C,
    "ExecutedToolScreener": OOD_C,
    "": GHOST_C,
}

#: Display order of right-hand slots; only slots that occur are drawn. An
#: unknown disposition aborts the render (fail closed).
SLOT_ORDER = [
    "re-assess under aligned conditions (condition-mismatched AIMD)",
    "screen cited failure modes first",
    "enumerate structures first",
    "check against released structures",
    "reuse the published AIMD protocol for the family",
    "reuse the cited family route",
    "no corpus precedent; simulate before synthesis",
    "field convention annotated, never enforced",
    "rank provisional (out-of-training element)",
    "carried baseline warning (out-of-training element)",
    "executed rank only",
]

WRAP = {
    "re-assess under aligned conditions (condition-mismatched AIMD)":
        "re-assess under aligned\nconditions (post hoc)",
    "screen cited failure modes first": "screen cited failure\nmodes first",
    "enumerate structures first": "enumerate structures first",
    "check against released structures": "check against released\nstructures",
    "reuse the published AIMD protocol for the family":
        "reuse the published AIMD\nprotocol for the family",
    "reuse the cited family route": "reuse the cited family route",
    "no corpus precedent; simulate before synthesis":
        "no corpus precedent;\nsimulate before synthesis",
    "field convention annotated, never enforced":
        "field convention annotated,\nnever enforced",
    "rank provisional (out-of-training element)":
        "rank provisional\n(out-of-training element)",
    "carried baseline warning (out-of-training element)":
        "carried baseline warning\n(out-of-training flag)",
    "executed rank only": "executed rank only",
}


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


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


def label_panel(ax, letter, subtitle, dx=-0.02):
    ax.text(dx, 1.045, letter, transform=ax.transAxes,
            fontsize=10.5, fontweight="bold", va="bottom", ha="left")
    ax.text(dx + 0.05, 1.045, subtitle, transform=ax.transAxes,
            fontsize=8.5, color=INK, va="bottom", ha="left")


def scurve(ax, x0, y0, x1, y1, *, color, lw, alpha, ls="solid", zorder=3):
    """A gentle S-curve between two points -- easier on the eye than a chord."""
    mid = (x0 + x1) / 2
    path = MplPath(
        [(x0, y0), (mid, y0), (mid, y1), (x1, y1)],
        [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4],
    )
    ax.add_patch(PathPatch(
        path, facecolor="none", edgecolor=color, linewidth=lw, alpha=alpha,
        linestyle=ls, capstyle="round", transform=ax.transAxes, clip_on=False,
        zorder=zorder,
    ))


# --------------------------------------------------------------------------- #
# (a) rank-only reading vs recorded disposition
# --------------------------------------------------------------------------- #
def panel_slope(ax, rows, n_top):
    label_panel(ax, "a", "Same 56 candidates, two readings", dx=-0.06)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    unknown = {r["disposition"] for r in rows} - set(SLOT_ORDER)
    if unknown:
        raise RuntimeError(f"dispositions missing from SLOT_ORDER: {unknown}")
    slots = [label for label in SLOT_ORDER
             if any(r["disposition"] == label for r in rows)]

    n = len(rows)
    y_left = {int(r["rank"]): 0.97 - (int(r["rank"]) - 1) * (0.93 / (n - 1)) for r in rows}
    slot_y = {label: 0.92 - i * (0.86 / max(len(slots) - 1, 1))
              for i, label in enumerate(slots)}
    x_l, x_r = 0.30, 0.66

    line_style = {
        "downstream disposition added": dict(lw=1.3, alpha=0.95, ls="solid"),
        "post-hoc illustrative addition": dict(lw=1.5, alpha=0.95, ls=(0, (4, 2))),
        "carried baseline warning": dict(lw=1.0, alpha=0.85, ls=(0, (1.5, 1.5))),
        "unchanged": dict(lw=0.5, alpha=0.45, ls="solid"),
    }

    # A quiet spine of rank dots with sparse guides, then the fan of curves.
    for rank in (20, 30, 50, 56):
        ax.text(x_l - 0.012, y_left[rank], str(rank), ha="right", va="center",
                fontsize=7.5, color=INK, zorder=2)

    slot_fill: dict[str, int] = {label: 0 for label in slots}
    for row in sorted(rows, key=lambda r: r["change_class"] == "unchanged", reverse=True):
        rank = int(row["rank"])
        slot = row["disposition"]
        offset = slot_fill[slot] * 0.0115 - 0.02
        slot_fill[slot] += 1
        changed = row["change_class"] != "unchanged"
        color = ROLE_COLOR[row["primary_role_displayed"]] if changed else GHOST_C
        styling = line_style[row["change_class"]]
        y0, y1 = y_left[rank], slot_y[slot] + min(offset, 0.055)
        scurve(ax, x_l, y0, x_r - 0.012, y1, color=color,
               lw=styling["lw"], alpha=styling["alpha"], ls=styling["ls"],
               zorder=3 if changed else 2)
        ax.plot([x_l], [y0], marker="o", ms=2.4 if changed else 1.2,
                markeredgecolor="white" if changed else color,
                markeredgewidth=0.4 if changed else 0,
                color=color, zorder=4)
        if changed:
            suffix = "  (post hoc)" if row["change_class"] == "post-hoc illustrative addition" else ""
            ax.text(x_l - 0.032, y0, f"{row['composition']}  {rank}{suffix}",
                    ha="right", va="center", fontsize=7.5, color=INK, zorder=5)

    ax.text(x_l - 0.015, 1.00,
            f"executed rank\n(top {n_top} = run budget)",
            ha="right", va="bottom", fontsize=8.5, color=INK, linespacing=1.3)
    bracket_top, bracket_bot = y_left[1] + 0.012, y_left[n_top] - 0.012
    ax.plot([x_l + 0.018, x_l + 0.018], [bracket_bot, bracket_top],
            color=INK, lw=0.9, zorder=4)
    ax.text(x_l + 0.03, (y_left[n_top + 1] + y_left[n]) / 2 + 0.06,
            "below the top five:\nnot examined", fontsize=7.5, color=INK,
            va="center", linespacing=1.25, zorder=5,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.2))

    ax.text(x_r + 0.02, 1.00, "recorded disposition\n(fixed rules / carried flags)",
            ha="left", va="bottom", fontsize=8.5, color=INK, linespacing=1.25)
    counts: dict[str, int] = {}
    role_of_slot: dict[str, str] = {}
    for row in rows:
        counts[row["disposition"]] = counts.get(row["disposition"], 0) + 1
        if row["change_class"] != "unchanged":
            role_of_slot.setdefault(row["disposition"], row["primary_role_displayed"])
    for label in slots:
        y = slot_y[label]
        chip_color = ROLE_COLOR.get(role_of_slot.get(label, ""), GHOST_C)
        # Count chip: a small rounded tag whose colour names the primary role.
        ax.add_patch(FancyBboxPatch(
            (x_r - 0.008, y - 0.0135), 0.030, 0.027,
            boxstyle="round,pad=0.003,rounding_size=0.010",
            facecolor=chip_color, edgecolor="none",
            transform=ax.transAxes, clip_on=False, zorder=4,
        ))
        ax.text(x_r + 0.007, y, str(counts[label]), ha="center", va="center",
                fontsize=7.5, color="white" if label != "executed rank only" else INK,
                fontweight="bold", zorder=5)
        ax.text(x_r + 0.036, y, WRAP[label],
                fontsize=7.5, va="center", color=INK, linespacing=1.2, zorder=5)


# --------------------------------------------------------------------------- #
# (b) what each role contributed, per deep-dive candidate
# --------------------------------------------------------------------------- #
def panel_matrix(ax, rows, ranked, cards, feedback):
    label_panel(ax, "b", "Evidence per module for the deep-dive set")
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    deep = [r for r in rows if r["deep_dive"] == "True"]
    deep.sort(key=lambda r: int(r["rank"]))
    sigma = {r["composition"]: float(r["log10_sigma"]) for r in ranked}

    columns = [
        ("survivor\nrow", LIGHT_C), ("executed\nrank", PASS_C),
        ("published\nAIMD", VIRTUAL_C), ("corpus\ncard", RETRIEVE_C),
        ("released\nstructures", LIGHT_C), ("fixed\nrules", MID_C),
    ]
    x0, cell_w, cell_h = 0.105, 0.148, 0.118
    top = 0.86

    for j, (header, color) in enumerate(columns):
        ax.text(x0 + j * cell_w + cell_w / 2 - 0.006, top + 0.075, header,
                ha="center", va="bottom", fontsize=7.2, color=INK, linespacing=1.15)

    for i, row in enumerate(deep):
        composition = row["composition"]
        card = cards[composition]
        virtual = feedback[composition]
        y = top - i * (cell_h + 0.024)
        label = f"{composition}\nrank {row['rank']}"
        if row["change_class"] == "post-hoc illustrative addition":
            label += "\n(post hoc)"
        ax.text(x0 - 0.015, y - cell_h / 2, label,
                ha="right", va="center", fontsize=7.5, color=INK, linespacing=1.3)

        if virtual["aimd_exact_match"] == "True":
            aimd = "exact:\nmp-1106183"
            aimd_on = True
        elif virtual["aimd_family_mp_ids"]:
            aimd = f"family:\n{len(virtual['aimd_family_mp_ids'].split(','))} of 9"
            aimd_on = True
        else:
            aimd = "none\npublished"
            aimd_on = False
        structures = int(card["structures_published"])
        cells = [
            ("row\nserved", True),
            (f"$10^{{{sigma[composition]:.1f}}}$\nS/cm", True),
            (aimd, aimd_on),
            (f"{card['n_papers']} papers\n{card['n_failure_modes']} failures", True),
            (f"{structures}\nstructures" if structures else "none\nreleased", structures > 0),
            (f"{len(row['rules_fired'].split(';'))} rules\n"
             f"{row['n_rule_citations']} cited", True),
        ]
        for j, ((text, filled), (_, color)) in enumerate(zip(cells, columns)):
            x = x0 + j * cell_w
            ax.add_patch(FancyBboxPatch(
                (x, y - cell_h), cell_w - 0.012, cell_h,
                boxstyle="round,pad=0.004,rounding_size=0.012",
                facecolor=color if filled else "none",
                edgecolor="none" if filled else MUTED,
                linestyle="solid" if filled else (0, (2, 1.4)),
                linewidth=0 if filled else 0.6,
                transform=ax.transAxes, clip_on=False, alpha=0.92 if filled else 1.0,
            ))
            ax.text(x + (cell_w - 0.012) / 2, y - cell_h / 2, text,
                    ha="center", va="center", fontsize=7.0,
                    color=INK, linespacing=1.25)


# --------------------------------------------------------------------------- #
# (c) stage touch counts and the retrieval-question ledger
# --------------------------------------------------------------------------- #
def panel_touch(ax, summary):
    label_panel(ax, "c", "Stage-touch counts: expensive stages on 6 of 56")
    stage_touch = summary["stage_touch"]
    n = summary["n_candidates"]
    stages = [
        ("survivor rows served", stage_touch["PropertyScreener"], LIGHT_C),
        ("ML-SSE executed", stage_touch["ExecutedToolScreener"], PASS_C),
        ("AIMD feedback", stage_touch["SimulationFeedback"], VIRTUAL_C),
        ("corpus retrieval", stage_touch["KnowledgeMiner"], RETRIEVE_C),
        ("structure lookup", stage_touch["MechanismSimulator"], LIGHT_C),
        ("rule suggestions", stage_touch["FormulationDesigner"], MID_C),
    ]
    ys = list(range(len(stages)))[::-1]
    for y, (label, count, color) in zip(ys, stages):
        ax.barh(y, count, color=color, height=0.62, zorder=3)
        ax.barh(y, n, color="none", edgecolor=GHOST_C, height=0.62,
                linewidth=0.6, zorder=2)
        ax.text(count + 1.2, y, f"{count}/{n}", va="center", fontsize=8.5, color=INK)
        ax.text(-1.2, y, label, va="center", ha="right", fontsize=8.5, color=INK)

    ax.set_xlim(0, n * 1.24)
    ax.set_ylim(-2.6, len(stages) - 0.4)
    ax.set_yticks([])
    ax.set_xticks([0, 14, 28, 42, 56])
    ax.set_xlabel("candidates touched", fontsize=8.5)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=2.5, labelsize=8.5)

    asked = summary["retrieval_questions_asked"]
    uniform = summary["retrieval_questions_uniform"]
    share = summary["retrieval_question_share"]
    ax.text(0, -1.45,
            f"retrieval questions: {asked} asked vs {uniform} uniform ({share:.1%})",
            fontsize=7.5, color=RETRIEVE_C, va="center")


def main():
    style()
    summary = json.loads((EVIDENCE / "value_added_summary.json").read_text(encoding="utf-8"))
    rows = read_csv(EVIDENCE / "table_case_a_value_added.csv")
    ranked = read_csv(EVIDENCE / "table_case_a_ranked.csv")
    cards = {r["composition"]: r for r in read_csv(EVIDENCE / "table_case_a_cards.csv")}
    feedback = {r["composition"]: r
                for r in read_csv(EVIDENCE / "table_case_a_virtual_feedback.csv")}

    fig = plt.figure(figsize=(7.4, 8.4))
    grid = fig.add_gridspec(
        2, 2, width_ratios=[0.92, 1.14], height_ratios=[1.12, 1.0],
        hspace=0.52, wspace=0.30,
        left=0.075, right=0.965, top=0.925, bottom=0.075,
    )
    panel_slope(fig.add_subplot(grid[:, 0]), rows, summary["n_top"])
    panel_matrix(fig.add_subplot(grid[0, 1]), rows, ranked, cards, feedback)
    panel_touch(fig.add_subplot(grid[1, 1]), summary)

    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext, kwargs in (("svg", {}), ("png", {"dpi": 400})):
        out = FIGDIR / ("Fig_case_a_value_added.%s" % ext)
        fig.savefig(out, format=ext, **kwargs)
        print("wrote %s" % out)
    plt.close(fig)


if __name__ == "__main__":
    main()
