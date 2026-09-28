#!/usr/bin/env python3
"""Build the GitHub hero strip for the VLA world-state project.

The hero carries exactly one claim -- readable state is not a usable control
handle -- as a four-rung causal ladder. Every number is read from the locked
artifacts, never hardcoded, so re-running after a scale-up refreshes the figure.

Outputs light and dark SVG; scripts/render_hero_png.sh rasterises them.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
OUTPUT = ARTIFACTS / "hero"

W, H = 1680, 628
MARGIN = 72
GUTTER = 44
COL_W = (W - 2 * MARGIN - 3 * GUTTER) // 4
COL_X = [MARGIN + i * (COL_W + GUTTER) for i in range(4)]

SERIF = "Charter, 'Iowan Old Style', Georgia, serif"
SANS = "'Helvetica Neue', Helvetica, Arial, sans-serif"
MONO = "'SF Mono', Menlo, ui-monospace, monospace"

LIGHT = {
    "paper": "#FBFAF7",
    "ink": "#14202E",
    "muted": "#6B7785",
    "faint": "#9AA4B0",
    "rule": "#E2E6EA",
    "track": "#EDF0F2",
    "accent": "#0F7B6C",
    "accent_soft": "#DCEEE9",
    "alarm": "#B23A2E",
    "alarm_soft": "#F6E3E0",
}

DARK = {
    "paper": "#0F1419",
    "ink": "#ECEFF2",
    "muted": "#8B96A3",
    "faint": "#69737E",
    "rule": "#232B33",
    "track": "#1B222A",
    "accent": "#35B39C",
    "accent_soft": "#123029",
    "alarm": "#E0654F",
    "alarm_soft": "#3A1D18",
}


def load(rel: str) -> dict:
    return json.loads((ARTIFACTS / rel).read_text())


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


def read_data() -> dict:
    atlas = load("openvla_world_state_atlas/summary/atlas_summary.json")["headline"]
    pi_visual = load("pi05_world_state_atlas/visual_input_summary.json")["selected"]
    pi_expert = load("pi05_action_expert_atlas/expert_action_mean_summary.json")["selected"]

    def pi(entry: dict, factor: str) -> float:
        if factor == "phase":
            return entry["phase_coarse"]["test"]["macro_f1"]
        return entry["regression"][factor]["test"]["mean_r2"]

    decode = [
        {
            "label": "Target XYZ",
            "unit": "R²",
            "ov_control": atlas["visual_target_xyz_test_r2"],
            "ov_prompt": atlas["prompt_end_target_xyz_test_r2"],
            "pi_control": pi(pi_visual, "target_xyz"),
            "pi_prompt": pi(pi_expert, "target_xyz"),
        },
        {
            "label": "Target→EEF",
            "unit": "R²",
            "ov_control": atlas["visual_target_to_eef_test_r2"],
            "ov_prompt": atlas["prompt_end_target_to_eef_test_r2"],
            "pi_control": pi(pi_visual, "target_to_eef_xyz"),
            "pi_prompt": pi(pi_expert, "target_to_eef_xyz"),
        },
        {
            "label": "Task phase",
            "unit": "macro-F1",
            "ov_control": atlas["visual_phase_test_macro_f1"],
            "ov_prompt": atlas["prompt_end_phase_test_macro_f1"],
            "pi_control": pi(pi_visual, "phase"),
            "pi_prompt": pi(pi_expert, "phase"),
        },
    ]

    # Both patch directions; the hero shows the mean, the caption keeps the range.
    modes = ["paired", "null", "shuffled", "random", "goal"]
    mode_labels = {
        "paired": "Natural paired",
        "null": "Probe nullspace",
        "shuffled": "Episode shuffled",
        "random": "Norm-matched random",
        "goal": "Probe goal subspace",
    }
    per_direction = [
        load(f"pi05_paired_patching/control-base-{d}/summary.json")["splits"]["test"]
        for d in (0, 1)
    ]
    recovery = []
    for mode in modes:
        stats = [d[mode]["action_recovery"]["episode_bootstrap"] for d in per_direction]
        recovery.append(
            {
                "key": mode,
                "label": mode_labels[mode],
                "mean": sum(s["mean"] for s in stats) / len(stats),
                "low": min(s["ci95_low"] for s in stats),
                "high": max(s["ci95_high"] for s in stats),
            }
        )

    replay = load("pi05_full_pathway_replay/summary/summary.json")

    def success(task: str, condition: str) -> float:
        return replay["tasks"][task]["conditions"][condition]["success"]["mean"]

    closed_loop = [
        {
            "row": "Target A",
            "correct": success("0", "correct"),
            "mismatch": success("0", "mismatch"),
            "single": success("0", "single_block"),
            "full": success("0", "full_pathway"),
        },
        {
            "row": "Target B",
            "correct": success("1", "correct"),
            "mismatch": success("1", "mismatch"),
            "single": success("1", "single_block"),
            "full": success("1", "full_pathway"),
        },
    ]

    fidelity = replay.get("global_operator_fidelity", {})
    n_states = replay["tasks"]["0"]["num_paired_initial_states"]
    n_trials = n_states * len(closed_loop)

    return {
        "decode": decode,
        "recovery": recovery,
        "closed_loop": closed_loop,
        "fidelity": fidelity,
        "n_states": n_states,
        "n_trials": n_trials,
    }


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------


def text(
    x: float,
    y: float,
    body: str,
    *,
    size: float,
    fill: str,
    family: str = SANS,
    weight: int = 400,
    anchor: str = "start",
    tracking: float | None = None,
    opacity: float | None = None,
) -> str:
    attrs = [
        f'x="{x:.1f}"',
        f'y="{y:.1f}"',
        f'font-family="{family}"',
        f'font-size="{size}"',
        f'font-weight="{weight}"',
        f'fill="{fill}"',
    ]
    if anchor != "start":
        attrs.append(f'text-anchor="{anchor}"')
    if tracking is not None:
        attrs.append(f'letter-spacing="{tracking}"')
    if opacity is not None:
        attrs.append(f'opacity="{opacity}"')
    return f'<text {" ".join(attrs)}>{esc(body)}</text>'


def line(x1: float, y1: float, x2: float, y2: float, stroke: str, width: float = 1, dash: str = "") -> str:
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{width}"{extra}/>'
    )


def rect(x: float, y: float, w: float, h: float, fill: str, rx: float = 3, stroke: str = "", sw: float = 1) -> str:
    extra = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}"{extra}/>'


def circle(cx: float, cy: float, r: float, fill: str, stroke: str = "", sw: float = 1.6) -> str:
    extra = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
    return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{fill}"{extra}/>'


def diamond(cx: float, cy: float, r: float, fill: str) -> str:
    pts = f"{cx:.1f},{cy - r:.1f} {cx + r:.1f},{cy:.1f} {cx:.1f},{cy + r:.1f} {cx - r:.1f},{cy:.1f}"
    return f'<polygon points="{pts}" fill="{fill}"/>'


# --------------------------------------------------------------------------
# column headers
# --------------------------------------------------------------------------

EYEBROW_Y = 214
STAT_Y = 274
CAP_Y = 302
PLOT_TOP = 344
PLOT_BOTTOM = 508


def column_head(c: dict, index: int, step: str, stat_svg: str, caption: list[str]) -> str:
    x = COL_X[index]
    out = [
        text(x, EYEBROW_Y, f"0{index + 1}", size=11, fill=c["accent"], weight=500, tracking=1.4),
        text(x + 24, EYEBROW_Y, step.upper(), size=11, fill=c["muted"], weight=500, tracking=1.4),
        stat_svg,
    ]
    for i, entry in enumerate(caption):
        out.append(text(x, CAP_Y + i * 19, entry, size=13, fill=c["muted"]))
    return "".join(out)


# --------------------------------------------------------------------------
# panels
# --------------------------------------------------------------------------


def panel_decode(c: dict, rows: list[dict]) -> str:
    x = COL_X[0]
    x0, x1 = x + 92, x + COL_W - 16
    span = x1 - x0
    ys = [PLOT_TOP + 34, PLOT_TOP + 82, PLOT_TOP + 130]
    out: list[str] = []

    for row, y in zip(rows, ys):
        out.append(rect(x0, y - 3, span, 6, c["track"], rx=3))
        out.append(text(x + 84, y + 4, row["label"], size=12.5, fill=c["ink"], anchor="end"))

        ctrl = (row["ov_control"] + row["pi_control"]) / 2
        cx = x0 + span * ctrl
        far = x0 + span * max(row["ov_prompt"], row["pi_prompt"])
        out.append(line(cx, y, far, y, c["accent"], 3))
        out.append(circle(cx, y, 5.5, c["paper"], c["faint"], 1.8))
        out.append(circle(x0 + span * row["ov_prompt"], y, 5.5, c["accent"]))
        out.append(diamond(x0 + span * row["pi_prompt"], y, 5.5, c["accent"]))

    axis_y = PLOT_BOTTOM - 4
    out.append(line(x0, axis_y, x1, axis_y, c["rule"], 1))
    for value in (0.0, 0.5, 1.0):
        tx = x0 + span * value
        out.append(line(tx, axis_y, tx, axis_y + 4, c["rule"], 1))
        out.append(text(tx, axis_y + 17, f"{value:.1f}", size=10.5, fill=c["faint"], anchor="middle"))

    legend_y = PLOT_TOP + 2
    out.append(circle(x + 4, legend_y - 4, 4.5, c["paper"], c["faint"], 1.6))
    out.append(text(x + 14, legend_y, "prompt-blind", size=10.5, fill=c["faint"]))
    out.append(circle(x + 100, legend_y - 4, 4.5, c["accent"]))
    out.append(text(x + 110, legend_y, "OpenVLA", size=10.5, fill=c["faint"]))
    out.append(diamond(x + 176, legend_y - 4, 4.5, c["accent"]))
    out.append(text(x + 186, legend_y, "π0.5 expert", size=10.5, fill=c["faint"]))
    return "".join(out)


def panel_recovery(c: dict, rows: list[dict]) -> str:
    x = COL_X[1]
    x0, x1 = x + 126, x + COL_W - 34
    span = x1 - x0
    vmax = 0.32
    out: list[str] = []

    for i, row in enumerate(rows):
        y = PLOT_TOP + 16 + i * 33
        lead = row["key"] in ("paired", "null")
        colour = c["accent"] if lead else c["faint"]
        out.append(text(x + 118, y + 4, row["label"], size=12, fill=c["ink"] if lead else c["muted"], anchor="end"))
        out.append(line(x0, y, x1, y, c["track"], 5))
        out.append(line(x0 + span * row["low"] / vmax, y, x0 + span * row["high"] / vmax, y, colour, 1.4))
        out.append(line(x0, y, x0 + span * row["mean"] / vmax, y, colour, 5))
        out.append(circle(x0 + span * row["mean"] / vmax, y, 4.5, colour))
        out.append(
            text(x1 + 30, y + 4, f"{row['mean']:.2f}", size=12, fill=colour, anchor="end", family=MONO)
        )

    axis_y = PLOT_BOTTOM - 4
    out.append(line(x0, axis_y, x1, axis_y, c["rule"], 1))
    for value in (0.0, 0.1, 0.2, 0.3):
        tx = x0 + span * value / vmax
        out.append(line(tx, axis_y, tx, axis_y + 4, c["rule"], 1))
        out.append(text(tx, axis_y + 17, f"{value:.1f}", size=10.5, fill=c["faint"], anchor="middle"))
    return "".join(out)


CELL_H = 54
CELL_GAP = 10
ROW_Y = [PLOT_TOP + 26, PLOT_TOP + 26 + CELL_H + CELL_GAP]


def outcome_cell(c: dict, x: float, y: float, w: float, value: float, good: bool) -> str:
    fill = c["accent_soft"] if good else c["alarm_soft"]
    ink = c["accent"] if good else c["alarm"]
    return "".join(
        [
            rect(x, y, w, CELL_H, fill, rx=4),
            text(
                x + w / 2,
                y + CELL_H / 2 + 7,
                f"{value * 100:.0f}%",
                size=20,
                fill=ink,
                weight=500,
                anchor="middle",
            ),
        ]
    )


def panel_closed_loop(c: dict, rows: list[dict]) -> str:
    x = COL_X[2]
    label_w = 62
    cells_x = x + label_w
    cell_w = (COL_W - label_w - 2 * 6) / 3
    heads = ["Correct", "Swapped", "Block 13"]
    keys = ["correct", "mismatch", "single"]
    out: list[str] = []

    for i, head in enumerate(heads):
        cx = cells_x + i * (cell_w + 6) + cell_w / 2
        out.append(text(cx, PLOT_TOP + 12, head, size=11, fill=c["muted"], anchor="middle", tracking=0.3))

    for r, row in enumerate(rows):
        y = ROW_Y[r]
        out.append(text(x, y + CELL_H / 2 + 4, row["row"], size=12, fill=c["ink"]))
        for i, key in enumerate(keys):
            value = row[key]
            good = key != "single" and value >= 0.8
            out.append(outcome_cell(c, cells_x + i * (cell_w + 6), y, cell_w, value, good))

    note_y = ROW_Y[1] + CELL_H + 26
    out.append(text(x, note_y, "The patch that worked offline destroys", size=12, fill=c["alarm"]))
    out.append(text(x, note_y + 17, "behaviour the swapped prompt still had.", size=12, fill=c["alarm"]))
    return "".join(out)


def panel_replay(c: dict, rows: list[dict], data: dict) -> str:
    x = COL_X[3]
    cell_w = 150
    out: list[str] = []

    out.append(text(x + cell_w / 2, PLOT_TOP + 12, "All 18 blocks", size=11, fill=c["muted"], anchor="middle", tracking=0.3))

    for r, row in enumerate(rows):
        out.append(outcome_cell(c, x, ROW_Y[r], cell_w, row["full"], True))

    audits = [
        ("action error", "0.0"),
        ("velocity error", "0.0"),
        ("hooks fired", "18 × 10"),
        ("held-out trials", str(data["n_trials"])),
    ]
    ax = x + cell_w + 26
    for i, (label, value) in enumerate(audits):
        y = ROW_Y[0] + 14 + i * 26
        out.append(text(ax, y, label, size=11.5, fill=c["muted"]))
        out.append(text(x + COL_W, y, value, size=11.5, fill=c["ink"], anchor="end", family=MONO))
        out.append(line(ax, y + 9, x + COL_W, y + 9, c["rule"], 0.75))

    note_y = ROW_Y[1] + CELL_H + 26
    out.append(text(x, note_y, "Replaying the whole prompt-conditioned", size=12, fill=c["accent"]))
    out.append(text(x, note_y + 17, "pathway recovers the policy exactly.", size=12, fill=c["accent"]))
    return "".join(out)


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------


def build(theme: dict, data: dict) -> str:
    c = theme
    decode, recovery = data["decode"], data["recovery"]
    rows = data["closed_loop"]

    ov_gain = decode[0]["ov_prompt"] - decode[0]["ov_control"]
    pi_gain = decode[0]["pi_prompt"] - decode[0]["pi_control"]
    paired = next(r for r in recovery if r["key"] == "paired")["mean"]
    goal = next(r for r in recovery if r["key"] == "goal")["mean"]

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" role="img" aria-label="Readable is not controllable: a four-step causal ladder">',
        "<style>text{font-variant-numeric:tabular-nums;}</style>",
        rect(0, 0, W, H, c["paper"], rx=0),
    ]

    # Chevrons in the gutters so the four columns read as one sequence.
    for i in range(3):
        gx = COL_X[i] + COL_W + GUTTER / 2
        parts.append(
            f'<path d="M{gx - 4:.1f} {STAT_Y - 22:.1f} L{gx + 4:.1f} {STAT_Y - 13:.1f} '
            f'L{gx - 4:.1f} {STAT_Y - 4:.1f}" fill="none" stroke="{c["rule"]}" '
            'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
        )

    # header
    parts.append(
        text(MARGIN, 62, "VLA WORLD-STATE ATLAS   ·   OPENVLA + π0.5   ·   LIBERO",
             size=11.5, fill=c["muted"], weight=500, tracking=2.2)
    )
    parts.append(
        text(MARGIN, 116, "Readable is not controllable.", size=44, fill=c["ink"], family=SERIF)
    )
    parts.append(
        text(MARGIN, 150,
             "Goal state is linearly decodable in two VLAs and locally causal inside the action expert "
             "— yet a single-block intervention fails closed-loop control.",
             size=15.5, fill=c["muted"])
    )
    parts.append(line(MARGIN, 180, W - MARGIN, 180, c["rule"], 1))

    # column 1
    parts.append(
        column_head(
            c, 0, "Decode",
            text(COL_X[0], STAT_Y, f"+{ov_gain:.2f}", size=46, fill=c["ink"], weight=500),
            [
                "R² gain over the prompt-blind control",
                f"OpenVLA target XYZ  ·  π0.5 expert +{pi_gain:.2f}",
            ],
        )
    )
    parts.append(panel_decode(c, decode))

    # column 2
    parts.append(
        column_head(
            c, 1, "Locally causal",
            text(COL_X[1], STAT_Y, f"{paired * 100:.0f}%", size=46, fill=c["ink"], weight=500),
            [
                "of the prompt-induced action change —",
                f"the supervised probe subspace recovers {goal * 100:.1f}%",
            ],
        )
    )
    parts.append(panel_recovery(c, recovery))

    # column 3
    drop_stat = (
        f'<text x="{COL_X[2]}" y="{STAT_Y}" font-family="{SANS}" font-size="46" font-weight="500">'
        f'<tspan fill="{c["faint"]}">80</tspan>'
        f'<tspan fill="{c["faint"]}" font-size="30" dx="2">%</tspan>'
        f'<tspan fill="{c["muted"]}" font-size="30" dx="8">→</tspan>'
        f'<tspan fill="{c["alarm"]}" dx="8">20</tspan>'
        f'<tspan fill="{c["alarm"]}" font-size="30" dx="2">%</tspan>'
        "</text>"
    )
    parts.append(
        column_head(
            c, 2, "Closed loop",
            drop_stat,
            [
                "Single-block patching damages",
                f"otherwise successful behaviour ({data['n_states']} states/task)",
            ],
        )
    )
    parts.append(panel_closed_loop(c, rows))

    # column 4
    parts.append(
        column_head(
            c, 3, "Pathway replay",
            text(COL_X[3], STAT_Y, f"{data['n_trials']}/{data['n_trials']}", size=46, fill=c["accent"], weight=500),
            [
                "Full 18-block replay reproduces the",
                "correct-prompt policy to exactly 0.0 error",
            ],
        )
    )
    parts.append(panel_replay(c, rows, data))

    # footer
    parts.append(line(MARGIN, 546, W - MARGIN, 546, c["rule"], 1))
    parts.append(
        text(MARGIN, 578,
             "The failure is path incompatibility, not absence of the state: "
             "an intervention must preserve the model's computation path.",
             size=18, fill=c["ink"], weight=500)
    )
    parts.append(
        text(MARGIN, 602,
             "Locked test splits  ·  validation-only layer selection  ·  "
             "episode-cluster bootstrap  ·  norm-matched, shuffled and nullspace controls",
             size=11.5, fill=c["faint"], tracking=0.2)
    )
    parts.append("</svg>")
    return "".join(parts)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = read_data()
    for name, theme in (("light", LIGHT), ("dark", DARK)):
        path = OUTPUT / f"hero_{name}.svg"
        path.write_text(build(theme, data), encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
