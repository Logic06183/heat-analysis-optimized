"""Figure S2-DAG - Directed acyclic graph for the temperature → biomarker analysis.

Scientific purpose:
    Section S2 of the supplementary appendix describes the DAG textually but
    no visual exists. This figure renders it so reviewers can see the assumed
    causal structure at a glance:

      - Ambient temperature (ERA5-Land, lags 0-30 d) is the exposure.
      - Biomarker level is the outcome.
      - Measured confounders (age, sex, calendar time, study source,
        HIV status, ward-level SES) affect both.
      - Mediators (e.g. dehydration → creatinine) are NOT blocked - SHAP
        therefore captures total effects.
      - Unmeasured confounders (diet, pollution, activity, adherence)
        are flagged as residual-confounding hazards.

    No fitting is involved; this is a purely diagrammatic figure.

Style:
    Matches the rest of the Parker 2026 supplement (Lancet RH-Africa target):
    Arial-family stack, MM = 1/25.4 in, organ-colour palette via _lancet_style.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

from _lancet_style import MM, apply_style
from _lancet_style import ARTWORK_MODE, midline_tick_labels, md  # house style (2026-09-04)
from _lancet_style import save_at_width

# Artwork runs write elsewhere so submission files do not overwrite the
# working previews. Set RP2_FIG_OUT to redirect.
OUT_DIR = Path(os.environ.get("RP2_FIG_OUT",
                              str(Path(__file__).resolve().parent)))


# Layout on a 100(w) × 100(h) canvas, organised as three horizontal bands:
#   - measured confounders in a shaded panel across the TOP
#   - exposure -> outcome across the MIDDLE
#   - unmeasured confounders in a distinct panel across the BOTTOM
# Arrows then fan cleanly from each band toward the middle instead of
# crossing through a central tangle.
NODES = {
    # (label, x, y, kind)
    # -- middle band: the causal question --
    "exposure":  ("Ambient temperature\n(ERA5-Land, lag 0-30 d)", 17, 50, "exposure"),
    "outcome":   ("Biomarker level\n(13 retained, 7 systems)",    83, 50, "outcome"),
    # -- top band: measured confounders (adjusted) --
    "age":       ("Age",                          11, 80, "confounder"),
    "sex":       ("Sex",                          27, 80, "confounder"),
    "month":     ("Calendar time\n(month, year)", 44, 80, "confounder"),
    "study":     ("Study source\n(14 JHB cohorts)", 60, 80, "confounder"),
    "hiv":       ("HIV status",                   76, 80, "confounder"),
    "ses":       ("Ward-level SES\n(GCRO IES)",   91, 80, "confounder"),
    # -- bottom band: unmeasured confounders (residual) --
    "u_diet":    ("Seasonal diet",        18, 18, "unmeasured"),
    "u_poll":    ("Air pollution",        39, 18, "unmeasured"),
    "u_activ":   ("Physical activity",    61, 18, "unmeasured"),
    "u_adher":   ("Medication\nadherence", 82, 18, "unmeasured"),
}

# Shaded band panels drawn behind everything (x0, x1, y0, y1, kind, label).
PANELS = [
    (4, 97, 70, 91, "measured",
     "MEASURED CONFOUNDERS  ·  adjusted in all models"),
    (4, 97, 8, 27, "unmeasured",
     "UNMEASURED  ·  potential residual confounding (Section S2)"),
]

# Directed edges. (source, target, kind)
EDGES = [
    # Total effect of interest
    ("exposure", "outcome",   "primary"),
    # Confounders → exposure
    ("age",      "exposure", "confound"),
    ("sex",      "exposure", "confound"),
    ("month",    "exposure", "confound"),
    ("study",    "exposure", "confound"),
    ("hiv",      "exposure", "confound"),
    ("ses",      "exposure", "confound"),
    # Confounders → outcome
    ("age",      "outcome", "confound"),
    ("sex",      "outcome", "confound"),
    ("month",    "outcome", "confound"),
    ("study",    "outcome", "confound"),
    ("hiv",      "outcome", "confound"),
    ("ses",      "outcome", "confound"),
    # Unmeasured → exposure & outcome (dashed)
    ("u_diet",   "exposure", "unmeasured"),
    ("u_diet",   "outcome",  "unmeasured"),
    ("u_poll",   "exposure", "unmeasured"),
    ("u_poll",   "outcome",  "unmeasured"),
    ("u_activ",  "exposure", "unmeasured"),
    ("u_activ",  "outcome",  "unmeasured"),
    ("u_adher",  "exposure", "unmeasured"),
    ("u_adher",  "outcome",  "unmeasured"),
]

# Visual mapping per node kind. Exposure/outcome carry weight (bold fills);
# confounders are quieter tinted boxes; unmeasured are open/dashed to read as
# "hypothesised, not in the model".
NODE_STYLE = {
    "exposure":   dict(fc="#E76F51", ec="#A33D27", lw=1.6, fontw="bold",   fc_text="white",   ls="-"),
    "outcome":    dict(fc="#2A9D8F", ec="#1F7268", lw=1.6, fontw="bold",   fc_text="white",   ls="-"),
    "confounder": dict(fc="#DCE6F1", ec="#4E79A7", lw=1.1, fontw="normal", fc_text="#2C3E50", ls="-"),
    "unmeasured": dict(fc="#FFFFFF", ec="#9AA0A6", lw=0.9, fontw="normal", fc_text="#6B7075", ls=(0, (4, 2))),
}

# Band-panel backgrounds.
PANEL_STYLE = {
    "measured":   dict(fc="#F2F6FB", ec="#C9D8EC", label_c="#4E79A7"),
    "unmeasured": dict(fc="#F7F7F7", ec="#DEDEDE", label_c="#9AA0A6"),
}

EDGE_STYLE = {
    "primary":    dict(color="#A33D27", lw=2.6, ls="-",   alpha=1.0,  arrowsize=18),
    "confound":   dict(color="#7FA0C4", lw=0.7, ls="-",   alpha=0.45, arrowsize=7),
    "unmeasured": dict(color="#B0B4B8", lw=0.7, ls="--",  alpha=0.50, arrowsize=7),
}


def draw_node(ax, label, x, y, kind):
    style = NODE_STYLE[kind]
    n_lines = label.count("\n") + 1
    w = max(10, min(22, max(len(ln) for ln in label.split("\n")) * 0.92))
    h = 5 + 3.2 * (n_lines - 1)
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.6,rounding_size=1.4",
        fc=style["fc"], ec=style["ec"], lw=style["lw"],
        linestyle=style.get("ls", "-"), zorder=3,
    )
    ax.add_patch(box)
    ax.text(
        x, y, label,
        ha="center", va="center",
        fontsize=6.8, color=style["fc_text"],
        fontweight=style["fontw"], zorder=4, linespacing=1.15,
    )
    return (x, y, w, h)


def edge_endpoint(node_geom, dx, dy):
    """Step the arrow start/end inward to the node boundary."""
    x, y, w, h = node_geom
    # Approximate: step to ellipse fit around the box
    import math
    angle = math.atan2(dy, dx)
    rx, ry = w / 2 + 0.5, h / 2 + 0.5
    ex = x + rx * math.cos(angle)
    ey = y + ry * math.sin(angle)
    return ex, ey


def draw_edge(ax, geom_a, geom_b, kind, rad=0.0):
    style = EDGE_STYLE[kind]
    ax_x, ax_y = geom_a[0], geom_a[1]
    bx_x, bx_y = geom_b[0], geom_b[1]
    dx, dy = bx_x - ax_x, bx_y - ax_y
    src = edge_endpoint(geom_a, dx, dy)
    dst = edge_endpoint(geom_b, -dx, -dy)
    arrow = FancyArrowPatch(
        src, dst,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>,head_width=0.32,head_length=0.5",
        mutation_scale=style["arrowsize"],
        color=style["color"], lw=style["lw"], linestyle=style["ls"],
        alpha=style["alpha"], zorder=2,
        shrinkA=0, shrinkB=0, capstyle="round",
    )
    ax.add_patch(arrow)


def _curved_rad(geom_a, geom_b, kind):
    """Give confounder/unmeasured edges a gentle, consistent curvature so the
    fan from each band reads as a sheaf of arcs rather than a crossing tangle.
    Sign is chosen from the source's horizontal offset relative to its target
    so arcs bow away from the centre line."""
    if kind == "primary":
        return 0.0
    ax_x = geom_a[0]
    tx = geom_b[0]
    base = 0.16
    # Bow left-side sources one way, right-side sources the other.
    return base if ax_x < tx else -base


def build_figure():
    apply_style(base_font_size=8.0)
    from matplotlib.patches import FancyBboxPatch as _FBP
    fig_w_in = 190 * MM
    fig_h_in = 108 * MM
    fig = plt.figure(figsize=(fig_w_in, fig_h_in))

    # Title block
    if not ARTWORK_MODE:
        fig.text(
            0.03, 0.972,
            "Figure S2-DAG  |  Adjustment-set diagram for the temperature-biomarker analysis",
            fontsize=10.0, fontweight="bold", ha="left", va="top",
        )
    fig.text(
        0.03, 0.930,
        "Identifying assumption: conditioning on the six measured confounders (top) blocks the back-door paths. "
        "Unmeasured\nconfounders (bottom, dashed) remain open and are discussed as residual-confounding hazards in Section S2.",
        fontsize=7.0, color="#555555", ha="left", va="top", linespacing=1.35,
    )

    ax = fig.add_axes([0.02, 0.02, 0.96, 0.85])
    ax.set_xlim(0, 100)
    ax.set_ylim(2, 100)
    ax.set_aspect("auto")
    ax.axis("off")

    # --- Band panels behind everything ---
    for x0, x1, y0, y1, kind, label in PANELS:
        ps = PANEL_STYLE[kind]
        panel = _FBP(
            (x0, y0), x1 - x0, y1 - y0,
            boxstyle="round,pad=0.4,rounding_size=2.0",
            fc=ps["fc"], ec=ps["ec"], lw=0.8, zorder=0,
        )
        ax.add_patch(panel)
        ax.text(x0 + 1.5, y1 - 1.4, label, ha="left", va="top",
                fontsize=6.3, color=ps["label_c"], fontweight="bold",
                zorder=1, style="italic")

    # Draw nodes first so we can compute boundary points
    geoms = {}
    for nid, (label, x, y, kind) in NODES.items():
        geoms[nid] = draw_node(ax, label, x, y, kind)

    # Edges: confounders/unmeasured behind boxes, primary drawn last on top.
    for src, dst, kind in EDGES:
        if kind == "primary":
            continue
        draw_edge(ax, geoms[src], geoms[dst], kind,
                  rad=_curved_rad(geoms[src], geoms[dst], kind))
    for src, dst, kind in EDGES:
        if kind == "primary":
            draw_edge(ax, geoms[src], geoms[dst], kind, rad=0.0)

    # Legend (bottom, own row under the diagram)
    legend_items = [
        Line2D([0], [0], color="#A33D27", lw=2.6, label="Primary effect (total, estimand)"),
        Line2D([0], [0], color="#7FA0C4", lw=1.1, label="Confounding path, adjusted"),
        Line2D([0], [0], color="#B0B4B8", lw=1.0, linestyle="--",
               label="Unmeasured confounding, residual"),
    ]
    ax.legend(
        handles=legend_items, loc="upper center", ncol=3, fontsize=6.8,
        frameon=False, bbox_to_anchor=(0.50, 0.055), handlelength=2.2,
        columnspacing=2.4, handletextpad=0.7,
    )

    svg_path = OUT_DIR / "figS_DAG.svg"
    png_path = OUT_DIR / "figS_DAG.png"
    midline_tick_labels(fig)
    save_at_width(fig, OUT_DIR, "figS_DAG", span="double")
    print(f"Saved: {svg_path}")
    print(f"Saved: {png_path}")
    plt.close(fig)
    return svg_path, png_path


if __name__ == "__main__":
    build_figure()
