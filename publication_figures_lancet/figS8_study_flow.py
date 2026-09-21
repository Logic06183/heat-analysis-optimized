"""Figure S8: study flow and data harmonisation. Black and white flow diagram (Lancet trial-profile style).
Counts are those reported in the manuscript Methods, Results and table 1 footnote."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({"font.family": "Liberation Sans", "svg.fonttype": "none", "pdf.fonttype": 42})
MM = 1 / 25.4
W, H = 190 * MM, 232 * MM
fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 190); ax.set_ylim(0, 232); ax.axis("off")
INK = "#1A1A1A"
MX, MW, BH = 18, 96, 17            # main column x, width, box height
SX, SW = 124, 62                   # side column
main = [
 ("14 Johannesburg study cohorts", "Six research networks, five clinic areas, 2005 to 2022"),
 ("9,759 participants harmonised", "Enrolled across the 14 cohorts"),
 ("30 biomarkers registered in the codebook", "598,433 harmonised observations across 52 biomarker labels"),
 ("24 biomarkers assessed", "260,166 observations over 59,896 participant-visits"),
 ("9,745 participants linked to temperature\nand socioeconomic status", "ERA5-Land daily mean (about 9 km); ward-level match to the GCRO survey"),
 ("24 biomarkers screened for a temperature signal (Stage 1)", "Participant-grouped cross-validated R²"),
 ("10 biomarkers retained", "Six physiological domains; cross-validated R² of 0·05 or more"),
 ("9,293 participants grouped by heat-related pattern (Stage 3)", "Principal component analysis, then Gaussian mixture model (k=3)"),
 ("Three heat-vulnerability profiles", "Cluster 1, n=4,377; Cluster 2, n=1,247; Cluster 3, n=3,669"),
]
side = {
 2: "6 not assessed: ESR and CD8 count (too few studies); eGFR (derived from creatinine); high-sensitivity CRP (no harmonised observations); fat mass index and total fat mass (not carried into the pipeline)",
 4: "14 participants excluded: MASC records with no visit date, which could not be linked to temperature. Visits without a temperature-lag value also excluded",
 5: "14 biomarkers below the retention threshold; no metabolic or liver biomarker retained",
 7: "452 participants excluded: no measurement of any retained biomarker",
}
import textwrap
top = 226; gap = 7.6
ys = [top - BH - i * (BH + gap) for i in range(len(main))]
for i, ((title, sub), y) in enumerate(zip(main, ys)):
    ax.add_patch(FancyBboxPatch((MX, y), MW, BH, boxstyle="square,pad=0", fc="white", ec=INK, lw=0.6))
    two = "\n" in title
    ax.text(MX + MW / 2, y + BH * (0.66 if sub else 0.5) + (0 if not two else -0.6), title, ha="center", va="center", fontsize=8.5, fontweight="bold", color=INK, linespacing=1.15)
    if sub:
        ax.text(MX + MW / 2, y + BH * (0.24 if not two else 0.17), sub, ha="center", va="center", fontsize=7, color=INK)
    if i < len(main) - 1:
        ax.annotate("", xy=(MX + MW / 2, ys[i + 1] + BH), xytext=(MX + MW / 2, y), arrowprops=dict(arrowstyle="-|>", lw=0.7, color=INK, shrinkA=0, shrinkB=0, mutation_scale=8))
    if i in side:
        txt = "\n".join(textwrap.wrap(side[i], 46))
        n = txt.count("\n") + 1; sh = 4 + n * 3.4
        # exclusions branch from the arrow leaving the box above the step they apply to
        ymid = y + BH + gap / 2 if i in (4, 7) else y + BH / 2
        if i in (4, 7):
            ax.plot([MX + MW / 2, SX], [ymid, ymid], color=INK, lw=0.6)
        else:
            ax.plot([MX + MW, SX], [ymid, ymid], color=INK, lw=0.6)
        ax.add_patch(FancyBboxPatch((SX, ymid - sh / 2), SW, sh, boxstyle="square,pad=0", fc="white", ec=INK, lw=0.6))
        ax.text(SX + 2, ymid, txt, ha="left", va="center", fontsize=7, color=INK, linespacing=1.2)
for ext in ("svg", "pdf"): fig.savefig(f"figS8_study_flow.{ext}")
fig.savefig("figS8_study_flow.png", dpi=400)
print("ok")
