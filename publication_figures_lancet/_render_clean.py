"""Render a supplementary figure script as clean artwork: no header narrative, no interpretive
text panels, 'participant' wording. Usage: python3 _render_clean.py figSx_script.py"""
import pathlib, re, runpy, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.figure as mfig
import matplotlib.axes as maxes
sys.path.insert(0, ".")
import _lancet_style as ls

SCRIPT = sys.argv[1]
DROP = ([r"^\s*•"] if not any(k in SCRIPT for k in ("figS3","figS4")) else []) + [r"^Key takeaways", r"^Takeaways", r"^\s+(cluster 2|SHAP balanced|acute haemo)", r"motivates HIV",
        r"^Why PC10", r"^Stability ARI", r"k-means ARI", r"^GMM soft boundaries", r"^k = 3 is interpretable",
        r"^PC10 retains", r"^above the 0\.80", r"^failed reproducibility", r"^allow probabilistic", r"^splits into",
        r"^sufficient signal", r"^Primary methods cite", r"^Original all-component"]
def _fix(s):
    s = re.sub(r"(\d)\.(\d)", "\\1\u00b7\\2", s)
    s = re.sub(r"(\d)-(\d)", "\\1\u2013\\2", s)
    s = s.replace(" - ", " \u2013 ")
    s = re.sub(r"(^|[\s(\[=])-(\d)", "\\1\u2212\\2", s)
    return s.replace("patient-visit", "participant-visit").replace("Patient-visit", "Participant-visit").replace("n patients", "n participants")
_ft = mfig.Figure.text
def fig_text(self, x, y, s, *a, **k):
    if isinstance(s, str) and y >= 0.90 and x <= 0.05:
        return _ft(self, x, y, "", *a, **k)
    return _ft(self, x, y, _fix(s) if isinstance(s, str) else s, *a, **k)
mfig.Figure.text = fig_text
_at = maxes.Axes.text
def ax_text(self, x, y, s, *a, **k):
    if isinstance(s, str):
        if any(re.search(p, s) for p in DROP):
            return _at(self, x, y, "", *a, **k)
        if "figS7" in SCRIPT and s == "d":
            return _at(self, x, y, "", *a, **k)
        s = _fix(s)
    return _at(self, x, y, s, *a, **k)
maxes.Axes.text = ax_text
_an = maxes.Axes.annotate
def ax_annotate(self, text, *a, **k):
    return _an(self, _fix(text) if isinstance(text, str) else text, *a, **k)
maxes.Axes.annotate = ax_annotate
for name in ("set_title", "set_xlabel", "set_ylabel"):
    orig = getattr(maxes.Axes, name)
    def make(orig):
        def f(self, label, *a, **k):
            return orig(self, _fix(label) if isinstance(label, str) else label, *a, **k)
        return f
    setattr(maxes.Axes, name, make(orig))

def _plain(fig, out_dir, stem, span="double", formats=("svg", "png", "pdf"), verbose=True):
    out_dir = pathlib.Path(out_dir); paths = []
    for ext in formats:
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.05, facecolor="white")
        paths.append(p)
    print("Saved:", *[str(p) for p in paths])
    return paths
ls.save_at_width = _plain
runpy.run_path(SCRIPT, run_name="__main__")
