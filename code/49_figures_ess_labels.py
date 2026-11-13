# Code Developer: Somnath Luitel (somnath.luitel416@topper.wku.edu)
"""Step 49: redraw the step 23 figures with AGU-style labels for the ESS submission.

Runs code/23_figures_v2.py unchanged in its data and plotting, after these label-only edits:
American spelling ("neighbor"), en dashes in year and day ranges, full years on the GHCN axis,
"<=" written as a symbol, and bias axes in percentage points per decade (a difference between
normalized trends). Output goes to figures/ess_labels/ so the step 23 originals are kept.
Manuscript mapping: Figure 3 = fig8, Figure 4 = fig15, Figure 6 = fig13; Supporting Information
Figure S1 = fig2_pit, S2 = fig11, S3 = fig12, S4 = fig14.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
src = (HERE / "23_figures_v2.py").read_text(encoding="utf-8")

EDITS = [
    ('OUT, FIG = ROOT / "outputs", ROOT / "figures"',
     'OUT, FIG = ROOT / "outputs", ROOT / "figures" / "ess_labels"\nFIG.mkdir(exist_ok=True)'),
    # Fig 3b (fig8): full years
    ('ax.set_xticks(x); ax.set_xticklabels([str(y)[2:] for y in yrs]); ax.set_xlabel("Year (20xx)")',
     'ax.set_xticks(x); ax.set_xticklabels([str(y) for y in yrs], fontsize=6.8); ax.set_xlabel("Year")'),
    ('[("o_Rx1day", "Rx1day (mm)"), ("o_PRCPTOT", "Annual total (mm)")]',
     '[("o_Rx1day", "Rx1day (mm)"), ("o_PRCPTOT", "annual total (mm)")]'),
    # Fig 5c (fig2_pit): American spelling
    ('ax.set_ylabel("Station minus neighbour PIT")', 'ax.set_ylabel("Station minus neighbor PIT")'),
    # Fig 4a (fig15): en dash
    ('"June-September"', '"June\\u2013September"'),
    # Fig 7a-c (fig13): bias unit
    ('ax.set_xlabel("Trend bias (% per decade)", fontsize=7.5)',
     'ax.set_xlabel("Trend bias\\n(percentage points per decade)", fontsize=7.5)'),
    # Fig 8 (fig12): en dashes in year ranges (data keys unchanged)
    ('ax.set_ylabel("Median station trend, 1981-2017 (% per decade)")',
     'ax.set_ylabel("Median station trend, 1981\\u20132017 (% per decade)")'),
    ('ax.bar(xs, [d.loc[s, "median_pct_dec"] for s in sub], 0.28, color=c, label=lab)',
     'ax.bar(xs, [d.loc[s, "median_pct_dec"] for s in sub], 0.28, color=c, label=lab.replace("-", "\\u2013"))'),
    # Fig 10a (fig11): symbols and en dashes
    ('"Short gaps (<= 30 d)"', '"Short gaps (\\u2264 30 d)"'),
    ('"Long gaps (31-365 d)"', '"Long gaps (31\\u2013365 d)"'),
    ('"2018-19 outage, other years"', '"2018\\u201319 outage, other years"'),
]
for a, b in EDITS:
    assert src.count(a) == 1, a
    src = src.replace(a, b)

g = {"__file__": str(HERE / "23_figures_v2.py"), "__name__": "__main__"}
exec(compile(src, "23_figures_v2[ess_labels]", "exec"), g)
