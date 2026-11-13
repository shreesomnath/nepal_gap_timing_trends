# Code Developer: Somnath Luitel (somnath.luitel416@topper.wku.edu)
"""Step 23: figures 8-12 and S1 for the 1981-2017 framing.

Fig 8   evidence that the 2018-2020 gap is in the archive and that archive values are original
Fig 9   how the treatment of partly observed years changes station-trend significance (1981-2017)
Fig 10  ERA5 as a gap-free truth: bias of the three treatments, real gap patterns and replayed outage
Fig 11  ERA5 diagnostics: missing vs observed days, 2018-2020 climate, ERA5 skill on recovered days
Fig 12  regional pattern with Karki et al. (2017) longitude zones and sub-regions
Fig S1  ERA5 vs gauge station trends
Every plotted number comes from a CSV in outputs/ (figure-source tables are saved as figures/*_source.csv).
Style follows step 11: Okabe-Ito palette, 180 mm width, thin marks, direct labels, grey for context.
GHCN-Daily day counts (Kathmandu Airport, Dhankuta) were computed from NOAA by-station files in step
18's notes and are stored in outputs/ghcn_days_per_year.csv.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT, FIG = ROOT / "outputs", ROOT / "figures"
BLUE, ORANGE, GREEN, PURPLE, VERM, GREY, SKY = "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#8C8C8C", "#56B4E9"
INK, LIGHT = "#222222", "#D9D9D9"
MM = 1 / 25.4
plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 8, "axes.labelsize": 8.5,
                     "axes.titlesize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
                     "axes.edgecolor": INK, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "pdf.fonttype": 42})
TREAT = {"R1": ("Eligible years only", GREY), "R4": ("All years, uncorrected", VERM), "R4corr": ("All years, corrected", BLUE)}


def clean(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=LIGHT, lw=0.5)
    ax.set_axisbelow(True)


def tag(ax, t):
    ax.text(-0.02, 1.05, t, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="right")


def save(fig, name, src=None):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)
    if src is not None:
        for k, df in src.items():
            df.to_csv(FIG / f"{name}_{k}_source.csv", index=False)


IDX = ["Rx1day", "Rx5day", "PRCPTOT", "R95pTOT"]
LAB = {"Rx1day": "Rx1day", "Rx5day": "Rx5day", "PRCPTOT": "Annual total", "R95pTOT": "R95pTOT"}

# ======================================================================= Fig 8
g = np.load(OUT / "daily_qc.npz", allow_pickle=True)
status = g["status"]; gd = pd.to_datetime(g["dates"]); sid = g["station_id"].astype(str)
obs = status == 0
months = pd.period_range("2015-01", "2021-12", freq="M")
cnt = []
for p in months:
    m = (gd.to_period("M") == p)
    cnt.append(int((obs[:, m].mean(1) >= 0.9).sum()))
mon = pd.DataFrame({"month": months.astype(str), "stations_ge90pct_days": cnt})
yrs = list(range(2014, 2024))
ghcn = pd.read_csv(OUT / "ghcn_days_per_year.csv")
pair = [list(sid).index("1030"), list(sid).index("1307")]      # archive stations matched to Kathmandu Airport and Dhankuta
arch = [int(obs[pair, :][:, gd.year == y].sum()) for y in yrs]
rec = pd.DataFrame({"year": yrs, "archive_days_2_stations": arch})
pm = pd.read_csv(OUT / "poudel_station_means_compare.csv", dtype={"station_id": str})

fig, axs = plt.subplots(2, 2, figsize=(180 * MM, 150 * MM))
ax = axs[0, 0]; clean(ax)
ax.plot(range(len(cnt)), cnt, color=BLUE, lw=1.4)
ax.set_xticks([k for k, p in enumerate(months) if p.month == 1])
ax.set_xticklabels([str(p.year) for p in months if p.month == 1])
ax.set_ylabel("Stations with at least 90% of days observed")
i17 = list(months.astype(str)).index("2017-12"); i20 = list(months.astype(str)).index("2020-12")
ax.annotate("Dec 2017", (i17, cnt[i17]), (i17 - 12, 120), fontsize=7, arrowprops=dict(arrowstyle="-", lw=0.5, color=GREY))
ax.annotate("Dec 2020", (i20, cnt[i20]), (i20 - 4, 60), fontsize=7, arrowprops=dict(arrowstyle="-", lw=0.5, color=GREY))
ax.set_ylim(0, 275); tag(ax, "a")
ax = axs[0, 1]; clean(ax)
w = 0.27; x = np.arange(len(yrs))
ax.bar(x - w / 2, rec.archive_days_2_stations, w, color=BLUE, label="DHM archive (2 stations)")
ax.bar(x + w / 2, ghcn.ktm_days + ghcn.dhankuta_days, w, color=GREEN, label="NOAA GHCN-Daily (same 2 stations)")
ax.set_xticks(x); ax.set_xticklabels([str(y)[2:] for y in yrs]); ax.set_xlabel("Year (20xx)")
ax.set_ylabel("Station-days with a value"); ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.02))
ax.set_ylim(0, 1000); ax.set_yticks(range(0, 1001, 200)); tag(ax, "b")
for k, (col, lab) in enumerate([("o_Rx1day", "Rx1day (mm)"), ("o_PRCPTOT", "Annual total (mm)")]):
    ax = axs[1, k]; clean(ax, grid=None)
    pcol = col.replace("o_", "p_")
    ax.scatter(pm[pcol], pm[col], s=12, color=BLUE, lw=0)
    lim = [0, max(pm[pcol].max(), pm[col].max()) * 1.05]
    ax.plot(lim, lim, color=GREY, lw=0.7)
    ax.set_xlim(lim); ax.set_ylim(lim)
    r = np.corrcoef(pm[pcol], pm[col])[0, 1]
    ax.text(0.04, 0.93, f"30 stations, r = {r:.3f}", transform=ax.transAxes, fontsize=7.5)
    ax.set_xlabel(f"Poudel et al. (2022), mean {lab}"); ax.set_ylabel(f"This archive, mean {lab}")
    tag(ax, "cd"[k])
fig.tight_layout()
save(fig, "fig8_archive_gap_evidence", {"a": mon, "b": rec, "ghcn": ghcn, "poudel": pm})

# ======================================================================= Fig 9
A = pd.read_csv(OUT / "sens_tests_pre2018.csv")
Bt = pd.read_csv(OUT / "sens_threshold_pre2018.csv")
BS = pd.read_csv(OUT / "bootstrap_shift_pre2018.csv")
fig, axs = plt.subplots(2, 2, figsize=(180 * MM, 150 * MM))
ax = axs[0, 0]; clean(ax)
mk = A[A.test == "MK"]
w = 0.26
for j, t in enumerate(["R1", "R4", "R4corr"]):
    v = [int(mk[(mk["index"] == n) & (mk.treatment == t)].fdr_declines.iloc[0]) for n in IDX]
    xs = np.arange(4) + (j - 1) * w
    ax.bar(xs, v, w * 0.92, color=TREAT[t][1], label=TREAT[t][0])
    for xx, vv in zip(xs, v):
        ax.text(xx, vv + 0.8, str(vv), ha="center", fontsize=6.5)
ax.set_xticks(range(4)); ax.set_xticklabels([LAB[n] for n in IDX])
ax.set_ylabel("Stations with an FDR-significant decline"); ax.legend(frameon=False, loc="upper left")
ax.set_title("219 stations, 1981-2017, Mann-Kendall", fontsize=8, loc="left"); tag(ax, "a")
ax = axs[0, 1]; clean(ax, grid="x")
rows = []
for k, n in enumerate(IDX):
    for j, (q, c, lab) in enumerate([("R4-R1", VERM, "uncorrected - eligible only"), ("Rcorr-R1", BLUE, "corrected - eligible only")]):
        r = BS[(BS["index"] == n) & (BS.quantity == q)].iloc[0]
        y = k + (0.15 if j == 0 else -0.15)
        ax.plot([r.ci_lo, r.ci_hi], [y, y], color=c, lw=1.2); ax.plot(r.point, y, "o", color=c, ms=4.5, mec="white", mew=0.6)
        rows.append((n, lab, r.point, r.ci_lo, r.ci_hi))
ax.axvline(0, color=INK, lw=0.6)
ax.set_yticks(range(4)); ax.set_yticklabels([LAB[n] for n in IDX]); ax.invert_yaxis()
ax.set_xlabel("Paired change in station trend (% per decade)")
ax.plot([], [], "o", color=VERM, label="Uncorrected - eligible only"); ax.plot([], [], "o", color=BLUE, label="Corrected - eligible only")
ax.legend(frameon=False, loc="upper left", fontsize=7); tag(ax, "b")
ax = axs[1, 0]; clean(ax)
pt = A[A["index"] == "PRCPTOT"]
w = 0.26
for j, t in enumerate(["R1", "R4", "R4corr"]):
    v = [int(pt[(pt.test == k) & (pt.treatment == t)].fdr_declines.iloc[0]) for k in ("MK", "MMK", "TFPW")]
    xs = np.arange(3) + (j - 1) * w
    ax.bar(xs, v, w * 0.92, color=TREAT[t][1])
    for xx, vv in zip(xs, v):
        ax.text(xx, vv + 0.8, str(vv), ha="center", fontsize=6.5)
ax.set_xticks(range(3)); ax.set_xticklabels(["Mann-Kendall", "Hamed-Rao\nmodified", "Trend-free\npre-whitened"])
ax.set_ylabel("Annual total: FDR-significant declines"); tag(ax, "c")
ax = axs[1, 1]; clean(ax)
ths = [22, 26, 30]
for n, ls, lab in [("PRCPTOT", "-", "Annual total"), ("Rx1day", "--", "Rx1day")]:
    for t in ["R1", "R4", "R4corr"]:
        v = [int(Bt[(Bt["index"] == n) & (Bt.variant == f"threshold>={th}") & (Bt.treatment == t)].fdr_declines.iloc[0]) for th in ths]
        ax.plot(ths, v, ls, color=TREAT[t][1], lw=1.3, marker="o", ms=3.5)
ax.set_xticks(ths); ax.set_xlabel("Minimum eligible years (of 37)"); ax.set_ylabel("FDR-significant declines")
ax.plot([], [], "-", color=GREY, label="Annual total (solid)"); ax.plot([], [], "--", color=GREY, label="Rx1day (dashed)")
ax.legend(frameon=False, loc="upper right", fontsize=7); tag(ax, "d")
fig.tight_layout()
save(fig, "fig9_partial_year_effect_1981_2017", {"a": mk, "b": pd.DataFrame(rows, columns=["index", "comparison", "point", "ci_lo", "ci_hi"]), "c": pt, "d": Bt})

# ======================================================================= Fig 10
fig, axs = plt.subplots(1, 2, figsize=(180 * MM, 70 * MM), sharey=True)
src = {}
for k, sc in enumerate(["real", "replay"]):
    d = pd.read_csv(OUT / f"era5_truth_{sc}.csv"); src[sc] = d
    ax = axs[k]; clean(ax, grid="x")
    for a_, n in enumerate(IDX):
        for j, t in enumerate(["R1", "R4", "R4corr"]):
            r = d[(d["index"] == n) & (d.treatment == t)].iloc[0]
            y = a_ + (j - 1) * 0.24
            ax.plot([r.ci_lo, r.ci_hi], [y, y], color=TREAT[t][1], lw=1.2)
            ax.plot(r.mean_bias_pct_dec, y, "o", color=TREAT[t][1], ms=4.2, mec="white", mew=0.5)
    ax.axvline(0, color=INK, lw=0.6)
    ax.set_yticks(range(4)); ax.set_yticklabels([LAB[n] for n in IDX]); ax.invert_yaxis()
    ax.set_xlabel("Bias of station trend relative to undegraded reference (% per decade)")
    ax.set_title(["Each year's real gap pattern", "2018-2020 outage replayed on 2008-2010"][k], fontsize=8, loc="left"); tag(ax, "ab"[k])
hs = [axs[1].plot([], [], "o", color=TREAT[t][1], label=TREAT[t][0])[0] for t in ["R1", "R4", "R4corr"]]
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.legend(handles=hs, frameon=False, loc="lower center", ncol=3, fontsize=7.5)
save(fig, "fig10_era5_truth_experiment", src)

# ======================================================================= Fig 11
TA = pd.read_csv(OUT / "era5_testA_missingness.csv")
TB = pd.read_csv(OUT / "era5_testB_nepal_annual.csv")
RG = pd.read_csv(OUT / "era5_vs_recovered_gauges.csv")
fig, axs = plt.subplots(1, 2, figsize=(180 * MM, 66 * MM), gridspec_kw={"width_ratios": [1.15, 1.2]})
ax = axs[0]; clean(ax, grid="x")
sel = [("short", "same_year", "Short gaps (<= 30 d)"), ("long", "same_year", "Long gaps (31-365 d)"),
       ("outage", "other_years", "2018-19 outage, other years"), ("control_kept", "other_years", "Kept stations, other years"),
       ("outage_DiD", "other_years", "Outage relative to kept")]
ta = TA[TA.region == "All"]
rows = []
for k, (gt, ref, lab) in enumerate(sel):
    r = ta[(ta.gap_type == gt) & (ta.reference == ref)].iloc[0]
    c = VERM if gt == "outage_DiD" else BLUE
    ax.plot([r.ratio_mean_era5_lo, r.ratio_mean_era5_hi], [k, k], color=c, lw=1.3)
    ax.plot(r.ratio_mean_era5, k, "o", color=c, ms=4.5, mec="white", mew=0.5)
    rows.append((lab, r.ratio_mean_era5, r.ratio_mean_era5_lo, r.ratio_mean_era5_hi))
ax.axvline(1, color=INK, lw=0.6)
ax.set_yticks(range(len(sel))); ax.set_yticklabels([s[2] for s in sel], fontsize=7); ax.invert_yaxis()
ax.set_xlabel("ERA5 rain on gauge-missing days\nrelative to observed days"); tag(ax, "a")
ax = axs[1]; clean(ax)
tb = TB[(TB.series == "Nepal_area_mean") & (TB["index"] == "annual_total")].sort_values("year")
col = [VERM if y else GREY for y in tb.in_2018_2020]
ax.bar(tb.year, tb.z_1981_2017, color=col, width=0.8)
ax.axhline(0, color=INK, lw=0.6)
ax.set_xlabel("Year"); ax.set_ylabel("ERA5 annual total, z-score")
y20 = tb[tb.year == 2020].iloc[0]
ax.annotate("2020: rank 2 of 43", (2020, y20.z_1981_2017), (1988, 2.5), fontsize=7, arrowprops=dict(arrowstyle="-", lw=0.5, color=GREY))
tag(ax, "b")
fig.tight_layout()
save(fig, "fig11_era5_diagnostics", {"a": pd.DataFrame(rows, columns=["comparison", "ratio", "lo", "hi"]), "b": tb})

# ======================================================================= Fig 12
T = pd.read_csv(OUT / "era5_vs_gauge_trends.csv", dtype={"station_id": str})
KR = pd.read_csv(OUT / "karki_regional_compare.csv")
rng = np.random.default_rng(23)
fig, axs = plt.subplots(1, 2, figsize=(180 * MM, 70 * MM), gridspec_kw={"width_ratios": [1.0, 1.25]})
ax = axs[0]; clean(ax)
zc = {"West": GREEN, "Central": ORANGE, "East": BLUE}
rows = []
for a_, n in enumerate(["Rx1day", "PRCPTOT", "R95pTOT"]):
    for j, z in enumerate(["West", "Central", "East"]):
        v = T[T.zone == z]["gauge_" + n].dropna().values
        bs = [np.median(rng.choice(v, len(v))) for _ in range(2000)]
        lo, hi = np.percentile(bs, [2.5, 97.5]); m = np.median(v)
        x = a_ + (j - 1) * 0.24
        ax.plot([x, x], [lo, hi], color=zc[z], lw=1.2); ax.plot(x, m, "o", color=zc[z], ms=4.5, mec="white", mew=0.5)
        rows.append((n, z, len(v), m, lo, hi))
ax.axhline(0, color=INK, lw=0.6)
ax.set_xticks(range(3)); ax.set_xticklabels(["Rx1day", "Annual total", "R95pTOT"])
ax.set_ylabel("Median station trend, 1981-2017 (% per decade)")
for z in zc:
    ax.plot([], [], "o", color=zc[z], label=f"{z} ({int((T.zone == z).sum())})")
ax.legend(frameon=False, loc="lower left", fontsize=7); tag(ax, "a")
ax = axs[1]; clean(ax)
sub = ["WL", "WM", "CL", "CM", "CH", "EL", "EM"]
exp = {"WM": "+", "CL": "+", "CH": "+", "CM": "-", "EM": "-"}
kr = KR[(KR["index"] == "Rx1day") & (KR.region.isin(sub))]
for j, (per, c, lab) in enumerate([("1981-2012", SKY, "1981-2012"), ("1981-2017", BLUE, "1981-2017")]):
    d = kr[kr.period == per].set_index("region")
    xs = np.arange(len(sub)) + (j - 0.5) * 0.3
    ax.bar(xs, [d.loc[s, "median_pct_dec"] for s in sub], 0.28, color=c, label=lab)
ax.axhline(0, color=INK, lw=0.6)
n_ = kr[kr.period == "1981-2012"].set_index("region").n_karki_stations
ax.set_xticks(range(len(sub))); ax.set_xticklabels([f"{s}\n({int(n_[s])})" for s in sub], fontsize=7)
ax.set_ylim(-8, 7.0)   # Karki's reported signs are compared like-for-like in outputs/karki_replication.csv, not drawn here
ax.set_ylabel("Median Rx1day trend, Karki's 76 stations\n(% per decade)"); ax.legend(frameon=False, loc="lower left", fontsize=7); tag(ax, "b")
fig.tight_layout()
save(fig, "fig12_regional_pattern", {"a": pd.DataFrame(rows, columns=["index", "zone", "n", "median", "lo", "hi"]), "b": kr})

# ======================================================================= Fig S1
fig, axs = plt.subplots(1, 2, figsize=(180 * MM, 70 * MM))
for k, (n, lab) in enumerate([("Rx1day", "Rx1day"), ("PRCPTOT", "Annual total")]):
    ax = axs[k]; clean(ax, grid=None)
    ok = T[["gauge_" + n, "era5_" + n]].dropna()
    ax.scatter(ok["era5_" + n], ok["gauge_" + n], s=9, color=BLUE, lw=0, alpha=0.8)
    ax.axhline(0, color=GREY, lw=0.5); ax.axvline(0, color=GREY, lw=0.5)
    from scipy import stats
    rho = stats.spearmanr(ok["era5_" + n], ok["gauge_" + n])[0]
    ax.text(0.04, 0.92, f"Spearman rho = {rho:.2f}, n = {len(ok)}", transform=ax.transAxes, fontsize=7.5)
    ax.set_xlabel(f"ERA5 {lab} trend (% per decade)"); ax.set_ylabel(f"Gauge {lab} trend (% per decade)"); tag(ax, "ab"[k])
fig.tight_layout()
save(fig, "figS1_era5_vs_gauge_trends", {"a": T})
# ======================================================================= Fig 13
GT = pd.read_csv(OUT / "gauge_truth_experiment.csv")
SCEN = [("random_timing", "Gaps in random years"), ("real_timing", "Gaps follow the archive's\nyear-by-year frequency"),
        ("station_timing", "Gaps cluster around each\nstation's own partial years")]
fig = plt.figure(figsize=(180 * MM, 130 * MM))
gs = fig.add_gridspec(2, 6, height_ratios=[1.05, 1])
for k, (sc, ttl) in enumerate(SCEN):
    ax = fig.add_subplot(gs[0, 2 * k:2 * k + 2], sharey=None if k == 0 else axp)
    if k == 0:
        axp = ax
    clean(ax, grid="x")
    for a_, n in enumerate(IDX):
        for j, t in enumerate(["R1", "R4", "R4corr"]):
            r = GT[(GT.scenario == sc) & (GT["index"] == n) & (GT.treatment == t)].iloc[0]
            y = a_ + (j - 1) * 0.24
            ax.plot([r.bias_lo, r.bias_hi], [y, y], color=TREAT[t][1], lw=1.1)
            ax.plot(r.bias_mean, y, "o", color=TREAT[t][1], ms=3.8, mec="white", mew=0.4)
    ax.axvline(0, color=INK, lw=0.6)
    ax.set_title(ttl, fontsize=7.3, loc="left")
    ax.set_xlabel("Trend bias (% per decade)", fontsize=7.5)
    if k == 0:
        ax.set_yticks(range(4)); ax.set_yticklabels([LAB[n] for n in IDX]); ax.invert_yaxis()
    else:
        plt.setp(ax.get_yticklabels(), visible=False)
    tag(ax, "abc"[k])
real = pd.read_csv(OUT / "sens_tests_pre2018.csv"); real = real[real.test == "MK"]
src13 = []
for k, (n, ttl) in enumerate([("PRCPTOT", "Annual total"), ("Rx1day", "Rx1day")]):
    ax = fig.add_subplot(gs[1, 3 * k:3 * k + 3]); clean(ax)
    groups = [(sc, lab.split("\n")[0]) for sc, lab in SCEN] + [("real", "Real archive")]
    w = 0.26
    for j, t in enumerate(["R1", "R4", "R4corr"]):
        v = []
        for sc, _ in groups:
            if sc == "real":
                v.append(int(real[(real["index"] == n) & (real.treatment == t)].fdr_declines.iloc[0]))
            else:
                v.append(float(GT[(GT.scenario == sc) & (GT["index"] == n) & (GT.treatment == t)].fdr_declines.iloc[0]))
        xs = np.arange(len(groups)) + (j - 1) * w
        ax.bar(xs, v, w * 0.92, color=TREAT[t][1])
        for xx, vv in zip(xs, v):
            ax.text(xx, vv + 0.5, f"{vv:.0f}", ha="center", fontsize=6.2)
        src13.append(pd.DataFrame({"index": n, "treatment": t, "group": [g[0] for g in groups], "fdr_declines": v}))
    truth = float(GT[(GT.scenario == "random_timing") & (GT["index"] == n) & (GT.treatment == "truth")].fdr_declines.iloc[0])
    ax.axhline(truth, color=INK, lw=0.7, ls="--"); ax.text(len(groups) - 0.45, truth + 0.5, "no gaps", fontsize=6.5, ha="right")
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(["Random\nyears", "Archive\nfrequency", "Station\nclusters", "Real\narchive"], fontsize=6.8)
    ax.set_ylabel(f"{ttl}: FDR-significant\ndeclines", fontsize=7.5); tag(ax, "de"[k])
hs = [fig.axes[0].plot([], [], "o", color=TREAT[t][1], label=TREAT[t][0])[0] for t in ["R1", "R4", "R4corr"]]
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.legend(handles=hs, frameon=False, loc="lower center", ncol=3, fontsize=7.5)
save(fig, "fig13_gauge_truth_experiment", {"bias": GT, "fdr": pd.concat(src13)})

# ======================================================================= Fig 2 (PIT, 1981-2017)
PT = pd.read_csv(OUT / "partial_year_pit_pre2018.csv")
fig, axs = plt.subplots(1, 3, figsize=(180 * MM, 60 * MM), gridspec_kw={"width_ratios": [1, 1, 0.8]})
for k, (c, lab) in enumerate([("pit_rx1day", "Rx1day"), ("pit_prcptot", "Annual total")]):
    ax = axs[k]; clean(ax)
    ax.hist(PT[c], bins=10, range=(0, 1), color=BLUE, edgecolor="white", lw=0.6)
    ax.axhline(len(PT) / 10, color=VERM, ls="--", lw=1.0)
    ax.set_xlabel("Rank of observed value (PIT)"); ax.set_title(lab, fontsize=8, loc="left")
    if k == 0:
        ax.set_ylabel("Partly observed station-years"); ax.text(0.98, 0.95, "dashed: uniform", transform=ax.transAxes, ha="right", va="top", fontsize=6.8, color=VERM)
    tag(ax, "ab"[k])
ax = axs[2]; clean(ax)
hn = PT[PT.n_neighbours >= 2]
dd = [hn.pit_rx1day - hn.nb_pit_rx1day, hn.pit_prcptot - hn.nb_pit_prcptot]
bp = ax.boxplot(dd, widths=0.5, showfliers=False, patch_artist=True, medianprops=dict(color=INK, lw=1.2))
for b in bp["boxes"]:
    b.set(facecolor="#C6DBEF", edgecolor=INK, lw=0.6)
ax.axhline(0, color=VERM, ls="--", lw=1.0)
ax.set_xticklabels(["Rx1day", "Annual\ntotal"]); ax.set_ylabel("Station minus neighbour PIT"); ax.set_title(f"n = {len(hn)}", fontsize=8, loc="left"); tag(ax, "c")
fig.tight_layout()
save(fig, "fig2_pit_1981_2017", {"a": PT})

# ======================================================================= Fig 15 (when the gaps fall)
elg = g["eligible"]; yrs_all = g["years"]; lonv = g["lon"].astype(float)
m_i = [list(yrs_all).index(y) for y in range(1981, 2018)]
selst = elg[:, m_i].sum(1) >= 26
Yr = np.arange(1981, 2018)
dy_ = gd.year.to_numpy(); dm_ = gd.month.to_numpy()
miss = np.zeros((12, len(Yr)))
for jj, y in enumerate(Yr):
    for mth in range(1, 13):
        cols = (dy_ == y) & (dm_ == mth)
        s_ = status[selst][:, cols]
        op = (s_ != 2).sum()
        miss[mth - 1, jj] = (s_ == 1).sum() / op if op else np.nan
zone_ = np.where(lonv < 83, "West", np.where(lonv < 86, "Central", "East"))
share = {z: [] for z in ("West", "Central", "East")}
for jj, y in enumerate(Yr):
    j = list(yrs_all).index(y)
    for z in share:
        ii = np.where(selst & (zone_ == z))[0]
        opn = [i for i in ii if (status[i, dy_ == y] != 2).any() and (status[i, dy_ == y] == 0).any()]
        part = [i for i in opn if not elg[i, j]]
        share[z].append(len(part) / max(len(opn), 1))
fig, axs = plt.subplots(1, 2, figsize=(180 * MM, 68 * MM), gridspec_kw={"width_ratios": [1.25, 1]})
ax = axs[0]
im = ax.imshow(miss * 100, aspect="auto", cmap="Blues", origin="lower", extent=[Yr[0] - 0.5, Yr[-1] + 0.5, 0.5, 12.5], vmin=0, vmax=15)
ax.set_yticks(range(1, 13)); ax.set_yticklabels(list("JFMAMJJASOND"))
ax.set_xlabel("Year"); ax.set_ylabel("Month")
cb = fig.colorbar(im, ax=ax, pad=0.02); cb.set_label("Operating station-days missing (%)", fontsize=7)
ax.axhspan(5.5, 9.5, fill=False, edgecolor=VERM, lw=0.9)
ax.text(1981, 9.9, "June-September", color=VERM, fontsize=6.8, va="bottom"); tag(ax, "a")
ax = axs[1]; clean(ax)
for z in ("West", "Central", "East"):
    ax.plot(Yr, np.array(share[z]) * 100, color=zc[z], lw=1.4, label=z)
ax.set_xlabel("Year"); ax.set_ylabel("Operating stations partly observed (%)")
ax.legend(frameon=False, loc="upper left", fontsize=7); tag(ax, "b")
fig.tight_layout()
save(fig, "fig15_gap_timing", {"a": pd.DataFrame(miss.T * 100, index=Yr, columns=[f"m{m}" for m in range(1, 13)]).reset_index().rename(columns={"index": "year"}),
                              "b": pd.DataFrame({"year": Yr, **{z: np.array(v) * 100 for z, v in share.items()}})})

# ======================================================================= Fig 14
GV = pd.read_csv(OUT / "gev_experiment_pre2018.csv")
M = {"strict": ("Eligible years\nonly", GREY), "naive": ("All years,\nuncorrected", VERM), "hyades": ("HYADES\nrule", BLUE),
     "sn_raw": ("Likelihood,\nf = days observed", GREEN), "sn_season": ("Likelihood,\nseasonal f", PURPLE)}
fig, axs = plt.subplots(1, 2, figsize=(180 * MM, 66 * MM), sharey=True)
rows = []
for k, col in enumerate(["err10_pct", "err20_pct"]):
    ax = axs[k]; clean(ax)
    for j, m in enumerate(M):
        e_ = GV[GV.method == m][col].dropna().values
        rm = np.sqrt(np.mean(e_ ** 2)); mb = e_.mean()
        ax.bar(j, rm, 0.62, color=M[m][1], alpha=0.35)
        ax.plot(j, mb, "D", color=M[m][1], ms=5, mec="white", mew=0.5)
        ax.text(j, rm + 0.12, f"{rm:.1f}", ha="center", fontsize=6.5)
        rows.append((col, m, rm, mb, len(e_)))
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xticks(range(len(M))); ax.set_xticklabels([v[0] for v in M.values()], fontsize=6.3)
    ax.set_title(["10-year", "20-year"][k] + " return level", fontsize=8, loc="left"); tag(ax, "ab"[k])
axs[0].set_ylabel("RMSE (bars) and mean bias (diamonds), %")
fig.tight_layout()
save(fig, "fig14_return_levels_1981_2017", {"a": pd.DataFrame(rows, columns=["horizon", "method", "rmse_pct", "mean_bias_pct", "n"])})
print("figures written")
