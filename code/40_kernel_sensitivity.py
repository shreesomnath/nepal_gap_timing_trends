# Code Developer: Somnath Luitel (somnath.luitel416@topper.wku.edu)
"""Step 40: sensitivity of the gauge-truth station-timing result to the clustering kernel.

Runs the step 25 experiment unchanged except for the scenario set: station_timing with kernel scale 1, 2, 4 and 8 years
(weight exp(-|t - p| / scale) around the station's own partly observed years p) and late_decade (the same station-level degradation
probability q, concentrated in 2008-2017: q scaled so the expected number of degraded years per station is unchanged).
Fewer replicates (default 20). Outputs: outputs/kernel_sensitivity_reps.csv, outputs/kernel_sensitivity.csv
Run: python -I code/40_kernel_sensitivity.py 20
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / "code" / "25_gauge_truth_experiment.py").read_text(encoding="utf-8")
src = src.replace("__file__", repr(str(ROOT / "code" / "25_gauge_truth_experiment.py")))
src = src.replace('for s_ in ("random_timing", "real_timing", "station_timing")',
                  'for s_ in ("kernel_1", "kernel_2", "kernel_4", "kernel_8", "late_decade")')
old = """        if scen == "random_timing":
            qq = q
        elif scen == "real_timing":
            qq = np.minimum(1.0, q * W[ey])
        else:                                              # station_timing: gaps cluster around this station's own partial years
            Kt = np.array([np.exp(-np.abs(t - ptime) / 2.0).sum() for t in ey]) if len(ptime) else np.zeros(len(ey))
            qq = np.minimum(1.0, q * Kt / Kt.mean()) if Kt.mean() > 0 else np.zeros(len(ey))"""
new = """        if scen == "late_decade":
            late = (Y[ey] >= 2008).astype(float)
            qq = np.minimum(1.0, q * late / late.mean()) if late.mean() > 0 else np.zeros(len(ey))
        else:
            sc_ = float(scen.split("_")[1])
            Kt = np.array([np.exp(-np.abs(t - ptime) / sc_).sum() for t in ey]) if len(ptime) else np.zeros(len(ey))
            qq = np.minimum(1.0, q * Kt / Kt.mean()) if Kt.mean() > 0 else np.zeros(len(ey))"""
assert old in src
src = src.replace(old, new)
src = src.replace("gauge_truth_experiment_reps.csv", "kernel_sensitivity_reps.csv").replace("gauge_truth_experiment.csv", "kernel_sensitivity.csv")
src = src.replace("gauge_truth_omitted_share.csv", "kernel_sensitivity_omitted_share.csv")
if len(sys.argv) < 2:
    sys.argv.append("20")
exec(compile(src, "step25_variant", "exec"), {"__name__": "__main__"})
