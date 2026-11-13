# Missing-data timing and station rainfall trends in Nepal

Code and derived data for the manuscript

> Luitel, S., Biswakarma, K., Thapa, S., Amgain, A., Rayamajhi, D., Bhattarai, M., & Pokharel, Y. *Missing-Data Timing Alters Station Rainfall Trend Estimates in Nepal.* Submitted to *Earth and Space Science* (AGU).

The study asks how the treatment of partly observed station-years, and the timing of the missing days, change station rainfall trend estimates. It uses 219 rain gauges of Nepal's Department of Hydrology and Meteorology (DHM) for 1981–2017. Three treatments of partly observed years are compared: eligible years only, all years uncorrected, and all years with a gap-pattern correction. The archive's real gap patterns are then imposed on gauge and ERA5 reference series under random and clustered timing.

## What is in this repository

| Folder | Contents |
|---|---|
| `code/` | The analysis steps that produce every number, table and figure in the paper and its Supporting Information (Python). |
| `outputs/` | Derived results: annual indices per station-year, eligible-year flags, gap masks, station trends, paired shifts, bootstrap and masking-experiment results. None of these files contains daily rainfall values. |
| `figures/` | The manuscript figures (Figures 1–6) and Supporting Information figures (S1–S4). |
| `data/` | Notes on the input data and how to obtain them. |

### What is not included, and why

- **Daily DHM station records** (`melted_ppt_from_1980.csv`) cannot be redistributed under DHM's data policy. Researchers can request them from DHM (https://www.dhm.gov.np). All scripts start from this file.
- **The processed daily array** (`outputs/daily_qc.npz`) and the monthly totals of the neighbor screen contain DHM values and are also withheld. Step 01 rebuilds the daily array from the DHM file.
- **ERA5 station-cell series** (`outputs/era5_station_daily.npz`, 29 MB) are not stored. Steps 14a or 15/15b re-extract them from the Copernicus Climate Data Store or the public ARCO-ERA5 store.
- **Nepal boundary shapefile** used for the map in Figure 2 (`gis/Local Unit/`) comes from the Survey Department of Nepal and is not redistributed.

## Reproducing the analysis

1. Install Python 3.11 or later and the packages in `requirements.txt` (`pip install -r requirements.txt`).
2. Put the DHM daily file in the repository root as `melted_ppt_from_1980.csv` (station-by-day table of 24 h accumulations, mm; empty cells are missing).
3. Run the steps in numerical order from the repository root, for example `python code/01_qc_inventory.py`. Each script reads from and writes to `outputs/`.
4. ERA5: run `code/14a_era5_download.py` (needs a free Copernicus account) or `code/15_arco_extract.py` and `code/15b_arco_finish.py` (no account), then `code/14_era5_checks.py`.
5. Homogeneity: `code/34_rhtests_prepare.py` writes the daily series for RHtests_dlyPrcp (Wang et al., 2010), which runs in R outside this repository; `code/35_rhtests_summary.py` reads its output. `outputs/rhtests_pre2018.csv` holds the changepoints used in the paper.
6. Bootstrap and masking steps use fixed random seeds; interval endpoints can still differ in the second decimal between platforms.

## Steps and where they appear in the paper

| Step | Purpose | Paper |
|---|---|---|
| 01 | Quality control, station inventory, status of every station-day | Table 1, section 2.1 |
| 02 | Annual indices (Rx1day, Rx5day, annual wet-day total, R95pTOT) | section 2.1 |
| 03 | Station trends by completeness rule (base table read by later steps) | input |
| 06 | Station regions and the first null experiment | input |
| 09, 32 | Relative SNHT homogeneity screen | Text S4, Table S11 |
| 14, 14a, 15, 15b | ERA5 extraction; ERA5 rain on gauge-missing days; 2018–2020 climate | section 3.2, Figure S2, Text S7 |
| 16 | Fixed network and end-year sensitivity | section 3.7, Text S7 |
| 17 | Station trends, paired shifts and field significance, 1981–2017 | section 3.3, Table S10 |
| 17b, 29 | PIT diagnostics with all-year and local references | section 3.2, Figure S1 |
| 18, 26 | Comparison and replication of Karki et al. (2017) | Text S5, Table S6, Figure S3 |
| 18b | Comparison with Poudel et al. (2022) | Figure 3c, d |
| 19 | ERA5 masking experiment (real and replayed gap patterns) | section 3.5, Table 4 |
| 19b | ERA5 on recovered 2018–2020 gauge days (read by step 23; not reported in the paper) | input |
| 20 | Alternative trend tests and minimum-year rules | Table S1, Figure 5c, d |
| 21 | Trend-free null experiment | section 3.3, Text S2 |
| 22 | ERA5 versus gauge station trends | Text S5 |
| 23, 49 | Figures 3, 4 and 6 and Figures S1–S4 (49 redraws 23 with final labels) | figures |
| 24 | Day alignment between gauge dates and ERA5 | Text S1 |
| 25 | Gauge masking experiment under three timing scenarios | section 3.5, Figure 6, Table 4, Table S2 |
| 27 | Return levels under degradation | Text S6, Figure S4, Table S8 |
| 30 | Uncertainty of significant-station counts | Table S5 |
| 31 | Trends by elevation band and homogeneity result | Table S11 |
| 34, 35 | RHtests_dlyPrcp input and summary | Text S4 |
| 36, 48 | Break-aware trends; paired shift by minimum eligible years; 1981 changepoints | Table S7, Table S4, Text S4 |
| 37 | Composition-only placebo | Text S3, Table S12 |
| 38 | Correction tested on real partly observed years | section 3.4, Table 3 |
| 39 | Two-way bootstrap intervals for the paired shift | section 3.3, Table 2 |
| 40 | Clustering-scale sensitivity of the gauge experiment | section 3.5, Table 5 |
| 41 | Temporal clustering of changepoints; comparison within strata | Text S4 |
| 42 | ERA5 versus gauge daily rainfall properties | Text S3 |
| 43 | Monthly neighbor screen | Text S1 |
| 45 | Field-test block length; representativeness of the 219 stations | section 4, Text S2 |
| 47 | Common normalization, block bootstrap, correction without exclusion | section 3.3, Table S9 |
| 50, 51, 52 | Figures 5, 2 and 1 | figures |

Step numbers that are missing (04, 05, 07, 08, 10–13, 28, 33, 44, 46) belong to an earlier exploratory version of the analysis on the 1980–2023 record or to scripts that built manuscript files. They are not needed to reproduce this paper and are not included.

## Key output files

| File | Contents |
|---|---|
| `annual_indices.csv` | Annual indices per station-year with observed-day counts |
| `eligible_years.csv` | 1 if a station-year is eligible (≥95% of days observed, no month with more than three missing days) |
| `gap_intervals.csv` | Gap masks: every run of missing, suspect or out-of-operation days per station (dates only, no values) |
| `station_inventory.csv`, `station_province.csv` | Station coordinates, elevation, record span, completeness and region |
| `station_trends_pre2018.csv` | Station Sen slopes (% of median per decade) under the three treatments, 1981–2017 |
| `bootstrap_shift_twoway_pre2018.csv` | Paired shifts with two-way bootstrap intervals (Table 2) |
| `review_checks_pre2018.csv` | Common-denominator, block-bootstrap and exclusion-rule checks (Table S9) |
| `gauge_truth_experiment*.csv`, `kernel_sensitivity*.csv` | Gauge masking experiments (Figure 6, Tables 4 and 5) |
| `era5_truth_*.csv` | ERA5 masking experiments (Table 4) |
| `correction_validation_real_years.csv` | Correction tested against neighbor predictions (Table 3) |

## Relation to other work

A separate study by the first author on monsoon onset and withdrawal dates uses the same DHM network and shares only the quality-control step (step 01). Its code and results are in a separate repository; the two papers do not share results.

## Code Developer

- **Code Developer:** Somnath Luitel (Western Kentucky University, somnath.luitel416@topper.wku.edu), ORCID: [0009-0009-8669-4245](https://orcid.org/0009-0009-8669-4245)

## License and citation

- Code: MIT License (`LICENSE`) — Copyright (c) 2026 Somnath Luitel.
- Derived data in `outputs/` and figures: Creative Commons Attribution 4.0 International (`LICENSE-data`).
- ERA5 contains modified Copernicus Climate Change Service information. GHCN-Daily is distributed by NOAA NCEI.
- Please cite the archived release (DOI on the Zenodo record) and the paper. Citation metadata are in `CITATION.cff`.

Contact: Somnath Luitel, Western Kentucky University (somnath.luitel416@topper.wku.edu), ORCID [0009-0009-8669-4245](https://orcid.org/0009-0009-8669-4245).
