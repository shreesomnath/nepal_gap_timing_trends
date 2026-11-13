# Input data

| Input | Source | In this repository |
|---|---|---|
| Daily precipitation, 277 DHM stations, 1980–2023 (`melted_ppt_from_1980.csv`, placed in the repository root) | Department of Hydrology and Meteorology, Government of Nepal, https://www.dhm.gov.np, on request | No (DHM data policy) |
| Two later DHM station files used in step 19b (`Precipitation_1980 to 2024.xlsx - 24h accumulated Precipitation.csv`, `Precipitation_2020-2024.xlsx - 24h accumulated Precipitation.csv`) | DHM, on request | No |
| ERA5 hourly total precipitation | Copernicus Climate Data Store (step 14a) or ARCO-ERA5 on Google Cloud (steps 15, 15b) | No (re-extracted by the scripts) |
| GHCN-Daily, Kathmandu Airport and Dhankuta | NOAA NCEI | Day counts only (`outputs/ghcn_days_per_year.csv`) |
| Station means of Poudel et al. (2022), Table 4 | Published paper | Comparison table (`outputs/poudel_station_means_compare.csv`) |
| Station list of Karki et al. (2017), Table 1 | Published paper | Match table (`outputs/karki_station_check.csv`) |
| Nepal local-unit boundaries (map in Figure 2) | Survey Department of Nepal | No |

The DHM file is a station-by-day table of 24 h accumulations in millimeters. Recorded zeros are observed dry days and empty cells are missing; step 01 keeps the two apart.
