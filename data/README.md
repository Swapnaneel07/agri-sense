# AgriSense AI, Module 4: Irrigation Datasets (Documentation)

Group 9 | Neural Network and Deep Learning | Role A handover to B and C
Purpose: time-series data to compare LSTM, GRU and BiLSTM for irrigation / soil-moisture forecasting.

This file describes **three datasets**, the **uniform attribute names** used for all of them, and where each one came from.

| Short name | Full name | Location | Rows | Interval | Period | Uniform file |
|---|---|---|---|---|---|---|
| `topraq` | TOPRAQ open-access drip-irrigation data | Turkey | 157,225 | 5 min | 2023-01-01 to 2024-06-26 | `topraq_uniform.csv` |
| `arnesano` | Multi-sector precision-irrigation field trial | Arnesano, Italy | 47,068 (5 sectors) | 10 min | 2025-02-11 to 2025-09-23 | `arnesano_uniform.csv` |
| `edge_iot` | Edge-IoT smart home-garden sensor data | not stated | 960 | 30 min | 2026-04-01 to 2026-04-20 | `edge_iot_uniform.csv` |

---

## 1. Recommendation (read this first)

1. **Main dataset: `topraq`.** It is the largest, almost complete (at most 2.1% missing in any weather column, none in soil columns), and only has 3 continuous blocks, so sliding windows work without special handling. Four soil depths are included.
2. **Second dataset: `arnesano`.** It is richer (5 sectors, 4 crops, measured irrigation water, ERA5 weather) but **heavily fragmented**: 4,746 continuous segments (all 5 sectors) with a median length of 3 rows. Only 40 segments are at least 168 rows (28 h) long. Build windows **inside `segment_id` only** (see Section 6), never across it.
3. **Debug only: `edge_iot`.** 960 rows is too small for the final comparison. Use it for quick code tests on Day 1.

A heads-up on `topraq`: soil moisture barely varies (mean 33.2, std 1.7, range 26.6 to 39.3). Report RMSE next to a naive "same as last step" baseline, otherwise every model will look good.

---

## 2. Uniform attribute names

All files use `snake_case`. The same concept has the same name and unit in every file. Columns a dataset does not have are simply absent.

| Uniform name | Meaning | Unit | topraq | arnesano | edge_iot |
|---|---|---|:-:|:-:|:-:|
| `timestamp` | Date and time (local time, no time zone) | datetime | yes | yes | yes |
| `sector_id` | Irrigation sector / plot | integer | 1 | 1-5 | 1 |
| `segment_id` | Id of a continuous, gap-free run of rows (made by us). In `arnesano_uniform.csv` it restarts for each sector, so always group by (`sector_id`, `segment_id`) | integer | yes | yes | yes |
| `soil_moisture` | Soil moisture (**main target**) | % | yes (mean of 4 depths) | yes | yes |
| `soil_moisture_20cm`, `_40cm`, `_60cm`, `_80cm` | Soil moisture per depth | % | yes | no | no |
| `soil_temperature` | Soil temperature | °C | yes (mean of 4 depths) | yes (0-7 cm) | no |
| `soil_temperature_20cm`, `_40cm`, `_60cm`, `_80cm` | Soil temperature per depth | °C | yes | no | no |
| `soil_temperature_deep` | Soil temperature, deeper ERA5 band | °C | no | yes | no |
| `air_temperature` | Air temperature | °C | yes | yes | yes |
| `relative_humidity` | Air relative humidity | % | yes | yes | yes |
| `rainfall` | Rain in the interval | mm | yes | yes | no |
| `air_pressure` | Surface pressure | hPa | no | yes | no |
| `wind_speed` | Wind speed | m/s | yes | yes | no |
| `solar_radiation` | Solar radiation | W/m² | yes | yes | no |
| `light_intensity` | Light sensor reading (unit not stated by the source) | raw | no | no | yes |
| `et0` | Reference evapotranspiration (same value all day) | mm/day | yes | no | no |
| `gdd` | Growing degree days, accumulated over the year | °C·day | yes | no | no |
| `soil_ec` | Soil electrical conductivity | µS/cm | no | yes | no |
| `soil_ph` | Soil pH | pH | no | yes | no |
| `irrigation_on` | 1 if irrigation / pump is on, else 0 | 0/1 | yes | yes | yes |
| `irrigation_depth_mm` | Water applied as a depth per interval | mm | yes | no | no |
| `irrigation_duration_min` | Minutes the valve was open in the 10-min bin | min | no | yes | no |
| `irrigation_volume_l` | Water applied in the bin | litres | no | yes | no |

**Columns common to all three datasets** (use these if one model must run on every dataset):
`timestamp`, `soil_moisture`, `air_temperature`, `relative_humidity`, `irrigation_on`.

Suggested target: `soil_moisture` at the next step (or 24 h ahead). Suggested inputs: the common columns plus whatever extra weather columns the chosen dataset has.

---

## 3. Dataset 1: `topraq` (main)

**What it is:** Drip-irrigated field in Turkey. Weather, four soil-moisture and soil-temperature probes at different depths, and the amount of irrigation water, recorded roughly every 5 minutes for about 18 months.

**Source:** TOPRAQ open-access data, https://topraq.ai/?p=41325. The data description file inside the zip says the timestamps are in Turkey local time. It does not name authors or a license. **Please read the license on the TOPRAQ page and cite it in the report.**

**Original files:** `data.csv` (no header row) and `data description.docx`. The two `open_access_data` zips you uploaded hold the same data (one also contains a nested copy of the other).

**Setup (from the data description):** drippers every 90 cm, 3 L/hour each. Probes at 20, 40, 60 and 80 cm root depth.

| Original column | Uniform name | Notes |
|---|---|---|
| `c_tstamp` | `timestamp` | |
| `c_et0` | `et0` | mm, same value for every row of a day |
| `c_temp` | `air_temperature` | °C |
| `c_solar_irradiation` | `solar_radiation` | W/m² |
| `c_solar_heat_flux_density` | dropped | always 0 according to the source |
| `c_wind` | `wind_speed` | m/s at 2 m height |
| `c_humid` | `relative_humidity` | % |
| `c_rain` | `rainfall` | mm |
| `c_irrig_mm` | `irrigation_depth_mm` | mm per interval, same value for every row while irrigation is on |
| `c_sensor_1_vwc` to `c_sensor_4_vwc` | `soil_moisture_20cm` to `_80cm` | % volumetric water content |
| (derived) | `soil_moisture` | mean of the four depths |
| `c_sensor_1_soil_temp` to `c_sensor_4_soil_temp` | `soil_temperature_20cm` to `_80cm` | °C |
| (derived) | `soil_temperature` | mean of the four depths |
| `c_gdd_val` | `gdd` | max(0, (Tmax+Tmin)/2 - 10), accumulated over the year |
| (derived) | `irrigation_on` | 1 if `irrigation_depth_mm` > 0 |
| (derived) | `segment_id` | new id whenever the gap to the previous row is more than 10 min |
| (constant) | `sector_id` | 1 |

**Checks done:** 157,225 rows, no duplicate timestamps. 98.7% of steps are exactly 5 min apart; 1,035 steps are 2 min and 785 are 3 min. Only two real gaps (1 h 17 min on 2023-02-21 and 53 min on 2023-03-09), so `segment_id` has 3 values (15,040, 4,646 and 137,539 rows).

**Missing values:** `wind_speed` 2.1%, `solar_radiation` 1.2%, `air_temperature`, `relative_humidity` and `rainfall` 0.7% each, `et0` 0.1%. Soil columns have none. Fill short gaps by interpolation inside a segment.

**Known issues:** `wind_speed` has a maximum of 52.5 m/s while the median is 0.2 m/s, which is almost certainly a sensor error. Clip or remove outliers before scaling. Irrigation is on in about 5.3% of rows.

---

## 4. Dataset 2: `arnesano`

**What it is:** A full growing season of an IoT-instrumented farm in Arnesano (Lecce, Apulia, Italy; 40.349 N, 18.082 E). Five sectors with four crop setups (open-field tomato in sectors 1 and 2, potted tomato in sector 3, zucchini in sector 4, blueberry in sector 5). Soil sensors, valve (actuator) telemetry and ERA5 weather. Irrigation events were rebuilt from noisy valve signals by the dataset authors.

**Source:** "Soil Moisture, Irrigation Actuator and Weather Dataset from a Multi-Sector Precision-Irrigation Field Trial (Arnesano, Italy, 2025)", Mendeley Data, V1, https://data.mendeley.com/datasets/c837v6p8ph/2. License: **CC BY 4.0**. The README in the zip still has the authors and DOI marked "to be completed", so copy them from the Mendeley page when you cite it. Weather values come from the Open-Meteo Historical Weather API (ERA5 reanalysis by ECMWF / Copernicus, CC BY 4.0), not from on-site instruments.

**Files used:** `02_processed_data/preprocessed/dataset_zone_{1..5}_preprocessed.csv` (cleaned by the authors). The source also ships raw sensor/valve exports, merged tables and its own code and `DATA_DICTIONARY.md`; those were not changed.

| Original column | Uniform name | Notes |
|---|---|---|
| `ts` | `timestamp` | 10-min bin start, local time (Europe/Rome) |
| (sector from file name) | `sector_id` | 1 to 5 |
| `soil_moisture` | `soil_moisture` | relative probe output, 0-100 % |
| `soil_temperature_0-7cm` | `soil_temperature` | °C, ERA5 |
| `soil_temperature_7-18cm` | `soil_temperature_deep` | °C, ERA5 (the source says the real band is 7-28 cm) |
| `weather_temp` | `air_temperature` | °C at 2 m |
| `weather_humidity` | `relative_humidity` | % |
| `weather_rain` | `rainfall` | mm |
| `weather_pressure` | `air_pressure` | hPa |
| `weather_wind_speed` | `wind_speed` | **converted from km/h to m/s** (÷ 3.6); ERA5 value at 10 m |
| `weather_radiation` | `solar_radiation` | W/m², direct radiation |
| `ec` | `soil_ec` | µS/cm; values outside 0-1000 set to empty (sensor error codes) |
| `ph` | `soil_ph` | values outside 3-9 set to empty (sensor error codes) |
| `irrigation_duration_minutes` | `irrigation_duration_min` | 0-10 min per bin |
| (derived) | `irrigation_volume_l` | duration × flow rate: **1.1 L/min, and 0.4 L/min for sector 5** (the source's preprocessed volumes wrongly used 1.1 for every sector) |
| (derived) | `irrigation_on` | 1 if duration > 0 |
| (derived) | `segment_id` | new id whenever the gap to the previous row is not exactly 10 min |

**Dropped on purpose:** `target_point_24h`, `target_mean_24h`, `water_vol_to_24h` and `real_moisture_delta` (they look into the future, so using them as inputs is leakage; build your own target from `soil_moisture`), `water_vol_past_4h` (recomputable), and the calendar columns `month`, `day`, `hour`, `day_period`, `quarter` (derive them from `timestamp` if wanted; the source warns that they let a model memorise the season).

**Rows per sector:** 1: 9,531 | 2: 11,929 | 3: 1,118 | 4: 12,676 | 5: 11,814 (total 47,068). Irrigation is on in about 5.1% of rows.

**Known issues:**
- **Fragmented series.** The authors drop rows when soil moisture is missing for more than 2 steps, so the file has 4,746 segments with a median of 3 rows. Only 40 segments have at least 168 rows (sector 1: 8, sector 2: 9, sector 4: 19, sector 5: 4; sector 3 has none). Because of this, "144 steps" is not always 24 hours.
- Sector 3 is short (1,118 rows) because its probe stopped working in late June. Leave it out of the main comparison.
- Weather is reanalysis data, so it is identical for all five sectors and can change slightly if re-downloaded.
- The sectors have different soil parameters and flow rates; train per sector or add `sector_id` as an input.

---

## 5. Dataset 3: `edge_iot` (small, for testing)

**What it is:** A small sensor log from a smart home garden with an edge-IoT controller: temperature, humidity, soil moisture, light and pump status every 30 minutes for 20 days (2026-04-01 to 2026-04-20).

**Source:** "Environmental Sensor Dataset for LSTM-Based Predictive Irrigation in Smart Home Gardening Using Edge IoT", Mendeley Data, https://data.mendeley.com/datasets/jtsyb593fy/1. The xlsx has no metadata sheet, so **authors and license are not in the file; copy them from the Mendeley page**.

**File used:** `Edge_IoT_Predictive_Irrigation_Dataset.xlsx` (one sheet).

| Original column | Uniform name | Notes |
|---|---|---|
| `Timestamp` | `timestamp` | |
| `Day`, `Time` | dropped | duplicate parts of `Timestamp` |
| `Temperature_C` | `air_temperature` | °C (range 25 to 35) |
| `Humidity_pct` | `relative_humidity` | % (range 61 to 88) |
| `Soil_Moisture_pct` | `soil_moisture` | % (range 46.6 to 85) |
| `Light_Intensity` | `light_intensity` | unit not stated by the source (0 to 949) |
| `Pump_Status` | `irrigation_on` | 0/1; pump is on in only 2.9% of rows (28 rows) |
| (constant) | `sector_id` | 1 |
| (derived) | `segment_id` | 1 for all rows: no gaps, no missing values |

**Known issues:** only 960 rows and 28 "pump on" rows, so a deep model will overfit. The values look like a short controlled run, so do not report this dataset's accuracy as the final result.

---

## 6. How to build sliding windows correctly

1. Sort by `sector_id`, then `timestamp`.
2. Make windows **inside each (`sector_id`, `segment_id`) group only**. A window must never contain rows from two segments.
3. Use a **chronological split** (for example 70/15/15 by time, no shuffling). Fit scalers on the training part only.
4. Keep the split identical for LSTM, GRU and BiLSTM, and also compute a naive "last value" baseline.
5. `build_uniform_datasets.py` recreates the three uniform CSVs from the original downloads, in case the data needs to be rebuilt.

---

## 7. Files delivered

| File | Content |
|---|---|
| `DATASET_DOCUMENTATION.md` | this document |
| `uniform_data/topraq_uniform.csv` | 157,225 rows × 22 columns |
| `uniform_data/arnesano_uniform.csv` | 47,068 rows × 17 columns |
| `uniform_data/edge_iot_uniform.csv` | 960 rows × 8 columns |
| `build_uniform_datasets.py` | script that produced the three uniform CSVs |
| `cleaned_data/*_clean.csv` | cleaned, unscaled data used for the windows (topraq is hourly) |
| `make_windows.py` | cleaning + sliding windows (writes `cleaned_data/` and `windows/`) |

## 8. Citation checklist for the report

- [ ] TOPRAQ: copy the license and attribution from https://topraq.ai/?p=41325
- [ ] Arnesano: authors and DOI from https://data.mendeley.com/datasets/c837v6p8ph/2 (CC BY 4.0); also credit Open-Meteo / ERA5 (ECMWF, Copernicus)
- [ ] Edge IoT: authors and license from https://data.mendeley.com/datasets/jtsyb593fy/1

---

## 9. Sliding windows (ready for B and C)

Made by `make_windows.py` from the uniform CSVs. Files are in `windows/`: `<dataset>_{train,val,test}.npz` and `<dataset>_meta.json`.

**Cleaning applied before windowing**
- `topraq`: `wind_speed` above 20 m/s set to empty (sensor spikes), then resampled from 5 min to **1 hour** (mean for state variables, sum for `rainfall` and `irrigation_depth_mm`).
- `arnesano`: sector 3 removed (probe failed); `soil_ec` and `soil_ph` not used as inputs (error codes, mostly empty).
- All: gaps of at most 3 steps are interpolated inside a sector; rows still missing are dropped; `segment_id` is rebuilt after cleaning.

**Window setup (same for every dataset)**
- Input: the last **24 steps** (`X` shape: N × 24 × features). Features are listed in `*_meta.json` (`hour_sin` and `hour_cos` are added; the target column `soil_moisture` is also an input).
- Target: `y` has 3 columns, `soil_moisture` **1, 6 and 24 steps ahead** (`horizons_steps` in the meta file). One step is 1 h for `topraq`, 10 min for `arnesano`, 30 min for `edge_iot`.
- Windows never cross a gap or a sector (each lies inside one `segment_id`).
- Split: chronological 70/15/15 by date (cutoffs in the meta file, same dates for all sectors). Windows are built after the split, so no window overlaps two splits.
- Scaling: z-score with mean and std from the **training rows only** (stored in the meta file). `X` and `y` are already scaled; undo with `value × std + mean` using the `soil_moisture` entry.
- Extra arrays in each file: `t` (timestamp of the last input step, nanoseconds) and `sector`.

| Dataset | Features | Train / Val / Test windows |
|---|---|---|
| `topraq` (1 h step) | 11 | 8,839 / 1,878 / 1,879 |
| `arnesano` (10 min step, sectors 1, 2, 4, 5) | 11 | 8,366 / 8,249 / 6,850 |
| `edge_iot` (30 min step) | 7 | 625 / 97 / 97 |

**Naive baseline to beat** ("same as last value", RMSE in scaled units, test split; horizons 1 / 6 / 24 steps): `topraq` 0.189 / 0.686 / 1.163, `arnesano` 0.033 / 0.033 / 0.037, `edge_iot` 0.262 / 0.780 / 1.001. A model that does not beat these numbers has learned nothing.

**Load example**
```python
import numpy as np, json
d = np.load("windows/topraq_train.npz"); X, y = d["X"], d["y"]   # X: (N,24,11), y: (N,3)
meta = json.load(open("windows/topraq_meta.json"))
```
