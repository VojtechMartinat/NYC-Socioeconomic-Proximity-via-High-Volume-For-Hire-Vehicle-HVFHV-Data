# NYC Socioeconomic Proximity via High-Volume For-Hire Vehicle (HVFHV) Data

This repository contains sample code, sample analysis pipelines and final report for a team research project investigating whether **High Volume For-Hire Vehicle (HVFHV)** taxi data can serve as a reliable proxy for urban socioeconomic patterns across New York City. This project received a grade of 85% (A+ equivalent, 4.0 GPA scale)

The **ISAMBARD Supercomputer (most powerful supercomputer in the UK)** was used for the data processing, cleaning and analysis portion of this project.

The study was conducted collaboratively by a 5-member research team, addressing three main questions:
1. **Crime Correlation:** Whether taxi activity correlates spatially with crime patterns.
2. **Predictive Modeling:** Whether fare and tipping behavior can be predicted from trip-level features using machine learning.
3. **Gentrification Tracking:** Whether shifts in taxi activity and urban infrastructure signal neighborhood gentrification.

The team was composed of: Vojtech Martinat, Lukasz Krepa, Rami Rached, James Burton, Luiz Manzione Raffaini
---

## Repository Structure
```text
├── data/                                         # Directory for samples of raw and processed datasets used
│   ├── Gentrification/                           # Gentrification data used, per each ZIP code zone. ALso includes a legend for column heading codes. Also includes examples of processed tip data, including average tips by zone and increases. 
│   ├── all_imputed.csv/                          # Final sheet with all processed and cleaned indicators per each TLC zone. This was used to find connections between desired variables.
│   ├── max_gentrification_per_focus.csv          # For each TLC zone, this sheet shows the most popular destination zones of high value rides, thus perhaps pointing to gentrification.
│   └── taxi_zone_lookup.csv/                     # Lookup table for each TLC zone, ZIP code etc. 
├── heatmaps/
│   ├── Crime/                                    # Contains various heatmaps generated that pertain to various crimes and TLC data
│   └── Gentrification/                           # For each gentrification indicator used, it contains a heatmap of the metric measured across ZIP zones and TLC zones to demonstrate the efficiency between converting between the different spatial metrics
├── notebooks/
│   ├── correlation_analysis/                     # Jupyter Notebook for correlation analysis between different variables and related data
│   ├── xgboost_classification/                   # Jupyter Notebook for XGBoost classification for fare prediction and related data
│   └── xgboost_regression/                       # Jupyter Notebook for XGBoost regression for fare prediction and related data
├── outputs/                                      # Outputs of various scripts, showing the raw data for some  crime heatmaps, including correlations between crime groups and dropoff zones etc.
├── src/                                          # Source scripts for data ingestion, cleaning, and ML pipelines
│   ├── fare_prediction/                          # Python code and SLURM batch job configurations required to train, hyperparameter-tune, evaluate, and save XGBoost regression and classification models for predicting taxi trip fares
│   ├── heatmap_generation/                       # Correlates taxi metrics with crime by zone. Aggregates high-volume for-hire vehicle trip data and merges it with spatial taxi zone shapefiles to generate and save geographic heatmaps of taxi data. 
│   ├── tip_prediction/                           # Provides a SLURM batch script to run the job, a core Python script that loads trip data and trains an XGBoost classifier to predict tip occurrences.   
│   ├── utils/                                    # Provides utility classes to load, sample, and process TLC high-volume for-hire vehicle trip parquet files and NYPD arrest CSV datasets
│   ├── gentrification.py                         # Computes trip metrics, growth rates, and a composite gentrification score relative to a specific focus location
│   ├── gentrification_correlation.py             # Combines socio-economic census changes and taxi tip metrics with TLC zones to calculate correlation matrices
│   ├── gentrification_find_highest.py            # Calculates trip metrics and gentrification scores, extracting the maximum scoring record for each focus zone
│   └── gentrification_merge_w_TLC.py             # Cleans ZIP-level census demographic data and aggregates it into NYC Taxi & Limousine Commission (TLC) zones using area-weighted mapping ratios to visualize regional heatmaps
├── README.md                                     # Project documentation
└── FINAL REPORT.pdf                                    # Final report in an academic paper style

```
---
## Methods of Analysis

* **Data Integration & Preprocessing:** Cleaned and merged multi-source datasets over unified spatial-temporal grids (combining taxi drop-off/pickup coordinates with neighborhood boundaries and census blocks).
* **Predictive Modeling (XGBoost):** 
  * **Fare Prediction:** Achieved strong performance (MAE = 5.09, R^2 = 0.82) using XGBoost, with trip distance and trip duration serving as dominant features.
  * **Tip Prediction:** Encountered higher prediction complexity (F_1 = 0.29) primarily due to the systematic absence of cash tips in the digital dataset.
* **Spatial & Crime Correlation:** Evaluated relationships between taxi demand and crime categories. Findings revealed a modest positive correlation between late-night drop-offs and disorderly conduct arrests. Drop-off demand correlated more strongly with arrests than pickup demand, and non-white-collar crime (r = 0.417) correlated higher than white-collar crime (r = 0.333).
* **Gentrification Analysis:** Explored spatial connections between taxi revenue, property values, and building permit trends (highlighting shifts from Midtown Manhattan to neighborhoods like Red Hook, Brooklyn). Results demonstrated that while direct correlations varied by zone, higher revenues frequently co-occurred in rapidly gentrifying areas.
---

### Using the ISAMBARD supercomputer in our workflow:
- **Ultra-High Memory Bandwidth for Spatial Joins:** The NVIDIA Grace CPU Superchip delivers high unified memory bandwidth, which significantly sped up spatial polygon intersections and geometric joins between millions of NYC TLC pickup/drop-off points, Census tracts, NYPD arrest sectors, and Department of Buildings permit locations.
- **Multi-Threaded Feature Preprocessing:** Utilizing 144 high-performance Arm CPU cores per node enabled massive parallel processing during large-scale feature engineering, data aggregation, and cross-dataset temporal aligns.
- **Scalable Parallel Model Training:** Parallelized CPU execution enabled efficient distributed grid searches, hyperparameter tuning, and cross-validation for XGBoost modeling across millions of records, reducing total pipeline processing times from days to hours.
---

## Data Sources

The study integrated several large-scale open data sources:
* **NYC Taxi & Limousine Commission (TLC):** Millions of HVFHV trip records containing timestamps, geographic coordinates, trip distance, duration, fares, and tips.
* **NYPD Arrest Data:** Historical arrest records mapped spatially and temporally to evaluate local crime rates.
* **US Census Bureau:** Socioeconomic indicators and demographic details across New York City census tracts.
* **NYC Department of Buildings (DOB):** Permit issuance data used to track property investment shifts and urban redevelopment.


##  Privacy Disclaimer

All data used throughout this project are anonymised at
source and do not identify any user or entry. The TLC trip
data do not contain any passenger or driver identifiers, with the
locations generalised using the zone-level pickup and drop-off
IDs. The NYPD arrest data carry a higher risk of identification
due to the inclusion of the unique coordinates of each arrest
as well as age group, ethnicity, gender and a detailed charge
description, though no individual can be identified from this
dataset alone

## Personal Reflection

Overall, I believe we were able to leverage a unique
vector of approach to this investigation. Specifically, the
relationship of crime to the taxi presence in a zone is
not something that is often examined. I believe we were
able to bring about findings and conclusions that can be
useful for our primary audience of city policy makers and
planners. Using our analysis, further conclusions can be
drawn on the effects of the presence of taxis, in order to
make cities safer.
On the contrary, we were not able to draw as strong
conclusions on the correlation between taxi presence in
certain zones and gentrification taking place. A large part
the of indicators and instruments we picked did not show
a significant relationship with the analysed taxi data. If
we were to do this project again, I would tackle this by
utilizing more robust indicators for gentrification (demo-
graphic change, new businesses opening etc.), alongside
the ones we picked. In addition, I would use data for
a longer period of time and use more frequent samples,
as gentrification might be tougher to spot with just a
simple change. I believe this would make our report more
robust.
Throughout this project, I was able to get invaluable
experience with working with and manipulating millions
of rows of data to get meaningful conclusions. It taught
me that simple relationships on spreadsheets can have a
big impact on the daily lives of many people, therefore
Big Data is an extremely useful tool for any kind
of policy research. I also learned the ability to shift
approaches when a method of data analysis does not
bring about meaningful results. It showed me that there
is always value in fundamentally rethinking my approach
to any such analysis, ultimately giving me the the tools
to succeed in this project.
