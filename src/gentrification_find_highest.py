import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import folium
from scipy.stats import zscore
from pathlib import Path

# config

DATA_DIR = Path("../data/")
SHAPEFILE_PATH = "taxi_zones/taxi_zones.shp"

YEARS_TO_COMPARE = [2020, 2019, 2025]
MONTHS_FILTER = None  # e.g.: [6,7,8] for summer only

OUTPUT_DIR = Path("output_maps/")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = Path("max_gentrification_per_focus.csv")

# load shp
zones = gpd.read_file(SHAPEFILE_PATH)
zones = zones.to_crs(epsg=4326)

# data load
def load_year_data(year):
    print(f"Loading data for {year}")
    files = sorted(DATA_DIR.glob(f"*{year}*.parquet"))

    df_list = []
    for f in files:
        temp = pd.read_parquet(f)
        df_list.append(temp)

    df = pd.concat(df_list)

    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['year'] = df['pickup_datetime'].dt.year
    df['month'] = df['pickup_datetime'].dt.month

    if MONTHS_FILTER:
        df = df[df['month'].isin(MONTHS_FILTER)]

    return df

# filter trips for location
def filter_focus_trips(df, focus_id):
    return df[
        (df['PULocationID'] == focus_id) |
        (df['DOLocationID'] == focus_id)
    ].copy()

# assemble destination counts
def compute_zone_metrics(df, focus_id):
    df['target_zone'] = np.where(
        df['PULocationID'] == focus_id,
        df['DOLocationID'],
        df['PULocationID']
    )

    metrics = df.groupby('target_zone').agg(
        trip_count=('target_zone', 'count'),
        avg_fare=('base_passenger_fare', 'mean'),
        avg_distance=('trip_miles', 'mean'),
        total_revenue=('driver_pay', 'sum')
    ).reset_index()

    return metrics

# pre-load all data for efficiency
all_year_data = {year: load_year_data(year) for year in YEARS_TO_COMPARE}

# store max gentrification rows
max_rows = []

# loop over focus locations
for focus_id in range(1, 290):  # 1 to 289 inclusive
    print(f"Processing FOCUS_LOCATION_ID = {focus_id}")

    all_year_metrics = {}

    for year in YEARS_TO_COMPARE:
        df_year = all_year_data[year]
        df_focus = filter_focus_trips(df_year, focus_id)
        metrics = compute_zone_metrics(df_focus, focus_id)
        all_year_metrics[year] = metrics

    # merge year comparisons
    merged = zones[['LocationID', 'geometry']].copy()
    for year, metrics in all_year_metrics.items():
        metrics = metrics.rename(columns={
            'trip_count': f'trips_{year}',
            'avg_fare': f'fare_{year}',
            'avg_distance': f'distance_{year}',
            'total_revenue': f'revenue_{year}'
        })
        merged = merged.merge(
            metrics,
            left_on='LocationID',
            right_on='target_zone',
            how='left'
        )
        merged.drop(columns=['target_zone'], inplace=True)

    merged.fillna(0, inplace=True)

    # growth metrics
    base_year = YEARS_TO_COMPARE[0]
    latest_year = YEARS_TO_COMPARE[-1]

    merged['trip_growth_pct'] = (
        (merged[f'trips_{latest_year}'] - merged[f'trips_{base_year}']) /
        (merged[f'trips_{base_year}'] + 1)
    )

    merged['revenue_growth_pct'] = (
        (merged[f'revenue_{latest_year}'] - merged[f'revenue_{base_year}']) /
        (merged[f'revenue_{base_year}'] + 1)
    )

    # standardise
    merged['growth_zscore'] = zscore(merged['trip_growth_pct'])  # type: ignore

    # Potential gentrification score
    merged['gentrification_score'] = (
        0.6 * merged['growth_zscore'] +
        0.4 * zscore(merged['revenue_growth_pct'])
    )

    # pick the row with the highest gentrification score
    if merged['gentrification_score'].notna().any():
        max_idx = merged['gentrification_score'].idxmax()
        max_row = merged.loc[max_idx].copy()
        max_row['focus_location'] = focus_id
        max_rows.append(max_row)
    else:
        print(f"No valid gentrification score for focus zone {focus_id}, skipping.")

# save all max rows
df_max = pd.DataFrame(max_rows)
df_max.to_csv(OUTPUT_FILE, index=False)
print(f"Saved max gentrification rows for all focus zones to {OUTPUT_FILE}")