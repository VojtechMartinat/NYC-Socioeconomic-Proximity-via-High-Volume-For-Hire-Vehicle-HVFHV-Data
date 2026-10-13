import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from scipy.stats import zscore
from pathlib import Path

# config

DATA_DIR = Path("../data/")
SHAPEFILE_PATH = "taxi_zones/taxi_zones.shp"

FOCUS_LOCATION_ID = 231
YEARS_TO_COMPARE = [2020, 2019, 2025]
MONTHS_FILTER = None  

OUTPUT_DIR = Path("output_maps/")
OUTPUT_DIR.mkdir(exist_ok=True)

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

def filter_focus_trips(df):
    return df[
        (df['PULocationID'] == FOCUS_LOCATION_ID) |
        (df['DOLocationID'] == FOCUS_LOCATION_ID)
        ].copy()


# assemble destination counts

def compute_zone_metrics(df):
    df['target_zone'] = np.where(
        df['PULocationID'] == FOCUS_LOCATION_ID,
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


# process

all_year_metrics = {}

for year in YEARS_TO_COMPARE:
    df_year = load_year_data(year)
    df_focus = filter_focus_trips(df_year)
    metrics = compute_zone_metrics(df_focus)
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
merged['growth_zscore'] = zscore(merged['trip_growth_pct']) # type: ignore

# Potential gentrification score
merged['gentrification_score'] = (
        0.6 * merged['growth_zscore'] +
        0.4 * zscore(merged['revenue_growth_pct'])
)

merged.to_csv(f'location_gentrification_{FOCUS_LOCATION_ID}.csv', index=False)


# static heatmap

def plot_heatmap(column, title, filename):
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    merged.plot(
        column=column,
        ax=ax,
        cmap="Reds",
        legend=True,
        missing_kwds={"color": "lightgrey"}
    )
    ax.set_title(title)
    ax.axis('off')
    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()


plot_heatmap(
    'gentrification_score',
    f"Gentrification Signal ({base_year} → {latest_year})",
    "gentrification_heatmap.png"
)

# interactive map

m = folium.Map(location=[40.7128, -74.0060], zoom_start=11)

folium.Choropleth(
    geo_data=merged,
    data=merged,
    columns=['LocationID', 'gentrification_score'],
    key_on='feature.properties.LocationID',
    fill_color='YlOrRd',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name="Gentrification Score"
).add_to(m)

m.save(OUTPUT_DIR / "interactive_gentrification_map.html")

print("Analysis Complete.")