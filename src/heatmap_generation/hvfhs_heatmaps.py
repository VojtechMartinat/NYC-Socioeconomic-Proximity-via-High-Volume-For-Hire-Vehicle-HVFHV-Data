from pathlib import Path
import zipfile

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = Path("../../data/")
SHAPE_ZIP = Path("../../data/taxi_zones.zip")
LOOKUP_FILE = Path("../../data/taxi_zone_lookup.csv")
EXTRACT_DIR = Path("../../data/taxi_zones_extracted")
OUTPUT_DIR = Path("Hvfhs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_TRIPS_FOR_AVG = 30

# -----------------------------
# LOADERS
# -----------------------------
REQUIRED_COLS = [
    "PULocationID",
    "DOLocationID",
    "trip_miles",
    "trip_time",
    "base_passenger_fare",
    "tips",
    "driver_pay",
    "shared_request_flag",
    "shared_match_flag",
    "access_a_ride_flag",
    "wav_request_flag",
    "wav_match_flag",
]

EXCLUDE_LOCATION_IDS = [264, 265]


def prepare_hvfhs(df: pd.DataFrame) -> pd.DataFrame:
    df = df[REQUIRED_COLS]

    numeric_cols = [
        "PULocationID",
        "DOLocationID",
        "trip_miles",
        "trip_time",
        "base_passenger_fare",
        "tips",
        "driver_pay",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in [
        "shared_request_flag",
        "shared_match_flag",
        "access_a_ride_flag",
        "wav_request_flag",
        "wav_match_flag",
    ]:
        df[col] = df[col].fillna("N").astype(str).str.upper().str.strip()

    mask = (
        df["PULocationID"].notna()
        & df["DOLocationID"].notna()
        & ~df["PULocationID"].isin(EXCLUDE_LOCATION_IDS)
        & ~df["DOLocationID"].isin(EXCLUDE_LOCATION_IDS)
        & (df["trip_miles"].fillna(0) >= 0)
        & (df["trip_time"].fillna(0) >= 0)
        & (df["base_passenger_fare"].fillna(0) >= 0)
        & (df["tips"].fillna(0) >= 0)
        & (df["driver_pay"].fillna(0) >= 0)
    )

    return df.loc[mask]


def ensure_shapefile(zip_path: Path, extract_dir: Path) -> Path:
    shp_path = extract_dir / "taxi_zones" / "taxi_zones.shp"
    if not shp_path.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
    return shp_path


# -----------------------------
# PREP
# -----------------------------
def aggregate_partial(df: pd.DataFrame, zone_col: str) -> pd.DataFrame:
    temp = pd.DataFrame(
        {
            "LocationID": df[zone_col].astype("Int64"),
            "trip_count": 1,
            "sum_fare": df["base_passenger_fare"].fillna(0),
            "sum_tips": df["tips"].fillna(0),
            "sum_distance": df["trip_miles"].fillna(0),
            "sum_duration_min": df["trip_time"].fillna(0) / 60.0,
            "sum_driver_pay": df["driver_pay"].fillna(0),
            "sum_shared_request": (df["shared_request_flag"] == "Y").astype(int),
            "sum_shared_match": (df["shared_match_flag"] == "Y").astype(int),
            "sum_access_a_ride": (df["access_a_ride_flag"] == "Y").astype(int),
            "sum_wav_request": (df["wav_request_flag"] == "Y").astype(int),
            "sum_wav_match": (df["wav_match_flag"] == "Y").astype(int),
        }
    )

    return temp.groupby("LocationID", as_index=False).sum()


def combine_partials(partials: list[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(partials, ignore_index=True)
    return combined.groupby("LocationID", as_index=False).sum()


def finalize_metrics(grouped: pd.DataFrame) -> pd.DataFrame:
    out = grouped.copy()

    out["avg_fare"] = out["sum_fare"] / out["trip_count"]
    out["avg_distance"] = out["sum_distance"] / out["trip_count"]
    out["avg_duration_min"] = out["sum_duration_min"] / out["trip_count"]
    out["avg_driver_pay"] = out["sum_driver_pay"] / out["trip_count"]

    out["tip_rate"] = np.where(
        out["sum_fare"] > 0,
        (out["sum_tips"] / out["sum_fare"]) * 100,
        np.nan,
    )

    out["shared_request_share"] = (out["sum_shared_request"] / out["trip_count"]) * 100
    out["shared_match_share"] = (out["sum_shared_match"] / out["trip_count"]) * 100
    out["access_a_ride_share"] = (out["sum_access_a_ride"] / out["trip_count"]) * 100
    out["wav_request_share"] = (out["sum_wav_request"] / out["trip_count"]) * 100
    out["wav_match_share"] = (out["sum_wav_match"] / out["trip_count"]) * 100

    avg_cols = [
        "avg_fare",
        "avg_distance",
        "avg_duration_min",
        "avg_driver_pay",
        "tip_rate",
        "shared_request_share",
        "shared_match_share",
        "access_a_ride_share",
        "wav_request_share",
        "wav_match_share",
    ]
    low_volume_mask = out["trip_count"] < MIN_TRIPS_FOR_AVG
    out.loc[low_volume_mask, avg_cols] = np.nan

    return out


# -----------------------------
# AGGREGATION
# -----------------------------
def aggregate_zone_metrics(df: pd.DataFrame, zone_col: str) -> pd.DataFrame:
    temp = df.copy()

    temp["trip_duration_min"] = temp["trip_time"] / 60.0
    temp["shared_request_num"] = (temp["shared_request_flag"] == "Y").astype(int)
    temp["shared_match_num"] = (temp["shared_match_flag"] == "Y").astype(int)
    temp["access_a_ride_num"] = (temp["access_a_ride_flag"] == "Y").astype(int)
    temp["wav_request_num"] = (temp["wav_request_flag"] == "Y").astype(int)
    temp["wav_match_num"] = (temp["wav_match_flag"] == "Y").astype(int)

    grouped = (
        temp.groupby(zone_col)
        .agg(
            trip_count=(zone_col, "size"),
            avg_fare=("base_passenger_fare", "mean"),
            total_fare=("base_passenger_fare", "sum"),
            avg_tip=("tips", "mean"),
            total_tips=("tips", "sum"),
            avg_distance=("trip_miles", "mean"),
            avg_duration_min=("trip_duration_min", "mean"),
            avg_driver_pay=("driver_pay", "mean"),
            shared_request_share=("shared_request_num", "mean"),
            shared_match_share=("shared_match_num", "mean"),
            access_a_ride_share=("access_a_ride_num", "mean"),
            wav_request_share=("wav_request_num", "mean"),
            wav_match_share=("wav_match_num", "mean"),
        )
        .reset_index()
    )

    grouped["tip_rate"] = np.where(
        grouped["total_fare"] > 0,
        grouped["total_tips"] / grouped["total_fare"],
        np.nan,
    )

    for col in [
        "shared_request_share",
        "shared_match_share",
        "access_a_ride_share",
        "wav_request_share",
        "wav_match_share",
        "tip_rate",
    ]:
        grouped[col] = grouped[col] * 100

    avg_cols = [
        "avg_fare",
        "avg_tip",
        "avg_distance",
        "avg_duration_min",
        "avg_driver_pay",
        "shared_request_share",
        "shared_match_share",
        "access_a_ride_share",
        "wav_request_share",
        "wav_match_share",
        "tip_rate",
    ]
    low_volume_mask = grouped["trip_count"] < MIN_TRIPS_FOR_AVG
    grouped.loc[low_volume_mask, avg_cols] = np.nan

    grouped = grouped.rename(columns={zone_col: "LocationID"})
    return grouped


# -----------------------------
# MAP
# -----------------------------
def merge_with_geometry(
    metrics: pd.DataFrame, zones: gpd.GeoDataFrame, lookup: pd.DataFrame
) -> gpd.GeoDataFrame:
    merged = zones.merge(metrics, on="LocationID", how="left")
    merged = merged.merge(
        lookup[["LocationID", "Borough", "Zone", "service_zone"]],
        on="LocationID",
        how="left",
    )
    return merged


def plot_heatmap(
    gdf: gpd.GeoDataFrame, column: str, title: str, filename: str, cmap: str = "Reds"
) -> None:
    fig, ax = plt.subplots(figsize=(12, 12))
    gdf.plot(
        column=column,
        ax=ax,
        cmap=cmap,
        linewidth=0.25,
        edgecolor="white",
        legend=True,
        missing_kwds={"color": "lightgrey", "label": "Sem dados"},
    )
    ax.set_title(title, fontsize=14)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


# -----------------------------
# MAIN
# -----------------------------
def main() -> None:
    print("#--------------#")
    print("Checking directory...")
    if not DATA_DIR.exists():
        raise FileNotFoundError("Dir not found!")

    files = sorted(DATA_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"None parquet file found at {DATA_DIR}")

    print("Directory found!")
    print("#--------------#")
    print("Extracting shapefile...")
    shp_path = ensure_shapefile(SHAPE_ZIP, EXTRACT_DIR)

    print("Reading shapefile...")
    zones = gpd.read_file(shp_path).to_crs(epsg=4326)
    zones = zones[~zones["LocationID"].isin(EXCLUDE_LOCATION_IDS)]

    print("Reading lookup file...")
    lookup = pd.read_csv(LOOKUP_FILE)
    lookup = lookup[~lookup["LocationID"].isin(EXCLUDE_LOCATION_IDS)]

    pickup_partials = []
    dropoff_partials = []

    print("#--------------#")
    print("Reading and aggregating parquet files...")
    for f in files:
        print(f"Processing {f.name}...")
        df = pd.read_parquet(f, columns=REQUIRED_COLS)
        df = prepare_hvfhs(df)

        pickup_partials.append(aggregate_partial(df, "PULocationID"))
        dropoff_partials.append(aggregate_partial(df, "DOLocationID"))

        del df

    print("#--------------#")
    print("Combining partial aggregates...")
    pickup_metrics = finalize_metrics(combine_partials(pickup_partials))
    dropoff_metrics = finalize_metrics(combine_partials(dropoff_partials))

    print("#--------------#")
    print("Merging with geometry...")
    pickup_map = merge_with_geometry(pickup_metrics, zones, lookup)
    dropoff_map = merge_with_geometry(dropoff_metrics, zones, lookup)

    print("#--------------#")
    print("Plotting heatmaps...")
    plot_heatmap(
        pickup_map,
        "trip_count",
        "HVFHS - Demand by Pickup Zone",
        "hvfhs_pickup_demand.png",
    )
    plot_heatmap(
        dropoff_map,
        "trip_count",
        "HVFHS - Demand by Dropoff Zone",
        "hvfhs_dropoff_demand.png",
    )
    plot_heatmap(
        pickup_map,
        "avg_fare",
        "HVFHS - Average Trip Value by Pickup Zone",
        "hvfhs_avg_fare_pickup.png",
    )
    plot_heatmap(
        pickup_map,
        "tip_rate",
        "HVFHS - Tip Rate (%) by Pickup Zone",
        "hvfhs_tip_rate_pickup.png",
    )
    plot_heatmap(
        pickup_map,
        "avg_distance",
        "HVFHS - Average Trip Distance by Pickup Zone",
        "hvfhs_avg_distance_pickup.png",
    )
    plot_heatmap(
        pickup_map,
        "wav_request_share",
        "HVFHS - WAV Request Rate (%) by Pickup Zone",
        "hvfhs_wav_request_pickup.png",
    )

    print("#--------------#")
    print("Heatmaps outputed into:", OUTPUT_DIR)


if __name__ == "__main__":
    print("Running script...")
    main()
