from pathlib import Path

import numpy as np
import pandas as pd

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = Path("../../data/")
OUTPUT_DIR = Path("Visual Correlations/Csvs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_TRIPS_FOR_AVG = 30
EXCLUDE_LOCATION_IDS = [264, 265]

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

ARRESTS_FILE = DATA_DIR / "arrests.csv"

SPECIFIC_CRIME_TYPES = [
    "DISORDERLY CONDUCT",
    "CRIMINAL MISCHIEF & RELATED OF",
    "ASSAULT 3 & RELATED OFFENSES",
]


def slugify(text: str) -> str:
    return (
        text.lower()
        .replace("&", "and")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
        .replace("__", "_")
    )


# -----------------------------
# TAXI PREP
# -----------------------------
def prepare_hvfhs(df: pd.DataFrame) -> pd.DataFrame:
    df = df[REQUIRED_COLS].copy()

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

    flag_cols = [
        "shared_request_flag",
        "shared_match_flag",
        "access_a_ride_flag",
        "wav_request_flag",
        "wav_match_flag",
    ]

    for col in flag_cols:
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

    return df.loc[mask].copy()


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

    avg_cols = [
        "avg_fare",
        "avg_distance",
        "avg_duration_min",
        "avg_driver_pay",
        "tip_rate",
    ]

    low_volume_mask = out["trip_count"] < MIN_TRIPS_FOR_AVG
    out.loc[low_volume_mask, avg_cols] = np.nan

    return out


def build_taxi_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    files = sorted(DATA_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found at {DATA_DIR}")

    pickup_partials = []
    dropoff_partials = []

    for f in files:
        print(f"Processing taxi file: {f.name}")
        df = pd.read_parquet(f, columns=REQUIRED_COLS)
        df = prepare_hvfhs(df)

        pickup_partials.append(aggregate_partial(df, "PULocationID"))
        dropoff_partials.append(aggregate_partial(df, "DOLocationID"))

        del df

    pickup_metrics = finalize_metrics(combine_partials(pickup_partials))
    dropoff_metrics = finalize_metrics(combine_partials(dropoff_partials))

    return pickup_metrics, dropoff_metrics


def build_taxi_crime_table(
    taxi_metrics: pd.DataFrame,
    crime_metrics: pd.DataFrame,
    demand_label: str,
) -> pd.DataFrame:
    out = taxi_metrics.merge(crime_metrics, on="LocationID", how="left")

    out["trip_count"] = pd.to_numeric(out["trip_count"], errors="coerce").fillna(0)
    out["crime_count"] = pd.to_numeric(out["crime_count"], errors="coerce").fillna(0)

    out["crime_per_1000_trips"] = np.where(
        out["trip_count"] > 0,
        (out["crime_count"] / out["trip_count"]) * 1000,
        np.nan,
    )

    out["demand_type"] = demand_label
    return out


# -----------------------------
# CRIME PREP
# -----------------------------
def load_crime_base() -> pd.DataFrame:
    print("Loading arrest data...")

    crime_df = pd.read_csv(
        ARRESTS_FILE,
        sep=";",
        usecols=["ARREST_DATE", "LocationID", "OFNS_DESC"],
    ).copy()

    crime_df["ARREST_DATE"] = pd.to_datetime(
        crime_df["ARREST_DATE"],
        format="%m/%d/%Y",
        errors="coerce",
    )

    crime_df["LocationID"] = pd.to_numeric(
        crime_df["LocationID"],
        errors="coerce",
    ).astype("Int64")

    crime_df["OFNS_DESC"] = crime_df["OFNS_DESC"].astype(str).str.strip().str.upper()

    crime_df = crime_df[crime_df["LocationID"].notna()]
    crime_df = crime_df[~crime_df["LocationID"].isin(EXCLUDE_LOCATION_IDS)]

    return crime_df


def build_crime_counts(
    crime_df: pd.DataFrame,
    crime_types: list[str] | None = None,
) -> pd.DataFrame:
    df = crime_df.copy()

    if crime_types:
        normalized_types = [c.upper().strip() for c in crime_types]
        df = df[df["OFNS_DESC"].isin(normalized_types)]

    return df.groupby("LocationID").size().reset_index(name="crime_count")


def build_crime_counts_wide(
    crime_df: pd.DataFrame,
    crime_types: list[str],
) -> pd.DataFrame:
    total_counts = (
        crime_df.groupby("LocationID").size().reset_index(name="crime_count_total")
    )

    wide = total_counts.copy()

    for crime_type in crime_types:
        crime_type_norm = crime_type.upper().strip()
        slug = slugify(crime_type_norm)

        temp = (
            crime_df[crime_df["OFNS_DESC"] == crime_type_norm]
            .groupby("LocationID")
            .size()
            .reset_index(name=f"{slug}_count")
        )

        wide = wide.merge(temp, on="LocationID", how="left")

    count_cols = [
        c for c in wide.columns if c.endswith("_count") or c == "crime_count_total"
    ]
    wide[count_cols] = wide[count_cols].fillna(0)

    return wide


# -----------------------------
# MAIN
# -----------------------------
def main():
    print("Loading taxi metrics...")
    pickup_metrics, dropoff_metrics = build_taxi_metrics()

    print("Loading and preparing crime base...")
    crime_base = load_crime_base()

    print("Generating total crime outputs...")
    total_crime_metrics = build_crime_counts(crime_base)

    pickup_total = build_taxi_crime_table(
        pickup_metrics,
        total_crime_metrics,
        "pickup",
    )
    dropoff_total = build_taxi_crime_table(
        dropoff_metrics,
        total_crime_metrics,
        "dropoff",
    )

    pickup_total.to_csv(OUTPUT_DIR / "pickup_crime_by_zone.csv", index=False)
    dropoff_total.to_csv(OUTPUT_DIR / "dropoff_crime_by_zone.csv", index=False)

    print("Generating specific crime outputs...")
    for crime_type in SPECIFIC_CRIME_TYPES:
        print(f"Processing crime type: {crime_type}")

        crime_metrics = build_crime_counts(crime_base, [crime_type])
        crime_slug = slugify(crime_type)

        pickup_specific = build_taxi_crime_table(
            pickup_metrics,
            crime_metrics,
            "pickup",
        )
        dropoff_specific = build_taxi_crime_table(
            dropoff_metrics,
            crime_metrics,
            "dropoff",
        )

        pickup_specific = pickup_specific.rename(
            columns={"crime_count": f"{crime_slug}_count"}
        )
        dropoff_specific = dropoff_specific.rename(
            columns={"crime_count": f"{crime_slug}_count"}
        )

        pickup_specific.to_csv(
            OUTPUT_DIR / f"pickup_{crime_slug}_by_zone.csv",
            index=False,
        )
        dropoff_specific.to_csv(
            OUTPUT_DIR / f"dropoff_{crime_slug}_by_zone.csv",
            index=False,
        )

    print("Generating crime breakdown outputs...")
    crime_wide = build_crime_counts_wide(crime_base, SPECIFIC_CRIME_TYPES)

    pickup_full = pickup_metrics.merge(crime_wide, on="LocationID", how="left")
    dropoff_full = dropoff_metrics.merge(crime_wide, on="LocationID", how="left")

    count_cols = [c for c in crime_wide.columns if c != "LocationID"]
    pickup_full[count_cols] = pickup_full[count_cols].fillna(0)
    dropoff_full[count_cols] = dropoff_full[count_cols].fillna(0)

    pickup_full.to_csv(
        OUTPUT_DIR / "pickup_crime_breakdown_by_zone.csv",
        index=False,
    )
    dropoff_full.to_csv(
        OUTPUT_DIR / "dropoff_crime_breakdown_by_zone.csv",
        index=False,
    )

    print(f"Done. Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
