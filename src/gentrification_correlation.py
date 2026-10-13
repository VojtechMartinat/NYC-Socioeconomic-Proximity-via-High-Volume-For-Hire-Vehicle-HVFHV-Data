import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from scipy.stats import zscore
from pathlib import Path

base_path = Path("../data/")


df_ZIP_TLC = pd.read_excel(base_path / "Gentrification/ZIP to TLC.xlsx")


df = pd.read_csv(base_path / "max_gentrification_per_focus.csv")

df2 = pd.read_csv(base_path / "Gentrification" / "Mean Income" / "2025" / "data.csv")
df3 = pd.read_csv(base_path / "Gentrification" / "Mean Income" / "2019" / "data.csv")

df4 = pd.read_csv(base_path / "Gentrification" / "Median Value of a House" / "2025" / "data.csv")
df5 = pd.read_csv(base_path / "Gentrification" / "Median Value of a House" / "2019" / "data.csv")

df6 = pd.read_csv(base_path / "Gentrification" / "Total number of people with a Bachelor's Degree" / "2025" / "data.csv")
df7 = pd.read_csv(base_path / "Gentrification" / "Total number of people with a Bachelor's Degree" / "2019" / "data.csv")


df8 = pd.read_csv(base_path / "Gentrification" / "tip_data" /  "tip_increase_by_dropoff_zone_2019_2025.csv")

df9 = pd.read_csv(base_path / "Gentrification" / "tip_data" /  "tip_increase_by_pickup_zone_2019_2025.csv")



import pandas as pd

# List of your datasets and corresponding years/metrics
datasets = [
    ("Mean Income 2025", df2),
    ("Mean Income 2019", df3),
    ("Median House Value 2025", df4),
    ("Median House Value 2019", df5),
    ("Bachelor's Degree 2025", df6),
    ("Bachelor's Degree 2019", df7),
    
]

# Clean ZIP-TLC mapping first
df_ZIP_TLC["ZCTA5CE20"] = pd.to_numeric(df_ZIP_TLC["ZCTA5CE20"], errors="coerce")
df_ZIP_TLC = df_ZIP_TLC[df_ZIP_TLC["ZCTA5CE20"].notna()]
df_ZIP_TLC["ZCTA5CE20"] = df_ZIP_TLC["ZCTA5CE20"].astype(int).astype(str).str.zfill(5)

# Function to calculate weighted values per TLC zone
def weighted_tlc(df_value, value_column="Value"):
    # Take only the first two columns (ZIP and value)
    df_value = df_value.iloc[:, 1:3].copy()

    
    
    # Ensure ZIP column is string and 5 digits
    df_value.columns = ["ZIP", value_column]

    # Now safely convert ZIP
    df_value["ZIP"] = df_value["ZIP"].astype(str).str.zfill(5)
    
    # Rename columns

    df_ZIP_TLC["ZCTA5CE20"] = (
        pd.to_numeric(df_ZIP_TLC["ZCTA5CE20"], errors="coerce")
        .astype("Int64")
        .astype(str)
        .str.zfill(5)
    )

    # Clean your data ZIPs
    df_value["ZIP"] = (
        pd.to_numeric(df_value["ZIP"], errors="coerce")
        .astype("Int64")
        .astype(str)
        .str.zfill(5)
    )
    
    # Merge with ZIP-TLC mapping
    df_mapping = df_ZIP_TLC.merge(df_value, left_on="ZCTA5CE20", right_on="ZIP", how="left")

    
    # Convert to numeric
    df_mapping[value_column] = pd.to_numeric(df_mapping[value_column], errors="coerce")
    df_mapping["ratio2"] = pd.to_numeric(df_mapping["ratio2"], errors="coerce")
    
    # Keep only valid rows
    df_valid = df_mapping[df_mapping[value_column].notna() & (df_mapping[value_column] != 0)].copy()
    
    # Weighted value per ZIP
    df_valid["WeightedValue"] = df_valid[value_column] * df_valid["ratio2"]
    
    # Aggregate per TLC zone (vectorized)
    df_zone = df_valid.groupby("OBJECTID").agg(
        WeightedSum=("WeightedValue", "sum"),
        RatioSum=("ratio2", "sum")
    ).reset_index()
    
    # Compute final weighted average, avoiding division by zero
    df_zone[value_column] = df_zone.apply(
        lambda row: row["WeightedSum"] / row["RatioSum"] if row["RatioSum"] > 0 else None,
        axis=1
    )
    
    # Keep only OBJECTID + final value
    return df_zone[["OBJECTID", value_column]]

# Apply to all datasets
weighted_results = {}
for name, df in datasets:
    weighted_results[name] = weighted_tlc(df)
    
# Example: access the weighted TLC values for Mean Income 2025
df_weighted_income_2025 = weighted_results["Mean Income 2025"]





def compute_change(df_2025, df_2019, value_column="Value", key="OBJECTID"):
    # Work on copies
    df_2025 = df_2025.copy()
    df_2019 = df_2019.copy()
    
    # Ensure key is numeric
    df_2025[key] = pd.to_numeric(df_2025[key], errors="coerce")
    df_2019[key] = pd.to_numeric(df_2019[key], errors="coerce")
    
    df_2025 = df_2025.dropna(subset=[key])
    df_2019 = df_2019.dropna(subset=[key])
    
    df_2025[key] = df_2025[key].astype(int)
    df_2019[key] = df_2019[key].astype(int)
    
    # Rename value columns so they don’t clash
    df_2025 = df_2025.rename(columns={value_column: "Value_2025"})
    df_2019 = df_2019.rename(columns={value_column: "Value_2019"})
    
    # Merge on OBJECTID
    merged = df_2025.merge(df_2019, on=key)
    
    # Convert to numeric
    merged["Value_2025"] = pd.to_numeric(merged["Value_2025"], errors="coerce")
    merged["Value_2019"] = pd.to_numeric(merged["Value_2019"], errors="coerce")
    
    # Keep valid rows
    merged = merged.dropna(subset=["Value_2025", "Value_2019"])
    
    # Compute delta
    merged["delta"] = merged["Value_2025"] - merged["Value_2019"]
    
    return merged[[key, "delta"]]



income_delta = compute_change(
    weighted_results["Mean Income 2025"],
    weighted_results["Mean Income 2019"]
)

housing_delta = compute_change(
    weighted_results["Median House Value 2025"],
    weighted_results["Median House Value 2019"]
)

edu_delta = compute_change(
    weighted_results["Bachelor's Degree 2025"],
    weighted_results["Bachelor's Degree 2019"]
)



income_delta = income_delta.rename(columns={"delta": "income_delta"})
housing_delta = housing_delta.rename(columns={"delta": "house_delta"})
edu_delta = edu_delta.rename(columns={"delta": "edu_delta"})



df = pd.read_csv(base_path / "taxi_zone_IDS.csv")




def add_all_metrics(df, income_delta, house_delta, edu_delta,
                    df8, df9):
    df = df.copy()
    
    # Ensure numeric IDs
    df["LocationID"] = pd.to_numeric(df["LocationID"], errors="coerce")
    
    # --- Existing maps ---
    income_map = income_delta.set_index("OBJECTID")["income_delta"]
    house_map = house_delta.set_index("OBJECTID")["house_delta"]
    edu_map = edu_delta.set_index("OBJECTID")["edu_delta"]
    
    df["income_change"] = df["LocationID"].map(income_map)
    
    df["housing_price_change"] = df["LocationID"].map(house_map)
    
    df["edu_change"] = df["LocationID"].map(edu_map)
    
    
    # --- Helper to add tip metrics ---
    def add_tip_metrics(df, tip_df, prefix):
        tip_df = tip_df.copy()
        tip_df["LocationID"] = pd.to_numeric(tip_df["LocationID"], errors="coerce")
        
        avg_map = tip_df.set_index("LocationID")["avg_diff_%"]
        sum_map = tip_df.set_index("LocationID")["sum_diff_%"]
        trip_map = tip_df.set_index("LocationID")["trip_count_diff_%"]
        
        # Map for LocationID (A)



        df[f"{prefix}_avg_tip_diff_%"] = df["LocationID"].map(avg_map)
        df[f"{prefix}_sum_tip_diff_%"] = df["LocationID"].map(sum_map)
        df[f"{prefix}_trip_count_diff_%"] = df["LocationID"].map(trip_map)
        
        # Map for max_location (B)
       
        
        return df
    
    
    df = add_tip_metrics(df, df8, "dropoff")
    df = add_tip_metrics(df, df9, "pickup")


    
    import glob
    import os

    def add_gentrification_metrics(df, folder_path):
        df = df.copy()

        # Find all matching files
        files = glob.glob(os.path.join(folder_path, "location_gentrification_*.csv"))

        for file in files:
            # Extract ID from filename
            filename = os.path.basename(file)
            location_id = filename.replace("location_gentrification_", "").replace(".csv", "")

            cols_needed = ["LocationID", "trip_growth_pct", "revenue_growth_pct", "gentrification_score"]

            g_df = pd.read_csv(file, usecols=cols_needed,low_memory=False)

            
            g_df["LocationID"] = pd.to_numeric(g_df["LocationID"], errors="coerce")
            

            g_df = g_df.dropna(thresh=3)

            g_df = g_df[g_df["LocationID"] <= 263]

            g_df["gentrification_score"] = pd.to_numeric(
                g_df["gentrification_score"], errors="coerce"
            )

            g_df = g_df[
                (g_df["gentrification_score"] >= -1) &
                (g_df["gentrification_score"] <= 1)
            ]


            print(g_df.head())


            # Create maps
            trip_map = g_df.set_index("LocationID")["trip_growth_pct"]
            revenue_map = g_df.set_index("LocationID")["revenue_growth_pct"]
            score_map = g_df.set_index("LocationID")["gentrification_score"]

             # Add columns (A side)
            df[f"gentrification_{location_id}_trip_growth_%"] = df["LocationID"].map(trip_map)
            df[f"gentrification_{location_id}_revenue_growth_%"] = df["LocationID"].map(revenue_map)
            df[f"gentrification_{location_id}_score_between"] = df["LocationID"].map(score_map)

        


        
        return df

            

           

    location_path = Path("../data/Gentrification/Location_gentrification")


    df = add_gentrification_metrics(df, location_path)
    
    return df

# Apply
df_all = add_all_metrics(df, income_delta, housing_delta, edu_delta, df8, df9)

print(df_all.columns)


cols = [
    'LocationID', 'income_change',
       'housing_price_change', 'edu_change', 'dropoff_avg_tip_diff_%',
       'dropoff_sum_tip_diff_%', 'dropoff_trip_count_diff_%',
       'pickup_avg_tip_diff_%', 'pickup_sum_tip_diff_%',
       'pickup_trip_count_diff_%', 'gentrification_132_trip_growth_%',
       'gentrification_132_revenue_growth_%',
       'gentrification_132_score_between', 'gentrification_161_trip_growth_%',
       'gentrification_161_revenue_growth_%',
       'gentrification_161_score_between', 'gentrification_230_trip_growth_%',
       'gentrification_230_revenue_growth_%',
       'gentrification_230_score_between', 'gentrification_261_trip_growth_%',
       'gentrification_261_revenue_growth_%',
       'gentrification_261_score_between', 'gentrification_87_trip_growth_%',
       'gentrification_87_revenue_growth_%', 'gentrification_87_score_between',
       'gentrification_88_trip_growth_%', 'gentrification_88_revenue_growth_%',
       'gentrification_88_score_between', 'gentrification_90_trip_growth_%',
       'gentrification_90_revenue_growth_%',
       'gentrification_90_score_between',

       
       'gentrification_164_revenue_growth_%',
       'gentrification_164_score_between', 'gentrification_164_trip_growth_%',
       'gentrification_163_revenue_growth_%',
       'gentrification_163_score_between', 'gentrification_163_trip_growth_%',
       'gentrification_162_revenue_growth_%',
       'gentrification_162_score_between', 'gentrification_162_trip_growth_%',
       'gentrification_162_revenue_growth_%',
       'gentrification_231_score_between', 'gentrification_231_trip_growth_%',
       'gentrification_231_revenue_growth_%', 
       'gentrification_186_score_between','gentrification_186_trip_growth_%', 
       'gentrification_186_revenue_growth_%',
       
       ]

df_corr = df_all[cols].copy()


df_corr.to_csv("all.csv", index=False)



corr_matrix = df_corr.corr(method="pearson")  # or "spearman", "kendall"
print(corr_matrix)

corr_matrix.to_csv("output.csv", index=False)

df_corr = pd.read_csv(base_path / "all_imputed.csv")




import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

def plot_with_fit(df, x_col, y_col, title, method="pearson"):
    # Keep the large base global size
    plt.rcParams.update({'font.size': 18}) 
    
    x = df[x_col]
    y = df[y_col]

    if method == "pearson":
        corr = x.corr(y, method="pearson")
        x_fit = x
        y_fit = y
    elif method == "spearman":
        corr, _ = spearmanr(x, y)
        x_fit = x.rank()
        y_fit = y.rank()

    # Fit line calculation
    m, b = np.polyfit(x_fit, y_fit, 1)
    line = m * x_fit + b
    sorted_idx = np.argsort(x_fit)

    # Maintain larger figure size to accommodate big text
    plt.figure(figsize=(12, 9)) 
    
    plt.scatter(x, y, alpha=0.5, s=80) 
    plt.plot(
        x.iloc[sorted_idx],
        line.iloc[sorted_idx],
        color="red", 
        linewidth=3 
    )

    # Large text sizes without bolding
    plt.xlabel(x_col, fontsize=22, labelpad=15) 
    plt.ylabel("manhattan_midtown_revenue_growth_%", fontsize=22, labelpad=15)
    
    # Title at size 28, fontweight removed
    # Set linespacing to 1.5 or 2.0 to give the multi-line title more air
    plt.title(f"{title}\n{method.capitalize()} $r$ = {corr:.3f}", 
              fontsize=28, 
              pad=30,
              linespacing=1.5)

    # Large Axis Numbers
    plt.tick_params(axis='both', which='major', labelsize=18)

    plt.tight_layout()
    
    # Save as PDF
    # 1. Create a "clean" title for the filename
    # We replace \n with an underscore first, then handle spaces
    safe_title = title.replace('\n', '_').replace(' ', '_')
    
    # 2. Use that safe title for the filename
    filename = f"{safe_title}_{method}.pdf"
    
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    print(f"Successfully saved: {filename}")
    
    plt.show()


# 2. pickup_avg_tip_diff_% vs housing_price_change
# This one works perfectly:
# ==========================================
# GRAPH 1: Revenue Growth vs Housing Price
# ==========================================
plot_with_fit(
    df_corr,
    "housing_price_change",
    "gentrification_161_revenue_growth_%",
    "Manhattan Midtown Center Revenue Growth vs\nHousing Price Change", 
    method="pearson"  # You can change this to "spearman" if you prefer
)

# ==========================================
# GRAPH 2: Trip Growth vs Income Change
# ==========================================
plot_with_fit(
    df_corr,
    "housing_price_change",
    "pickup_avg_tip_diff_%",
    "Average Tip Difference per Pickup Zone vs\n Housing Price Change", 
    method="pearson"  # You can change this to "spearman" if you prefer
)

# --- END OF FILE ---