import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from scipy.stats import zscore
from pathlib import Path


def plot_tlc_heatmap(data_root, year, value_column):

    base_path = Path(data_root)

    if value_column == "B25077_001E":
        category_folder = "Gentrification/Median Value of a House"

    if value_column == "S1902_C01_001E":
        category_folder = "Gentrification/Mean Income"

    if value_column == "B15003_022E":
        category_folder = "Gentrification/Total number of people with a Bachelor's Degree"
    

    category = category_folder[15:]

    TLCzones = gpd.read_file(base_path / "taxi_zones_extracted/taxi_zones/taxi_zones.shp")

    df = pd.read_csv(base_path / category_folder / year / "data.csv")
    df_allowed = pd.read_excel(base_path / "Gentrification/ZIPs_in_NYC.xlsx")
    df_ZIP_TLC = pd.read_excel(base_path / "Gentrification/ZIP to TLC.xlsx")


    #Clean zip code census area data
    df_allowed["ZIP"] = df_allowed["ZIP"].astype(str).str.zfill(5)

    df = df[df["ZIP"].notna()]
    df["ZIP"] = pd.to_numeric(df["ZIP"], errors="coerce")
    df = df[df["ZIP"].notna()]
    df["ZIP"] = df["ZIP"].astype(int).astype(str).str.zfill(5)

    # Filter to NYC ZIPs
    df_filtered = df[df["ZIP"].isin(df_allowed["ZIP"])].copy()

    ZIPzones = gpd.read_file(base_path / "Gentrification/qgis/map1/map.shp")

    # Ensure ZIP formats match
    ZIPzones["ZCTA5CE20"] = ZIPzones["ZCTA5CE20"].astype(str).str.zfill(5)
    df_filtered["ZIP"] = df_filtered["ZIP"].astype(str).str.zfill(5)




    # Merge ZIP data with geometry
    gdf_zip = ZIPzones.merge(
        df_filtered,
        left_on="ZCTA5CE20",
        right_on="ZIP",
        how="inner"
    )



    gdf_zip[value_column] = (
        gdf_zip[value_column]
        .replace("-", 0)
        .astype(str)  
        .str.replace(",", "", regex=False)
    )
    gdf_zip[value_column] = pd.to_numeric(gdf_zip[value_column], errors="coerce")


    #Plot just the original data on the ZIP census geometry 
    plt.rcParams.update({'font.size': 16}) 

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    gdf_zip.plot(
        column=value_column,
        cmap="YlOrRd",
        linewidth=0.2,
        legend=True,
        ax=ax,
        vmin=0,
        vmax=gdf_zip[value_column].max()  
    )

    # 2. Increase the title fontsize manually (e.g., from 16 to 24)
    ax.set_title(f"Mean Income per ZCTA ({year})", fontsize=24)
    ax.axis("off")

    fig.savefig(f"../data/output/{category},{year},ZIP.pdf", dpi=300, bbox_inches="tight")    

    plt.show()


    # Clean ZIP mapping
    df_ZIP_TLC["ZCTA5CE20"] = pd.to_numeric(df_ZIP_TLC["ZCTA5CE20"], errors="coerce")
    df_ZIP_TLC = df_ZIP_TLC[df_ZIP_TLC["ZCTA5CE20"].notna()]
    df_ZIP_TLC["ZCTA5CE20"] = df_ZIP_TLC["ZCTA5CE20"].astype(int).astype(str).str.zfill(5)

   #Merge with data
    df_mapping = df_ZIP_TLC.merge(
        df_filtered[["ZIP", value_column]],
        left_on="ZCTA5CE20",
        right_on="ZIP",
        how="left"
    )

    #Clean
    df_mapping[value_column] = pd.to_numeric(df_mapping[value_column], errors="coerce")

    df_mapping["ratio2"] = pd.to_numeric(df_mapping["ratio2"], errors="coerce")

    valid_mask = (df_mapping[value_column].notna()) & (df_mapping[value_column] != 0)

    df_valid = df_mapping[valid_mask].copy()


    # Calculate weighed value based on the area taken up
    df_valid["WeightedValue"] = df_valid[value_column] * df_valid["ratio2"]

    # Aggregate per TLC zone
    df_zone = df_valid.groupby("OBJECTID").apply(
        lambda g: g["WeightedValue"].sum() / g["ratio2"].sum()
        if g["ratio2"].sum() > 0 else None
    ).reset_index(name="Value")



    # Merge with geometry
    TLCzones["OBJECTID"] = TLCzones["OBJECTID"].astype(int)
    gdf = TLCzones.merge(df_zone, on="OBJECTID", how="left")

    

    #Plot on TLC geometry 
    plt.rcParams.update({'font.size': 16}) 

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    gdf.plot(
        column="Value",
        cmap="YlOrRd",
        linewidth=0.2,
        legend=True,
        ax=ax
        # Notice I removed legend_kwds here!
    )

    ax.set_title(f"Mean Income per TLC Zone ({year})", fontsize=24)
    ax.axis("off")


    fig.savefig(f"../data/output/Mean Income {year} TLC.pdf", dpi=300, bbox_inches="tight")    


    plt.show()

    return gdf


#B25077_001E - median house value
#S1902_C01_001E - mean income
#B15003_022E - total with bachelors degree

plot_tlc_heatmap(
    "../data/",
    "2019",
    "S1902_C01_001E"
)