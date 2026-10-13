import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIRECTORY = PROJECT_ROOT / "data"
HVFHV_COLUMNS = [
    "hvfhs_license_num",
    "dispatching_base_num",
    "originating_base_num",
    "request_datetime",
    "on_scene_datetime",
    "pickup_datetime",
    "dropoff_datetime",
    "PULocationID",
    "DOLocationID",
    "trip_miles",
    "trip_time",
    "base_passenger_fare",
    "tolls",
    "bcf",
    "sales_tax",
    "congestion_surcharge",
    "airport_fee",
    "tips",
    "driver_pay",
    "shared_request_flag",
    "shared_match_flag",
    "access_a_ride_flag",
    "wav_request_flag",
    "wav_match_flag",
    "cbd_congestion_fee",
]

ARREST_DATA_COLUMNS = [
    "ARREST_KEY",
    "ARREST_DATE",
    "PD_DESC",
    "KY_CD",
    "OFNS_DESC",
    "LAW_CODE",
    "AGE_GROUP",
    "PERP_SEX",
    "PERP_RACE",
    "Latitude",
    "Longitude",
    "LocationID",
    "Borough",
    "WHITE_COLLAR_CRIME"
]

# Class which is responsible for loading TLC HVFHV trip data from parquet files. 
# It can load individual files or all files for a given year, with optional column selection and row sampling.
# Example usage:
# loader = TLCHVLoader(year=2025)
# df = loader.load_all_files_in_folder(columns=["pickup_datetime", "dropoff_datetime", "trip_miles"], max_n_rows=100000, seed=42)

class TLCHVLoader():
    def __init__(self, year = 2025, data_directory = DATA_DIRECTORY):
        """
        Args:
            year: The year of trip data to load.
            data_directory: Root data directory. Expects subfolder fhvhv_{year}/.
        """

        self.year = int(year)
        self.data_directory = data_directory / f"fhvhv_{self.year}"
        self.columns = HVFHV_COLUMNS

    def _files(self):
        """
        Get and sort all parquet files for the given year.
        """

        pattern = f"fhvhv_tripdata_{self.year}-*.parquet"
        files = list(self.data_directory.glob(pattern))
        files.sort()

        if not files or len(files) == 0:
            raise FileNotFoundError(f"No parquet files found in {self.data_directory} matching pattern {pattern}")
        return files

    def _select_files(self, months = None):
        """
        Get parquet files for the year, optionally filtered to specific months.

        Args:
            months: Optional iterable of month numbers (1-12).

        Returns:
            list[Path]
        """

        files = self._files()

        if months is None:
            return files

        month_set = set()
        for month in months:
            month_int = int(month)
            if month_int < 1 or month_int > 12:
                raise ValueError(f"Month must be between 1 and 12, got {month}")
            month_set.add(month_int)

        selected_files = []
        for file_path in files:
            file_month = int(file_path.stem.split("-")[-1])
            if file_month in month_set:
                selected_files.append(file_path)

        if len(selected_files) == 0:
            raise FileNotFoundError(f"No parquet files matched months={sorted(month_set)} in {self.data_directory}")

        return selected_files

    def load_parquet_file(self, file_path, columns = None, max_n_rows = None, seed = 42):
        """
        Load a single parquet file.

        Args:
            file_path: Path to the parquet file.
            columns: List of columns to load. Defaults to all HVFHV columns.
            max_n_rows: If set, randomly sample this many rows. Capped at file length.
            seed: Random seed for reproducibility. Pass None for a random sample.

        Returns:
            pd.DataFrame
        """

        file_path = Path(file_path)

        if columns is None:
            columns_to_read = self.columns
        else:
            columns_to_read = columns
        try:
            df = pd.read_parquet(file_path, columns=columns_to_read, engine="pyarrow")
        except Exception as e:
            print(f"Error loading {file_path.name}: {e}")
            return pd.DataFrame(columns=columns_to_read)
        
        if max_n_rows is not None:
            n = min(max_n_rows, len(df))
            df = df.sample(n=n, random_state=seed).reset_index(drop=True)
            print(f"Sampled {n:,} rows (seed={seed})")
        
        print(f"Loaded {len(df):,} rows from {file_path.name}")
        return df

    def load_files(self, columns = None, max_n_rows = None, seed = 42, months = None):
        """
        Load and concatenate parquet files for the year.

        When max_n_rows is set, sampling is done per file (max_n_rows // num_files + 1 rows each) to keep memory low, then the combined result is trimmed to exactly max_n_rows.
        Note: this is approximate uniform sampling, where months with more trips are slightly underrepresented. 
        For true uniform sampling across the full dataset, ideally use HPC.

        Args:
            columns: List of columns to load. Defaults to all HVFHV columns.
            max_n_rows: If set, randomly sample this many rows from the full dataset.
            seed: Random seed for reproducibility. Pass None for a random sample.
            months: Optional iterable of month numbers (1-12) to limit which parquet files are read.

        Returns:
            pd.DataFrame
        """

        files = self._select_files(months=months)
        print(f"Found {len(files)} files in {self.data_directory}")

        if max_n_rows is not None:
            rows_per_file = (max_n_rows // len(files)) + 1
        else:
            rows_per_file = None

        dfs = []
        for file_path in files:
            df_part = self.load_parquet_file(file_path, columns=columns, max_n_rows=rows_per_file, seed=seed)
            dfs.append(df_part)

        if len(dfs) == 0:
            if columns is None:
                return pd.DataFrame(columns=self.columns)
            else:
                return pd.DataFrame(columns=columns)
            
        if max_n_rows is not None:
            total_rows = 0
            for df in dfs:
                total_rows += len(df)
            n = min(max_n_rows, total_rows)
            df = pd.concat(dfs, ignore_index=True)
            df = df.sample(n=n, random_state=seed).reset_index(drop=True)
            print(f"Sampled {n:,} rows (seed={seed})")
        else:
            df = pd.concat(dfs, ignore_index=True)

        print(f"Total rows: {len(df):,}")
        return df

    def load_all_files_in_folder(self, columns = None, max_n_rows = None, seed = 42):
        """
        Backwards-compatible wrapper for loading all parquet files for the year.
        """

        return self.load_files(columns=columns, max_n_rows=max_n_rows, seed=seed, months=None)

# Class which is responsible for loading NYPD arrest data from a CSV file.
# It supports optional column selection and row sampling.
# Example usage:
# arrest_loader = ArrestDataLoader()
# df_arrest = arrest_loader.load_csv(max_n_rows=10_000, seed=42)

class ArrestDataLoader():
    def __init__(self, data_directory=DATA_DIRECTORY):
        """
        Args:
            data_directory: Root data directory. Expects arrests.csv inside.
        """
        self.data_directory = data_directory
        self.columns = ARREST_DATA_COLUMNS

    def _file(self):
        """Find the arrest data CSV file."""
        files = list(self.data_directory.glob("arrests.csv"))
        if not files or len(files) == 0:
            raise FileNotFoundError(f"No arrest data CSV found in {self.data_directory}")
        return files[0]

    def load_csv(
        self,
        columns=None,
        max_n_rows=None,
        seed=42,
        file_path=None,
        delimiter=";",
        parse_dates=("ARREST_DATE",),
    ):
        """
        Load the NYPD arrest data CSV.

        Args:
            columns: List of columns to load. Defaults to all arrest columns.
            max_n_rows: If set, randomly sample this many rows.
            seed: Random seed for reproducibility. Pass None for a random sample.

        Returns:
            pd.DataFrame
        """
        if file_path is None:
            file_path = self._file()
        else:
            file_path = Path(file_path)
        if columns is not None:
            columns_to_read = columns
        else:
            columns_to_read = self.columns

        try:
            print(f"Loading {file_path.name}...")
            read_csv_kwargs = {"usecols": columns_to_read, "sep": delimiter}
            if parse_dates is not None:
                read_csv_kwargs["parse_dates"] = parse_dates
            df = pd.read_csv(file_path, **read_csv_kwargs)
        except Exception as e:
            print(f"Error loading {file_path.name}: {e}")
            return pd.DataFrame(columns=columns_to_read)

        if max_n_rows is not None:
            n = min(max_n_rows, len(df))
            df = df.sample(n=n, random_state=seed).reset_index(drop=True)
            print(f"Sampled {n:,} rows (seed={seed})")

        print(f"Loaded {len(df):,} rows from {file_path.name}")
        return df
