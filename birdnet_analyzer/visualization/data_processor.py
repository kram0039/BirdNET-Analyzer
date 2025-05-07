"""
DataProcessor class for handling and transforming prediction data.

This module defines the DataProcessor class, which processes prediction data
from one or multiple files, prepares a consolidated DataFrame, and provides
methods for filtering that data.
"""

import os
import datetime
import re
import pathlib
import itertools
import functools
from typing import Dict, List, Optional, Tuple, Any

import warnings
import numpy as np
import pandas as pd
import gradio as gr

from birdnet_analyzer.evaluation.preprocessing.utils import (
    extract_recording_filename,
    extract_recording_filename_from_filename,
    read_and_concatenate_files_in_directory,
)
from birdnet_analyzer.visualization.common import convert_timestamp_to_datetime

# Pre-compiled regex patterns for datetime extraction
DATETIME_PATTERNS = [
    # YYYYMMDD_HHMMSS or YYYYMMDDHHMMSS
    (re.compile(r'(\d{4}\d{2}\d{2}_\d{2}\d{2}\d{2})'), '%Y%m%d_%H%M%S'),
    (re.compile(r'(\d{4}\d{2}\d{2}\d{2}\d{2}\d{2})'), '%Y%m%d%H%M%S'), # No underscore
    # YYYYMMDD_HHMM (New pattern)
    (re.compile(r'(\d{4}\d{2}\d{2}_\d{2}\d{2})'), '%Y%m%d_%H%M'),
    # YYYY-MM-DD_HH-MM-SS or YYYY-MM-DD-HH-MM-SS
    (re.compile(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})'), '%Y-%m-%d_%H-%M-%S'),
    (re.compile(r'(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})'), '%Y-%m-%d-%H-%M-%S'), # Hyphen separator
    # YYYY.MM.DD_HH.MM.SS
    (re.compile(r'(\d{4}\.\d{2}\.\d{2}_\d{2}\.\d{2}\.\d{2})'), '%Y.%m.%d_%H.%M.%S'),
    # MM-DD-YYYY_HH-MM-SS
    (re.compile(r'(\d{2}-\d{2}-\d{4}_\d{2}-\d{2}-\d{2})'), '%m-%d-%Y_%H-%M-%S'),
    # DD-MM-YYYY_HH-MM-SS (Note: same regex as above, relies on format string)
    (re.compile(r'(\d{2}-\d{2}-\d{4}_\d{2}-\d{2}-\d{2})'), '%d-%m-%Y_%H-%M-%S'),
    # YYYY_MM_DD_HH_MM_SS
    (re.compile(r'(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})'), '%Y_%m_%d_%H_%M_%S'),
]

class DataProcessor:
    """
    Processor for handling and transforming prediction data.

    This class loads prediction files, merges optional metadata, creates derived
    columns (datetime parts, coordinates), and provides a method to filter
    the complete dataset based on various criteria.

    Attributes:
        prediction_directory_path (str): Path to the folder containing prediction files.
        prediction_file_name (Optional[str]): Name of a single prediction file, if specified.
        columns_predictions (Dict[str, str]): Column mappings for prediction files.
        raw_predictions_df (pd.DataFrame): Raw data loaded from prediction files.
        predictions_df (pd.DataFrame): Processed prediction data without metadata.
        metadata_df (Optional[pd.DataFrame]): Processed and validated site metadata (site_name, latitude, longitude).
        complete_df (pd.DataFrame): Fully processed, merged data including metadata and derived columns.
        classes (tuple[str]): Sorted tuple of unique class names found in the data.
        metadata_centroid (Optional[Tuple[float, float]]): Mean latitude and longitude of sites in metadata_df, if available.
    """

    # Default column mappings for predictions
    DEFAULT_COLUMNS_PREDICTIONS = {
        "Start Time": "Start Time",
        "End Time": "End Time",
        "Class": "Class",
        "Recording": "Recording",
        "Duration": "Duration",
        "Confidence": "Confidence",
        "Correctness": "Correctness",
    }

    def __init__(
        self,
        prediction_directory_path: str,
        prediction_file_name: Optional[str] = None,
        columns_predictions: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Initializes the DataProcessor by loading and preparing prediction data.

        Args:
            prediction_directory_path (str): Path to the folder containing prediction files.
            prediction_file_name (Optional[str]): Name of a single prediction file to process.
                If None, all `.csv` or `.tsv` files in the directory will be loaded.
            columns_predictions (Optional[Dict[str, str]], optional): Custom column mappings for
                prediction files. If None, default mappings are used.
        """
        self.prediction_directory_path: str = prediction_directory_path
        self.prediction_file_name: Optional[str] = prediction_file_name
        self.columns_predictions: Dict[str, str] = (
            columns_predictions if columns_predictions is not None
            else self.DEFAULT_COLUMNS_PREDICTIONS.copy()
        )

        # Internal DataFrames
        self.raw_predictions_df: pd.DataFrame = pd.DataFrame()  # raw on disk
        self.predictions_df: pd.DataFrame = pd.DataFrame()  # processed – **no metadata**
        self.metadata_df: Optional[pd.DataFrame] = None  # cleaned site meta
        self.complete_df: pd.DataFrame = pd.DataFrame()  # final merged result
        # Cached geometric centre of all valid sites
        self.metadata_centroid: Optional[Tuple[float, float]] = None
        self._predictions_ready = False # Added for lazy loading

        self._validate_columns()
        self.load_data()

    def _validate_columns(self) -> None:
        """Validates essential prediction column mappings."""
        required_columns = ["Start Time", "Class", "Confidence", "Recording"]  # Recording is needed for filename extraction
        missing_pred_columns = [
            col
            for col in required_columns
            if col not in self.columns_predictions or self.columns_predictions[col] is None
        ]
        if missing_pred_columns:
            if "Recording" in missing_pred_columns and self.columns_predictions.get("Recording") is None:
                pass  # Allow missing Recording if source_file will be used
            else:
                raise ValueError(f"Missing or None prediction columns: {', '.join(missing_pred_columns)}")

    def load_data(self) -> None:
        """Loads the raw prediction data into self.raw_predictions_df."""
        if self.prediction_file_name is None:
            self.raw_predictions_df = read_and_concatenate_files_in_directory(
                self.prediction_directory_path
            )
        else:
            full_path = os.path.join(self.prediction_directory_path, self.prediction_file_name)
            try:
                self.raw_predictions_df = pd.read_csv(full_path, sep="\t", on_bad_lines='warn')
                if len(self.raw_predictions_df.columns) <= 1 and '\t' not in open(full_path).readline():
                    print("Reading as TSV resulted in few columns, trying CSV.")
                    self.raw_predictions_df = pd.read_csv(full_path, on_bad_lines='warn')
            except Exception as e_tsv:
                print(f"Failed to read as TSV ({e_tsv}), trying CSV.")
                try:
                    self.raw_predictions_df = pd.read_csv(full_path, on_bad_lines='warn')
                except Exception as e_csv:
                    raise ValueError(f"Failed to read prediction file '{full_path}' as both TSV and CSV: {e_csv}")

        if "source_file" not in self.raw_predictions_df.columns:
            default_source = self.prediction_file_name if self.prediction_file_name else "unknown_source"
            self.raw_predictions_df["source_file"] = default_source

        rename_map = {v: k for k, v in self.columns_predictions.items() if v in self.raw_predictions_df.columns and k != v}
        standard_rename_map = {}
        for standard_name, user_name in self.columns_predictions.items():
            if user_name not in self.raw_predictions_df.columns and standard_name in self.raw_predictions_df.columns:
                pass
            elif user_name in self.raw_predictions_df.columns:
                standard_rename_map[user_name] = standard_name

        self.raw_predictions_df.rename(columns=standard_rename_map, inplace=True)

    @functools.lru_cache(maxsize=None)  # Cache results for performance
    def _extract_datetime_from_filename(self, filename: str) -> Any:  # Return pd.Timestamp or pd.NaT
        """Extracts datetime from filename using pre-compiled regex patterns and pd.to_datetime."""
        if not isinstance(filename, str):
            return pd.NaT  # Return NaT for non-string input

        for compiled_regex, fmt in DATETIME_PATTERNS:
            match = compiled_regex.search(filename)
            if match:
                try:
                    # Use the matched string directly with pd.to_datetime
                    dt_str = match.group(1)  # Group 1 contains the core datetime string
                    dt = pd.to_datetime(dt_str, format=fmt, errors='coerce')
                    if not pd.isna(dt):
                        return dt  # Return the first successful parse as Timestamp
                except ValueError:
                    # pd.to_datetime with errors='coerce' should handle most cases,
                    # but catch explicit ValueError just in case.
                    continue
        return pd.NaT  # Return NaT if no pattern matched successfully

    def _match_site_id(
        self,
        recording_path: str | None,
        valid_site_ids: set[str] | None = None,
    ) -> str | None:
        """
        Match *recording_path* against *valid_site_ids* using three cascading rules.

        Returns the **first** match found (priority 1 → 3) or ``None``.
        """
        if not recording_path or not isinstance(recording_path, str) or not valid_site_ids:
            return None

        # 1️⃣  whole stem -----------------------------------------------------------
        path = pathlib.PurePath(recording_path)
        stem = path.stem                      # e.g., "ABC123_20240503_120000"
        if stem in valid_site_ids:
            return stem

        # 2️⃣  underscore-delimited parts -----------------------------------------
        for part in stem.split("_"):
            if part in valid_site_ids:
                return part

        # 3️⃣  directory names (outer → inner) ------------------------------------
        #    PurePath.parts gives ('/', 'sites', 'ABC123', 'rec', 'xyz.wav') or ('c:\\', 'sites', ...)
        #    Skip root (parts[0]) & filename (parts[-1]), walk top-to-bottom.
        for folder in path.parts[1:-1]:
            if folder in valid_site_ids:
                return folder

        return None

    def _process_predictions(self) -> None:
        """
        Heavy, one-time transformation of raw prediction files.
        Does **not** look at metadata at all.
        Populates self.predictions_df.
        """
        if self.raw_predictions_df.empty:
            self.predictions_df = pd.DataFrame()
            return

        df = self.raw_predictions_df.copy()

        # Get the original names of essential columns from the input DataFrame
        start_time_col = self.get_column_name("Start Time")
        class_col = self.get_column_name("Class")
        conf_col = self.get_column_name("Confidence")
        # recording_col is the name of the column in df that contains the original recording identifier
        recording_col = self.get_column_name("Recording")
        correctness_col = self.get_column_name("Correctness")

        # Ensure essential columns exist, fill with defaults if not
        if start_time_col not in df.columns: df[start_time_col] = 0.0
        if class_col not in df.columns: df[class_col] = "Unknown"
        if conf_col not in df.columns: df[conf_col] = 0.0
        if correctness_col not in df.columns: df[correctness_col] = None # Will be handled later

        # Coerce types
        df[start_time_col] = pd.to_numeric(df[start_time_col], errors='coerce').fillna(0.0)
        df[conf_col] = pd.to_numeric(df[conf_col], errors='coerce').fillna(0.0)

        # --- Derive recording path, filename, and stem ---
        try:
            if recording_col in df.columns and df[recording_col].notna().any():
                 df["recording_path_str"] = df[recording_col].astype(str)
            elif "source_file" in df.columns: # Fallback to source_file
                 df["recording_path_str"] = df["source_file"].astype(str)
            else:
                 df["recording_path_str"] = "unknown_path"

            df["recording_filename_with_ext"] = df["recording_path_str"].apply(
                lambda x: os.path.basename(str(x)) if pd.notnull(x) else ""
            )
            df["recording_filename"] = df["recording_filename_with_ext"].apply(
                lambda x: os.path.splitext(x)[0] if pd.notnull(x) else ""
            )
            # Clean up potential empty strings
            for col_name in ["recording_path_str", "recording_filename_with_ext", "recording_filename"]:
                if col_name in df.columns:
                    df[col_name] = df[col_name].str.strip().replace('', pd.NA)
        except Exception as e:
            print(f"Warning: Error processing recording filenames/paths: {e}")
            df["recording_path_str"] = pd.NA
            df["recording_filename_with_ext"] = pd.NA
            df["recording_filename"] = pd.NA

        # --- Datetime processing ---
        df['recording_datetime'] = df['recording_filename_with_ext'].apply(self._extract_datetime_from_filename)
        df['prediction_time'] = pd.NaT
        valid_dt_mask = df['recording_datetime'].notna()
        if valid_dt_mask.any():
            df.loc[valid_dt_mask, 'prediction_time'] = df.loc[valid_dt_mask].apply(
                lambda row: row['recording_datetime'] + pd.to_timedelta(row[start_time_col], unit='s'),
                axis=1
            )
        df['prediction_time'] = pd.to_datetime(df['prediction_time'], errors='coerce')

        dt_series = df['prediction_time'].dt
        df['prediction_date'] = dt_series.normalize()
        df['prediction_year'] = dt_series.year.astype('Int64')
        df['prediction_month'] = dt_series.month.astype('Int64')
        df['prediction_day'] = dt_series.day.astype('Int64')
        df['prediction_hour'] = dt_series.hour.astype('Int64')
        df['prediction_minute'] = dt_series.minute.astype('Int64')
        df['prediction_second'] = dt_series.second.astype('Int64')
        df['prediction_dayofyear'] = dt_series.dayofyear.astype('Int64')
        df['prediction_weekofyear'] = dt_series.isocalendar().week.astype('Int64')
        df['prediction_weekday'] = dt_series.weekday.astype('Int64')
        df['prediction_time_of_day'] = dt_series.time

        # --- Correctness processing ---
        if correctness_col in df.columns:
            df[correctness_col] = df[correctness_col].astype(str).str.lower().str.strip()
            correctness_map = {
                'true': True, 'correct': True, '1': True, '1.0': True,
                'false': False, 'incorrect': False, '0': False, '0.0': False,
            }
            # Create the boolean 'Correctness' column
            df['Correctness'] = df[correctness_col].apply(
                lambda x: correctness_map.get(x, None) if isinstance(x, str) else (x if isinstance(x, bool) else None)
            )
        else:
            df['Correctness'] = None # Ensure the column exists even if source is missing
        df['Correctness'] = df['Correctness'].astype('boolean')


        # --- Renaming and column finalization for predictions_df ---
        rename_actions = {}

        # Generated date/time parts
        rename_actions.update({
            'prediction_date': 'Date', 'prediction_time_of_day': 'Time',
            'prediction_year': 'Year', 'prediction_month': 'Month',
            'prediction_day': 'Day', 'prediction_hour': 'Hour',
        })

        # 'recording_filename' is the processed version and should become the 'Recording' column.
        # If a column named 'Recording' already exists and it's not 'recording_filename' itself,
        # drop it to prevent duplicates when 'recording_filename' is renamed.
        if 'Recording' in df.columns and 'Recording' != 'recording_filename':
            df.drop(columns=['Recording'], inplace=True, errors='ignore')
        rename_actions['recording_filename'] = 'Recording' # This will be the definitive 'Recording' column

        # Map original input column names (if different) to standard output names.
        # These are the columns like start_time_col, class_col, etc.
        if start_time_col != 'Start Time': rename_actions[start_time_col] = 'Start Time'
        if class_col != 'Class': rename_actions[class_col] = 'Class'
        if conf_col != 'Confidence': rename_actions[conf_col] = 'Confidence'
        # The boolean 'Correctness' column is already named 'Correctness'.
        # The original correctness_col (text version) will be handled below if redundant.

        # Apply renames: only if the source column (key) exists in df and key is different from value.
        active_renames = {
            k: v for k, v in rename_actions.items()
            if k in df.columns and k != v
        }
        df.rename(columns=active_renames, inplace=True)

        # Drop original source columns if they are now superseded by the standardized ones and had different names.
        # The original 'recording_col' (e.g., "Path", "File Name") is superseded by the new 'Recording' column.
        # If its original name (recording_col) is still in df and is not 'Recording', drop it.
        if recording_col in df.columns and recording_col != 'Recording':
            df.drop(columns=[recording_col], inplace=True, errors='ignore')

        # If the original correctness_col (text version) still exists,
        # and it's not named 'Correctness' (which is the name of the boolean column), drop it.
        if correctness_col in df.columns and correctness_col != 'Correctness' and 'Correctness' in df.columns:
            df.drop(columns=[correctness_col], inplace=True, errors='ignore')

        self.predictions_df = df

    def _process_metadata(self, metadata_df: pd.DataFrame,
                          site_col="Site", lat_col="Latitude", lon_col="Longitude") -> None:
        """Clean & validate site metadata, cache centroid."""
        # Clear previous centroid before processing new metadata
        self.metadata_centroid = None

        # ── 0. Early-exit on empty ────────────────────────────────────────────────
        if metadata_df is None or metadata_df.empty:
            self.metadata_df = None
            print("Metadata is empty, clearing existing metadata.")
            return

        # ── 1. Column checks ─────────────────────────────────────────────────────
        required_cols = {site_col, lat_col, lon_col}
        missing = required_cols.difference(metadata_df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns in metadata: {', '.join(sorted(missing))}"
            )

        # Work on a copy so we do not mutate the caller’s DataFrame
        meta = metadata_df.copy()

        # ── 2. Standardise column names & dtypes ─────────────────────────────────
        meta.rename(
            columns={site_col: "site_name", lat_col: "latitude", lon_col: "longitude"},
            inplace=True,
        )

        # ↳ keep ALL site IDs as canonical strings
        meta["site_name"] = meta["site_name"].astype(str).str.strip()

        # ── 3. Coordinate cleanup / validation ───────────────────────────────────
        meta["latitude"] = meta["latitude"].apply(self._clean_coordinate)
        meta["longitude"] = meta["longitude"].apply(self._clean_coordinate)

        bad_lat = ~meta["latitude"].between(-90, 90, inclusive="both") & meta["latitude"].notna()
        bad_lon = ~meta["longitude"].between(-180, 180, inclusive="both") & meta["longitude"].notna()
        if bad_lat.any() or bad_lon.any():
            bad_rows = meta[bad_lat | bad_lon]
            err_msg = (
                "Invalid coordinates found in metadata. "
                "Latitude must be in [-90, 90] and longitude in [-180, 180].\n"
                "Problematic rows:\n"
                f"{bad_rows[['site_name', 'latitude', 'longitude']].head().to_string(index=False)}"
            )
            gr.Error(err_msg)
            print(f"ERROR: {err_msg}")
            meta = meta[~(bad_lat | bad_lon)]
            gr.Warning(f"Removed {len(bad_rows)} rows with invalid coordinates from metadata.")

        # ── 4. Duplicate / inconsistent site IDs ────────────────────────────────
        # Use an absolute tolerance instead of rounding
        TOL = 5e-5              # ≈ 5 metre in latitude / longitude

        dups = meta[meta.duplicated('site_name', keep=False)]
        inconsistent_sites_to_remove: set[str] = set()
        consistent_duplicate_count = 0

        def coords_consistent(grp, tol=TOL) -> bool:
            """True if all coords in *grp* are within *tol* of one another."""
            lat_ok = grp['latitude'].max() - grp['latitude'].min() < tol
            lon_ok = grp['longitude'].max() - grp['longitude'].min() < tol
            return lat_ok and lon_ok

        if not dups.empty:
            for site, grp in dups.groupby('site_name'):
                if coords_consistent(grp):
                    # same spot → keep one copy
                    consistent_duplicate_count += len(grp) - 1
                else:
                    # coordinates differ by > tol → drop everything for this site
                    inconsistent_sites_to_remove.add(site)

        if inconsistent_sites_to_remove:
            msg = (
                "Found duplicate site IDs with different coordinates in metadata. "
                "These sites will be removed:\n"
                f"{', '.join(sorted(list(inconsistent_sites_to_remove)))}\n"
                "Please correct the metadata file."
            )
            gr.Warning(msg)
            print(f"WARNING: {msg}")
            # Remove all rows associated with inconsistent site IDs
            meta = meta[~meta['site_name'].isin(inconsistent_sites_to_remove)]

        # Handle consistent duplicates (keep only the first occurrence)
        # This needs to run *after* inconsistent sites are removed
        if meta.duplicated(subset=["site_name"], keep=False).any():
            # Recalculate count after removing inconsistent sites
            n_consistent_dupes_to_drop = meta.duplicated(subset=["site_name"], keep='first').sum()
            meta = meta.drop_duplicates(subset=["site_name"], keep="first")
            if n_consistent_dupes_to_drop > 0:
                print(f"Removed {n_consistent_dupes_to_drop} duplicate metadata entries for sites with identical coordinates.")

        # Drop temporary columns before proceeding
        meta = meta.drop(columns=["lat_round", "lon_round"], errors='ignore')

        # ── 5. Drop sites without coordinates ────────────────────────────────────
        no_coord = meta["latitude"].isna() | meta["longitude"].isna()
        if no_coord.any():
            n_missing = int(no_coord.sum())
            sample_missing = meta.loc[no_coord, "site_name"].unique()[:5]
            print(
                "Warning: "
                f"{n_missing} sites in metadata have missing/invalid coordinates "
                f"(e.g., '{', '.join(map(str, sample_missing))}') and will be "
                "excluded from spatial analysis."
            )
            meta = meta.dropna(subset=["latitude", "longitude"])

        # ── 6. Persist & re-process main predictions ─────────────────────────────
        self.metadata_df = meta[["site_name", "latitude", "longitude"]].copy()
        print(
            f"Successfully processed metadata for "
            f"{len(self.metadata_df)} unique sites with valid coordinates."
        )

        # Calculate and cache centroid if metadata is valid and non-empty
        # Assumes self.metadata_df contains cleaned, numeric lat/lon at this point
        if self.metadata_df is not None and not self.metadata_df.empty:
            try:
                mean_lat = self.metadata_df['latitude'].mean()
                mean_lon = self.metadata_df['longitude'].mean()
                if pd.notna(mean_lat) and pd.notna(mean_lon):
                    # Basic check for valid range
                    if -90 <= mean_lat <= 90 and -180 <= mean_lon <= 180:
                        self.metadata_centroid = (mean_lat, mean_lon)
                        print(f"Cached metadata centroid: ({mean_lat:.4f}, {mean_lon:.4f})")
                    else:
                        print(f"Warning: Calculated centroid ({mean_lat:.4f}, {mean_lon:.4f}) is outside valid geographic bounds. Centroid not cached.")
                        self.metadata_centroid = None
                else:
                    print("Warning: Could not calculate metadata centroid (mean lat/lon is NaN).")
                    self.metadata_centroid = None
            except Exception as e:
                print(f"Error during centroid calculation: {e}. Centroid not cached.")
                self.metadata_centroid = None

    def _finalize_output_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Shared by predictions-only and merge paths – rename/output order."""
        final_rename_map = {
            'prediction_date': 'Date',
            'prediction_time_of_day': 'Time',
            'prediction_year': 'Year',
            'prediction_month': 'Month',
            'prediction_day': 'Day',
            'prediction_hour': 'Hour',
            'site_name': 'Site',
            'latitude': 'Latitude',
            'longitude': 'Longitude',
            'Start Time': 'Start Time',
            'Class': 'Class',
            'Confidence': 'Confidence',
            'Correctness': 'Correctness',
            'recording_filename': 'Recording',
        }
        df = df.rename(columns={k: v for k, v in final_rename_map.items() if k in df.columns and k != v})

        wanted = [
            'Recording', 'Date', 'Time', 'Year', 'Month', 'Day', 'Hour',
            'Start Time', 'Confidence', 'Class', 'Correctness',
            'Site', 'Latitude', 'Longitude',
            'prediction_time',
            'prediction_minute', 'prediction_second', 'prediction_dayofyear',
            'prediction_weekofyear', 'prediction_weekday',
            'recording_filename_with_ext', 'source_file', 'recording_path_str'
        ]
        ordered = [c for c in wanted if c in df.columns] + [c for c in df.columns if c not in wanted]
        return df.loc[:, ordered]

    def _merge_predictions_metadata(self) -> None:
        """Join already-processed predictions with (optional) metadata."""
        if self.predictions_df.empty:
            self.complete_df = pd.DataFrame()
            return

        # start from a copy so we never mutate predictions_df
        df = self.predictions_df.copy()

        if self.metadata_df is None or self.metadata_df.empty:
            # No metadata – the prediction table *is* the final result (after final column ordering)
            self.complete_df = self._finalize_output_columns(df)
            return

        # -------- site matching --------
        valid_site_ids = set(self.metadata_df['site_name'])
        # Ensure 'recording_path_str' exists from _process_predictions
        if 'recording_path_str' not in df.columns:
            # This case should ideally be handled by _process_predictions ensuring the column exists
            # For robustness, if it's missing, we can't match sites.
            print("Warning: 'recording_path_str' missing from predictions_df, cannot merge site metadata.")
            self.complete_df = self._finalize_output_columns(df) # Finalize what we have
            return

        df['site_name'] = df['recording_path_str'].apply(
            lambda p: self._match_site_id(p, valid_site_ids)
        )

        meta_lookup = self.metadata_df.set_index('site_name')[['latitude', 'longitude']]
        df['latitude']  = df['site_name'].map(meta_lookup['latitude'])
        df['longitude'] = df['site_name'].map(meta_lookup['longitude'])

        # Convert coordinates to numeric, coercing errors
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

        # -------- final tidy-up (rename/order columns) --------
        df = self._finalize_output_columns(df)

        self.complete_df = df

        # --- Warning for unmatched sites (moved here) ---
        try:
            if (
                not self.complete_df.empty
                and "Recording" in self.complete_df.columns
                and "Site"      in self.complete_df.columns # Site column should exist now if metadata was provided
            ):
                mask_no_site = self.complete_df["Site"].isna()
                # Only warn if metadata was provided but some sites still couldn't be matched
                if mask_no_site.any() and self.metadata_df is not None and not self.metadata_df.empty:
                    missing_recs = (
                        self.complete_df.loc[mask_no_site, "Recording"]
                        .dropna()
                        .astype(str)
                        .unique()
                    )
                    if len(missing_recs) > 0:
                        num_missing = len(missing_recs)
                        # Use 'Site' column here as 'site_name' might have been renamed by _finalize_output_columns
                        sample_unmatched_site_names_from_preds = self.complete_df.loc[mask_no_site & self.complete_df['Site'].notna(), 'Site'].unique()[:3]

                        warning_message_parts = [f"No site could be matched for {num_missing} recording(s)."]
                        if len(sample_unmatched_site_names_from_preds) > 0:
                             warning_message_parts.append(f"  (Examples of site names from recordings that were not in metadata or had no coords: {', '.join(sample_unmatched_site_names_from_preds)})")

                        if num_missing > 10:
                            display_recs = missing_recs[:10]
                            rec_list = "\n".join(display_recs)
                            warning_message_parts.append(f"  Showing first 10 affected recordings:\n{rec_list}\n  ... (and {num_missing - 10} more)")
                        else:
                            rec_list = "\n".join(missing_recs)
                            warning_message_parts.append(f"  Affected recordings:\n{rec_list}")
                        gr.Warning("\n".join(warning_message_parts))
        except Exception as exc:
            print(f"WARNING-handler for unmatched sites failed: {exc!r}")

    def _clean_coordinate(self, value):
        """Cleans coordinate values (string or numeric) to float or None."""
        if pd.isna(value):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        try:
            cleaned_val = str(value).strip().replace(',', '.')
            cleaned_val = re.sub(r'[^\d.-]+', '', cleaned_val)
            if cleaned_val.count('.') > 1:
                parts = cleaned_val.split('.')
                cleaned_val = parts[0] + '.' + parts[1]
            if '-' in cleaned_val[1:]:
                cleaned_val = cleaned_val.replace('-', '')
                if str(value).strip().startswith('-'):
                    cleaned_val = '-' + cleaned_val

            return float(cleaned_val)
        except (ValueError, TypeError):
            return None

    def set_metadata(
        self,
        metadata_df: pd.DataFrame,
        site_col: str = "Site",
        lat_col: str = "Latitude",
        lon_col: str = "Longitude",
    ) -> None:
        """Public entry point – cleans metadata then (re)merges."""
        self._process_metadata(metadata_df, site_col, lat_col, lon_col)
        self._merge_predictions_metadata()

    def _ensure_predictions(self):
        if not self._predictions_ready:
            self._process_predictions()
            self._merge_predictions_metadata() # This should be called after _process_predictions and potentially after metadata is set
            
            # Gather unique classes after processing
            class_col = self.get_column_name("Class")
            if class_col in self.predictions_df.columns and not self.predictions_df.empty:
                self.classes: Tuple[str, ...] = tuple(
                    sorted(self.predictions_df[class_col].dropna().astype(str).unique())
                )
            else:
                self.classes = tuple()
            
            self._predictions_ready = True

    def get_column_name(self, field_name: str) -> str:
        """
        Retrieves the actual column name in the DataFrame for a standard field.
        Falls back to the field_name itself if no mapping exists or column not present.
        """
        if field_name is None:
            raise TypeError("field_name cannot be None.")

        mapped_name = self.columns_predictions.get(field_name)

        if mapped_name:
            if mapped_name in self.raw_predictions_df.columns:
                return mapped_name
            if mapped_name in self.predictions_df.columns:
                return mapped_name
            if mapped_name in self.complete_df.columns:
                return mapped_name

        if field_name in self.complete_df.columns:
            return field_name
        if field_name in self.predictions_df.columns:
            return field_name
        if field_name in self.raw_predictions_df.columns:
            return field_name

        return field_name

    def get_complete_data(self) -> pd.DataFrame:
        self._ensure_predictions()
        return self.complete_df.copy()

    def get_filtered_data(
        self,
        selected_classes: Optional[List[str]] = None,
        selected_recordings: Optional[List[str]] = None,
        selected_sites: Optional[List[Any]] = None, # Allow Any type for None
        date_range_start: Optional[Any] = None,
        date_range_end: Optional[Any] = None,
        time_start: Optional[datetime.time] = None,
        time_end: Optional[datetime.time] = None,
        correctness_mode: str = "All",
        class_thresholds: Optional[pd.DataFrame] = None,
        min_confidence: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Filters the complete dataset based on the provided criteria.

        Args:
            selected_classes: List of classes to include.
            selected_recordings: List of recording filenames (without extension) to include.
            selected_sites: List of site names to include. Can contain None to filter for null sites.
            date_range_start: Start date for filtering (inclusive).
            date_range_end: End date for filtering (inclusive).
            time_start: Start time of day for filtering (inclusive).
            time_end: End time of day for filtering (inclusive).
            correctness_mode: Filter by correctness flag ("All", "Correct", "Incorrect", "Unspecified").
            class_thresholds: DataFrame with class-specific confidence thresholds.
            min_confidence: A global minimum confidence threshold (use class_thresholds preferably).

        Returns:
            A filtered pandas DataFrame.
        """
        self._ensure_predictions()

        if self.complete_df.empty:
            return pd.DataFrame()

        df = self.complete_df.copy()

        if selected_recordings:
            norm_selected_recordings = {rec.lower().strip() for rec in selected_recordings if rec}
            if 'Recording' in df.columns:
                df['Recording_lower'] = df['Recording'].astype(str).str.lower().str.strip()
                df = df[df['Recording_lower'].isin(norm_selected_recordings)]
                df = df.drop(columns=['Recording_lower'])
            else:
                print("Warning: 'Recording' column not found for filtering.")

        if selected_sites:
            if 'Site' in df.columns:
                filter_for_null_sites = None in selected_sites
                # Convert actual site names to string for consistent matching, excluding None
                actual_site_names = {str(site).strip() for site in selected_sites if site is not None}

                conditions = []
                if actual_site_names:
                    # Filter by actual site names. Ensure df['Site'] is also treated as string for comparison.
                    # NaNs in df['Site'] will become 'nan' string here, which is fine as actual_site_names won't contain pd.NA or float nan.
                    conditions.append(df['Site'].astype(str).isin(actual_site_names))
                
                if filter_for_null_sites:
                    conditions.append(df['Site'].isna())
                
                if conditions:
                    combined_mask = pd.Series(False, index=df.index)
                    for cond in conditions:
                        combined_mask |= cond
                    df = df[combined_mask]
                elif not actual_site_names and not filter_for_null_sites: # e.g. selected_sites was an empty list
                    pass # No site filter to apply if list was empty after processing
                elif not conditions: # e.g. selected_sites was [None] but no null sites, or ["KnownSite"] but no such site
                    df = df.iloc[0:0] # Return empty DataFrame if no conditions matched

            else:
                print("Warning: 'Site' column not found for filtering.")

        if selected_classes:
            if 'Class' in df.columns:
                df = df[df['Class'].isin(selected_classes)]
            else:
                print("Warning: 'Class' column not found for filtering.")

        if date_range_start is not None or date_range_end is not None:
            if 'Date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Date']):
                start_date = convert_timestamp_to_datetime(date_range_start)
                end_date = convert_timestamp_to_datetime(date_range_end)

                start_date = pd.Timestamp(start_date.date()) if start_date else None
                end_date = pd.Timestamp(end_date.date()) if end_date else None

                if start_date and end_date:
                    df = df[df['Date'].between(start_date, end_date, inclusive='both')]
                elif start_date:
                    df = df[df['Date'] >= start_date]
                elif end_date:
                    df = df[df['Date'] <= end_date]
            else:
                print("Warning: 'Date' column not found or not datetime type for filtering.")

        if time_start is not None or time_end is not None:
            if 'Time' in df.columns:
                is_time_object = df['Time'].apply(lambda x: isinstance(x, datetime.time)).all()
                if is_time_object:
                    if time_start and time_end:
                        if time_start <= time_end:
                            df = df[df['Time'].between(time_start, time_end, inclusive='both')]
                        else:
                            df = df[(df['Time'] >= time_start) | (df['Time'] <= time_end)]
                    elif time_start:
                        df = df[df['Time'] >= time_start]
                    elif time_end:
                        df = df[df['Time'] <= time_end]
                else:
                    print("Warning: 'Time' column does not contain time objects for filtering.")
            else:
                print("Warning: 'Time' column not found for filtering.")

        if correctness_mode != "All":
            if 'Correctness' in df.columns and pd.api.types.is_bool_dtype(df['Correctness'].dtype):
                if correctness_mode == "Correct":
                    df = df[df['Correctness'] == True]
                elif correctness_mode == "Incorrect":
                    df = df[df['Correctness'] == False]
                elif correctness_mode == "Unspecified":
                    df = df[df['Correctness'].isna()]
            else:
                print(f"Warning: 'Correctness' column not found or not boolean type for filtering by '{correctness_mode}'.")

        if class_thresholds is not None and not class_thresholds.empty:
            if 'Class' in df.columns and 'Confidence' in df.columns:
                if 'Class' in class_thresholds.columns and 'Threshold' in class_thresholds.columns:
                    thresholds_map = class_thresholds.set_index('Class')['Threshold']
                    thresholds_map = pd.to_numeric(thresholds_map, errors='coerce').fillna(0.0)

                    df['class_threshold'] = df['Class'].map(thresholds_map)
                    df['class_threshold'] = df['class_threshold'].fillna(0.01).clip(lower=0.01)

                    df = df[df['Confidence'] >= df['class_threshold']]
                    df = df.drop(columns=['class_threshold'])
                else:
                    print("Warning: Class thresholds DataFrame missing 'Class' or 'Threshold' column.")
            else:
                print("Warning: 'Class' or 'Confidence' column missing for applying class thresholds.")
        elif min_confidence is not None:
            if 'Confidence' in df.columns:
                df = df[df['Confidence'] >= min_confidence]
            else:
                print("Warning: 'Confidence' column not found for applying global minimum confidence.")

        return df
