"""
DataProcessor class for handling and transforming prediction data.

This module defines the DataProcessor class, which processes prediction data
from one or multiple files, prepares a consolidated DataFrame, and provides
methods for filtering that data.
"""

import os
import datetime
import re
import pathlib  # <-- Add pathlib import
import itertools # <-- Add itertools import
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


class DataProcessor:
    """
    Processor for handling and transforming prediction data.

    This class loads prediction files, merges optional metadata, creates derived
    columns (datetime parts, coordinates), and provides a method to filter
    the complete dataset based on various criteria.
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
        self.raw_predictions_df: pd.DataFrame = pd.DataFrame()  # Raw loaded data
        self.metadata_df: Optional[pd.DataFrame] = None  # Processed metadata
        self.complete_df: pd.DataFrame = pd.DataFrame()  # Fully processed, unfiltered data

        self._validate_columns()
        self.load_data()
        self._process_data()  # Initial processing without metadata

        # Gather unique classes after initial processing
        class_col = self.get_column_name("Class")
        if class_col in self.complete_df.columns:
            self.classes = tuple(
                sorted(self.complete_df[class_col].dropna().unique())
            )
        else:
            self.classes = tuple()

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

    def _extract_datetime_from_filename(self, filename: str) -> Optional[datetime.datetime]:
        """Extracts datetime from filename using flexible regex patterns."""
        if not isinstance(filename, str):
            return None

        datetime_patterns = [
            (r'(\d{4})(\d{2})(\d{2})[_]?(\d{2})(\d{2})(\d{2})', '%Y%m%d_%H%M%S'),
            (r'(\d{4})-(\d{2})-(\d{2})[_|-](\d{2})-(\d{2})-(\d{2})', '%Y-%m-%d_%H-%M-%S'),
            (r'(\d{4})\.(\d{2})\.(\d{2})[_](\d{2})\.(\d{2})\.(\d{2})', '%Y.%m.%d_%H.%M.%S'),
            (r'(\d{2})-(\d{2})-(\d{4})[_](\d{2})-(\d{2})-(\d{2})', '%m-%d-%Y_%H-%M-%S'),
            (r'(\d{2})-(\d{2})-(\d{4})[_](\d{2})-(\d{2})-(\d{2})', '%d-%m-%Y_%H-%M-%S'),
            (r'(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})', '%Y_%m_%d_%H_%M_%S'),
        ]

        for pattern, format_str in datetime_patterns:
            match = re.search(pattern, filename)
            if match:
                try:
                    dt_str_parts = match.groups()
                    if format_str == '%Y%m%d_%H%M%S':
                        dt_str = f"{dt_str_parts[0]}{dt_str_parts[1]}{dt_str_parts[2]}_{dt_str_parts[3]}{dt_str_parts[4]}{dt_str_parts[5]}"
                    elif format_str == '%Y-%m-%d_%H-%M-%S':
                        dt_str = f"{dt_str_parts[0]}-{dt_str_parts[1]}-{dt_str_parts[2]}_{dt_str_parts[3]}-{dt_str_parts[4]}-{dt_str_parts[5]}"
                    elif format_str == '%Y.%m.%d_%H.%M.%S':
                        dt_str = f"{dt_str_parts[0]}.{dt_str_parts[1]}.{dt_str_parts[2]}_{dt_str_parts[3]}.{dt_str_parts[4]}.{dt_str_parts[5]}"
                    elif format_str == '%m-%d-%Y_%H-%M-%S':
                        dt_str = f"{dt_str_parts[0]}-{dt_str_parts[1]}-{dt_str_parts[2]}_{dt_str_parts[3]}-{dt_str_parts[4]}-{dt_str_parts[5]}"
                    elif format_str == '%d-%m-%Y_%H-%M-%S':
                        dt_str = f"{dt_str_parts[0]}-{dt_str_parts[1]}-{dt_str_parts[2]}_{dt_str_parts[3]}-{dt_str_parts[4]}-{dt_str_parts[5]}"
                    elif format_str == '%Y_%m_%d_%H_%M_%S':
                        dt_str = f"{dt_str_parts[0]}_{dt_str_parts[1]}_{dt_str_parts[2]}_{dt_str_parts[3]}_{dt_str_parts[4]}_{dt_str_parts[5]}"
                    else:
                        continue

                    parse_format = format_str
                    if pattern == r'(\d{4})(\d{2})(\d{2})[_]?(\d{2})(\d{2})(\d{2})' and '_' not in dt_str:
                        parse_format = '%Y%m%d%H%M%S'
                    elif pattern == r'(\d{4})-(\d{2})-(\d{2})[_|-](\d{2})-(\d{2})-(\d{2})' and '-' in match.group(0) and '_' not in match.group(0):
                        parse_format = '%Y-%m-%d-%H-%M-%S'

                    return datetime.datetime.strptime(dt_str, parse_format)
                except ValueError:
                    continue
        return None

    def _extract_site_id_from_filename(self, filename: str, valid_site_ids: Optional[set] = None) -> Optional[str]:
        """Extracts site ID by checking filename parts against valid IDs. (DEPRECATED - use _match_site_id)"""
        # ↪ deprecated – keep for back-compat, wraps the new logic
        return self._match_site_id(filename, valid_site_ids)

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

    def _process_data(self) -> None:
        """
        Processes raw prediction data to create the complete, unfiltered DataFrame.
        Merges metadata if available. Creates derived columns.
        """
        if self.raw_predictions_df.empty:
            self.complete_df = pd.DataFrame()
            return

        df = self.raw_predictions_df.copy()

        start_time_col = self.get_column_name("Start Time")
        class_col = self.get_column_name("Class")
        conf_col = self.get_column_name("Confidence")
        recording_col = self.get_column_name("Recording")
        correctness_col = self.get_column_name("Correctness")

        if start_time_col not in df.columns:
            df[start_time_col] = 0.0
        if class_col not in df.columns:
            df[class_col] = "Unknown"
        if conf_col not in df.columns:
            df[conf_col] = 0.0
        if correctness_col not in df.columns:
            df[correctness_col] = None

        df[start_time_col] = pd.to_numeric(df[start_time_col], errors='coerce').fillna(0.0)
        df[conf_col] = pd.to_numeric(df[conf_col], errors='coerce').fillna(0.0)

        try:
            # Store the original path string first
            if recording_col in df.columns and df[recording_col].notna().any():
                 df["recording_path_str"] = df[recording_col].astype(str)
            elif "source_file" in df.columns:
                 df["recording_path_str"] = df["source_file"].astype(str)
            else:
                 df["recording_path_str"] = "unknown_path" # Should ideally not happen if source_file is guaranteed

            # Now extract basename and stem
            if recording_col in df.columns and df[recording_col].notna().any():
                df["recording_filename_with_ext"] = df[recording_col].apply(
                    lambda x: os.path.basename(str(x)) if pd.notnull(x) else ""
                )
            elif "source_file" in df.columns:
                # Fallback to source_file if Recording column is missing/empty
                df["recording_filename_with_ext"] = df["source_file"].apply(
                    lambda x: os.path.basename(str(x)) if pd.notnull(x) else ""
                )
            else:
                # If neither Recording nor source_file is useful, assign a default
                df["recording_filename_with_ext"] = "unknown_recording.ext"

            df["recording_filename"] = df["recording_filename_with_ext"].apply(
                lambda x: os.path.splitext(x)[0] if pd.notnull(x) else ""
            )
            # Clean up potential empty strings after extraction
            df["recording_filename"] = df["recording_filename"].str.strip().replace('', pd.NA)
            df["recording_filename_with_ext"] = df["recording_filename_with_ext"].str.strip().replace('', pd.NA)
            df["recording_path_str"] = df["recording_path_str"].str.strip().replace('', pd.NA)


        except Exception as e:
            print(f"Warning: Error processing recording filenames/paths: {e}")
            df["recording_path_str"] = pd.NA
            df["recording_filename"] = pd.NA
            df["recording_filename_with_ext"] = pd.NA

        df['recording_datetime'] = df['recording_filename_with_ext'].apply(self._extract_datetime_from_filename)

        df['prediction_time'] = pd.NaT
        valid_dt_mask = df['recording_datetime'].notna()
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

        if correctness_col in df.columns:
            df[correctness_col] = df[correctness_col].astype(str).str.lower().str.strip()
            correctness_map = {
                'true': True, 'correct': True, '1': True, '1.0': True,
                'false': False, 'incorrect': False, '0': False, '0.0': False,
            }
            df['Correctness'] = df[correctness_col].apply(
                lambda x: correctness_map.get(x, None) if isinstance(x, str) else (x if isinstance(x, bool) else None)
            )
        else:
            df['Correctness'] = None
        df['Correctness'] = df['Correctness'].astype('boolean')

        # --- Site ID Extraction and Metadata Merging ---
        valid_site_ids = set(self.metadata_df['site_name'].dropna().astype(str).unique()) if self.metadata_df is not None else None

        # Apply site ID extraction using the new multi-level matching logic
        df['site_name'] = df['recording_path_str'].apply(
            lambda p: self._match_site_id(p, valid_site_ids)
        )

        if self.metadata_df is not None:
            meta_to_merge = self.metadata_df[['site_name', 'latitude', 'longitude']].drop_duplicates(subset=['site_name']).set_index('site_name')

            # Perform the merge using map
            df['latitude'] = df['site_name'].map(meta_to_merge['latitude'])
            df['longitude'] = df['site_name'].map(meta_to_merge['longitude'])

            unmatched_sites = df['site_name'].notna() & df['latitude'].isna()  # Check where site_name exists but coords are NaN after merge
            if unmatched_sites.any():
                num_unmatched = unmatched_sites.sum()
                sample_unmatched_names = df.loc[unmatched_sites, 'site_name'].unique()[:5]
                print(f"Warning: {num_unmatched} records have site names ('{', '.join(sample_unmatched_names)}', ...) that were extracted from filenames but not found in the metadata (or metadata lacks coords).")
        else:
            df['latitude'] = pd.NA
            df['longitude'] = pd.NA

        # Convert coordinates to numeric, coercing errors
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

        # --- Renaming Logic ---
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

        final_rename_map_filtered = {}
        current_columns = set(df.columns)
        for source_col, target_col in final_rename_map.items():
            if source_col in current_columns:
                if target_col not in current_columns or source_col == target_col:
                    final_rename_map_filtered[source_col] = target_col

        df.rename(columns=final_rename_map_filtered, inplace=True)

        # --- Column Selection Logic ---
        final_columns = [
            'Recording', 'Date', 'Time', 'Year', 'Month', 'Day', 'Hour',
            'Start Time', 'Confidence', 'Class', 'Correctness',
            'Site', 'Latitude', 'Longitude',
            'prediction_time',
            'prediction_minute', 'prediction_second', 'prediction_dayofyear',
            'prediction_weekofyear', 'prediction_weekday',
            'recording_filename_with_ext', 'source_file', 'recording_path_str'
        ]
        if 'Recording' not in df.columns and 'recording_filename' in df.columns:
            if 'recording_filename' not in final_columns:
                final_columns.insert(1, 'recording_filename')

        existing_final_columns = [col for col in final_columns if col in df.columns]
        other_cols = [col for col in df.columns if col not in existing_final_columns]

        self.complete_df = df[existing_final_columns + other_cols].copy()

        duplicates = self.complete_df.columns[self.complete_df.columns.duplicated()]
        if not duplicates.empty:
            print(f"Warning: Duplicate columns found after processing: {duplicates.tolist()}")

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
        """
        Validate, clean and attach site-level metadata (site ID ⇢ lat/lon) to
        the processor, then re-run the internal data-processing pipeline.

        Parameters
        ----------
        metadata_df : pd.DataFrame
            Table that must contain at least the three columns given in
            *site_col*, *lat_col* and *lon_col*.
        site_col, lat_col, lon_col : str, default see signature
            Column names for site ID, latitude and longitude respectively.
        """

        # ── 0. Early-exit on empty ────────────────────────────────────────────────
        if metadata_df is None or metadata_df.empty:
            self.metadata_df = None
            print("Metadata is empty, clearing existing metadata.")
            self._process_data()
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

        # ── 4. Duplicate / inconsistent site IDs ─────────────────────────────────
        meta["lat_round"] = meta["latitude"].round(5)
        meta["lon_round"] = meta["longitude"].round(5)

        dups = meta[meta.duplicated(subset=["site_name"], keep=False)]
        inconsistent_sites_to_remove: set[str] = set()
        consistent_duplicate_count = 0 # Keep track for potential logging if needed

        if not dups.empty:
            for site, grp in dups.groupby("site_name"):
                # Check if all rounded coordinates within the group are the same
                if grp[['lat_round', 'lon_round']].nunique().max() > 1:
                    # Inconsistent coordinates found for this site ID
                    inconsistent_sites_to_remove.add(site)
                else:
                    # Consistent coordinates, count duplicates beyond the first
                    consistent_duplicate_count += len(grp) - 1

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

        # Re-run processing so that predictions gain lat/lon/site info
        self._process_data()


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
            elif mapped_name in self.complete_df.columns:
                return mapped_name

        if field_name in self.complete_df.columns:
            return field_name
        if field_name in self.raw_predictions_df.columns:
            return field_name

        return field_name

    def get_complete_data(self) -> pd.DataFrame:
        """Retrieves a copy of the complete, processed, unfiltered DataFrame."""
        return self.complete_df.copy()

    def get_filtered_data(
        self,
        selected_classes: Optional[List[str]] = None,
        selected_recordings: Optional[List[str]] = None,
        selected_sites: Optional[List[str]] = None,
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
            selected_sites: List of site names to include.
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
                norm_selected_sites = {site.strip() for site in selected_sites if site}
                df = df[df['Site'].isin(norm_selected_sites)]
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
