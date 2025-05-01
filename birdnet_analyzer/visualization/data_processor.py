"""
DataProcessor class for handling and transforming prediction data.

This module defines the DataProcessor class, which processes prediction data
from one or multiple files, prepares a consolidated DataFrame, and provides
methods for filtering that data.
"""

import os
import datetime
import re
from typing import Dict, List, Optional, Tuple

import warnings
import numpy as np
import pandas as pd
import gradio as gr  # Add this import

from birdnet_analyzer.evaluation.preprocessing.utils import (
    extract_recording_filename,
    extract_recording_filename_from_filename,
    read_and_concatenate_files_in_directory,
)


class DataProcessor:
    """
    Processor for handling and transforming prediction data.

    This class loads prediction files (either a single file or all files in
    a specified directory), prepares them into a unified DataFrame, and
    provides methods to filter the prediction data by recording, class,
    or confidence.
    """

    # Default column mappings for predictions
    DEFAULT_COLUMNS_PREDICTIONS = {
        "Start Time": "Start Time",
        "End Time": "End Time",  # Keep this for backwards compatibility
        "Class": "Class",
        "Recording": "Recording",
        "Duration": "Duration",
        "Confidence": "Confidence",
        "Correctness": "Correctness",  # Add Correctness field with default column name
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
                prediction files (e.g., {"Start Time": "begin", "End Time": "end"}). If None,
                default mappings are used.
        """
        # Paths and filenames
        self.prediction_directory_path: str = prediction_directory_path
        self.prediction_file_name: Optional[str] = prediction_file_name

        # Use provided column mappings or defaults
        self.columns_predictions: Dict[str, str] = (
            columns_predictions if columns_predictions is not None
            else self.DEFAULT_COLUMNS_PREDICTIONS.copy()
        )

        # Internal DataFrame to hold all predictions
        self.predictions_df: pd.DataFrame = pd.DataFrame()

        # Metadata DataFrame
        self.metadata_df: Optional[pd.DataFrame] = None

        # Validate column mappings
        self._validate_columns()

        # Load and prepare data
        self.load_data()
        self.predictions_df = self._prepare_dataframe(self.predictions_df)
        # Create the initial merged dataset as a copy of predictions
        self.merged_df = self.predictions_df.copy()

        # Ensure that the confidence column is numeric.
        conf_col = self.get_column_name("Confidence")
        if conf_col in self.predictions_df.columns:
            self.predictions_df[conf_col] = self.predictions_df[conf_col].astype(float)

        # Gather unique classes (if "Class" column exists)
        class_col = self.get_column_name("Class")
        if class_col in self.predictions_df.columns:
            self.classes = tuple(
                sorted(self.predictions_df[class_col].dropna().unique())
            )
        else:
            self.classes = tuple()

    def _validate_columns(self) -> None:
        """
        Validates that essential columns are provided in the prediction column mappings.

        Raises:
            ValueError: If required columns are missing or have None values.
        """
        # Required columns for predictions - removed "End Time" from required list
        required_columns = ["Start Time", "Class"]

        missing_pred_columns = [
            col
            for col in required_columns
            if col not in self.columns_predictions or self.columns_predictions[col] is None
        ]
        if missing_pred_columns:
            raise ValueError(f"Missing or None prediction columns: {', '.join(missing_pred_columns)}")

    def load_data(self) -> None:
        """
        Loads the prediction data into a DataFrame.

        - If `prediction_file_name` is None, all CSV/TSV files in `prediction_directory_path`
          are concatenated.
        - Otherwise, only the specified file is read.
        """
        if self.prediction_file_name is None:
            # Load all files in the directory
            self.predictions_df = read_and_concatenate_files_in_directory(
                self.prediction_directory_path
            )
        else:
            # Load a single specified file
            full_path = os.path.join(self.prediction_directory_path, self.prediction_file_name)
            # Attempt TSV read first; if it fails, try CSV
            try:
                self.predictions_df = pd.read_csv(full_path, sep="\t")
            except pd.errors.ParserError:
                self.predictions_df = pd.read_csv(full_path)

        # Ensure 'source_file' column exists for traceability
        if "source_file" not in self.predictions_df.columns:
            # If a single file was loaded, each row is from that file
            default_source = self.prediction_file_name if self.prediction_file_name else ""
            self.predictions_df["source_file"] = default_source

    def _extract_datetime_from_filename(self, filename: str) -> Tuple[str, datetime.datetime, str]:
        """
        Extracts site name and datetime from filename using strict format:
        SITE_SAMPLERATE_SITEID_YYYYMMDD_HHMMSS.wav
        
        Returns: (site_name, datetime_obj, original_filename)
        """
        if not isinstance(filename, str):
            return ("", None, str(filename))
            
        # Strict pattern matching the required format
        pattern = r"^([^_]+)_([^_]+)_([^_]+)_(\d{8})_(\d{6})(?:\.wav)?$"
        match = re.match(pattern, filename)
        if not match:
            return ("", None, filename)
        
        _, _, site_id, date_str, time_str = match.groups()
        try:
            date_time = datetime.datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
            return (site_id, date_time, filename)
        except ValueError:
            return ("", None, filename)

    def _add_metadata_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds metadata information (latitude, longitude) to the DataFrame."""
        if self.metadata_df is None or df.empty:
            return df

        # Create a copy and extract site names
        df = df.copy()
        df['site_name'] = df['recording_filename'].apply(
            lambda x: self._extract_datetime_from_filename(x)[0]
        )

        # Get valid sites and find missing ones
        valid_sites = set(self.metadata_df['site_name'].unique())
        found_sites = set(df['site_name'].unique())
        missing_sites = found_sites - valid_sites
        
        if missing_sites:
            missing_sites_str = sorted(list(missing_sites))
            gr.Warning(f"Site IDs not found in metadata: {missing_sites_str}")

        # Filter to valid sites
        df = df[df['site_name'].isin(valid_sites)]

        if df.empty:
            return df

        # Merge with metadata
        df = pd.merge(
            df,
            self.metadata_df[['site_name', 'latitude', 'longitude']],
            on='site_name',
            how='inner'  # Only keep records with matching site names
        )

        return df

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced preparation of DataFrame including datetime processing."""
        recording_col = self.get_column_name("Recording")
        
        try:
            # Extract and clean recording filenames
            if recording_col in df.columns:
                df["recording_filename"] = df[recording_col].apply(
                    lambda x: os.path.splitext(os.path.basename(str(x)))[0]
                    if pd.notnull(x) else ""
                )
            else:
                df["recording_filename"] = df["source_file"].apply(
                    lambda x: os.path.splitext(os.path.basename(str(x)))[0]
                    if pd.notnull(x) else ""
                )
            
            # Additional cleanup
            df["recording_filename"] = df["recording_filename"].str.strip()
            df["recording_filename"] = df["recording_filename"].replace('', pd.NA)
            
        except Exception as e:
            print(f"Warning: Error processing filenames: {e}")
            df["recording_filename"] = pd.NA

        # Extract datetime information
        datetime_info = df['recording_filename'].apply(self._extract_datetime_from_filename)
        df['site_name'] = datetime_info.apply(lambda x: x[0])
        df['recording_datetime'] = datetime_info.apply(lambda x: x[1])
        
        # Calculate actual prediction times
        start_time_col = self.get_column_name("Start Time")
        if start_time_col in df.columns and 'recording_datetime' in df.columns:
            df['prediction_time'] = df.apply(
                lambda row: row['recording_datetime'] + 
                          datetime.timedelta(seconds=float(row[start_time_col]))
                if pd.notnull(row['recording_datetime']) else None,
                axis=1
            )
            
            # Extract additional time components
            # Extract date components
            df['year'] = df['prediction_time'].dt.year.astype('Int64')  # Use Int64 for integer type with NA support
            df['month'] = df['prediction_time'].dt.month
            df['month_name'] = df['prediction_time'].dt.strftime('%B')  # Full month name
            df['day'] = df['prediction_time'].dt.day
            df['weekday'] = df['prediction_time'].dt.weekday  # Monday=0, Sunday=6
            df['weekday_name'] = df['prediction_time'].dt.strftime('%A')  # Full weekday name
            
            # Extract time components
            df['hour'] = df['prediction_time'].dt.hour
            df['minute'] = df['prediction_time'].dt.minute
            df['second'] = df['prediction_time'].dt.second
            
            # Add time period indicators
            df['is_weekend'] = df['weekday'].isin([5, 6])  # Saturday=5, Sunday=6
            df['day_period'] = pd.cut(df['hour'], 
                                    bins=[-1, 5, 11, 16, 21, 24],
                                    labels=['Night', 'Morning', 'Midday', 'Afternoon', 'Evening'])
        
        # Process correctness column - normalize values to True, False, or None
        correctness_col = self.get_column_name("Correctness")
        if correctness_col in df.columns:
            # Convert to lowercase strings first (handling NaN/None values)
            df[correctness_col] = df[correctness_col].astype(str).str.lower()
            
            # Map values to True, False, or None
            df[correctness_col] = df[correctness_col].apply(
                lambda x: True if x in ['true', 'correct'] 
                      else False if x in ['false', 'incorrect'] 
                      else None if x in ['nan', 'none', ''] or pd.isna(x) 
                      else None
            )
        else:
            # Create correctness column with all None values if it doesn't exist
            df[correctness_col] = None

        # Add metadata information
        df = self._add_metadata_info(df)
        
        return df

    def set_metadata(self, metadata_df: pd.DataFrame, 
                     site_col: str = 'Site',
                     lat_col: str = 'Latitude',
                     lon_col: str = 'Longitude') -> None:
        """
        Sets the metadata DataFrame with standardized column names and merges it with predictions_df.
        
        Args:
            metadata_df: DataFrame containing site metadata
            site_col: Name of the site identifier column
            lat_col: Name of the latitude column (decimal degrees)
            lon_col: Name of the longitude column (decimal degrees)
        """
        # Ensure required columns exist
        for col in [site_col, lat_col, lon_col]:
            if col not in metadata_df.columns:
                raise ValueError(f"Missing column '{col}' in metadata.")

        self.metadata_df = metadata_df.copy()
        
        # Standardize column names
        mapping = {site_col: 'site_name', lat_col: 'latitude', lon_col: 'longitude'}
        self.metadata_df.rename(columns=mapping, inplace=True)
        
        # Convert coordinates to numeric and validate ranges
        self.metadata_df['latitude'] = pd.to_numeric(self.metadata_df['latitude'], errors='coerce')
        self.metadata_df['longitude'] = pd.to_numeric(self.metadata_df['longitude'], errors='coerce')
        
        # Validate coordinate ranges
        invalid_lat = ~self.metadata_df['latitude'].between(-90, 90)
        invalid_lon = ~self.metadata_df['longitude'].between(-180, 180)
        
        if invalid_lat.any() or invalid_lon.any():
            raise ValueError("Invalid coordinates found. Latitude must be between -90 and 90, Longitude between -180 and 180.")
            
        # Merge metadata into predictions
        merged = self.predictions_df.copy()
        merged["latitude"] = merged["site_name"].map(self.metadata_df.set_index("site_name")["latitude"])
        merged["longitude"] = merged["site_name"].map(self.metadata_df.set_index("site_name")["longitude"])
        
        if merged["latitude"].isnull().all():
            raise ValueError("All latitude values are missing after merging metadata.")
        
        self.merged_df = merged.copy()

    def get_column_name(self, field_name: str, prediction: bool = True) -> str:
        """
        Retrieves the appropriate column name for the specified field.

        Args:
            field_name (str): The name of the field (e.g., "Class", "Start Time").
            prediction (bool): Whether to fetch from predictions mapping (True)
                             or annotations mapping (False). 
                             In visualization, this parameter is ignored since we only 
                             have prediction data.

        Returns:
            str: The column name corresponding to the field.

        Raises:
            TypeError: If field_name is None.
        """
        if field_name is None:
            raise TypeError("field_name cannot be None.")

        if field_name in self.columns_predictions and self.columns_predictions[field_name] is not None:
            return self.columns_predictions[field_name]

        return field_name

    def get_data(self) -> pd.DataFrame:
        """
        Retrieves a copy of the merged prediction DataFrame.
        """
        # Return the complete merged dataset if available.
        if hasattr(self, "merged_df") and not self.merged_df.empty:
            return self.merged_df.copy()
        return self.predictions_df.copy()

    def filter_data(
        self,
        selected_recordings: Optional[List[str]] = None,
        selected_classes: Optional[List[str]] = None,
        min_confidence: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Returns a filtered version of the prediction DataFrame.

        Args:
            selected_recordings (List[str], optional): A list of recording filenames to include.
                If None, no filtering by recording is applied.
            selected_classes (List[str], optional): A list of classes to include.
                If None, no filtering by class is applied.
            min_confidence (float, optional): Minimum confidence threshold for inclusion.
                If None, no filtering by confidence is applied.

        Returns:
            pd.DataFrame: The filtered DataFrame.
        """
        df = self.get_data()  # Work on a copy of the data

        # Filter by recordings – clean each selection consistently
        if selected_recordings:
            clean_selected = [os.path.splitext(os.path.basename(r.strip()))[0] 
                              for r in selected_recordings if r.strip()]
            df = df[df["recording_filename"].isin(clean_selected)]

        # Filter by classes
        class_col = self.get_column_name("Class")
        if selected_classes is not None and class_col in df.columns:
            df = df[df[class_col].isin(selected_classes)]

        # Filter by confidence
        confidence_col = self.get_column_name("Confidence")
        if min_confidence is not None and confidence_col in df.columns:
            df = df[df[confidence_col] >= min_confidence]

        return df

    def get_aggregated_locations(self, selected_classes: Optional[List[str]] = None) -> pd.DataFrame:
        """Returns aggregated prediction counts by location and class."""
        df = self.get_data()
        
        # Apply metadata and remove invalid records.
        df = self._add_metadata_info(df)
        if df.empty:
            raise ValueError("No valid predictions with matching site IDs found")
        
        # Ensure metadata columns exist.
        for col in ['latitude', 'longitude']:
            if col not in df.columns:
                raise ValueError(f"Metadata column '{col}' is missing. Please set metadata with valid latitude and longitude fields.")
        
        class_col = self.get_column_name("Class")
        if selected_classes:
            df = df[df[class_col].isin(selected_classes)]
            
        # Group by location and class, count occurrences
        agg_df = df.groupby([
            'site_name',
            'latitude',
            'longitude',
            class_col
        ]).size().reset_index(name='count')
        
        return agg_df
