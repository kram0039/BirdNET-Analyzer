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
        Extracts site name and datetime from filename using flexible regex patterns.
        
        Returns: (site_name, datetime_obj, original_filename)
        """
        if not isinstance(filename, str):
            return ("", None, str(filename))
        
        # Extract parts for later site_id matching
        self._extract_site_id_from_filename(filename)
        
        # Define patterns for date and time formats
        datetime_patterns = [
            # YYYYMMDD_HHMMSS (original format)
            (r'(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})', 
             '%Y%m%d_%H%M%S'),
            
            # YYYY-MM-DD_HH-MM-SS
            (r'(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})', 
             '%Y-%m-%d_%H-%M-%S'),
            
            # YYYYMMDDHHMMSS (continuous)
            (r'(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})', 
             '%Y%m%d%H%M%S'),
            
            # YYYY-MM-DD-HH-MM-SS
            (r'(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})', 
             '%Y-%m-%d-%H-%M-%S'),
            
            # YYYY.MM.DD_HH.MM.SS
            (r'(\d{4})\.(\d{2})\.(\d{2})_(\d{2})\.(\d{2})\.(\d{2})', 
             '%Y.%m.%d_%H.%M.%S'),
            
            # MM-DD-YYYY_HH-MM-SS
            (r'(\d{2})-(\d{2})-(\d{4})_(\d{2})-(\d{2})-(\d{2})', 
             '%m-%d-%Y_%H-%M-%S'),
            
            # DD-MM-YYYY_HH-MM-SS
            (r'(\d{2})-(\d{2})-(\d{4})_(\d{2})-(\d{2})-(\d{2})', 
             '%d-%m-%Y_%H-%M-%S'),
            
            # YYYY_MM_DD_HH_MM_SS
            (r'(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})',
             '%Y_%m_%d_%H_%M_%S'),
        ]
        
        # Try each pattern
        for pattern, format_str in datetime_patterns:
            # Extract the datetime parts from anywhere in the filename
            datetime_match = re.search(pattern, filename)
            if datetime_match:
                try:
                    # Reconstruct the datetime string based on matched groups
                    parts = datetime_match.groups()
                    if len(parts) == 6:  # Year, month, day, hour, minute, second
                        if format_str == '%m-%d-%Y_%H-%M-%S':
                            datetime_str = f"{parts[0]}-{parts[1]}-{parts[2]}_{parts[3]}-{parts[4]}-{parts[5]}"
                        elif format_str == '%d-%m-%Y_%H-%M-%S':
                            datetime_str = f"{parts[0]}-{parts[1]}-{parts[2]}_{parts[3]}-{parts[4]}-{parts[5]}"
                        else:
                            datetime_str = f"{parts[0]}{parts[1]}{parts[2]}_{parts[3]}{parts[4]}{parts[5]}"
                        
                        # Create datetime object
                        date_time = datetime.datetime.strptime(datetime_str, format_str)
                        return (None, date_time, filename)
                except ValueError:
                    continue  # Try the next pattern if this one doesn't work
        
        # If no pattern matched, return with no datetime
        return (None, None, filename)

    def _extract_site_id_from_filename(self, filename: str) -> None:
        """
        Store filename parts for later site_id matching.
        
        Splits the filename by underscores and stores parts for later matching.
        """
        if not isinstance(filename, str) or not filename:
            return
        
        # Remove file extension if present
        basename = os.path.splitext(filename)[0]
        
        # Simply split by underscore and store all parts
        parts = basename.split('_')
        
        # Store parts with this filename for later matching in _update_site_ids
        if not hasattr(self, '_filename_parts_map'):
            self._filename_parts_map = {}
        self._filename_parts_map[filename] = parts

    def _add_metadata_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds metadata information (latitude, longitude) to the DataFrame."""
        if self.metadata_df is None or df.empty:
            return df

        # Create a copy
        df = df.copy()
        
        # Site IDs are now set by _update_site_ids() when metadata is loaded
        # Just merge lat/lon based on existing site_name column
        
        # Get valid sites
        valid_sites = set(self.metadata_df['site_name'].unique())
        
        # For rows with valid site names, add lat/lon data
        site_mask = df['site_name'].isin(valid_sites)
        
        if site_mask.any():
            # Only merge records with valid site names
            valid_records = df[site_mask].copy()
            valid_records = pd.merge(
                valid_records,
                self.metadata_df[['site_name', 'latitude', 'longitude']],
                on='site_name',
                how='left'
            )
            
            # Update the original dataframe with the merged values
            df.loc[site_mask, 'latitude'] = valid_records['latitude'].values
            df.loc[site_mask, 'longitude'] = valid_records['longitude'].values
        
        return df

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced preparation of DataFrame including datetime processing."""
        recording_col = self.get_column_name("Recording")
        
        try:
            # Extract and clean recording filenames
            if recording_col in df.columns:
                # Store the full filename with extension
                df["recording_filename_with_ext"] = df[recording_col].apply(
                    lambda x: os.path.basename(str(x))
                    if pd.notnull(x) else ""
                )
                # Also store without extension for backward compatibility
                df["recording_filename"] = df[recording_col].apply(
                    lambda x: os.path.splitext(os.path.basename(str(x)))[0]
                    if pd.notnull(x) else ""
                )
            else:
                # Store the full filename with extension
                df["recording_filename_with_ext"] = df["source_file"].apply(
                    lambda x: os.path.basename(str(x))
                    if pd.notnull(x) else ""
                )
                # Also store without extension for backward compatibility
                df["recording_filename"] = df["source_file"].apply(
                    lambda x: os.path.splitext(os.path.basename(str(x)))[0]
                    if pd.notnull(x) else ""
                )
            
            # Additional cleanup
            df["recording_filename"] = df["recording_filename"].str.strip()
            df["recording_filename"] = df["recording_filename"].replace('', pd.NA)
            df["recording_filename_with_ext"] = df["recording_filename_with_ext"].str.strip()
            df["recording_filename_with_ext"] = df["recording_filename_with_ext"].replace('', pd.NA)
            
        except Exception as e:
            print(f"Warning: Error processing filenames: {e}")
            df["recording_filename"] = pd.NA
            df["recording_filename_with_ext"] = pd.NA

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

    def _clean_coordinate(self, value):
        """
        Clean coordinate values by handling different formats:
        - Replace comma decimal separators with periods
        - Remove degree symbols and other non-numeric characters
        - Handle N/A values
        
        Args:
            value: The coordinate value to clean
            
        Returns:
            Cleaned numeric value or None if invalid
        """
        if pd.isna(value) or value == 'N/A' or value == 'NA' or value == '':
            return None
            
        if isinstance(value, (int, float)):
            return value
            
        # Convert to string if not already
        value_str = str(value).strip()
        
        # Handle N/A variants
        if value_str.upper() in ('N/A', 'NA', 'NONE', 'NULL'):
            return None
            
        # Remove degree symbols and other non-numeric characters except for decimal separators
        # Keep minus sign for negative coordinates
        cleaned = ''
        for char in value_str:
            if char.isdigit() or char == '-':
                cleaned += char
            elif char == ',':  # Convert comma to period for decimal separator
                cleaned += '.'
            elif char == '.':  # Keep periods
                cleaned += '.'
                
        # Try to convert to float
        try:
            return float(cleaned)
        except ValueError:
            print(f"Warning: Could not convert coordinate value: '{value_str}' -> '{cleaned}'")
            return None

    def set_metadata(self, metadata_df: pd.DataFrame, 
                     site_col: str = 'Site',
                     lat_col: str = 'Latitude',
                     lon_col: str = 'Longitude') -> None:
        """
        Sets the metadata DataFrame with standardized column names and merges it with predictions_df.
        """
        # Ensure required columns exist
        for col in [site_col, lat_col, lon_col]:
            if col not in metadata_df.columns:
                raise ValueError(f"Missing column '{col}' in metadata.")

        # Check for duplicate site IDs in metadata BEFORE any processing
        metadata_df = metadata_df.copy()
        
        # EARLY CHECK FOR DUPLICATE SITES WITH DIFFERENT COORDINATES
        # Standardize column names for the check
        metadata_df['site_name'] = metadata_df[site_col]
        metadata_df['latitude'] = pd.to_numeric(metadata_df[lat_col], errors='coerce')
        metadata_df['longitude'] = pd.to_numeric(metadata_df[lon_col], errors='coerce')
        
        # Round coordinates for stable comparison
        metadata_df['latitude_rounded'] = metadata_df['latitude'].round(6)
        metadata_df['longitude_rounded'] = metadata_df['longitude'].round(6)
        
        # Find duplicate site names
        duplicated_sites = metadata_df['site_name'].duplicated(keep=False)
        if duplicated_sites.any():
            # Get list of sites with duplicates
            dup_sites_df = metadata_df[duplicated_sites].copy()
            
            # Check if each duplicate site has consistent coordinates
            inconsistent_sites = []
            for site, group in dup_sites_df.groupby('site_name'):
                # Check if any coordinate is different after conversion
                lat_values = group['latitude_rounded'].dropna().unique()
                lon_values = group['longitude_rounded'].dropna().unique()
                
                # If we have more than one unique value for lat or lon, it's inconsistent
                if len(lat_values) > 1 or len(lon_values) > 1:
                    # Find the different coordinates for this site
                    coords = group[['latitude', 'longitude']].drop_duplicates()
                    coord_strs = [f"({row['latitude']}, {row['longitude']})" for _, row in coords.iterrows()]
                    inconsistent_sites.append((site, coord_strs))
            
            # If any sites have inconsistent coordinates, raise a specific Gradio error
            if inconsistent_sites:
                error_msg = "Found duplicate site IDs with different coordinates in metadata file.\n\n"
                for site, coords in inconsistent_sites:
                    error_msg += f"- Site ID '{site}' maps to multiple locations: {', '.join(coords)}\n"
                error_msg += "\nPlease correct your metadata file to ensure each site ID has a unique location."
                
                print(f"ERROR: {error_msg}")
                gr.Error(error_msg)
                return
        
        # Debug: Print metadata site column info
        print(f"\nDEBUG - Metadata DataFrame shape: {metadata_df.shape}")
        print(f"DEBUG - Total unique site values in metadata: {metadata_df['site_name'].nunique()}")
        
        # Now proceed with normal metadata processing with the duplicate check already done
        self.metadata_df = metadata_df.copy()
        
        # Standardize column names (again to be sure)
        mapping = {site_col: 'site_name', lat_col: 'latitude', lon_col: 'longitude'}
        self.metadata_df.rename(columns=mapping, inplace=True)
        
        # Clean coordinates
        self.metadata_df['latitude'] = self.metadata_df['latitude'].apply(self._clean_coordinate)
        self.metadata_df['longitude'] = self.metadata_df['longitude'].apply(self._clean_coordinate)
        
        # Drop duplicate sites (keeping only the first occurrence since we know they're consistent)
        if metadata_df['site_name'].duplicated().any():
            duplicate_count = self.metadata_df[self.metadata_df['site_name'].duplicated(keep=False)]['site_name'].nunique()
            self.metadata_df = self.metadata_df.drop_duplicates(subset=['site_name'], keep='first')
            gr.Info(f"Found {duplicate_count} sites with identical coordinates for duplicate entries. Using first occurrence.")
        
        # Identify and report rows with invalid coordinates
        invalid_lat = ~(self.metadata_df['latitude'].between(-90, 90) | self.metadata_df['latitude'].isna())
        invalid_lon = ~(self.metadata_df['longitude'].between(-180, 180) | self.metadata_df['longitude'].isna())
        
        if invalid_lat.any() or invalid_lon.any():
            # Get sample of problematic values for better error reporting
            problem_rows = self.metadata_df[invalid_lat | invalid_lon].head(5)
            sample_issues = problem_rows[['site_name', 'latitude', 'longitude']].to_string(index=False)
            error_msg = f"Invalid coordinates found. Latitude must be between -90 and 90, Longitude between -180 and 180.\n"
            error_msg += f"Sample problematic rows:\n{sample_issues}"
            gr.Error(error_msg)
            return
        
        # Handle rows with valid site names but missing coordinates
        missing_coords_count = self.metadata_df[self.metadata_df['latitude'].isna() | self.metadata_df['longitude'].isna()].shape[0]
        if missing_coords_count > 0:
            print(f"Warning: {missing_coords_count} sites have missing coordinates - these sites will be excluded from spatial analysis")
        
        # Remove rows with missing coordinates entirely to prevent later issues
        self.metadata_df = self.metadata_df.dropna(subset=['latitude', 'longitude'])
            
        # Debug: Before final merge, show what we're working with
        print(f"DEBUG - Final metadata shape after cleaning: {self.metadata_df.shape}")
        print(f"DEBUG - Predictions DataFrame shape: {self.predictions_df.shape}")
        print(f"DEBUG - Predictions site_name column exists: {'site_name' in self.predictions_df.columns}")
        
        # Merge metadata into predictions
        merged = self.predictions_df.copy()
        
        # Debug the site_name values before mapping
        sample_site_names = []
        if not merged.empty:
            sample_site_names = merged["site_name"].sample(min(5, len(merged))).tolist()
        print(f"DEBUG - Sample site_name values before mapping: {sample_site_names}")
        
        # Check if we have any site_name matches between predictions and metadata
        if 'site_name' in merged.columns:
            # Create a set of all site names in metadata for fast lookup
            metadata_sites = set(self.metadata_df['site_name'].dropna().astype(str))
            
            # Count how many matches we have
            matching_sites_count = merged['site_name'].isin(metadata_sites).sum()
            print(f"DEBUG - Found {matching_sites_count} prediction records with matching site IDs")
            
            if matching_sites_count == 0:
                gr.Warning(
                    "No prediction records have site IDs that match the metadata file. "
                    "Please check that the recording filenames contain site IDs exactly matching those in the metadata."
                )
        
        # Now do the actual mapping
        merged["latitude"] = merged["site_name"].map(self.metadata_df.set_index("site_name")["latitude"])
        merged["longitude"] = merged["site_name"].map(self.metadata_df.set_index("site_name")["longitude"])
        
        # Debug after mapping
        print(f"DEBUG - Latitude column after mapping - null count: {merged['latitude'].isnull().sum()} out of {len(merged)}")
        print(f"DEBUG - Any non-null latitude values: {not merged['latitude'].isnull().all()}")
        
        if merged["latitude"].isnull().all():
            error_msg = ("All latitude values are missing after merging metadata. "
                        "This typically occurs when no site IDs in predictions match site IDs in metadata.")
            print(f"ERROR - {error_msg}")
            gr.Error(error_msg)
            return
        
        self.merged_df = merged.copy()

    def _update_site_ids(self) -> None:
        """
        Updates site ID matches in the predictions DataFrame using available metadata.
        Called after metadata is set to match site IDs correctly.
        Only considers exact matches between filename parts and site IDs.
        """
        if self.metadata_df is None or self.predictions_df.empty:
            print("DEBUG - Cannot update site IDs: metadata is None or predictions are empty")
            return
        
        # Get valid site IDs from metadata
        valid_site_ids = set(self.metadata_df['site_name'].dropna().astype(str).unique())
        print(f"DEBUG - Valid metadata site IDs: {sorted(list(valid_site_ids))}")
        
        # Check filename examples
        sample_filenames = self.predictions_df['recording_filename'].dropna().head(5).tolist()
        print(f"DEBUG - Sample filenames for site ID extraction: {sample_filenames}")
        
        # Iterate through each filename to find possible matches
        site_id_mapping = {}
        unmatched_filenames = set()
        
        for filename in self.predictions_df['recording_filename'].dropna().unique():
            # Extract potential parts from the filename
            parts = str(filename).split('_')
            print(f"DEBUG - Parts from '{filename}': {parts}")
            
            found_match = False
            # ONLY try exact matches - no case-insensitive or substring matching
            for part in parts:
                if part in valid_site_ids:
                    site_id_mapping[filename] = part
                    found_match = True
                    print(f"DEBUG - Found exact match for '{filename}': {part}")
                    break
            
            if not found_match:
                unmatched_filenames.add(filename)
                print(f"DEBUG - No matching site ID found for '{filename}'")
        
        # Apply the site ID mapping to all rows
        self.predictions_df['site_name'] = self.predictions_df['recording_filename'].map(site_id_mapping)
        
        # Debug: Show results
        match_count = self.predictions_df['site_name'].notna().sum()
        print(f"DEBUG - Records with matched site IDs: {match_count}/{len(self.predictions_df)} ({match_count/len(self.predictions_df)*100:.1f}%)")
        
        # Update merged DataFrame with new site names
        self.merged_df = self.predictions_df.copy()
        
        # Map coordinates from metadata to predictions
        self.merged_df["latitude"] = self.merged_df["site_name"].map(
            self.metadata_df.set_index("site_name")["latitude"])
        self.merged_df["longitude"] = self.merged_df["site_name"].map(
            self.metadata_df.set_index("site_name")["longitude"])
        
        # Check if we have any valid coordinates
        has_lat = not self.merged_df["latitude"].isnull().all()
        print(f"DEBUG - Has any valid latitude values: {has_lat}")
        print(f"DEBUG - Null latitude count: {self.merged_df['latitude'].isnull().sum()} out of {len(self.merged_df)}")
        
        # Show warning for unmatched files
        if unmatched_filenames:
            sample_unmatched = sorted(list(unmatched_filenames))[:10]  # Show first 10 examples
            display_unmatched = ", ".join([f"'{f}'" for f in sample_unmatched])
            if len(unmatched_filenames) > 10:
                display_unmatched += f" and {len(unmatched_filenames) - 10} more"
            
            valid_examples = ", ".join([f"'{s}'" for s in sorted(list(valid_site_ids))[:5]])
            warning_message = (
                f"No site ID match found for {len(unmatched_filenames)} recordings: {display_unmatched}\n"
                f"Valid site IDs in metadata include: {valid_examples}\n"
                f"Make sure each recording filename contains a site ID exactly matching one from the metadata."
            )
            gr.Warning(warning_message)

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
        
        # Debug: Print recording filename samples
        print(f"\nDEBUG - Original DataFrame shape: {df.shape}")
        if "recording_filename" in df.columns:
            sample_recordings = df["recording_filename"].sample(min(5, len(df))).tolist()
            print(f"DEBUG - Sample recording_filename values: {sample_recordings}")
        if "recording_filename_with_ext" in df.columns:
            sample_recordings_ext = df["recording_filename_with_ext"].sample(min(5, len(df))).tolist()
            print(f"DEBUG - Sample recording_filename_with_ext values: {sample_recordings_ext}")
        if "site_name" in df.columns:
            site_counts = df["site_name"].value_counts().head(5).to_dict()
            print(f"DEBUG - Top 5 site_name values and counts: {site_counts}")

        # Filter by recordings - handle both with and without extensions
        if selected_recordings:
            # Process each selection to handle paths and extensions flexibly
            clean_with_ext = []
            clean_without_ext = []
            
            for recording in selected_recordings:
                if not recording or not recording.strip():
                    continue
                    
                # Get just the basename (filename without path)
                basename = os.path.basename(recording.strip())
                
                # Store both with and without extension versions
                clean_with_ext.append(basename)
                clean_without_ext.append(os.path.splitext(basename)[0])
            
            # Debug: Show what we're filtering with
            if clean_with_ext:
                print(f"DEBUG - First few clean_with_ext: {clean_with_ext[:3]}")
            if clean_without_ext:
                print(f"DEBUG - First few clean_without_ext: {clean_without_ext[:3]}")
                
            # Match on either the full filename with extension or just the basename
            df = df[df["recording_filename_with_ext"].isin(clean_with_ext) | 
                    df["recording_filename"].isin(clean_without_ext)]
            
            # Debug: Result after filtering
            print(f"DEBUG - Filtered DataFrame shape: {df.shape}")

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
