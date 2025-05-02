"""
Common components for visualization modules.
Contains shared types and functions to avoid circular imports.
"""

import typing
import datetime
import pandas as pd
import gradio as gr
from birdnet_analyzer.visualization.data_processor import DataProcessor

class ProcessorState(typing.NamedTuple):
    """State of the DataProcessor."""
    processor: DataProcessor
    prediction_dir: str
    metadata_dir: str
    color_map: typing.Optional[typing.Dict[str, str]] = None
    class_thresholds: typing.Optional[pd.DataFrame] = None


def convert_timestamp_to_datetime(timestamp):
    """Convert Gradio DateTime timestamp to pandas datetime."""
    if timestamp is None:
        return None
    try:
        # Convert timestamp to pandas datetime
        return pd.to_datetime(timestamp, unit='s')
    except:
        return None


def apply_datetime_filters(df, date_range_start, date_range_end, 
                         time_start_hour, time_start_minute, 
                         time_end_hour, time_end_minute):
    """Apply date and time filters to DataFrame."""
    if df.empty or 'prediction_time' not in df.columns:
        return df

    # Create a copy to avoid modifying original
    filtered_df = df.copy()
    
    # Apply date range filter if dates are provided
    if date_range_start is not None and date_range_end is not None:
        try:
            filtered_df = filtered_df[filtered_df['prediction_time'].notna()]
            start_date = convert_timestamp_to_datetime(date_range_start)
            end_date = convert_timestamp_to_datetime(date_range_end)
            
            if start_date and end_date:
                # Convert to pandas datetime if needed
                if not pd.api.types.is_datetime64_any_dtype(filtered_df['prediction_time']):
                    filtered_df['prediction_time'] = pd.to_datetime(filtered_df['prediction_time'])
                
                # Fix timezone issues by using normalized dates
                start_date = pd.Timestamp(start_date.date()) 
                end_date = pd.Timestamp(end_date.date()) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                
                # Filter by date range
                filtered_df = filtered_df[filtered_df['prediction_time'].dt.normalize().between(start_date, end_date)]
        
        except Exception as e:
            print(f"Date filtering error: {e}")
            return filtered_df

    # Apply time range filter if all time components are provided
    if all(x is not None for x in [time_start_hour, time_start_minute, time_end_hour, time_end_minute]):
        try:
            # Convert times to datetime.time objects
            start_time = datetime.time(int(time_start_hour), int(time_start_minute), 0, 0)
            end_time = datetime.time(int(time_end_hour), int(time_end_minute), 59, 999999)
            
            # Filter by time of day if prediction_time is datetime
            if pd.api.types.is_datetime64_any_dtype(filtered_df['prediction_time']):
                filtered_df = filtered_df[filtered_df['prediction_time'].dt.time.between(start_time, end_time)]
                
        except Exception as e:
            print(f"Time filtering error: {e}")
            return filtered_df

    return filtered_df


def apply_class_thresholds(df: pd.DataFrame, thresholds_df: pd.DataFrame, class_col: str, conf_col: str) -> pd.DataFrame:
    """Apply class-specific confidence thresholds."""
    if df.empty:
        return df
    if thresholds_df is None or thresholds_df.empty:
        return df

    try:
        # Ensure threshold column is numeric and clip values
        thresholds_df = thresholds_df.copy()
        thresholds_df['Threshold'] = pd.to_numeric(thresholds_df['Threshold'], errors='coerce')
        thresholds_df['Threshold'] = thresholds_df['Threshold'].fillna(0.10).clip(0.01, 0.99) # Default 0.10 if invalid

        # Prepare for merge
        threshold_map = thresholds_df.set_index('Class')['Threshold']
        
        # Map thresholds to the main dataframe
        df['class_threshold'] = df[class_col].map(threshold_map)
        
        # Apply default threshold if class not in map (shouldn't happen with proper init)
        df['class_threshold'] = df['class_threshold'].fillna(0.10) 

        # Filter based on class-specific threshold
        filtered_df = df[df[conf_col] >= df['class_threshold']].copy()
        
        # Drop the temporary threshold column
        filtered_df.drop(columns=['class_threshold'], inplace=True)
        
        return filtered_df

    except Exception as e:
        print(f"Error applying class thresholds: {e}")
        # Return original df if error occurs
        return df
