import pandas as pd
import gradio as gr
import os
from typing import List, Optional, Dict, Any
import datetime

# Import from common module instead of from visualization.py
from birdnet_analyzer.visualization.common import ProcessorState, apply_class_thresholds, apply_datetime_filters

def calculate_detection_counts(
    proc_state: ProcessorState,
    selected_classes_list: Optional[List[str]] = None,
    selected_recordings_list: Optional[List[str]] = None,
    date_range_start: Optional[float] = None,
    date_range_end: Optional[float] = None,
    time_start_hour: Optional[str] = None,
    time_start_minute: Optional[str] = None,
    time_end_hour: Optional[str] = None,
    time_end_minute: Optional[str] = None,
    correctness_mode: str = "Ignore correctness flags"
) -> Dict[str, Any]:
    """
    Count detections for each class with the applied filters.
    
    Args:
        proc_state: ProcessorState containing the DataProcessor and related state
        selected_classes_list: List of classes to include (if None, includes all)
        selected_recordings_list: List of recordings to include (if None, includes all)
        date_range_start: Start date for filtering (timestamp)
        date_range_end: End date for filtering (timestamp)
        time_start_hour: Start hour for time-of-day filtering
        time_start_minute: Start minute for time-of-day filtering
        time_end_hour: End hour for time-of-day filtering
        time_end_minute: End minute for time-of-day filtering
        correctness_mode: How to handle correctness flags 
            ("Ignore correctness flags", "Show only correct", "Show only incorrect", "Show only unspecified")
            
    Returns:
        Dictionary with Gradio update information for the results table
    """
    if not proc_state or not proc_state.processor:
        raise gr.Error("Please load predictions first")
        
    # Correctly check if thresholds DataFrame is None or empty
    if proc_state.class_thresholds is None or proc_state.class_thresholds.empty:
        raise gr.Error("Class thresholds not initialized or are empty. Load data and optionally JSON thresholds first.")
        
    # Validate thresholds from state
    validated_thresholds_df = proc_state.class_thresholds
        
    # Get data and apply filters
    df = proc_state.processor.get_data()
    if df.empty:
        raise gr.Error("No predictions to analyze")
        
    # Apply class and recording filters
    col_class = proc_state.processor.get_column_name("Class")
    conf_col = proc_state.processor.get_column_name("Confidence")
    corr_col = proc_state.processor.get_column_name("Correctness")
    
    # Debug which correctness column is being used
    print(f"Using correctness column: '{corr_col}'")
    print(f"Available columns: {df.columns.tolist()}")
    
    # Check if correctness column exists; if not, try both capitalization forms
    if corr_col not in df.columns:
        # Try both 'correctness' and 'Correctness'
        alt_corr_col = 'correctness' if corr_col == 'Correctness' else 'Correctness'
        if alt_corr_col in df.columns:
            print(f"Switching to alternative correctness column: '{alt_corr_col}'")
            corr_col = alt_corr_col
        else:
            # Create an empty correctness column if none exists
            print(f"No correctness column found, creating a placeholder")
            df[corr_col] = None
    
    # Filter by selected classes if provided
    if selected_classes_list:
        df = df[df[col_class].isin(selected_classes_list)]
        
    # Filter by selected recordings if provided
    selected_recordings_list = [rec.lower() for rec in selected_recordings_list]
    df["recording_filename"] = df["recording_filename"].apply(
        lambda x: os.path.splitext(os.path.basename(x.strip()))[0].lower() 
        if isinstance(x, str) else x
    )
    df = df[df["recording_filename"].isin(selected_recordings_list)]
        
    # Apply class-specific confidence thresholds using validated thresholds
    df = apply_class_thresholds(df, validated_thresholds_df, col_class, conf_col)
    
    # Apply date and time filters
    df = apply_datetime_filters(
        df, 
        date_range_start, 
        date_range_end,
        time_start_hour,
        time_start_minute,
        time_end_hour,
        time_end_minute
    )
    
    # Log count before filtering
    print(f"Records before correctness filter: {len(df)}")
    print(f"Correctness mode: {correctness_mode}")
    
    # Make sure corr_col has proper boolean values (not strings)
    if corr_col in df.columns:
        # Convert string 'true'/'false' to proper boolean
        df[corr_col] = df[corr_col].map({
            'true': True, 'True': True, True: True, 1: True,
            'false': False, 'False': False, False: False, 0: False,
            'nan': None, 'none': None, '': None, 'null': None
        }, na_action='ignore')
    
    # Apply correctness filter based on selected mode
    if correctness_mode == "Show only correct":
        df = df[(df[corr_col] == True) | (df[corr_col] == 'True')]
        print(f"Records after filtering for correct: {len(df)}")
    elif correctness_mode == "Show only incorrect":
        df = df[(df[corr_col] == False) | (df[corr_col] == 'False')]
        print(f"Records after filtering for incorrect: {len(df)}")
    elif correctness_mode == "Show only unspecified":
        df = df[(df[corr_col].isna()) | (df[corr_col] == '') | (df[corr_col] == 'nan')]
        print(f"Records after filtering for unspecified: {len(df)}")
    # "Ignore correctness flags" mode doesn't filter the data
    
    if df.empty:
        raise gr.Error("No data matches the selected filters")
        
    # Generate counts table
    class_counts = df[col_class].value_counts().reset_index()
    class_counts.columns = ["Species", "Count"]
    
    # Add percentage column
    total = class_counts["Count"].sum()
    class_counts["Percentage"] = (class_counts["Count"] / total * 100).round(1).astype(str) + "%"
    
    # Sort by detection count (descending)
    class_counts = class_counts.sort_values("Count", ascending=False)
    
    # Add total row
    total_row = pd.DataFrame({
        "Species": ["Total"],
        "Count": [total],
        "Percentage": ["100.0%"]
    })
    
    result_df = pd.concat([class_counts, total_row])
    
    # Set column widths for better display
    column_widths = {
        "Species": "200px",
        "Count": "110px",
        "Percentage": "110px"
    }
    
    return gr.update(
        value=result_df, 
        visible=True, 
        column_widths=[column_widths.get(col, "120px") for col in result_df.columns]
    )
