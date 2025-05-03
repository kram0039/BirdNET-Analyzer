import pandas as pd
import gradio as gr
import os
from typing import Dict, Any

def calculate_detection_counts(
    filtered_df: pd.DataFrame, # Accept a pre-filtered DataFrame
) -> pd.DataFrame:
    """
    Count detections for each class from a pre-filtered DataFrame.

    Args:
        filtered_df (pd.DataFrame): DataFrame containing already filtered prediction data.
            Must contain at least a 'Class' column.

    Returns:
        DataFrame with detection counts and percentages for each class
    """
    if filtered_df is None or filtered_df.empty:
        # Return an empty DataFrame if input is empty
        return pd.DataFrame(columns=["Species", "Count", "Percentage"])

    # Assume standard column names are present after filtering
    col_class = "Class"

    if col_class not in filtered_df.columns:
        raise gr.Error(f"Required column '{col_class}' not found in the provided data.")

    # Generate counts table directly from the filtered data
    class_counts = filtered_df[col_class].value_counts().reset_index()
    class_counts.columns = ["Species", "Count"]

    # Add percentage column
    total = class_counts["Count"].sum()
    if total > 0:
        class_counts["Percentage"] = (class_counts["Count"] / total * 100).round(1).astype(str) + "%"
    else:
        class_counts["Percentage"] = "0.0%"

    # Sort by detection count (descending)
    class_counts = class_counts.sort_values("Count", ascending=False)

    # Add total row if there are counts
    if total > 0:
        total_row = pd.DataFrame({
            "Species": ["Total"],
            "Count": [total],
            "Percentage": ["100.0%"]
        })
        result_df = pd.concat([class_counts, total_row])
    else:
        # If no counts, just show the empty frame structure
        result_df = pd.DataFrame(columns=["Species", "Count", "Percentage"])

    # Return the DataFrame directly
    return result_df
