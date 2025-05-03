"""
Common components for visualization modules.
Contains shared types and functions to avoid circular imports.
"""

import typing
import datetime
import pandas as pd
import gradio as gr

# Add TYPE_CHECKING block for static analysis if needed (optional but good practice)
if typing.TYPE_CHECKING:
    from birdnet_analyzer.visualization.data_processor import DataProcessor

class ProcessorState(typing.NamedTuple):
    """State of the DataProcessor."""
    processor: 'DataProcessor'  # Use string literal
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
