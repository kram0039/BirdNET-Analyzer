"""
Common components for visualization modules.
Contains shared types and functions to avoid circular imports.
"""

import typing
import datetime
from dataclasses import dataclass
import pandas as pd
import gradio as gr

# Add TYPE_CHECKING block for static analysis if needed (optional but good practice)
if typing.TYPE_CHECKING:
    from birdnet_analyzer.visualization.data_processor import DataProcessor

NO_SITE_LABEL = "[No Site Assigned]"

@dataclass
class ProcessorState:
    """State of the DataProcessor."""
    processor: 'DataProcessor'  # Use string literal
    prediction_dir: str
    metadata_dir: typing.Optional[str] = None
    color_map: typing.Optional[typing.Dict[str, str]] = None
    class_thresholds: typing.Optional[pd.DataFrame] = None


def convert_timestamp_to_datetime(timestamp):
    """
    Convert the epoch seconds that the <DateTime> component sends back
    into a *local* (Europe/Berlin) wall-clock datetime.
    """
    if timestamp is None:
        return None
    try:
        # (1) build an *aware* UTC datetime
        dt_utc = pd.to_datetime(timestamp, unit="s", utc=True)

        # (2) convert to the user’s zone
        dt_local = dt_utc.tz_convert("Europe/Berlin")

        # (3) drop the tz info (Gradio expects naive datetimes)
        return dt_local.tz_localize(None)
    except Exception:
        return None
