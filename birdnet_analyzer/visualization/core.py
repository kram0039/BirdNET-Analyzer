"""
Core script for loading and optionally plotting prediction data.

This script:
  - Loads prediction data via the `DataProcessor` class
  - Allows optional filtering based on classes or recordings
  - Uses the ConfidencePlotter to create distribution plots
"""

import argparse
import json
import os
from typing import Optional, Dict, List

import matplotlib.pyplot as plt

# Import the simplified DataProcessor that only handles predictions
from birdnet_analyzer.visualization.data_processor import DataProcessor

# Import the ConfidencePlotter class 
# (Adjust this import path as needed to match your project structure)
from birdnet_analyzer.visualization.plotting.confidences import ConfidencePlotter


def process_predictions(
    prediction_path: str,
    columns_predictions: Optional[Dict[str, str]] = None,
    selected_classes: Optional[List[str]] = None,
    selected_recordings: Optional[List[str]] = None,
):
    """
    Loads prediction data, applies optional filters, and returns the resulting DataFrame.

    Args:
        prediction_path (str): Path to the prediction file or directory.
        columns_predictions (Optional[Dict[str, str]]): Custom column mappings for predictions.
        selected_classes (Optional[List[str]]): Classes to include; if None, keeps all.
        selected_recordings (Optional[List[str]]): Recording filenames to include; if None, keeps all.

    Returns:
        pd.DataFrame: Filtered predictions DataFrame.
    """
    # Determine directory and file
    if os.path.isfile(prediction_path):
        prediction_dir = os.path.dirname(prediction_path)
        prediction_file = os.path.basename(prediction_path)
    else:
        prediction_dir = prediction_path
        prediction_file = None

    # Initialize the DataProcessor (simplified, prediction-only version)
    processor = DataProcessor(
        prediction_directory_path=prediction_dir,
        prediction_file_name=prediction_file,
        columns_predictions=columns_predictions,
    )

    # Get the loaded predictions as a DataFrame
    df = processor.get_data()

    # If needed, apply filters
    if selected_classes or selected_recordings:
        df_filtered = processor.filter_data(
            selected_recordings=selected_recordings,
            selected_classes=selected_classes,
        )
        return df_filtered

    return df


def plot_predictions(predictions_df, output_dir=None):
    """
    Plots the distribution of confidence scores by class using ConfidencePlotter.

    By default, we'll create a ridgeline plot (matplotlib) and a histogram (plotly).
    Adjust or remove the one you don't need.
    """
    if predictions_df.empty:
        print("No data to plot.")
        return

    # Create a ConfidencePlotter that expects columns named "Class" and "Confidence"
    # Adjust these if your DataFrame uses different column names.
    plotter = ConfidencePlotter(
        data=predictions_df,
        class_col="Class",       # <-- change if needed
        conf_col="Confidence"    # <-- change if needed
    )

    # ----- 1) Ridgeline Plot (matplotlib) -----
    # We modify the ConfidencePlotter so it returns the figure instead of showing automatically.
    fig = plotter.plot_ridgeline_matplotlib(
        figsize=(8, 10),
        bandwidth=0.2,     # adjust for more/less smoothing
        overlap=0.6,
        fill=True,
        alpha=0.7,
        cmap="Spectral_r",
        title="Ridgeline of Confidence Scores by Class"
    )
    # 'fig' should be the current figure. If your ConfidencePlotter calls plt.show(), remove that.

    if output_dir:
        ridgeline_path = os.path.join(output_dir, "confidence_ridgeline.png")
        fig.savefig(ridgeline_path)
        print(f"Ridgeline plot saved to {ridgeline_path}")
    else:
        plt.show()

    # ----- 2) Histogram Plot (plotly) -----
    # This returns a Plotly Figure. We'll just display it in the console or save as HTML if needed.
    hist_fig = plotter.plot_histogram_plotly(
        nbins=30,
        facet_col=False,
        title="Histogram of Confidence Scores by Class"
    )
    if output_dir:
        hist_path = os.path.join(output_dir, "confidence_histogram.html")
        hist_fig.write_html(hist_path)
        print(f"Histogram plot saved to {hist_path}")


def main():
    """
    Entry point for loading and optionally plotting prediction data.
    """
    parser = argparse.ArgumentParser(
        description="Script for loading and plotting prediction data (no annotations)."
    )

    parser.add_argument(
        "--prediction_path",
        required=True,
        help="Path to prediction file or directory",
    )
    parser.add_argument(
        "--columns_predictions",
        type=json.loads,
        help='JSON string for columns_predictions (e.g., \'{"Start Time":"begin", "End Time":"end"}\')',
    )
    parser.add_argument(
        "--selected_classes",
        nargs="+",
        help="List of selected classes to filter on (optional)",
    )
    parser.add_argument(
        "--selected_recordings",
        nargs="+",
        help="List of selected recording filenames to filter on (optional)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="If set, plots the confidence distributions for each class",
    )
    parser.add_argument(
        "--output_dir",
        help="Optional directory to save plots or outputs",
    )

    args = parser.parse_args()

    # Process predictions (load + optional filter)
    predictions_df = process_predictions(
        prediction_path=args.prediction_path,
        columns_predictions=args.columns_predictions,
        selected_classes=args.selected_classes,
        selected_recordings=args.selected_recordings,
    )

    # Show basic info about loaded predictions
    print(f"Loaded {len(predictions_df)} predictions.")
    if predictions_df.empty:
        print("No predictions to show after filtering.")
        return

    # Optionally plot
    if args.plot:
        # Create output directory if needed
        if args.output_dir and not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        plot_predictions(predictions_df, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
