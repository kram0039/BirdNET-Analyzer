import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Optional, Dict


class ConfidencePlotter:
    """
    A helper class to plot distribution (histogram) plots of confidence scores
    for each class using only matplotlib and plotly.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        class_col: str = "class",
        conf_col: str = "confidence"
    ):
        """
        Initialize the ConfidencePlotter with a DataFrame and column names.

        Args:
            data (pd.DataFrame): Data containing confidence scores and class labels.
            class_col (str): Name of the column that indicates class/species/label.
            conf_col (str): Name of the column that indicates confidence score.
        """
        self.data = data.copy()
        self.class_col = class_col
        self.conf_col = conf_col

        # Ensure these columns exist
        if self.class_col not in self.data.columns:
            raise ValueError(f"Column '{self.class_col}' not found in data.")
        if self.conf_col not in self.data.columns:
            raise ValueError(f"Column '{self.conf_col}' not found in data.")

        # Gather unique classes
        self.classes = sorted(self.data[self.class_col].dropna().unique())

    def _get_color_map(self, classes: List[str]) -> Dict[str, str]:
        """Create consistent color mapping for classes."""
        sorted_classes = sorted(classes)
        
        # Base colors used if number of classes is 7 or less.
        base_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']
        
        color_dict: Dict[str, str]
        if len(sorted_classes) <= len(base_colors): # Effectively len(sorted_classes) <= 7
            color_dict = {cls: base_colors[i % len(base_colors)] for i, cls in enumerate(sorted_classes)}
        else:
            # Use a more diverse color palette if more than 7 classes
            extended_colors = px.colors.qualitative.Alphabet # Has 26 distinct colors
            # Cycle through extended_colors if more classes than available colors
            color_dict = {cls: extended_colors[i % len(extended_colors)] for i, cls in enumerate(sorted_classes)}
        return color_dict

    def _confidence_bins(self, step: float = 0.1
                     ) -> tuple[np.ndarray, list[str]]:
        """
        Build equally-wide bins between 0 and 1 **and** readable labels
        that accurately reflect numpy.histogram's binning convention.
        Example (step=0.1):
            edges  = np.array([0.0, 0.1, 0.2, ..., 1.0])
            labels = ['[0.0, 0.1)', '[0.1, 0.2)', ..., '[0.9, 1.0]']
        """
        # Generate edges (inclusive lower, inclusive upper for last bin)
        edges = np.round(np.arange(0, 1 + step, step), 2)
        # Build labels matching numpy.histogram behavior
        labels = []
        num_bins = len(edges) - 1
        for i in range(num_bins):
            lower_edge = edges[i]
            upper_edge = edges[i+1]
            if i < num_bins - 1:  # For all bins except the last one
                labels.append(f"[{lower_edge:.1f}, {upper_edge:.1f})")
            else:  # For the last bin
                labels.append(f"[{lower_edge:.1f}, {upper_edge:.1f}]")
        return edges, labels

    def plot_histogram_matplotlib(
        self,
        bins: int = 30,
        smooth: bool = False,
        alpha: float = 0.6,
        title: str = "Histogram of Confidence Scores by Class",
        figsize: tuple = (6, 8)
    ):
        """
        Creates a per-class histogram plot using matplotlib.
        One histogram is plotted for each class with vertical offsets.
        If smooth is True, the histogram counts are smoothed via convolution.
        The style (colors, transparency, labeling) matches that of the KDE/ridgeline plot.
        """
        fig, ax = plt.subplots(figsize=figsize)
        cm = plt.get_cmap("Spectral_r")
        num_classes = len(self.classes)
        # Define vertical offsets (similar to ridgeline)
        y_offsets = np.linspace(0, -(num_classes - 1) * 0.6, num_classes)
        
        for i, cls in enumerate(self.classes):
            cls_data = self.data.loc[self.data[self.class_col] == cls, self.conf_col].dropna().values
            if len(cls_data) < 1:
                continue
            # Compute histogram normalized to density
            counts, bin_edges = np.histogram(cls_data, bins=bins, density=True)
            # Compute center values for bins
            x_vals = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            # Optionally smooth the counts using a simple moving average
            if smooth:
                window = np.ones(5) / 5
                counts = np.convolve(counts, window, mode="same")
            y_offset = y_offsets[i]
            color = cm(i / num_classes)
            ax.fill_between(x_vals, y_offset, counts + y_offset, color=color, alpha=alpha)
            ax.plot(x_vals, counts + y_offset, color=color, alpha=alpha)
            # Label the left side with class name
            ax.text(bin_edges[0], y_offset + 0.02, cls, ha="right", va="bottom", fontsize=9)
        
        ax.set_yticks([])
        ax.set_ylabel("")
        ax.set_xlabel("Confidence Score")
        ax.set_title(title)
        plt.tight_layout()
        plt.show()
        return fig

    def plot_histogram_plotly(
        self,
        title: str = "Histogram of Confidence Scores by Class"
    ) -> go.Figure:
        """
        Creates a per-class histogram plot using Plotly.
        Each class is plotted as a grouped bar chart using exactly 10 bins.
        Each bin represents a 0.1-width range of confidence scores.
        """
        if self.data.empty:
            # Return a figure with an annotation if data is empty, instead of raising an error
            fig = go.Figure()
            fig.add_annotation(
                text="No detections available – nothing to plot",
                showarrow=False,
                x=0.5, y=0.5, xref="paper", yref="paper",
                font=dict(size=18))
            fig.update_layout(title=title)
            return fig
        
        # Set up confidence bins (10 bins from 0 to 1)
        bin_edges, bin_labels = self._confidence_bins(step=0.1)
        x_title = "Confidence score (bin width 0.1)"
        
        # Create figure
        fig = go.Figure()
        classes = sorted(self.data[self.class_col].unique())
        color_map = self._get_color_map(classes)

        # Detect global emptiness (GUI-friendly: return figure w/ annotation)
        if self.data[self.conf_col].dropna().empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No detections available – nothing to plot",
                showarrow=False,
                x=0.5, y=0.5, xref="paper", yref="paper",
                font=dict(size=18))
            fig.update_layout(title=title)
            return fig
        
        # Plot histogram for each class
        for cls in classes:
            cls_data = self.data[self.data[self.class_col] == cls]
            
            # Skip empty classes
            if cls_data.empty or cls_data[self.conf_col].dropna().empty:
                continue
            
            # Calculate histogram for detections
            counts, _ = np.histogram(cls_data[self.conf_col].dropna(), bins=bin_edges)
            
            # Add bar trace
            fig.add_trace(go.Bar(
                name=str(cls),
                x=bin_labels,
                y=counts,
                marker_color=color_map.get(cls),
                opacity=0.6,
                hovertemplate=(
                    "Bin: %{x}<br>"
                    "Species: " + str(cls) + "<br>"
                    "Count: %{y}<extra></extra>"
                )
            ))
        
        # Update layout - exactly matching the TimeDistributionPlotter layout
        fig.update_layout(
            barmode='group',
            title=title,
            xaxis_title=x_title,
            yaxis_title='Count', # Updated y-axis title
            legend_title='Species',
            legend=dict(x=1.02, y=1),
            margin=dict(r=150),
            showlegend=True
        )
        
        return fig
