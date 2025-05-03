from typing import List, Optional, Dict, Tuple
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import matplotlib.colors as mcolors  # For color conversion
import calendar  # Import calendar for month/day names

class TimeDistributionPlotter:
    def __init__(self, data: pd.DataFrame, class_col: str):
        self.data = data
        self.class_col = class_col
        self.base_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']
        # Define mappings for month and weekday
        self.month_map = {i: calendar.month_name[i] for i in range(1, 13)}
        self.weekday_map = {i: calendar.day_name[i] for i in range(7)}

    def _get_color_map(self, classes: List[str]) -> Dict[str, str]:
        """Create consistent color mapping for classes."""
        colors = self.base_colors * (1 + len(classes) // len(self.base_colors))
        return {cls: colors[i] for i, cls in enumerate(sorted(classes))}

    def _color_to_rgba(self, color: str, opacity: float = 0.5) -> str:
        """Convert any color format to rgba string."""
        # Convert named color or hex to RGB values
        rgb = mcolors.to_rgb(color)
        # Scale to 0-255 range and build rgba string
        return f'rgba({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)},{opacity})'

    def _aggregate_by_period(self, cls_data: pd.DataFrame, time_period: str):
        """Aggregate data by specified time period using standard column names."""
        # Ensure required columns exist
        required_cols = {
            'hour': ['Year', 'Month', 'Day', 'Hour'],
            'day': ['Year', 'Month', 'Day', 'prediction_weekday'],
            'month': ['Year', 'Month'],
            'year': ['Year', 'Month', 'Day']
        }
        if time_period not in required_cols:
            raise ValueError(f"Invalid time period: {time_period}")

        missing_cols = [col for col in required_cols[time_period] if col not in cls_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for time period '{time_period}': {', '.join(missing_cols)}")

        if time_period == 'hour':
            # Group by Year, Month, Day, Hour - then group result by Hour
            return (cls_data.groupby(['Year', 'Month', 'Day', 'Hour'])
                   .size().reset_index(name='count')
                   .groupby('Hour'))
        elif time_period == 'day':
            # Map numeric weekday to name before grouping
            cls_data['weekday_name'] = cls_data['prediction_weekday'].map(self.weekday_map)
            # Group by Year, Month, Day, weekday_name - then group result by weekday_name
            return (cls_data.groupby(['Year', 'Month', 'Day', 'weekday_name'])
                   .size().reset_index(name='count')
                   .groupby('weekday_name'))
        elif time_period == 'month':
            # Map numeric month to name before grouping
            cls_data['month_name'] = cls_data['Month'].map(self.month_map)
            # Group by Year, month_name - then group result by month_name
            return (cls_data.groupby(['Year', 'month_name'])
                   .size().reset_index(name='count')
                   .groupby('month_name'))
        else:  # year
            # Group by Year, Month, Day - then group result by Year
            return (cls_data.groupby(['Year', 'Month', 'Day'])
                   .size().reset_index(name='count')
                   .groupby('Year'))

    def plot_distribution(self, time_period: str, use_boxplot: bool = False, title: str = None) -> go.Figure:
        """Plot species counts distribution over the specified time period."""
        if self.data.empty:
            raise ValueError("No data to plot")

        # Ensure base columns for aggregation exist
        base_cols = ['Year', 'Month', 'Day', 'Hour', 'prediction_weekday']
        missing_base = [col for col in base_cols if col not in self.data.columns and col in ['Year', 'Month', 'Day', 'Hour', 'prediction_weekday']]
        if missing_base:
            raise ValueError(f"Missing base columns needed for aggregation: {', '.join(missing_base)}")

        # Convert relevant columns to numeric if they aren't already, handling potential errors
        for col in ['Year', 'Month', 'Day', 'Hour', 'prediction_weekday']:
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        # Drop rows where essential numeric columns became NaN
        self.data.dropna(subset=missing_base, inplace=True)
        # Convert to integer type after handling NaNs
        for col in missing_base:
             self.data[col] = self.data[col].astype(int)

        # Set up period-specific configurations
        if time_period == 'hour':
            x_title = 'Hour of Day'
            x_values = list(range(24))
            x_text = [f"{h:02d}:00" for h in range(24)]
        elif time_period == 'day':
            x_title = 'Day of Week'
            # Use calendar day names in standard order
            x_values = list(calendar.day_name)
            x_text = x_values
        elif time_period == 'month':
            x_title = 'Month'
            # Use calendar month names in standard order
            x_values = list(calendar.month_name)[1:]  # Skip empty string at index 0
            x_text = x_values
        elif time_period == 'year':
            x_title = 'Year'
            # Ensure 'Year' column exists and is numeric before getting unique values
            if 'Year' not in self.data.columns or not pd.api.types.is_numeric_dtype(self.data['Year']):
                 raise ValueError("Column 'Year' is missing or not numeric.")
            x_values = sorted(self.data['Year'].dropna().unique().astype(int))
            x_text = [str(y) for y in x_values]
        else:
            raise ValueError(f"Invalid time period: {time_period}")

        # Create figure
        fig = go.Figure()
        classes = sorted(self.data[self.class_col].unique())
        color_map = self._get_color_map(classes)

        if use_boxplot:
            # Offset boxplots for each class to avoid overlap
            num_classes = len(classes)
            offsets = np.linspace(-0.3, 0.3, num_classes)

            for i, cls in enumerate(classes):
                cls_data = self.data[self.data[self.class_col] == cls].copy()  # Use copy to avoid SettingWithCopyWarning

                # Prepare data for boxplot
                period_values = []
                y_values = []
                x_positions = []

                # Get data for each period
                agg_data_grouped = self._aggregate_by_period(cls_data, time_period)

                for j, period in enumerate(x_values):
                    # Find data for current period
                    if period in agg_data_grouped.groups:
                        period_counts = agg_data_grouped.get_group(period)['count'].values
                        if len(period_counts) > 0:
                            # Store data for boxplot
                            period_values.extend([str(period)] * len(period_counts))
                            y_values.extend(period_counts)
                            # Calculate offset position
                            x_positions.extend([j + offsets[i]] * len(period_counts))

                if len(period_values) > 0:
                    # Convert color to rgba format
                    color = color_map.get(cls)
                    rgba_color = self._color_to_rgba(color, 0.5)

                    # Create a single boxplot for this class with custom positions
                    fig.add_trace(go.Box(
                        x=x_positions,
                        y=y_values,
                        name=str(cls),
                        marker_color=color,
                        boxpoints='outliers',  # Show outliers
                        jitter=0.3,           # Add jitter to points
                        pointpos=0,           # Offset of points from box
                        line=dict(width=2),   # Box line width
                        fillcolor=rgba_color,  # Use properly converted color
                        hovertemplate=(
                            "Species: " + str(cls) + "<br>" +
                            "Period: " + str(period) + "<br>" +  # Add period info to hover
                            "Count: %{y}<br>" +
                            "Median: %{median}<br>" +
                            "Q1: %{q1}<br>" +
                            "Q3: %{q3}<br>" +
                            "<extra></extra>"
                        )
                    ))

            # Set custom x-axis ticks and labels
            fig.update_layout(
                xaxis=dict(
                    tickmode='array',
                    tickvals=list(range(len(x_text))),
                    ticktext=x_text
                )
            )
        else:
            # Original histogram plotting logic
            for cls in classes:
                cls_data = self.data[self.data[self.class_col] == cls].copy()  # Use copy
                grouped = self._aggregate_by_period(cls_data, time_period)

                counts = []
                for val in x_values:
                    count = (grouped.get_group(val)['count'].sum()
                            if val in grouped.groups else 0)
                    counts.append(count)

                fig.add_trace(go.Bar(
                    name=str(cls),
                    x=x_text,
                    y=counts,
                    marker_color=color_map.get(cls),
                    opacity=0.6,
                    hovertemplate=(
                        f"{x_title}: %{{x}}<br>"
                        "Species: " + str(cls) + "<br>"
                        "Count: %{y}<br>"
                        "<extra></extra>"
                    )
                ))

        # Update layout
        plot_type = "Boxplots" if use_boxplot else "Distribution"
        fig.update_layout(
            barmode='group' if not use_boxplot else None,
            boxmode='group' if use_boxplot else None,
            title=title or f'Species {plot_type} by {x_title}',
            xaxis_title=x_title,
            yaxis_title='Count',
            legend_title='Species',
            legend=dict(x=1.02, y=1),
            margin=dict(r=150),
            showlegend=True
        )

        return fig
