from typing import List, Optional, Dict, Tuple
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px  # Added import for plotly.express
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
        sorted_classes = sorted(classes)
        
        color_dict: Dict[str, str]
        # self.base_colors is defined in __init__ and has 7 colors.
        if len(sorted_classes) <= len(self.base_colors): # Effectively len(sorted_classes) <= 7
            color_dict = {cls: self.base_colors[i % len(self.base_colors)] for i, cls in enumerate(sorted_classes)}
        else:
            # Use a more diverse color palette if more than 7 classes
            extended_colors = px.colors.qualitative.Alphabet # Has 26 distinct colors
            # Cycle through extended_colors if more classes than available colors
            color_dict = {cls: extended_colors[i % len(extended_colors)] for i, cls in enumerate(sorted_classes)}
        return color_dict

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
            # Group by Year, Month, Day, Hour - then group result by Hour (formatted string)
            df = (cls_data.groupby(['Year', 'Month', 'Day', 'Hour'])
                  .size().reset_index(name='count'))
            df['Hour_display'] = df['Hour'].apply(lambda h: f"{h:02d}:00")
            return df.groupby('Hour_display')
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
        elif time_period == 'year': # Changed from else to elif
            # Group by Year, Month, Day - then group result by Year (as string)
            df = (cls_data.groupby(['Year', 'Month', 'Day'])
                  .size().reset_index(name='count'))
            df['Year_display'] = df['Year'].astype(str)
            return df.groupby('Year_display')
        else: # Add a final else to catch unexpected time_period values
            raise ValueError(f"Unhandled time period: {time_period}")

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
            # For grouped box plots, Plotly handles offsets if x is categorical and boxmode='group'
            for i, cls in enumerate(classes):
                cls_data = self.data[self.data[self.class_col] == cls].copy()

                y_values_for_species = []
                x_categories_for_species = [] # Store textual categories

                agg_data_grouped = self._aggregate_by_period(cls_data, time_period)

                # x_text contains the ordered list of category names (e.g., ['00:00', '01:00', ...] or ['Monday', ...])
                for period_name_text in x_text:
                    if period_name_text in agg_data_grouped.groups:
                        period_counts = agg_data_grouped.get_group(period_name_text)['count'].values
                        if len(period_counts) > 0:
                            y_values_for_species.extend(period_counts)
                            x_categories_for_species.extend([period_name_text] * len(period_counts))
                
                if y_values_for_species:
                    color = color_map.get(cls)
                    rgba_color = self._color_to_rgba(color, 0.6) # Adjusted alpha for better visibility

                    fig.add_trace(go.Box(
                        y=y_values_for_species,
                        x=x_categories_for_species, # Use textual categories for x
                        name=str(cls),
                        marker_color=color,
                        boxpoints=False, # Changed from 'outliers' to False
                        jitter=0.3,
                        pointpos=0, # Outliers relative to box
                        line=dict(width=1),
                        fillcolor=rgba_color,
                        hoveron='boxes', # Added to disable hover for outliers
                        hovertemplate=(
                            f"Species: %{{name}}<br>" +
                            f"{x_title}: %{{x}}<br>" + # %{x} will now be the textual category
                            "Median: %{median:.2f}<br>" +
                            "Q1: %{q1:.2f}<br>" +
                            "Q3: %{q3:.2f}<br>" +
                            "Min: %{lowerfence:.2f}<br>" + # Using lowerfence for robust min
                            "Max: %{upperfence:.2f}<br>" + # Using upperfence for robust max
                            "<extra></extra>"
                        )
                    ))
            
            # Layout updates for boxplot mode
            fig.update_layout(
                boxmode='group', # Key for grouped box plots
                xaxis=dict(
                    categoryorder='array', # Ensure x-axis categories follow the order in x_text
                    categoryarray=x_text,
                    title_text=x_title # Explicitly set x_title here
                )
                # yaxis_title='Count' will be set in the common layout update section
            )
        else:
            # Original histogram plotting logic
            for cls in classes:
                cls_data = self.data[self.data[self.class_col] == cls].copy()  # Use copy
                grouped = self._aggregate_by_period(cls_data, time_period)

                counts = []
                # Iterate over x_text (string labels) which match the keys in `grouped`
                for text_val in x_text: 
                    count = (grouped.get_group(text_val)['count'].sum()
                            if text_val in grouped.groups else 0)
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

            # Specific layout for bar charts (barmode)
            fig.update_layout(barmode='group', xaxis_title=x_title)

        # Update layout (common for both boxplot and histogram)
        plot_type_title = "Boxplots" if use_boxplot else "Distribution"
        final_title = title or f'Species {plot_type_title} by {x_title}'
        
        fig.update_layout(
            title=final_title,
            yaxis_title='Count',
            legend_title='Species',
            legend=dict(x=1.02, y=1),
            margin=dict(r=150),
            showlegend=True
        )

        return fig
