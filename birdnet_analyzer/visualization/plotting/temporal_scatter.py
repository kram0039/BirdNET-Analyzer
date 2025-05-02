import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional
import datetime
from pathlib import Path

class TemporalScatterPlotter:
    """
    Class for creating temporal scatter plots showing detections by date and time of day.
    This visualization helps identify daily activity patterns across different dates.
    """
    
    def __init__(self, data: pd.DataFrame, class_col: str, conf_col: str):
        """
        Initialize the TemporalScatterPlotter with a DataFrame and column names.

        Args:
            data (pd.DataFrame): DataFrame containing prediction data with datetime information
            class_col (str): Name of the column containing class/species labels
            conf_col (str): Name of the column containing confidence scores
        """
        self.data = data.copy()
        self.class_col = class_col
        self.conf_col = conf_col
        self.color_map = {}
        # Match the base colors with other plotters
        self.base_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']
        
        # Ensure required columns exist
        if 'prediction_time' not in self.data.columns or self.data['prediction_time'].isnull().all():
            raise ValueError("Prediction time data is not available or all null")
        
        # Extract date and time components for plotting
        self.data['date'] = self.data['prediction_time'].dt.date
        
        # Convert time to decimal hours for plotting (e.g. 14:30 = 14.5)
        self.data['decimal_time'] = (self.data['prediction_time'].dt.hour + 
                                    self.data['prediction_time'].dt.minute/60 + 
                                    self.data['prediction_time'].dt.second/3600)
    
    def _get_color_map(self, classes: List[str]) -> Dict[str, str]:
        """Create consistent color mapping for classes."""
        # Sort classes alphabetically first - ensuring consistent ordering with other plotters
        sorted_classes = sorted(classes)
        colors = self.base_colors * (1 + len(sorted_classes) // len(self.base_colors))
        self.color_map = {cls: colors[i] for i, cls in enumerate(sorted_classes)}
        return self.color_map

    def plot(self, title: str = "Temporal Distribution of Detections") -> go.Figure:
        """
        Creates a temporal scatter plot showing detections by date and time of day.
        
        Args:
            title (str): The title for the plot
            
        Returns:
            plotly.graph_objects.Figure: The generated plotly figure
        """
        if self.data.empty:
            raise ValueError("No data to plot")
        
        # Get all classes and sort them alphabetically for consistent ordering
        all_classes = sorted(self.data[self.class_col].unique())
        
        # Get or create color map
        color_map = self.color_map or self._get_color_map(all_classes)
        
        # Create the scatter plot - explicitly setting category_orders for consistent legend
        fig = px.scatter(
            self.data,
            x='date',
            y='decimal_time',
            color=self.class_col,
            color_discrete_map=color_map,
            category_orders={self.class_col: all_classes},  # Explicitly set order
            hover_data=[self.class_col, self.conf_col],
            opacity=0.3,
            title=title
        )
        
        # Format hover template
        for trace in fig.data:
            trace.hovertemplate = (
                "Date: %{x|%Y-%m-%d}<br>" +
                "Time: %{y:.2f} hrs<br>" +
                "Species: %{customdata[0]}<br>" +
                "Confidence: %{customdata[1]:.3f}<br>" +
                "<extra></extra>"
            )
        
        # Customize layout
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Time of Day (hours)",
            yaxis=dict(
                tickmode='array',
                tickvals=[0, 3, 6, 9, 12, 15, 18, 21, 24],
                ticktext=['00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00', '24:00']
            ),
            legend_title="Species",
            legend=dict(
                x=1.02, 
                y=1,
                itemsizing='constant',  # Makes legend symbols the same size
                traceorder='normal'     # Use the order specified in category_orders
            ),
            margin=dict(r=150),  # Add right margin for legend
        )
        
        return fig
    
    def add_sunrise_sunset(self, fig: go.Figure, latitude: float, longitude: float) -> go.Figure:
        """
        Adds sunrise and sunset lines to the plot.
        
        Args:
            fig (plotly.graph_objects.Figure): The figure to add lines to
            latitude (float): Latitude for sunrise/sunset calculations
            longitude (float): Longitude for sunrise/sunset calculations
            
        Returns:
            plotly.graph_objects.Figure: The updated figure
        """
        try:
            from astral import LocationInfo
            from astral.sun import sun
            from datetime import timedelta
            
            # Get unique dates in the data
            unique_dates = sorted(self.data['date'].unique())
            if not unique_dates:
                return fig
                
            # Create location info for the site's average location
            site = LocationInfo("RecordingSite", "Region", "UTC", latitude, longitude)
            
            # Define color for sunrise/sunset lines
            sun_line_color = "purple"
            
            # Calculate sunrise/sunset only every 10 days, starting from the first date
            first_date = min(unique_dates)
            last_date = max(unique_dates)
            
            # Find dates to calculate (every 10 days)
            calculation_dates = []
            current_date = first_date
            while current_date <= last_date:
                calculation_dates.append(current_date)
                current_date += timedelta(days=10)
            
            # Include last date if it's not already included
            if calculation_dates[-1] != last_date:
                calculation_dates.append(last_date)
                
            # Calculate sunrise/sunset for the selected dates
            sunrise_data = {}  # {date: decimal_time}
            sunset_data = {}   # {date: decimal_time}
            
            for calc_date in calculation_dates:
                try:
                    sun_info = sun(site.observer, date=calc_date)
                    
                    # Extract sunrise and sunset times
                    sunrise_time = sun_info['sunrise']
                    sunset_time = sun_info['sunset']
                    
                    # Remove timezone info if present
                    if sunrise_time.tzinfo is not None:
                        sunrise_time = sunrise_time.replace(tzinfo=None)
                    if sunset_time.tzinfo is not None:
                        sunset_time = sunset_time.replace(tzinfo=None)
                    
                    # Convert to decimal hours for plotting
                    sunrise_decimal = sunrise_time.hour + sunrise_time.minute/60 + sunrise_time.second/3600
                    sunset_decimal = sunset_time.hour + sunset_time.minute/60 + sunset_time.second/3600
                    
                    # Store calculations
                    sunrise_data[calc_date] = sunrise_decimal
                    sunset_data[calc_date] = sunset_decimal
                    
                except Exception as e:
                    print(f"Error calculating sunrise/sunset for {calc_date}: {str(e)}")
            
            # Create continuous sunrise and sunset lines
            sunrise_x = []
            sunrise_y = []
            sunset_x = []
            sunset_y = []
            
            # Convert calculated points to lists for plotting
            for date_val in sorted(sunrise_data.keys()):
                sunrise_x.append(date_val)
                sunrise_y.append(sunrise_data[date_val])
                
                sunset_x.append(date_val)
                sunset_y.append(sunset_data[date_val])
            
            # Add sunrise line
            fig.add_trace(
                go.Scatter(
                    x=sunrise_x,
                    y=sunrise_y,
                    mode='lines',
                    line=dict(color=sun_line_color, width=2),
                    name='Sunrise',
                    showlegend=True
                )
            )
            
            # Add sunset line
            fig.add_trace(
                go.Scatter(
                    x=sunset_x,
                    y=sunset_y,
                    mode='lines',
                    line=dict(color=sun_line_color, width=2),
                    name='Sunset',
                    showlegend=True
                )
            )
                
        except ImportError:
            print("Astral is not installed. Install with: pip install astral")
            
        except Exception as e:
            print(f"Error adding sunrise/sunset lines: {str(e)}")
            
        return fig
