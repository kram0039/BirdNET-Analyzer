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
    
    def add_sunrise_sunset(
        self,
        fig: go.Figure,
        latitude: float,
        longitude: float
    ) -> go.Figure:
        """
        Add local-time sunrise and sunset lines to *fig*.

        Requires:
        ─ pip install astral timezonefinder             # (pytz if Python < 3.9)

        Parameters
        ----------
        fig : plotly.graph_objects.Figure
            Figure returned by `plot()`.
        latitude, longitude : float
            Site coordinates in decimal degrees.
        """
        try:
            from astral import LocationInfo
            from astral.sun import sun
            from timezonefinder import TimezoneFinder
            from datetime import timedelta, date

            # ── tz helper ──────────────────────────────────────────────────
            try:                     # Python ≥ 3.9
                from zoneinfo import ZoneInfo
                def _tz(name): return ZoneInfo(name)
            except ModuleNotFoundError:     # older Python
                import pytz                 # pip install pytz
                def _tz(name): return pytz.timezone(name)

            # ── 1.  Get IANA zone name for the lat/lon ─────────────────────
            tz_name = TimezoneFinder().timezone_at(lat=latitude, lng=longitude)
            if tz_name is None:           # ocean / fallback
                tz_name = "UTC"
            tz = _tz(tz_name)

            site = LocationInfo("Site", tz_name, tz_name,
                                latitude=latitude, longitude=longitude)

            # ── 2.  Choose dates to evaluate (every 10 d + last) ───────────
            unique_dates = sorted(self.data["date"].unique())
            if not unique_dates:
                return fig

            first_day, last_day = unique_dates[0], unique_dates[-1]

            calc_days: list[date] = []
            d = first_day
            while d <= last_day:
                calc_days.append(d)
                d += timedelta(days=10)
            if calc_days[-1] != last_day:
                calc_days.append(last_day)

            # ── 3.  Collect sunrise/sunset decimal-hours ───────────────────
            sunrise_x, sunrise_y = [], []
            sunset_x,  sunset_y  = [], []

            for d in calc_days:
                s = sun(site.observer, date=d, tzinfo=tz)  # tz-aware dt

                for key, x_list, y_list in (("sunrise", sunrise_x, sunrise_y),
                                            ("sunset",  sunset_x,  sunset_y )):
                    t_local = s[key].astimezone(tz)        # local wall-clock
                    dec = (t_local.hour +
                           t_local.minute / 60 +
                           t_local.second / 3600)
                    x_list.append(d)
                    y_list.append(dec)

            # ── 4.  Plot lines ─────────────────────────────────────────────
            line_style = dict(color="purple", width=2)

            fig.add_trace(go.Scatter(
                x=sunrise_x, y=sunrise_y, mode="lines",
                line=line_style, name="Sunrise"
            ))
            fig.add_trace(go.Scatter(
                x=sunset_x,  y=sunset_y,  mode="lines",
                line=line_style, name="Sunset"
            ))

        except ImportError as e:
            print("Missing dependency:", e)
            print("Install with:  pip install astral timezonefinder pytz")
        except Exception as e:
            print("Sunrise/sunset calculation failed:", e)

        return fig

