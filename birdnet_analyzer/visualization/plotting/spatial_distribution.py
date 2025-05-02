import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple, Set
import numpy as np

class SpatialDistributionPlotter:
    """
    Class for creating spatial distribution plots showing detections by location.
    This visualization helps identify geographic patterns of species distributions.
    """
    
    def __init__(self, class_col: str = "Class"):
        """
        Initialize the SpatialDistributionPlotter.
        
        Args:
            class_col (str): Name of the column containing class/species labels
        """
        self.class_col = class_col
        self.color_map = {}
        # IMPORTANT: Use exactly the same base colors as TimeDistributionPlotter
        self.base_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']
    
    def _get_color_map(self, classes: List[str]) -> Dict[str, str]:
        """Create consistent color mapping for classes.
        
        Exactly matching TimeDistributionPlotter._get_color_map
        """
        # Sort classes alphabetically for consistent assignment
        sorted_classes = sorted(classes)
        colors = self.base_colors * (1 + len(sorted_classes) // len(self.base_colors))
        return {cls: colors[i] for i, cls in enumerate(sorted_classes)}
    
    def plot(self, 
             agg_df: pd.DataFrame, 
             all_locations_df: pd.DataFrame = None,
             title: str = "Spatial Distribution of Predictions by Class") -> go.Figure:
        """
        Creates a spatial distribution plot showing detections by location.
        """
        if agg_df.empty and all_locations_df is None:
            raise ValueError("No data to plot and no location data provided")
        
        # Handle empty results - show all locations with "No Detections"
        if agg_df.empty and all_locations_df is not None:
            agg_df = all_locations_df.copy()
            agg_df[self.class_col] = "No Detections"
            agg_df['count'] = 0
        
        # Find locations with no detections if all_locations_df is provided
        if not agg_df.empty and all_locations_df is not None:
            active_locations = set(agg_df[['site_name', 'latitude', 'longitude']].itertuples(index=False, name=None))
            all_locations = set(all_locations_df.itertuples(index=False, name=None))
            missing_locations = all_locations - active_locations
            
            if missing_locations:
                missing_df = pd.DataFrame(list(missing_locations), columns=['site_name', 'latitude', 'longitude'])
                missing_df[self.class_col] = "No Detections"
                missing_df['count'] = 0
                
                # Combine with aggregated data
                agg_df = pd.concat([agg_df, missing_df], ignore_index=True)
        
        # Get unique classes for consistent color mapping
        classes = sorted([c for c in agg_df[self.class_col].unique() if c != "No Detections"])
        
        # If we don't have a color map already, create it using the standard method
        if not self.color_map:
            self.color_map = self._get_color_map(classes)
        
        # Make sure all classes have colors and No Detections is always black
        for cls in classes:
            if cls not in self.color_map:
                # We need to rebuild the entire color map to ensure consistency
                tmp_map = self._get_color_map(list(self.color_map.keys()) + [cls])
                # Only update the missing class
                self.color_map[cls] = tmp_map[cls]
                
        # Always ensure "No Detections" has black color
        self.color_map["No Detections"] = 'black'
        
        # Get sorted classes for category order, with "No Detections" first
        sorted_classes = sorted([c for c in agg_df[self.class_col].unique() if c != "No Detections"])
        if "No Detections" in agg_df[self.class_col].unique():
            sorted_classes = ["No Detections"] + sorted_classes
        
        # Debug: Print the color assignments
        print("COLOR MAP USED FOR SPATIAL PLOT:")
        for cls in sorted(classes):
            print(f"  {cls}: {self.color_map.get(cls)}")
        
        # Create scatter mapbox plot
        fig = px.scatter_mapbox(
            agg_df,
            lat='latitude',
            lon='longitude',
            size='count',
            color=self.class_col,
            category_orders={self.class_col: sorted_classes},
            color_discrete_map=self.color_map,
            hover_data=['site_name', 'count'],
            size_max=50,
            zoom=10,
            height=600,
            title=title
        )
        
        # Adjust marker sizes - special case for "No Detections"
        for i, trace in enumerate(fig.data):
            if trace.name == "No Detections":
                # Set fixed small size for "No Detections" markers
                fig.data[i].marker.size = 8
                fig.data[i].marker.sizemode = "diameter"
                fig.data[i].marker.sizeref = 1
                fig.data[i].marker.sizemin = 8
            else:
                # Scale other markers by detection count
                max_count = max(agg_df['count']) if len(agg_df[agg_df['count'] > 0]) > 0 else 1
                size_scale = 50
                sizeref = 2.0 * max_count / (size_scale**2)
                fig.data[i].marker.sizeref = sizeref
                fig.data[i].marker.sizemin = 5
                fig.data[i].marker.sizemode = 'area'
            
            # Set opacity for all markers
            fig.data[i].marker.opacity = 0.8
            
            # Update hover template based on whether it's a "No Detections" point
            if trace.name == "No Detections":
                fig.data[i].hovertemplate = (
                    "Site: %{customdata[0]}<br>"
                    "Status: No Detections<br>"
                    "Latitude: %{lat:.2f}<br>"
                    "Longitude: %{lon:.2f}<br>"
                    "<extra></extra>"
                )
            else:
                fig.data[i].hovertemplate = (
                    "Site: %{customdata[0]}<br>"
                    "Count: %{customdata[1]}<br>"
                    "Latitude: %{lat:.2f}<br>"
                    "Longitude: %{lon:.2f}<br>"
                    "<extra></extra>"
                )
        
        # Set layout properties
        fig.update_layout(
            mapbox_style='open-street-map',
            margin={"r":0,"t":30,"l":0,"b":0},
            legend_title="Class",
            showlegend=True,
            mapbox=dict(
                center=dict(
                    lat=agg_df['latitude'].mean(), 
                    lon=agg_df['longitude'].mean()
                ),
                zoom=10
            ),
            modebar=dict(
                orientation='h',
                bgcolor='rgba(255, 255, 255, 0.8)',
                color='#333333',
                activecolor='#FF4B4B'
            ),
            modebar_add=[
                'zoom', 'zoomIn', 'zoomOut', 'resetViews'
            ],
            margin_pad=10,
            margin_t=50
        )
        
        return fig
