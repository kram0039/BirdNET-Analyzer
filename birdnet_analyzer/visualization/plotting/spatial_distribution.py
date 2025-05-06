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
            class_col (str): Name of the column containing class/species labels.
                             Defaults to "Class" (standard name from DataProcessor).
        """
        self.class_col = class_col # Should typically be "Class"
        self.color_map = {}
        # IMPORTANT: Use exactly the same base colors as TimeDistributionPlotter
        self.base_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']

    def _get_color_map(self, classes: List[str]) -> Dict[str, str]:
        """Create consistent color mapping for classes."""
        sorted_classes = sorted(classes) # Sort once
        
        color_dict: Dict[str, str]
        # self.base_colors is defined in __init__ and has 7 colors.
        if len(sorted_classes) <= len(self.base_colors): # Effectively len(sorted_classes) <= 7
            # Cycle through base_colors.
            color_dict = {cls: self.base_colors[i % len(self.base_colors)] for i, cls in enumerate(sorted_classes)}
        else:
            # Use a more diverse color palette if more than 7 classes
            extended_colors = px.colors.qualitative.Alphabet # Has 26 distinct colors
            # Cycle through extended_colors if more classes than available colors
            color_dict = {cls: extended_colors[i % len(extended_colors)] for i, cls in enumerate(sorted_classes)}
        return color_dict

    def plot(self,
             agg_df: pd.DataFrame, # Expected columns: Site, Latitude, Longitude, Class, count
             all_locations_df: pd.DataFrame = None, # Expected columns: site_name, latitude, longitude
             title: str = "Spatial Distribution of Predictions by Class") -> go.Figure:
        """
        Creates a spatial distribution plot showing detections by location.
        Uses standard column names: Site, Latitude, Longitude, Class.
        """
        # Standard column names expected/used internally
        site_col = "Site"
        lat_col = "Latitude"
        lon_col = "Longitude"
        class_col = self.class_col # Typically "Class"

        if agg_df.empty and all_locations_df is None:
            raise ValueError("No data to plot and no location data provided")

        # --- Standardize all_locations_df columns ---
        if all_locations_df is not None:
            all_locations_df = all_locations_df.copy()
            # Rename columns to standard names if they exist with old names
            rename_map = {}
            if 'site_name' in all_locations_df.columns and site_col not in all_locations_df.columns:
                rename_map['site_name'] = site_col
            if 'latitude' in all_locations_df.columns and lat_col not in all_locations_df.columns:
                rename_map['latitude'] = lat_col
            if 'longitude' in all_locations_df.columns and lon_col not in all_locations_df.columns:
                rename_map['longitude'] = lon_col
            if rename_map:
                all_locations_df.rename(columns=rename_map, inplace=True)

            # Check if standard columns are now present
            required_loc_cols = [site_col, lat_col, lon_col]
            missing_loc_cols = [col for col in required_loc_cols if col not in all_locations_df.columns]
            if missing_loc_cols:
                 raise ValueError(f"all_locations_df is missing required columns after standardization: {', '.join(missing_loc_cols)}")


        # Handle empty results - show all locations with "No Detections"
        if agg_df.empty and all_locations_df is not None:
            agg_df = all_locations_df[[site_col, lat_col, lon_col]].copy() # Use standard names
            agg_df[class_col] = "No Detections"
            agg_df['count'] = 0

        # Find locations with no detections if all_locations_df is provided
        elif not agg_df.empty and all_locations_df is not None:
            # Ensure agg_df has the required columns
            required_agg_cols = [site_col, lat_col, lon_col, class_col, 'count']
            missing_agg_cols = [col for col in required_agg_cols if col not in agg_df.columns]
            if missing_agg_cols:
                 raise ValueError(f"agg_df is missing required columns: {', '.join(missing_agg_cols)}")

            # Use standard column names for comparison
            active_locations = set(agg_df[[site_col, lat_col, lon_col]].itertuples(index=False, name=None))
            all_locations = set(all_locations_df[[site_col, lat_col, lon_col]].itertuples(index=False, name=None))
            missing_locations = all_locations - active_locations

            if missing_locations:
                missing_df = pd.DataFrame(list(missing_locations), columns=[site_col, lat_col, lon_col]) # Use standard names
                missing_df[class_col] = "No Detections"
                missing_df['count'] = 0

                # Combine with aggregated data
                agg_df = pd.concat([agg_df, missing_df], ignore_index=True)

        # Check required columns again after potential modifications
        final_required_cols = [site_col, lat_col, lon_col, class_col, 'count']
        missing_final_cols = [col for col in final_required_cols if col not in agg_df.columns]
        if missing_final_cols:
            raise ValueError(f"Final agg_df is missing required columns: {', '.join(missing_final_cols)}")


        # Get unique classes for consistent color mapping
        classes = sorted([c for c in agg_df[class_col].unique() if c != "No Detections"])

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
        sorted_classes = sorted([c for c in agg_df[class_col].unique() if c != "No Detections"])
        if "No Detections" in agg_df[class_col].unique():
            sorted_classes = ["No Detections"] + sorted_classes

        # Debug: Print the color assignments
        print("COLOR MAP USED FOR SPATIAL PLOT:")
        for cls in sorted(classes): # Use 'classes' which excludes "No Detections"
            print(f"  {cls}: {self.color_map.get(cls)}")
        if "No Detections" in self.color_map:
             print(f"  No Detections: {self.color_map['No Detections']}")


        # Create scatter mapbox plot using standard column names
        fig = px.scatter_mapbox(
            agg_df,
            lat=lat_col,        # Use standard Latitude
            lon=lon_col,        # Use standard Longitude
            size='count',
            color=class_col,    # Use standard Class
            category_orders={class_col: sorted_classes},
            color_discrete_map=self.color_map,
            hover_data=[site_col, 'count'], # Use standard Site
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
            # Uses standard column names via customdata mapping
            if trace.name == "No Detections":
                fig.data[i].hovertemplate = (
                    f"{site_col}: %{{customdata[0]}}<br>" # Use standard Site name
                    "Status: No Detections<br>"
                    f"{lat_col}: %{{lat:.2f}}<br>"       # Use standard Latitude name
                    f"{lon_col}: %{{lon:.2f}}<br>"       # Use standard Longitude name
                    "<extra></extra>"
                )
            else:
                fig.data[i].hovertemplate = (
                    f"Species: {trace.name}<br>" + # Added Species
                    f"{site_col}: %{{customdata[0]}}<br>" 
                    "Count: %{customdata[1]}<br>"
                    f"{lat_col}: %{{lat:.2f}}<br>"
                    f"{lon_col}: %{{lon:.2f}}<br>"
                    "<extra></extra>"
                )

        # Set layout properties
        fig.update_layout(
            mapbox_style='open-street-map',
            margin={"r":0,"t":30,"l":0,"b":0},
            legend_title="Class", # Use standard Class name
            showlegend=True,
            mapbox=dict(
                center=dict(
                    lat=agg_df[lat_col].mean(), # Use standard Latitude
                    lon=agg_df[lon_col].mean()  # Use standard Longitude
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
