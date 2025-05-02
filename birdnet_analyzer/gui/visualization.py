import json
import os
import shutil
import tempfile
import typing
import io
from pathlib import Path
import datetime

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import birdnet_analyzer.gui.localization as loc
import birdnet_analyzer.gui.utils as gu
from birdnet_analyzer.visualization.data_processor import DataProcessor
from birdnet_analyzer.visualization.plotting.confidences import ConfidencePlotter
from birdnet_analyzer.visualization.plotting.time_distributions import TimeDistributionPlotter
from birdnet_analyzer.visualization.plotting.temporal_scatter import TemporalScatterPlotter
from birdnet_analyzer.visualization.plotting.spatial_distribution import SpatialDistributionPlotter
from birdnet_analyzer.visualization.detection_counts import calculate_detection_counts

# Import from common module instead of defining here
from birdnet_analyzer.visualization.common import ProcessorState, apply_class_thresholds, apply_datetime_filters

def get_date_range(df: pd.DataFrame) -> tuple:
    """Get the earliest and latest dates from predictions."""
    try:
        if 'prediction_time' not in df.columns or df.empty:
            return None, None
        
        min_date = df['prediction_time'].min()
        max_date = df['prediction_time'].max()
        
        if pd.isna(min_date) or pd.isna(max_date):
            return None, None
            
        # Set time to start and end of day
        start_date = min_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = max_date.replace(hour=23, minute=59, second=59, microsecond=999999)
        
        return start_date, end_date
    except Exception:
        return None, None

def build_visualization_tab():
    """
    Builds a Gradio tab for loading and plotting prediction data only,
    using ConfidencePlotter. Annotation logic and metric calculations
    have been removed.
    """

    # Default columns for predictions (kept for user convenience)
    prediction_default_columns = {
        "Start Time": "File Offset (s)",
        "Class": "Common Name",
        "Recording": "Begin Path",
        "Confidence": "Confidence",
        "Correctness": "correctness",
    }

    # Default columns for metadata
    metadata_default_columns = {
        "Site": "Site",
        "X": "lat",
        "Y": "lon",
    }

    localized_column_labels = {
        "Start Time": loc.localize("eval-tab-column-start-time-label"),
        "Class": loc.localize("eval-tab-column-class-label"),
        "Recording": loc.localize("eval-tab-column-recording-label"),
        "Confidence": loc.localize("eval-tab-column-confidence-label"),
        "Correctness": loc.localize("viz-tab-column-correctness-label"),
        "Site": loc.localize("viz-tab-column-site-label"),
        "X": loc.localize("viz-tab-column-latitude-label"),
        "Y": loc.localize("viz-tab-column-longitude-label"),
    }

    def get_columns_from_uploaded_files(files):
        """
        Reads the header row of each file to discover available columns.
        """
        columns = set()
        if files:
            for file_obj in files:
                try:
                    df = pd.read_csv(file_obj, sep=None, engine="python", nrows=0)
                    columns.update(df.columns)
                except Exception as e:
                    print(f"Error reading file {file_obj}: {e}")
                    gr.Warning(f"{loc.localize('eval-tab-warning-error-reading-file')} {file_obj}")
        return sorted(list(columns))

    def save_uploaded_files(files):
        """
        Saves uploaded files into a temporary directory and returns its path.
        """
        if not files:
            return None
        temp_dir = tempfile.mkdtemp()
        for file_obj in files:
            dest_path = os.path.join(temp_dir, os.path.basename(file_obj))
            shutil.copy(file_obj, dest_path)
        return temp_dir

    def initialize_processor(
        prediction_files,
        pred_start_time,
        pred_class,
        pred_confidence,
        pred_recording,
        pred_correctness=None,  # Add correctness parameter with default
        prediction_dir=None,
    ):
        """Creates a simplified DataProcessor for predictions only."""
        if not prediction_files:
            return [], [], None, None

        try:
            # Initialize processor
            if prediction_dir is None:
                prediction_dir = save_uploaded_files(prediction_files)

            # Debug just number of files
            print(f"\nProcessing {len(prediction_files)} prediction files")
            print(f"Using prediction directory: {prediction_dir}")

            # Set up column mappings with proper fallbacks
            cols_pred = {}
            for key, default in prediction_default_columns.items():
                if key == "Start Time":
                    cols_pred[key] = pred_start_time or default
                elif key == "Class":
                    cols_pred[key] = pred_class or default
                elif key == "Confidence":
                    cols_pred[key] = pred_confidence or default
                elif key == "Recording":
                    cols_pred[key] = pred_recording or default
                elif key == "Correctness":
                    cols_pred[key] = pred_correctness or default

            print("Using column mappings:", cols_pred)

            proc = DataProcessor(
                prediction_directory_path=prediction_dir,
                prediction_file_name=None,
                columns_predictions=cols_pred,
            )
            
            # Get data and extract unique values
            df = proc.get_data()
            
            # Debug dataframe shape and columns
            print(f"\nLoaded DataFrame shape: {df.shape}")
            print(f"DataFrame columns: {df.columns}")
            
            avail_classes = list(proc.classes) 
            print(f"\nFound {len(avail_classes)} unique classes")
            
            # Get and clean up recording names consistently (strip and lowercase)
            recordings = df["recording_filename"].dropna().unique()
            avail_recordings = []
            for rec in recordings:
                if isinstance(rec, str):
                    clean_name = os.path.splitext(os.path.basename(rec.strip()))[0].lower()
                    if clean_name and clean_name not in avail_recordings:
                        avail_recordings.append(clean_name)
            avail_recordings.sort()
            print(f"Found {len(avail_recordings)} unique recordings")

            return avail_classes, avail_recordings, proc, prediction_dir

        except Exception as e:
            print(f"Error in initialize_processor: {e}")
            raise gr.Error(f"Error initializing processor: {str(e)}")

    def update_prediction_columns(uploaded_files):
        """
        Called when user selects prediction files. Reads headers and updates dropdowns.
        """
        cols = get_columns_from_uploaded_files(uploaded_files)
        cols = [""] + cols
        updates = []
        for label in ["Start Time", "Class", "Confidence", "Recording", "Correctness"]:
            default_val = prediction_default_columns.get(label)
            val = default_val if default_val in cols else None
            updates.append(gr.update(choices=cols, value=val))
        return updates

    def update_metadata_columns(uploaded_files):
        """
        Called when user selects metadata files. Reads headers and updates dropdowns.
        """
        cols = set()
        if uploaded_files:
            for file_obj in uploaded_files:
                try:
                    # Try reading with CSV engine first
                    df = pd.read_csv(file_obj, nrows=0)
                    cols.update(df.columns)
                except Exception:
                    try:
                        # Fallback to python engine with automatic delimiter detection
                        df = pd.read_csv(file_obj, sep=None, engine="python", nrows=0)
                        cols.update(df.columns)
                    except Exception as e:
                        print(f"Error reading file {file_obj}: {e}")
                        gr.Warning(f"{loc.localize('eval-tab-warning-error-reading-file')} {file_obj}")
        
        cols = [""] + sorted(list(cols))
        updates = []
        for label in ["Site", "X", "Y"]:
            default_val = metadata_default_columns.get(label)
            val = default_val if default_val in cols else None
            updates.append(gr.update(choices=cols, value=val))
        return updates

    def update_selections(
        prediction_files,
        metadata_files,
        pred_start_time,
        pred_class,
        pred_confidence,
        pred_recording,
        pred_correctness,
        meta_site,
        meta_x,
        meta_y,
        current_classes,
        current_recordings,
    ):
        """
        Called to create or update the processor based on current file uploads and column selections.
        Preserves any selected classes/recordings that remain valid.
        Initializes the class threshold DataFrame and makes the update button visible.
        """
        prediction_dir = save_uploaded_files(prediction_files)
        metadata_dir = save_uploaded_files(metadata_files)
        
        avail_classes, avail_recordings, proc, prediction_dir = initialize_processor(
            prediction_files,
            pred_start_time,
            pred_class,
            pred_confidence,
            pred_recording,
            pred_correctness,
            prediction_dir,
        )
        
        # Initialize class thresholds DataFrame
        class_thresholds_init_df = None
        threshold_df_update = gr.update(visible=False, value=None) # Default to hidden
        threshold_json_btn_update = gr.update(visible=False) # Update for JSON select button
        threshold_download_btn_update = gr.update(visible=False) # Update for download button
        
        if proc:
            # Create DataFrame for thresholds
            class_thresholds_init_df = pd.DataFrame({
                'Class': sorted(avail_classes),
                'Threshold': [0.10] * len(avail_classes) # Default threshold
            })
            threshold_df_update = gr.update(visible=True, value=class_thresholds_init_df)
            threshold_json_btn_update = gr.update(visible=True) # Make JSON select button visible
            threshold_download_btn_update = gr.update(visible=True) # Make download button visible

            state = ProcessorState(
                processor=proc, 
                prediction_dir=prediction_dir, 
                metadata_dir=metadata_dir,
                class_thresholds=class_thresholds_init_df # Store initial thresholds in state
            )
        else:
            state = None # No processor created

        # Keep current selections if they exist in available options
        new_classes = []
        new_recordings = []

        if current_classes:
            new_classes = [c for c in current_classes if c in avail_classes]
        if current_recordings:
            normalized_current = [os.path.splitext(os.path.basename(r.strip()))[0].lower() 
                               for r in current_recordings if isinstance(r, str)]
            new_recordings = [r for r in normalized_current if r in avail_recordings]

        # Default to all available if no valid selections remain
        if not new_classes:
            new_classes = avail_classes
        if not new_recordings:
            new_recordings = avail_recordings

        return (
            gr.update(choices=avail_classes, value=new_classes),
            gr.update(choices=avail_recordings, value=new_recordings),
            state,
            threshold_df_update, # Return update for the DataFrame UI
            threshold_json_btn_update, # Return update for JSON select button
            threshold_download_btn_update # Return update for download button
        )

    def update_datetime_defaults(processor_state):
        """Update the default date range based on available predictions."""
        if not processor_state or not processor_state.processor:
            return [gr.update()] * 6  # Updated for 6 outputs (2 dates + 4 time dropdowns)
        
        df = processor_state.processor.get_data()
        start_date, end_date = get_date_range(df)
        
        return [
            gr.update(value=start_date),
            gr.update(value=end_date),
            gr.update(value="00"),  # Start hour
            gr.update(value="00"),  # Start minute
            gr.update(value="23"),  # End hour
            gr.update(value="59"),  # End minute
        ]

    def combine_time_components(hour, minute):
        """Combine hour and minute components into a time string."""
        return f"{hour}:{minute}"

    def plot_predictions_action(
        proc_state: ProcessorState,
        selected_classes_list,
        selected_recordings_list,
        date_range_start,
        date_range_end,
        time_start_hour,
        time_start_minute,
        time_end_hour,
        time_end_minute,
        correctness_mode="Ignore correctness flags"  # Add correctness_mode parameter with default
    ):
        """Uses ConfidencePlotter to plot confidence distributions for the selected classes."""
        if not proc_state or not proc_state.processor:
            raise gr.Error(loc.localize("eval-tab-error-calc-metrics-first"))

        df = proc_state.processor.get_data()
        if df.empty:
            raise gr.Error("No predictions to show.")

        # Apply class and recording filters first
        col_class = proc_state.processor.get_column_name("Class")
        conf_col = proc_state.processor.get_column_name("Confidence")
        corr_col = proc_state.processor.get_column_name("Correctness")
        
        # Check if correctness column exists; if not, try both capitalization forms
        if corr_col not in df.columns:
            # Try both 'correctness' and 'Correctness'
            alt_corr_col = 'correctness' if corr_col == 'Correctness' else 'Correctness'
            if alt_corr_col in df.columns:
                print(f"Switching to alternative correctness column: '{alt_corr_col}'")
                corr_col = alt_corr_col
            else:
                # Create an empty correctness column if none exists
                print(f"No correctness column found, creating a placeholder")
                df[corr_col] = None
        
        if selected_classes_list:
            df = df[df[col_class].isin(selected_classes_list)]
        if selected_recordings_list:
            selected_recordings_list = [rec.lower() for rec in selected_recordings_list]
            df["recording_filename"] = df["recording_filename"].apply(lambda x: os.path.splitext(os.path.basename(x.strip()))[0].lower() if isinstance(x, str) else x)
            df = df[df["recording_filename"].isin(selected_recordings_list)]

        # Apply date and time filters
        df = apply_datetime_filters(
            df, 
            date_range_start, 
            date_range_end, 
            time_start_hour, 
            time_start_minute, 
            time_end_hour, 
            time_end_minute
        )
        
        if df.empty:
            raise gr.Error("No predictions match the selected date/time filters.")
            
        # Normalize correctness values
        df[corr_col] = df[corr_col].map({
            'true': True, 'True': True, True: True, 1: True,
            'false': False, 'False': False, False: False, 0: False,
            'nan': None, 'none': None, '': None, 'null': None, 'NA': None
        }, na_action='ignore')
        
        # Apply correctness filter based on selected mode
        if correctness_mode == "Show only correct":
            df = df[df[corr_col] == True]
        elif correctness_mode == "Show only incorrect":
            df = df[df[corr_col] == False]
        elif correctness_mode == "Show only unspecified":
            df = df[df[corr_col].isna()]
        
        if df.empty:
            raise gr.Error(f"No predictions match the selected {correctness_mode.lower()} filter.")

        # Create histogram plot (using Plotly) with fixed 10 bins (handled in the method)
        plotter = ConfidencePlotter(
            data=df,
            class_col=col_class,
            conf_col=conf_col
        )

        try:
            fig_hist = plotter.plot_histogram_plotly(title="Histogram of Confidence Scores by Class")
            new_state = ProcessorState(
                processor=proc_state.processor,
                prediction_dir=proc_state.prediction_dir,
                metadata_dir=proc_state.metadata_dir,
                color_map=proc_state.color_map,
                class_thresholds=proc_state.class_thresholds
            )
            return [new_state, gr.update(visible=True, value=fig_hist)]
        except Exception as e:
            raise gr.Error(f"Error creating plots: {str(e)}")

    def plot_temporal_scatter(
        proc_state: ProcessorState,
        selected_classes_list,
        selected_recordings_list,
        date_range_start,
        date_range_end,
        time_start_hour,
        time_start_minute,
        time_end_hour,
        time_end_minute,
        meta_x=None,
        meta_y=None,
        correctness_mode="Ignore correctness flags"  # Add correctness_mode parameter with default
    ):
        """Creates a temporal scatter plot showing detections by date and time of day."""
        if not proc_state or not proc_state.processor:
            raise gr.Error("Please load predictions first")
            
        if proc_state.class_thresholds is None:
            raise gr.Error("Class thresholds not initialized. Load data first.")
            
        # Validate thresholds from state
        validated_thresholds_df = proc_state.class_thresholds
            
        # Get data and apply all filters
        df = proc_state.processor.get_data()
        
        # Apply class and recording filters
        col_class = proc_state.processor.get_column_name("Class")
        conf_col = proc_state.processor.get_column_name("Confidence")
        corr_col = proc_state.processor.get_column_name("Correctness")
        
        # Check if correctness column exists; if not, try both capitalization forms
        if corr_col not in df.columns:
            # Try both 'correctness' and 'Correctness'
            alt_corr_col = 'correctness' if corr_col == 'Correctness' else 'Correctness'
            if alt_corr_col in df.columns:
                print(f"Switching to alternative correctness column: '{alt_corr_col}'")
                corr_col = alt_corr_col
            else:
                # Create an empty correctness column if none exists
                print(f"No correctness column found, creating a placeholder")
                df[corr_col] = None
            
        if selected_classes_list:
            df = df[df[col_class].isin(selected_classes_list)]
        if selected_recordings_list:
            selected_recordings_list = [rec.lower() for rec in selected_recordings_list]
            df["recording_filename"] = df["recording_filename"].apply(
                lambda x: os.path.splitext(os.path.basename(x.strip()))[0].lower() 
                if isinstance(x, str) else x
            )
            df = df[df["recording_filename"].isin(selected_recordings_list)]
            
        # Apply class-specific confidence thresholds
        df = apply_class_thresholds(df, validated_thresholds_df, col_class, conf_col)
        
        # Apply date and time filters
        df = apply_datetime_filters(
            df, 
            date_range_start, 
            date_range_end,
            time_start_hour,
            time_start_minute,
            time_end_hour,
            time_end_minute
        )
        
        # Normalize correctness values
        df[corr_col] = df[corr_col].map({
            'true': True, 'True': True, True: True, 1: True,
            'false': False, 'False': False, False: False, 0: False,
            'nan': None, 'none': None, '': None, 'null': None, 'NA': None
        }, na_action='ignore')
        
        # Apply correctness filter based on selected mode
        if correctness_mode == "Show only correct":
            df = df[df[corr_col] == True]
        elif correctness_mode == "Show only incorrect":
            df = df[df[corr_col] == False]
        elif correctness_mode == "Show only unspecified":
            df = df[df[corr_col].isna()]
        
        if df.empty:
            raise gr.Error("No data matches the selected filters")
        
        try:
            # Use the new TemporalScatterPlotter class
            plotter = TemporalScatterPlotter(
                data=df,
                class_col=col_class,
                conf_col=conf_col
            )
            
            fig = plotter.plot(title="Temporal Distribution of Detections")
            
            # Calculate mean latitude and longitude for sunrise/sunset calculations
            # Try to get coordinates from the data or use a default
            try:
                # First check if we have latitude/longitude in the dataframe
                if 'latitude' in df.columns and 'longitude' in df.columns and not df['latitude'].isnull().all() and not df['longitude'].isnull().all():
                    mean_lat = df['latitude'].mean()
                    mean_lon = df['longitude'].mean()
                    print(f"Using coordinates from data: {mean_lat}, {mean_lon}")
                    
                    # Add sunrise/sunset lines using the plotter method
                    fig = plotter.add_sunrise_sunset(fig, mean_lat, mean_lon)
                    
                # If not, see if we can get them from metadata using provided column names
                elif proc_state.metadata_dir:
                    meta_files = list(Path(proc_state.metadata_dir).glob("*.csv"))
                    if not meta_files:
                        raise ValueError("No metadata CSV files found in the metadata directory")
                        
                    metadata_df = pd.read_csv(meta_files[0])
                    print(f"Metadata columns available: {metadata_df.columns.tolist()}")
                        
                    # First try using the explicitly provided meta_x and meta_y from UI
                    lat_col = meta_x
                    lon_col = meta_y
                        
                    # If not provided, try common column names
                    if not lat_col or not lon_col or lat_col not in metadata_df.columns or lon_col not in metadata_df.columns:
                        print("Metadata column names not provided or invalid, trying common column names...")
                        lat_cols = ['latitude', 'lat', 'y', 'Y', 'LAT', 'Latitude']
                        lon_cols = ['longitude', 'lon', 'long', 'x', 'X', 'LON', 'Longitude']
                            
                        lat_col = next((col for col in lat_cols if col in metadata_df.columns), None)
                        lon_col = next((col for col in lon_cols if col in metadata_df.columns), None)
                    
                    if lat_col is None or lon_col is None:
                        raise ValueError(f"Could not find latitude/longitude columns in metadata. Available columns: {', '.join(metadata_df.columns)}")
                            
                    print(f"Using metadata columns: {lat_col} for latitude, {lon_col} for longitude")
                    mean_lat = pd.to_numeric(metadata_df[lat_col], errors='coerce').mean()
                    mean_lon = pd.to_numeric(metadata_df[lon_col], errors='coerce').mean()
                    
                    # Check that we have valid coordinates
                    if pd.isna(mean_lat) or pd.isna(mean_lon):
                        raise ValueError("Invalid coordinates (NaN values)")
                            
                    print(f"Using average location: {mean_lat}, {mean_lon}")
                    fig = plotter.add_sunrise_sunset(fig, mean_lat, mean_lon)
                            
            except Exception as e:
                print(f"Error adding sunrise/sunset lines: {str(e)}")
                import traceback
                traceback.print_exc()
            
            # Update state with new color map for consistency
            new_state = ProcessorState(
                processor=proc_state.processor,
                prediction_dir=proc_state.prediction_dir,
                metadata_dir=proc_state.metadata_dir,
                color_map=plotter.color_map,
                class_thresholds=validated_thresholds_df
            )
            
            return [new_state, gr.update(value=fig, visible=True)]
            
        except Exception as e:
            print(f"Error creating temporal scatter plot: {str(e)}")
            import traceback
            traceback.print_exc()  # Print the full traceback for debugging
            raise gr.Error(f"Error creating temporal scatter plot: {str(e)[:100]}...")

    def plot_spatial_distribution(
        proc_state: ProcessorState,
        meta_x,
        meta_y,
        meta_site,
        date_range_start,
        date_range_end,
        time_start_hour,
        time_start_minute,
        time_end_hour,
        time_end_minute,
        correctness_mode="Ignore correctness flags"
    ):
        """Plot spatial distribution of predictions by class."""
        if not proc_state or not proc_state.processor:
            raise gr.Error("Please load predictions first")
            
        if proc_state.class_thresholds is None:
            raise gr.Error("Class thresholds not initialized. Load data first.") 
            
        try:
            # Validate thresholds from state
            validated_thresholds_df = proc_state.class_thresholds

            # Read metadata file from the provided directory
            meta_files = list(Path(proc_state.metadata_dir).glob("*.csv"))
            if not meta_files:
                raise gr.Error("No metadata files found")
            metadata_df = pd.read_csv(meta_files[0])
            
            # Ensure expected source columns exist
            if meta_x not in metadata_df.columns or meta_y not in metadata_df.columns:
                raise gr.Error("Metadata file missing expected coordinate columns.")
            # Convert columns and create standardized 'latitude' and 'longitude'
            metadata_df = metadata_df.copy()
            metadata_df['latitude'] = pd.to_numeric(metadata_df[meta_x], errors='coerce')
            metadata_df['longitude'] = pd.to_numeric(metadata_df[meta_y], errors='coerce')
            
            # Store all unique locations from metadata before filtering
            all_locations_df = metadata_df.copy()
            all_locations_df = all_locations_df.rename(columns={meta_site: 'site_name'})
            all_locations_df = all_locations_df[['site_name', 'latitude', 'longitude']].drop_duplicates()
            
            # Set metadata into processor
            proc_state.processor.set_metadata(metadata_df, site_col=meta_site, lat_col='latitude', lon_col='longitude')
            
            # Get prediction data and apply filters
            df = proc_state.processor.get_data()
            
            # Apply class-specific confidence threshold filter using validated thresholds
            conf_col = proc_state.processor.get_column_name("Confidence")
            class_col = proc_state.processor.get_column_name("Class")
            corr_col = proc_state.processor.get_column_name("Correctness")
            
            # Check if correctness column exists; if not, try both capitalization forms
            if corr_col not in df.columns:
                # Try both 'correctness' and 'Correctness'
                alt_corr_col = 'correctness' if corr_col == 'Correctness' else 'Correctness'
                if alt_corr_col in df.columns:
                    print(f"Switching to alternative correctness column: '{alt_corr_col}'")
                    corr_col = alt_corr_col
                else:
                    # Create an empty correctness column if none exists
                    print(f"No correctness column found, creating a placeholder")
                    df[corr_col] = None
                    
            df = apply_class_thresholds(df, validated_thresholds_df, class_col, conf_col)
            
            # Apply date and time filters
            df = apply_datetime_filters(
                df, 
                date_range_start, 
                date_range_end, 
                time_start_hour, 
                time_start_minute, 
                time_end_hour, 
                time_end_minute
            )
            
            # Normalize correctness values
            df[corr_col] = df[corr_col].map({
                'true': True, 'True': True, True: True, 1: True,
                'false': False, 'False': False, False: False, 0: False,
                'nan': None, 'none': None, '': None, 'null': None, 'NA': None
            }, na_action='ignore')
            
            # Apply correctness filter based on selected mode
            if correctness_mode == "Show only correct":
                df = df[df[corr_col] == True]
            elif correctness_mode == "Show only incorrect":
                df = df[df[corr_col] == False]
            elif correctness_mode == "Show only unspecified":
                df = df[df[corr_col].isna()]
            
            # Ensure that latitude and longitude exist in the data after merge
            for col in ['latitude', 'longitude']:
                if col not in df.columns:
                    raise gr.Error(f"Column '{col}' is missing after merging metadata.")
            
            # Create aggregated dataframe with counts by location and class
            agg_df = df.groupby(['site_name', 'latitude', 'longitude', class_col]).size().reset_index(name='count')
            
            # Get existing color map or create one
            plotter = SpatialDistributionPlotter(class_col=class_col)
            
            # Init color map with TimeDistributionPlotter's exact algorithm if needed
            if not proc_state.color_map:
                # If we have no color map yet, create one using all classes for consistency
                all_classes = sorted(df[class_col].unique())
                # Get a TimeDistributionPlotter to initialize colors consistently
                time_plotter = TimeDistributionPlotter(data=df, class_col=class_col)
                shared_color_map = time_plotter._get_color_map(all_classes)
                # Debug output
                print("Creating new color map from TimeDistributionPlotter:")
                for cls in sorted(all_classes):
                    print(f"  {cls}: {shared_color_map.get(cls)}")
                plotter.color_map = shared_color_map
            else:
                # Use existing color map - fix color consistency
                print("Using existing color map:")
                for cls in sorted(proc_state.color_map.keys()):
                    print(f"  {cls}: {proc_state.color_map.get(cls)}")
                plotter.color_map = proc_state.color_map.copy()
            
            # Generate the plot
            fig = plotter.plot(
                agg_df=agg_df,
                all_locations_df=all_locations_df,
                title="Spatial Distribution of Predictions by Class"
            )
            
            # Update state with new color map for consistency
            new_state = ProcessorState(
                processor=proc_state.processor,
                prediction_dir=proc_state.prediction_dir,
                metadata_dir=proc_state.metadata_dir,
                color_map=plotter.color_map,
                class_thresholds=validated_thresholds_df
            )
            
            return [new_state, gr.update(value=fig, visible=True)]
        except Exception as e:
            print(f"Error creating map: {str(e)}")
            import traceback
            traceback.print_exc()  # Print the full traceback for debugging
            raise gr.Error(f"Error creating map: {str(e)}")

    def plot_time_distribution(
        proc_state: ProcessorState,
        time_period: str,
        use_boxplot: bool,
        selected_classes_list,
        selected_recordings_list,
        date_range_start,
        date_range_end,
        time_start_hour,
        time_start_minute,
        time_end_hour,
        time_end_minute,
        correctness_mode="Ignore correctness flags"  # Add correctness_mode parameter with default
    ):
        """Creates time distribution plot with all filters applied."""
        if not proc_state or not proc_state.processor:
            raise gr.Error("Please load predictions first")
            
        if proc_state.class_thresholds is None:
            raise gr.Error("Class thresholds not initialized. Load data first.")
            
        # Validate thresholds from state
        validated_thresholds_df = proc_state.class_thresholds
            
        # Get data and apply all filters
        df = proc_state.processor.get_data()
        
        # Apply class and recording filters
        col_class = proc_state.processor.get_column_name("Class")
        conf_col = proc_state.processor.get_column_name("Confidence")
        corr_col = proc_state.processor.get_column_name("Correctness")
        
        # Check if correctness column exists; if not, try both capitalization forms
        if corr_col not in df.columns:
            # Try both 'correctness' and 'Correctness'
            alt_corr_col = 'correctness' if corr_col == 'Correctness' else 'Correctness'
            if alt_corr_col in df.columns:
                print(f"Switching to alternative correctness column: '{alt_corr_col}'")
                corr_col = alt_corr_col
            else:
                # Create an empty correctness column if none exists
                print(f"No correctness column found, creating a placeholder")
                df[corr_col] = None
                
        if selected_classes_list:
            df = df[df[col_class].isin(selected_classes_list)]
        if selected_recordings_list:
            selected_recordings_list = [rec.lower() for rec in selected_recordings_list]
            df["recording_filename"] = df["recording_filename"].apply(
                lambda x: os.path.splitext(os.path.basename(x.strip()))[0].lower() 
                if isinstance(x, str) else x
            )
            df = df[df["recording_filename"].isin(selected_recordings_list)]
            
        # Apply class-specific confidence thresholds using validated thresholds
        df = apply_class_thresholds(df, validated_thresholds_df, col_class, conf_col)
        
        # Apply date and time filters
        df = apply_datetime_filters(
            df, 
            date_range_start, 
            date_range_end,
            time_start_hour,
            time_start_minute,
            time_end_hour,
            time_end_minute
        )
        
        # Normalize correctness values
        df[corr_col] = df[corr_col].map({
            'true': True, 'True': True, True: True, 1: True,
            'false': False, 'False': False, False: False, 0: False,
            'nan': None, 'none': None, '': None, 'null': None, 'NA': None
        }, na_action='ignore')
        
        # Apply correctness filter based on selected mode
        if correctness_mode == "Show only correct":
            df = df[df[corr_col] == True]
        elif correctness_mode == "Show only incorrect":
            df = df[df[corr_col] == False]
        elif correctness_mode == "Show only unspecified":
            df = df[df[corr_col].isna()]
        
        if df.empty:
            raise gr.Error("No data matches the selected filters")
            
        # Create plotter and generate plot
        plotter = TimeDistributionPlotter(
            data=df,
            class_col=col_class
        )
        
        try:
            fig = plotter.plot_distribution(
                time_period=time_period,
                use_boxplot=use_boxplot,
                title=f"Species {'Boxplots' if use_boxplot else 'Counts'} by {time_period.capitalize()}"
            )
            return gr.update(value=fig, visible=True)
        except Exception as e:
            raise gr.Error(f"Error creating time distribution plot: {str(e)}")

    def get_selection_tables(directory):
        """Reads prediction txt files and metadata csv files from directory."""
        from pathlib import Path
        directory = Path(directory)
        files = list(directory.glob("*.txt")) + list(directory.glob("*.csv"))
        return files

    def download_threshold_template(proc_state: ProcessorState):
        """Saves the current threshold table as a JSON file template."""
        if not proc_state or proc_state.class_thresholds is None or proc_state.class_thresholds.empty:
            raise gr.Error("No thresholds available to save. Load data first.")
        
        try:
            thresholds_dict = {}
            for _, row in proc_state.class_thresholds.iterrows():
                thresholds_dict[row['Class']] = float(row['Threshold'])
            
            file_location = gu.save_file_dialog(
                state_key="viz-threshold-template",
                filetypes=("JSON (*.json)",),
                default_filename="threshold_template.json",
            )
            
            if file_location:
                with open(file_location, "w") as f:
                    json.dump(thresholds_dict, f, indent=4)
                
                gr.Info("Threshold template saved successfully")
        except Exception as e:
            print(f"Error saving threshold template: {e}")
            raise gr.Error(f"Error saving threshold template: {e}")

    def select_threshold_json_file(proc_state: ProcessorState):
        """Opens a file dialog to select a JSON threshold file and loads its contents."""
        if not proc_state or not proc_state.processor:
            gr.Warning("Processor not initialized. Load data first.")
            return proc_state, gr.update()
        
        try:
            # Call select_file - ensure filetypes is a tuple of strings
            file_path = gu.select_file(
                filetypes=('JSON files (*.json)', 'All files (*.*)'), # Corrected format
                state_key="viz-threshold-json"
            )
            
            if not file_path:
                # User canceled or no file selected (file_path is None or empty string)
                return proc_state, gr.update()
                
            # Ensure file_path is a string before opening
            if not isinstance(file_path, str):
                 # This case should ideally not happen if gu.select_file works correctly
                 raise TypeError(f"Expected a file path string, but got {type(file_path)}")

            with open(file_path, 'r') as f:
                json_data = json.load(f)

            if not isinstance(json_data, dict):
                raise ValueError("JSON content must be a dictionary (object).")

            if proc_state.class_thresholds is None or proc_state.class_thresholds.empty:
                gr.Warning("Class thresholds not initialized. Cannot update.")
                return proc_state, gr.update()

            updated_thresholds_df = proc_state.class_thresholds.copy()
            updated_thresholds_df.set_index('Class', inplace=True)

            loaded_count = 0
            warning_messages = []

            for cls, threshold in json_data.items():
                if not isinstance(cls, str):
                    warning_messages.append(f"Skipping non-string class key: {cls}")
                    continue
                if not isinstance(threshold, (int, float)):
                    warning_messages.append(f"Skipping non-numeric threshold for class '{cls}': {threshold}")
                    continue
                
                valid_threshold = float(threshold)
                clipped_threshold = max(0.01, min(0.99, valid_threshold))
                if clipped_threshold != valid_threshold:
                    warning_messages.append(f"Threshold for '{cls}' ({valid_threshold}) clipped to {clipped_threshold}.")
                    
                if cls in updated_thresholds_df.index:
                    updated_thresholds_df.loc[cls, 'Threshold'] = clipped_threshold
                    loaded_count += 1
                else:
                    warning_messages.append(f"Class '{cls}' from JSON not found in loaded data.")

            updated_thresholds_df.reset_index(inplace=True)

            if warning_messages:
                gr.Warning("\n".join(warning_messages))

            new_state = ProcessorState(
                processor=proc_state.processor,
                prediction_dir=proc_state.prediction_dir,
                metadata_dir=proc_state.metadata_dir,
                color_map=proc_state.color_map,
                class_thresholds=updated_thresholds_df
            )
            
            gr.Info(f"Successfully loaded thresholds for {loaded_count} classes from JSON.")
            return new_state, gr.update(value=updated_thresholds_df)
            
        except Exception as e:
            print(f"Error loading thresholds from JSON: {e}")
            gr.Error(f"Error loading thresholds from JSON: {str(e)}")
            return proc_state, gr.update()

    with gr.Tab(loc.localize("visualization-tab-title")):
        gr.Markdown(
            """
            <style>
            .custom-checkbox-group {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                grid-gap: 8px;
            </style>
            """
        )

        # States
        processor_state = gr.State()
        prediction_files_state = gr.State()
        metadata_files_state = gr.State()

        # File Selection UI
        with gr.Row():
            with gr.Column():
                prediction_select_directory_btn = gr.Button(loc.localize("eval-tab-prediction-selection-button-label"))
                prediction_directory_input = gr.Matrix(
                    interactive=False,
                    headers=[loc.localize("eval-tab-selections-column-file-header")],
                )
            with gr.Column():
                metadata_select_directory_btn = gr.Button(loc.localize("viz-tab-metadata-selection-button-label"))
                metadata_directory_input = gr.Matrix(
                    interactive=False,
                    headers=[loc.localize("eval-tab-selections-column-file-header")],
                )

        # Prediction columns box (using gr.Group as in evaluation tab)
        with gr.Group(visible=False) as prediction_group:
            with gr.Accordion(loc.localize("eval-tab-prediction-col-accordion-label"), open=True):
                with gr.Row():
                    prediction_columns: dict[str, gr.Dropdown] = {}
                    for col in ["Start Time", "Class", "Confidence", "Recording", "Correctness"]:
                        prediction_columns[col] = gr.Dropdown(choices=[], label=localized_column_labels.get(col, col))

        # Metadata columns box
        with gr.Group(visible=False) as metadata_group:
            with gr.Accordion(loc.localize("viz-tab-metadata-col-accordion-label"), open=True):
                with gr.Row():
                    metadata_columns: dict[str, gr.Dropdown] = {}
                    for col in ["Site", "X", "Y"]:
                        label = localized_column_labels[col]
                        if col == "X":
                            label += " (Decimal Degrees)"
                        elif col == "Y":
                            label += " (Decimal Degrees)"
                        metadata_columns[col] = gr.Dropdown(choices=[], label=label)

        # Class and Recording Selection Box
        with gr.Group(visible=True) as class_recording_group:
            with gr.Accordion(loc.localize("viz-tab-class-recording-accordion-label"), open=False):
                with gr.Row():
                    with gr.Column():
                        select_classes_checkboxgroup = gr.CheckboxGroup(
                            choices=[],
                            value=[],
                            label=loc.localize("viz-tab-classes-label"),
                            info=loc.localize("viz-tab-classes-info"),
                            interactive=True,
                            elem_classes="custom-checkbox-group",
                        )
                    with gr.Column():
                        select_recordings_checkboxgroup = gr.CheckboxGroup(
                            choices=[],
                            value=[],
                            label=loc.localize("viz-tab-recordings-label"),
                            info=loc.localize("viz-tab-recordings-info"),
                            interactive=True,
                            elem_classes="custom-checkbox-group",
                        )

        # Parameters Box
        with gr.Group():
            with gr.Accordion(loc.localize("viz-tab-parameters-accordion-label"), open=False):
                with gr.Row():
                    date_range_start = gr.DateTime(
                        label=loc.localize("viz-tab-date-range-start-label"),
                        info=loc.localize("viz-tab-date-range-start-info"),
                        interactive=True,
                        show_label=True,
                        include_time=False
                    )
                    date_range_end = gr.DateTime(
                        label=loc.localize("viz-tab-date-range-end-label"),
                        info=loc.localize("viz-tab-date-range-end-info"),
                        interactive=True,
                        show_label=True,
                        include_time=False
                    )

                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            time_start_hour = gr.Dropdown(
                                choices=[f"{i:02d}" for i in range(24)],
                                value="00",
                                label=loc.localize("viz-tab-start-time-label-hour"),
                                interactive=True
                            )
                            time_start_minute = gr.Dropdown(
                                choices=[f"{i:02d}" for i in range(60)],
                                value="00",
                                label=loc.localize("viz-tab-start-time-label-minute"),
                                interactive=True
                            )
                    
                    with gr.Column(): 
                        with gr.Row():
                            time_end_hour = gr.Dropdown(
                                choices=[f"{i:02d}" for i in range(24)],
                                value="23",
                                label=loc.localize("viz-tab-end-time-label-hour"),
                                interactive=True
                            )
                            time_end_minute = gr.Dropdown(
                                choices=[f"{i:02d}" for i in range(60)],
                                value="59",
                                label=loc.localize("viz-tab-end-time-label-minute"),
                                interactive=True
                            )

                with gr.Row():
                    time_distribution_period = gr.Dropdown(
                        choices=["hour", "day", "month", "year"],
                        value="hour",
                        label="Time Distribution Period",
                        info="Select period for time distribution plot"
                    )
                    use_boxplot = gr.Checkbox(
                        label="Use Box Plots",
                        info="Show distribution as box plots instead of counts",
                        value=False
                    )
                
                # Threshold components 
                with gr.Row():
                     class_thresholds_df = gr.DataFrame(
                         headers=["Class", "Threshold"],
                         datatype=["str", "number"],
                         label="Class Confidence Thresholds (Read-only)",
                         interactive=False, # Make DataFrame read-only
                         visible=False,
                         col_count=(2, "fixed")
                     )
                # Replace JSON Upload with Select JSON button and add Download button
                with gr.Row():
                    threshold_json_select_btn = gr.Button(
                        "Select Threshold JSON File",
                        variant="secondary",
                        visible=False,
                        scale=2
                    )
                    threshold_template_download_btn = gr.Button(
                        "Download Threshold Template",
                        variant="secondary",
                        visible=False,
                        scale=2
                    )

                # Add correctness mode selection
                correctness_mode = gr.Radio(
                    choices=[
                        "Ignore correctness flags",
                        "Show only correct",
                        "Show only incorrect",
                        "Show only unspecified"
                    ],  # Removed "Distinguish all" option
                    value="Ignore correctness flags",
                    label="Correctness Filter Mode",
                    info="Select how to handle the correctness flags in visualizations",
                    interactive=True
                )

        # Warning message about model validation and interpretation
        gr.Markdown(
            """
            <div style="background-color: #FFF3CD; color: #856404; padding: 10px; margin: 10px 0; 
                      border-left: 5px solid #FFDD33; border-radius: 4px;">
              <span style="font-weight: bold;">⚠️ Warning:</span> Visualizations should be interpreted with caution. 
              Please verify model performance for your target species and environment before drawing conclusions. 
              Confidence thresholds significantly affect detection rates - lower values increase detections but may 
              introduce false positives.
            </div>
            """
        )

        # Action button and output for smooth distribution plot
        plot_predictions_btn = gr.Button(
            loc.localize("viz-tab-plot-distributions-button-label"), 
            variant="huggingface"
        )
        smooth_distribution_output = gr.Plot(label=loc.localize("viz-tab-distribution-plot-label"), visible=False)

        # Add map button and output after the existing plot components
        plot_map_btn = gr.Button(
            loc.localize("viz-tab-plot-map-button-label"), 
            variant="huggingface"
        )
        map_output = gr.Plot(label=loc.localize("viz-tab-map-plot-label"), visible=False)

        # Add plot time distribution button after existing plot buttons
        plot_time_distribution_btn = gr.Button(
            "Plot Time Distribution", 
            variant="huggingface"
        )
        time_distribution_output = gr.Plot(
            label="Time Distribution Plot",
            visible=False
        )
        
        # Add temporal scatter plot button and output
        plot_temporal_scatter_btn = gr.Button(
            "Plot Temporal Scatter", 
            variant="huggingface"
        )
        temporal_scatter_output = gr.Plot(
            label="Temporal Scatter Plot",
            visible=False
        )

        # Add calculate detections button and output table
        calculate_detections_btn = gr.Button(
            "Calculate Detections", 
            variant="huggingface"
        )
        detections_table = gr.DataFrame(
            show_label=False,
            type="pandas",
            visible=False,
            interactive=False,
            wrap=True,  # Enable text wrapping for better readability
            column_widths=[200, 110, 80, 130, 90, 150, 110, 110, 80]  # Set widths for all columns
        )

        # Interactions
        def get_selection_func(state_key, on_select):
            def select_directory_on_empty():
                folder = gu.select_folder(state_key=state_key)
                if folder:
                    files = get_selection_tables(folder)
                    files_to_display = files[:100] + [["..."]] if len(files) > 100 else files
                    return [files, files_to_display, gr.update(visible=True)] + on_select(files)
                return ["", [[loc.localize("eval-tab-no-files-found")]], gr.update(visible=False)] + [gr.update(visible=False)] * 5
            return select_directory_on_empty

        prediction_select_directory_btn.click(
            get_selection_func("eval-predictions-dir", update_prediction_columns),
            outputs=[prediction_files_state, prediction_directory_input, prediction_group]
            + [prediction_columns[label] for label in ["Start Time", "Class", "Confidence", "Recording", "Correctness"]],
            show_progress=True,
        )

        metadata_select_directory_btn.click(
            get_selection_func("viz-metadata-dir", update_metadata_columns),
            outputs=[
                metadata_files_state,
                metadata_directory_input,
                metadata_group,
                metadata_columns["Site"],
                metadata_columns["X"],
                metadata_columns["Y"]
            ],
            show_progress=True,
        )

        # Add visibility toggle for metadata group
        metadata_directory_input.change(
            lambda x: gr.update(visible=bool(x)),
            inputs=[metadata_files_state],
            outputs=[metadata_group],
        )

        # Update processor and selections when columns change or files change
        # Consolidate triggers for update_selections
        update_triggers = [
            prediction_files_state,
            metadata_files_state,
            prediction_columns["Start Time"],
            prediction_columns["Class"],
            prediction_columns["Confidence"],
            prediction_columns["Recording"],
            prediction_columns["Correctness"],
            metadata_columns["Site"],
            metadata_columns["X"],
            metadata_columns["Y"],
        ]

        for trigger in update_triggers:
            trigger.change(
                fn=update_selections,
                inputs=[
                    prediction_files_state,
                    metadata_files_state,
                    prediction_columns["Start Time"],
                    prediction_columns["Class"],
                    prediction_columns["Confidence"],
                    prediction_columns["Recording"],
                    prediction_columns["Correctness"],
                    metadata_columns["Site"],
                    metadata_columns["X"],
                    metadata_columns["Y"],
                    select_classes_checkboxgroup, # Pass current selections
                    select_recordings_checkboxgroup, # Pass current selections
                ],
                outputs=[
                    select_classes_checkboxgroup,
                    select_recordings_checkboxgroup,
                    processor_state,
                    class_thresholds_df, # Output to update the DataFrame UI
                    threshold_json_select_btn, # Update for JSON select button
                    threshold_template_download_btn # Update for download button
                ],
                # Trigger date updates only on success of processor update
            ).success(
                fn=update_datetime_defaults,
                inputs=[processor_state],
                outputs=[
                    date_range_start,
                    date_range_end,
                    time_start_hour,
                    time_start_minute,
                    time_end_hour,
                    time_end_minute
                ]
            )

        # Add click handlers for the new buttons
        threshold_json_select_btn.click(
            fn=select_threshold_json_file,
            inputs=[processor_state],
            outputs=[processor_state, class_thresholds_df],
            api_name="select_threshold_json" # Add unique API name for better traceability
        )
        
        threshold_template_download_btn.click(
            fn=download_threshold_template,
            inputs=[processor_state]
        )

        # Plot button action (Histogram - ignores class thresholds)
        plot_predictions_btn.click(
            fn=plot_predictions_action,
            inputs=[
                processor_state,
                select_classes_checkboxgroup,
                select_recordings_checkboxgroup,
                date_range_start,
                date_range_end,
                time_start_hour,
                time_start_minute,
                time_end_hour,
                time_end_minute,
                correctness_mode,  # Add correctness_mode input
            ],
            outputs=[processor_state, smooth_distribution_output]
        )

        # Add click handler for map button (Uses class thresholds from state)
        plot_map_btn.click(
            fn=plot_spatial_distribution,
            inputs=[
                processor_state,
                metadata_columns["X"],
                metadata_columns["Y"],
                metadata_columns["Site"],
                date_range_start,
                date_range_end,
                time_start_hour,
                time_start_minute,
                time_end_hour,
                time_end_minute,
                correctness_mode,  # Add correctness_mode input
            ],
            outputs=[processor_state, map_output]
        )

        # Add click handler for time distribution button (Uses class thresholds from state)
        plot_time_distribution_btn.click(
            fn=plot_time_distribution,
            inputs=[
                processor_state,
                time_distribution_period,
                use_boxplot,
                select_classes_checkboxgroup,
                select_recordings_checkboxgroup,
                date_range_start,
                date_range_end,
                time_start_hour,
                time_start_minute,
                time_end_hour,
                time_end_minute,
                correctness_mode,  # Add correctness_mode input
            ],
            outputs=[time_distribution_output]
        )
        
        # Add click handler for temporal scatter button
        plot_temporal_scatter_btn.click(
            fn=plot_temporal_scatter,
            inputs=[
                processor_state,
                select_classes_checkboxgroup,
                select_recordings_checkboxgroup,
                date_range_start,
                date_range_end,
                time_start_hour,
                time_start_minute,
                time_end_hour,
                time_end_minute,
                metadata_columns["X"],
                metadata_columns["Y"],
                correctness_mode,  # Add correctness_mode input
            ],
            outputs=[processor_state, temporal_scatter_output]
        )

        # Add click handler for calculate detections button - using the imported function
        calculate_detections_btn.click(
            fn=calculate_detection_counts,
            inputs=[
                processor_state,
                select_classes_checkboxgroup,
                select_recordings_checkboxgroup,
                date_range_start,
                date_range_end,
                time_start_hour,
                time_start_minute,
                time_end_hour,
                time_end_minute,
                correctness_mode,  # Add correctness_mode input
            ],
            outputs=[detections_table]
        )


if __name__ == "__main__":
    gu.open_window(build_visualization_tab)
