import json
import os
import shutil
import tempfile
import typing
import io
from pathlib import Path
import datetime
import traceback

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import birdnet_analyzer.gui.localization as loc
import birdnet_analyzer.gui.utils as gu
import birdnet_analyzer.visualization.data_processor as dp
from birdnet_analyzer.visualization.plotting.confidences import ConfidencePlotter
from birdnet_analyzer.visualization.plotting.time_distributions import TimeDistributionPlotter
from birdnet_analyzer.visualization.plotting.temporal_scatter import TemporalScatterPlotter
from birdnet_analyzer.visualization.plotting.spatial_distribution import SpatialDistributionPlotter
from birdnet_analyzer.visualization.detection_counts import calculate_detection_counts

from birdnet_analyzer.visualization.common import ProcessorState, convert_timestamp_to_datetime

def get_date_range(df: pd.DataFrame) -> tuple:
    """Get the earliest and latest dates from the 'Date' column."""
    try:
        if 'Date' not in df.columns or df.empty or not pd.api.types.is_datetime64_any_dtype(df['Date']):
            return None, None

        min_date = df['Date'].min()
        max_date = df['Date'].max()

        if pd.isna(min_date) or pd.isna(max_date):
            return None, None

        start_date = min_date.to_pydatetime()
        end_date = max_date.to_pydatetime()

        return start_date, end_date
    except Exception as e:
        print(f"Error getting date range: {e}")
        return None, None

def build_visualization_tab():
    """
    Builds a Gradio tab for loading and plotting prediction data only,
    using ConfidencePlotter. Annotation logic and metric calculations
    have been removed.
    """

    prediction_default_columns = {
        "Start Time": "File Offset (s)",
        "Class": "Common Name",
        "Recording": "Begin Path",
        "Confidence": "Confidence",
        "Correctness": "correctness",
    }

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

    MAX_DEFAULT_CLASSES = 5  # Define the class limit

    def get_columns_from_uploaded_files(files):
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
        pred_correctness=None,
        prediction_dir=None,
    ):
        if not prediction_files:
            return [], [], None, None

        try:
            if prediction_dir is None:
                prediction_dir = save_uploaded_files(prediction_files)

            print(f"\nProcessing {len(prediction_files)} prediction files")
            print(f"Using prediction directory: {prediction_dir}")

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

            proc = dp.DataProcessor(
                prediction_directory_path=prediction_dir,
                prediction_file_name=None,
                columns_predictions=cols_pred,
            )

            df = proc.get_complete_data()

            print(f"\nLoaded DataFrame shape: {df.shape}")
            print(f"DataFrame columns: {df.columns}")

            # Check for duplicate 'Recording' column before accessing
            if isinstance(df.columns, pd.MultiIndex):
                raise gr.Error("DataFrame has a MultiIndex, which is not supported.")
            if 'Recording' not in df.columns:
                raise gr.Error("The processed DataFrame is missing the 'Recording' column.")

            # Explicitly handle potential duplicate columns when selecting for .unique()
            recording_col_data = df['Recording']
            if isinstance(recording_col_data, pd.DataFrame):
                print("Warning: Duplicate 'Recording' columns found. Using the first instance.")
                # Select the first column named 'Recording' if duplicates exist
                recording_col_data = df.loc[:, 'Recording'].iloc[:, 0]

            # Now apply dropna and unique to the Series
            recordings = recording_col_data.dropna().astype(str).unique()
            recordings = sorted([rec for rec in recordings if rec])  # Ensure strings and remove empty

            # Similarly for 'Class'
            if 'Class' not in df.columns:
                raise gr.Error("The processed DataFrame is missing the 'Class' column.")

            class_col_data = df['Class']
            if isinstance(class_col_data, pd.DataFrame):
                print("Warning: Duplicate 'Class' columns found. Using the first instance.")
                class_col_data = df.loc[:, 'Class'].iloc[:, 0]

            classes = class_col_data.dropna().astype(str).unique()
            classes = sorted([cls for cls in classes if cls])  # Ensure strings and remove empty

            print(f"Found {len(classes)} unique classes")

            print(f"Found {len(recordings)} unique recordings")

            return classes, recordings, proc, prediction_dir

        except ValueError as e:
            print(f"Error in initialize_processor: {str(e)}")
            traceback.print_exc()  # Print full traceback for debugging
            raise gr.Error(f"Error initializing processor: {str(e)}")
        except Exception as e:
            print(f"Unexpected error in initialize_processor: {str(e)}")
            traceback.print_exc()  # Print full traceback for debugging
            raise gr.Error(f"An unexpected error occurred: {str(e)}")

    def update_prediction_columns(uploaded_files):
        cols = get_columns_from_uploaded_files(uploaded_files)
        cols = [""] + cols
        updates = []
        for label in ["Start Time", "Class", "Confidence", "Recording", "Correctness"]:
            default_val = prediction_default_columns.get(label)
            val = default_val if default_val in cols else None
            updates.append(gr.update(choices=cols, value=val))
        return updates

    def update_metadata_columns(uploaded_files):
        cols = set()
        if uploaded_files:
            for file_obj in uploaded_files:
                file_path = str(file_obj)
                print(f"Attempting to read metadata from: {file_path}")

                try:
                    if file_path.lower().endswith('.xlsx'):
                        try:
                            df = pd.read_excel(file_path, sheet_name=0, nrows=0)
                            print(f"Successfully read Excel headers: {list(df.columns)}")
                            cols.update(df.columns)
                            continue
                        except Exception as e:
                            print(f"Error reading Excel file: {e}")

                    try:
                        df = pd.read_csv(file_path, sep=';', encoding='latin1', nrows=1)
                        if len(df.columns) > 1:
                            print(f"Successfully read with semicolon delimiter: {list(df.columns)}")
                            cols.update(df.columns)
                            continue
                        else:
                            print("Only got one column with semicolon delimiter, trying other options")
                    except Exception as e:
                        print(f"Failed with semicolon delimiter: {e}")

                    encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']
                    for encoding in encodings:
                        try:
                            df = pd.read_csv(file_path, sep=',', encoding=encoding, nrows=1)
                            if len(df.columns) > 1:
                                print(f"Successfully read with comma delimiter and {encoding} encoding: {list(df.columns)}")
                                cols.update(df.columns)
                                break
                        except Exception as e:
                            print(f"Failed with comma and {encoding}: {e}")

                    if df is not None and len(df.columns) > 1:
                        continue

                    try:
                        df = pd.read_csv(file_path, sep=None, engine='python', nrows=1)
                        print(f"Successfully read with Python auto-detection: {list(df.columns)}")
                        cols.update(df.columns)
                    except Exception as e:
                        print(f"Python auto-detection failed: {e}")

                        try:
                            with open(file_path, 'r', encoding='latin1') as f:
                                header_line = f.readline().strip()

                            if ';' in header_line:
                                headers = header_line.split(';')
                                print(f"Manual semicolon split found {len(headers)} headers")
                                cols.update([h.strip('"\'') for h in headers])
                            elif ',' in header_line:
                                headers = header_line.split(',')
                                print(f"Manual comma split found {len(headers)} headers")
                                cols.update([h.strip('"\'') for h in headers])
                        except Exception as e:
                            print(f"Manual reading failed: {e}")

                except Exception as e:
                    print(f"Error reading headers from {file_path}: {e}")
                    gr.Warning(f"{loc.localize('eval-tab-warning-error-reading-file')} {file_path}")

        cols = [""] + sorted(list(cols))
        print(f"Final columns found: {cols}")
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

        # 2) NEW ► attach the metadata if everything is available ──────────────
        if (
            proc                                           # processor exists
            and metadata_files                             # user chose meta files
            and metadata_dir                               # metadata directory exists
            and meta_site and meta_x and meta_y            # user mapped the columns
        ):
            try:
                meta_path = Path(metadata_dir)
                meta_file_list = list(meta_path.glob("*.csv")) + list(meta_path.glob("*.xlsx"))
                if meta_file_list:
                    first_meta_file = meta_file_list[0]
                    print(f"Attempting to load metadata early from: {first_meta_file}")
                    if str(first_meta_file).lower().endswith('.xlsx'):
                        meta_df = pd.read_excel(first_meta_file, sheet_name=0)
                    else:
                        try:
                            meta_df = pd.read_csv(first_meta_file, sep=None, engine="python")
                        except Exception as e_utf8:
                            print(f"UTF-8 read failed for {first_meta_file}, trying latin1: {e_utf8}")
                            meta_df = pd.read_csv(first_meta_file, sep=None, engine="python", encoding='latin1')

                    # let DataProcessor validate / clean the table
                    proc.set_metadata(meta_df, site_col=meta_site, lat_col=meta_x, lon_col=meta_y)
                    print("Metadata loaded early into processor.")
                else:
                    print("Metadata files selected, but no CSV/XLSX found in the directory for early loading.")
            except Exception as e:
                print(f"Error loading metadata early in update_selections: {e}")
                # Optional: Raise a gr.Warning or just log the error
                # gr.Warning(f"Could not load metadata during initial setup: {e}")

        class_thresholds_init_df = None
        threshold_df_update = gr.update(visible=False, value=None)
        threshold_json_btn_update = gr.update(visible=False)
        threshold_download_btn_update = gr.update(visible=False)

        if proc:
            class_thresholds_init_df = pd.DataFrame({
                'Class': sorted(avail_classes),
                'Threshold': [0.10] * len(avail_classes)
            })
            threshold_df_update = gr.update(visible=True, value=class_thresholds_init_df)
            threshold_json_btn_update = gr.update(visible=True)
            threshold_download_btn_update = gr.update(visible=True)

            state = ProcessorState(
                processor=proc,
                prediction_dir=prediction_dir,
                metadata_dir=metadata_dir,
                class_thresholds=class_thresholds_init_df
            )
        else:
            state = None

        new_classes = []
        new_recordings = []

        # Determine default classes based on limit
        default_classes = avail_classes
        if len(avail_classes) > MAX_DEFAULT_CLASSES:
            default_classes = avail_classes[:MAX_DEFAULT_CLASSES]
            gr.Warning(
                f"Found {len(avail_classes)} classes. Selecting the first {MAX_DEFAULT_CLASSES} "
                f"by default for visualization clarity. You can change the selection in the Select classes and Recordings section."
            )

        if current_classes:
            new_classes = [c for c in current_classes if c in avail_classes]
        if current_recordings:
            normalized_current = [str(r).strip() for r in current_recordings if isinstance(r, str)]
            new_recordings = [r for r in normalized_current if r in avail_recordings]

        # If no valid current classes or initially empty, use the default set
        if not new_classes:
            new_classes = default_classes
        # If no valid current recordings or initially empty, select all available
        if not new_recordings:
            new_recordings = avail_recordings

        return (
            gr.update(choices=avail_classes, value=new_classes),  # Keep all choices available
            gr.update(choices=avail_recordings, value=new_recordings),
            state,
            threshold_df_update,
            threshold_json_btn_update,
            threshold_download_btn_update,
            avail_classes,          # NEW – goes to classes_full_list_state
            avail_recordings        # NEW – goes to recordings_full_list_state
        )

    def update_datetime_defaults(processor_state):
        if not processor_state or not processor_state.processor:
            return [gr.update()] * 6

        df = processor_state.processor.get_complete_data()
        start_date, end_date = get_date_range(df)

        return [
            gr.update(value=start_date),
            gr.update(value=end_date),
            gr.update(value="00"),
            gr.update(value="00"),
            gr.update(value="23"),
            gr.update(value="59"),
        ]

    def update_site_choices(proc_state: ProcessorState):
        if not proc_state or not proc_state.processor or proc_state.processor.complete_df.empty:
            # return both the UI update *and* the raw list (empty)
            return (gr.update(choices=[], value=[]), [])

        if 'Site' in proc_state.processor.complete_df.columns:
            sites = sorted(proc_state.processor.complete_df['Site'].dropna().unique())
            # return both the UI update *and* the raw list
            return (gr.update(choices=sites, value=sites), sites)
        else:
            # return both the UI update *and* the raw list (empty)
            return (gr.update(choices=[],   value=[]),   [])

    def combine_time_components(hour, minute) -> typing.Optional[datetime.time]:
        if hour is None or minute is None:
            return None
        try:
            return datetime.time(int(hour), int(minute))
        except ValueError:
            return None

    def plot_predictions_action(
        proc_state: ProcessorState,
        selected_classes_list,
        selected_recordings_list,
        selected_sites_list,
        date_range_start,
        date_range_end,
        time_start_hour,
        time_start_minute,
        time_end_hour,
        time_end_minute,
        correctness_mode="Ignore correctness flags"
    ):
        if not proc_state or not proc_state.processor:
            raise gr.Error(loc.localize("eval-tab-error-calc-metrics-first"))

        processor = proc_state.processor
        thresholds_df = proc_state.class_thresholds

        time_start = combine_time_components(time_start_hour, time_start_minute)
        time_end = combine_time_components(time_end_hour, time_end_minute)

        try:
            filtered_df = processor.get_filtered_data(
                selected_classes=selected_classes_list,
                selected_recordings=selected_recordings_list,
                selected_sites=selected_sites_list,
                date_range_start=date_range_start,
                date_range_end=date_range_end,
                time_start=time_start,
                time_end=time_end,
                correctness_mode=correctness_mode,
                class_thresholds=thresholds_df,
            )
        except Exception as e:
            print(f"Error during filtering: {e}")
            traceback.print_exc()
            raise gr.Error(f"Error filtering data: {str(e)}")

        if filtered_df.empty:
            raise gr.Error("No predictions match the selected filters.")

        col_class = "Class"
        conf_col = "Confidence"

        plotter = ConfidencePlotter(
            data=filtered_df,
            class_col=col_class,
            conf_col=conf_col
        )

        try:
            fig_hist = plotter.plot_histogram_plotly(title="Histogram of Confidence Scores by Class")
            return [proc_state, gr.update(visible=True, value=fig_hist)]
        except Exception as e:
            print(f"Error creating confidence plot: {e}")
            traceback.print_exc()
            raise gr.Error(f"Error creating confidence plot: {str(e)}")

    def plot_temporal_scatter(
        proc_state: ProcessorState,
        selected_classes_list,
        selected_recordings_list,
        selected_sites_list,
        date_range_start,
        date_range_end,
        time_start_hour,
        time_start_minute,
        time_end_hour,
        time_end_minute,
        meta_x=None,
        meta_y=None,
        correctness_mode="Ignore correctness flags"
    ):
        if not proc_state or not proc_state.processor:
            raise gr.Error("Please load predictions first")

        processor = proc_state.processor
        thresholds_df = proc_state.class_thresholds

        if thresholds_df is None:
            raise gr.Error("Class thresholds not initialized. Load data first.")

        time_start = combine_time_components(time_start_hour, time_start_minute)
        time_end = combine_time_components(time_end_hour, time_end_minute)

        try:
            filtered_df = processor.get_filtered_data(
                selected_classes=selected_classes_list,
                selected_recordings=selected_recordings_list,
                selected_sites=selected_sites_list,
                date_range_start=date_range_start,
                date_range_end=date_range_end,
                time_start=time_start,
                time_end=time_end,
                correctness_mode=correctness_mode,
                class_thresholds=thresholds_df,
            )
        except Exception as e:
            print(f"Error during filtering: {e}")
            traceback.print_exc()
            raise gr.Error(f"Error filtering data: {str(e)}")

        if filtered_df.empty:
            raise gr.Error("No data matches the selected filters")

        col_class = "Class"
        conf_col = "Confidence"

        try:
            plotter = TemporalScatterPlotter(
                data=filtered_df,
                class_col=col_class,
                conf_col=conf_col
            )

            fig = plotter.plot(title="Temporal Distribution of Detections")

            try:
                if 'Latitude' in filtered_df.columns and 'Longitude' in filtered_df.columns and \
                   filtered_df['Latitude'].notna().any() and filtered_df['Longitude'].notna().any():
                    mean_lat = filtered_df['Latitude'].mean()
                    mean_lon = filtered_df['Longitude'].mean()
                    print(f"Using coordinates from filtered data: {mean_lat}, {mean_lon}")
                    fig = plotter.add_sunrise_sunset(fig, mean_lat, mean_lon)
                elif proc_state.metadata_dir and meta_x and meta_y:
                    meta_dir = Path(proc_state.metadata_dir)
                    meta_files = list(meta_dir.glob("*.csv")) + list(meta_dir.glob("*.xlsx"))
                    if meta_files:
                        first_meta_file = meta_files[0]
                        if str(first_meta_file).lower().endswith('.xlsx'):
                            metadata_df = pd.read_excel(first_meta_file, sheet_name=0)
                        else:
                            metadata_df = pd.read_csv(first_meta_file)

                        lat_col = meta_x
                        lon_col = meta_y
                        if lat_col in metadata_df.columns and lon_col in metadata_df.columns:
                             mean_lat = pd.to_numeric(metadata_df[lat_col], errors='coerce').mean()
                             mean_lon = pd.to_numeric(metadata_df[lon_col], errors='coerce').mean()
                             if pd.notna(mean_lat) and pd.notna(mean_lon):
                                 print(f"Using average location from metadata file: {mean_lat}, {mean_lon}")
                                 fig = plotter.add_sunrise_sunset(fig, mean_lat, mean_lon)
                             else:
                                 print("Could not calculate mean coordinates from metadata file.")
                        else:
                             print(f"Specified metadata columns '{lat_col}' or '{lon_col}' not found in file.")
                    else:
                        print("No metadata file found for sunrise/sunset fallback.")

            except Exception as e:
                print(f"Error adding sunrise/sunset lines: {str(e)}")
                traceback.print_exc()

            new_state = ProcessorState(
                processor=proc_state.processor,
                prediction_dir=proc_state.prediction_dir,
                metadata_dir=proc_state.metadata_dir,
                color_map=plotter.color_map,
                class_thresholds=thresholds_df
            )

            return [new_state, gr.update(value=fig, visible=True)]

        except Exception as e:
            print(f"Error creating temporal scatter plot: {str(e)}")
            traceback.print_exc()
            raise gr.Error(f"Error creating temporal scatter plot: {str(e)}")

    def plot_spatial_distribution(
        proc_state: ProcessorState,
        selected_classes_list,
        selected_recordings_list,
        selected_sites_list,
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
        if not proc_state or not proc_state.processor:
            raise gr.Error("Please load predictions first")
        if not proc_state.metadata_dir:
             raise gr.Error("Metadata directory not set. Please select metadata files.")

        processor = proc_state.processor
        thresholds_df = proc_state.class_thresholds

        if thresholds_df is None:
            raise gr.Error("Class thresholds not initialized. Load data first.")

        try:
            meta_dir = Path(proc_state.metadata_dir)
            meta_files = list(meta_dir.glob("*.csv")) + list(meta_dir.glob("*.xlsx"))
            if not meta_files:
                raise gr.Error("No metadata CSV or XLSX files found in the specified directory.")

            first_meta_file = meta_files[0]
            print(f"Loading metadata from: {first_meta_file}")
            if str(first_meta_file).lower().endswith('.xlsx'):
                metadata_df = pd.read_excel(first_meta_file, sheet_name=0)
            else:
                 try:
                      metadata_df = pd.read_csv(first_meta_file, sep=None, engine='python', encoding='utf-8-sig')
                 except Exception as e_utf8:
                      print(f"Metadata read failed with utf-8-sig ({e_utf8}), trying latin1...")
                      try:
                           metadata_df = pd.read_csv(first_meta_file, sep=None, engine='python', encoding='latin1')
                      except Exception as e_latin1:
                           raise gr.Error(f"Failed to read metadata file '{first_meta_file.name}' with common encodings: {e_latin1}")

            if meta_site not in metadata_df.columns: raise gr.Error(f"Metadata site column '{meta_site}' not found.")
            if meta_x not in metadata_df.columns: raise gr.Error(f"Metadata latitude column '{meta_x}' not found.")
            if meta_y not in metadata_df.columns: raise gr.Error(f"Metadata longitude column '{meta_y}' not found.")

            processor.set_metadata(metadata_df, site_col=meta_site, lat_col=meta_x, lon_col=meta_y)
            print("Metadata set in processor.")

            all_locations_df = processor.metadata_df[['site_name', 'latitude', 'longitude']].drop_duplicates().copy()
            print(f"Found {len(all_locations_df)} unique locations in metadata.")

            time_start = combine_time_components(time_start_hour, time_start_minute)
            time_end = combine_time_components(time_end_hour, time_end_minute)

            filtered_df = processor.get_filtered_data(
                selected_classes=selected_classes_list,
                selected_recordings=selected_recordings_list,
                selected_sites=selected_sites_list,
                date_range_start=date_range_start,
                date_range_end=date_range_end,
                time_start=time_start,
                time_end=time_end,
                correctness_mode=correctness_mode,
                class_thresholds=thresholds_df,
            )
            print(f"Filtered data shape: {filtered_df.shape}")

            if filtered_df.empty:
                raise gr.Error("No data matches the selected filters")

            if 'Latitude' not in filtered_df.columns or 'Longitude' not in filtered_df.columns:
                 raise gr.Error("Latitude/Longitude columns missing after filtering. Check metadata processing.")
            if filtered_df['Latitude'].isna().all() or filtered_df['Longitude'].isna().all():
                 raise gr.Error("All Latitude/Longitude values are missing in the filtered data. Check site ID matching and metadata.")

            class_col = "Class"
            site_col = "Site"
            lat_col = "Latitude"
            lon_col = "Longitude"

            agg_df = filtered_df.groupby([site_col, lat_col, lon_col, class_col]).size().reset_index(name='count')
            print(f"Aggregated data shape for plotting: {agg_df.shape}")

            plotter = SpatialDistributionPlotter(class_col=class_col)

            if proc_state.color_map:
                print("Using existing color map from state.")
                plotter.color_map = proc_state.color_map.copy()
            else:
                filtered_classes = sorted(filtered_df[class_col].unique())
                temp_plotter = TimeDistributionPlotter(data=filtered_df, class_col=class_col)
                plotter.color_map = temp_plotter._get_color_map(filtered_classes)
                print("Created new color map based on filtered data.")

            fig = plotter.plot(
                agg_df=agg_df,
                all_locations_df=all_locations_df,
                title="Spatial Distribution of Predictions by Class"
            )

            new_state = ProcessorState(
                processor=processor,
                prediction_dir=proc_state.prediction_dir,
                metadata_dir=proc_state.metadata_dir,
                color_map=plotter.color_map,
                class_thresholds=thresholds_df
            )

            return [new_state, gr.update(value=fig, visible=True)]
        except Exception as e:
            print(f"Error creating map: {str(e)}")
            traceback.print_exc()
            raise gr.Error(f"Error creating map: {str(e)}")

    def plot_time_distribution(
        proc_state: ProcessorState,
        time_period: str,
        use_boxplot: bool,
        selected_classes_list,
        selected_recordings_list,
        selected_sites_list,
        date_range_start,
        date_range_end,
        time_start_hour,
        time_start_minute,
        time_end_hour,
        time_end_minute,
        correctness_mode="Ignore correctness flags"
    ):
        if not proc_state or not proc_state.processor:
            raise gr.Error("Please load predictions first")

        processor = proc_state.processor
        thresholds_df = proc_state.class_thresholds

        if thresholds_df is None:
            raise gr.Error("Class thresholds not initialized. Load data first.")

        time_start = combine_time_components(time_start_hour, time_start_minute)
        time_end = combine_time_components(time_end_hour, time_end_minute)

        try:
            filtered_df = processor.get_filtered_data(
                selected_classes=selected_classes_list,
                selected_recordings=selected_recordings_list,
                selected_sites=selected_sites_list,
                date_range_start=date_range_start,
                date_range_end=date_range_end,
                time_start=time_start,
                time_end=time_end,
                correctness_mode=correctness_mode,
                class_thresholds=thresholds_df,
            )
        except Exception as e:
            print(f"Error during filtering: {e}")
            traceback.print_exc()
            raise gr.Error(f"Error filtering data: {str(e)}")

        if filtered_df.empty:
            raise gr.Error("No data matches the selected filters")

        col_class = "Class"

        plotter = TimeDistributionPlotter(
            data=filtered_df,
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
            print(f"Error creating time distribution plot: {e}")
            traceback.print_exc()
            raise gr.Error(f"Error creating time distribution plot: {str(e)}")

    def calculate_detections_action(
        proc_state: ProcessorState,
        selected_classes_list,
        selected_recordings_list,
        selected_sites_list,
        date_range_start,
        date_range_end,
        time_start_hour,
        time_start_minute,
        time_end_hour,
        time_end_minute,
        correctness_mode="Ignore correctness flags"
    ):
        if not proc_state or not proc_state.processor:
            raise gr.Error("Please load predictions first")

        processor = proc_state.processor
        thresholds_df = proc_state.class_thresholds

        if thresholds_df is None:
            raise gr.Error("Class thresholds not initialized. Load data first.")

        time_start = combine_time_components(time_start_hour, time_start_minute)
        time_end = combine_time_components(time_end_hour, time_end_minute)

        try:
            filtered_df = processor.get_filtered_data(
                selected_classes=selected_classes_list,
                selected_recordings=selected_recordings_list,
                selected_sites=selected_sites_list,
                date_range_start=date_range_start,
                date_range_end=date_range_end,
                time_start=time_start,
                time_end=time_end,
                correctness_mode=correctness_mode,
                class_thresholds=thresholds_df,
            )
        except Exception as e:
            print(f"Error during filtering: {e}")
            traceback.print_exc()
            raise gr.Error(f"Error filtering data: {str(e)}")

        if filtered_df.empty:
            # Return an empty DataFrame wrapped in gr.update to clear the table
            return gr.update(value=pd.DataFrame(columns=["Species", "Count", "Percentage"]), visible=True)

        try:
            counts_df = calculate_detection_counts(filtered_df)
            # Return the calculated DataFrame wrapped in gr.update
            return gr.update(value=counts_df, visible=True)
        except Exception as e:
            print(f"Error calculating detection counts: {e}")
            traceback.print_exc()
            raise gr.Error(f"Error calculating detection counts: {str(e)}")

    def get_selection_tables(
        directory,
        patterns = ("*.txt", "*.csv", "*.tsv", "*.xlsx"),
        recursive = True,
    ):
        """
        Collect all prediction-table files under *directory*.

        Parameters
        ----------
        directory : str | pathlib.Path
            Root folder that the user selected in the GUI.
        patterns  : iterable of str, default ("*.txt", "*.csv", "*.tsv", "*.xlsx")
            Shell-style glob patterns to match (case-insensitive).
            Adjust if you need a tighter filter.
        recursive : bool, default True
            • True  → walk the entire directory tree with Path.rglob  
            • False → only look at files directly inside *directory*

        Returns
        -------
        list[pathlib.Path]
            Sorted list of matching files.
        """
        root = Path(directory).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(root)

        # Use rglob for deep search, glob for single level
        gather = root.rglob if recursive else root.glob

        files: list[Path] = []
        for pattern in patterns:
            files.extend(gather(pattern))

        # Sort for consistent order (relative paths for readability)
        return sorted(files, key=lambda p: p.as_posix().lower())

    def download_threshold_template(proc_state: ProcessorState):
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
        if not proc_state or not proc_state.processor:
            gr.Warning("Processor not initialized. Load data first.")
            return proc_state, gr.update()

        try:
            file_path = gu.select_file(
                filetypes=('JSON files (*.json)', 'All files (*.*)'),
                state_key="viz-threshold-json"
            )

            if not file_path:
                return proc_state, gr.update()

            if not isinstance(file_path, str):
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

        processor_state = gr.State()
        prediction_files_state = gr.State()
        metadata_files_state = gr.State()
        # NEW ─ keeps a copy of the *full* choice lists so the
        #       “Select-All / Deselect-All” buttons know what to do
        classes_full_list_state     = gr.State([])
        recordings_full_list_state  = gr.State([])
        sites_full_list_state       = gr.State([])

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

        with gr.Group(visible=False) as prediction_group:
            with gr.Accordion(loc.localize("eval-tab-prediction-col-accordion-label"), open=True):
                with gr.Row():
                    prediction_columns: dict[str, gr.Dropdown] = {}
                    for col in ["Start Time", "Class", "Confidence", "Recording", "Correctness"]:
                        prediction_columns[col] = gr.Dropdown(choices=[], label=localized_column_labels.get(col, col))

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

        with gr.Group(visible=True) as selection_group:
            with gr.Accordion(loc.localize("viz-tab-class-recording-accordion-label"), open=False):
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            classes_select_all_btn = gr.Button("✓ All",   size="sm")
                            classes_deselect_all_btn = gr.Button("✕ None", size="sm")
                        select_classes_checkboxgroup = gr.CheckboxGroup(
                            choices=[],
                            value=[],
                            label=loc.localize("viz-tab-classes-label"),
                            info=loc.localize("viz-tab-classes-info"),
                            interactive=True,
                            elem_classes="custom-checkbox-group",
                        )
                    with gr.Column():
                        with gr.Row():
                            recordings_select_all_btn = gr.Button("✓ All",   size="sm")
                            recordings_deselect_all_btn = gr.Button("✕ None", size="sm")
                        select_recordings_checkboxgroup = gr.CheckboxGroup(
                            choices=[],
                            value=[],
                            label=loc.localize("viz-tab-recordings-label"),
                            info=loc.localize("viz-tab-recordings-info"),
                            interactive=True,
                            elem_classes="custom-checkbox-group",
                        )
                    with gr.Column():
                        with gr.Row():
                            sites_select_all_btn = gr.Button("✓ All",   size="sm")
                            sites_deselect_all_btn = gr.Button("✕ None", size="sm")
                        select_sites_checkboxgroup = gr.CheckboxGroup(
                            choices=[],
                            value=[],
                            label=loc.localize("viz-tab-sites-label"),
                            info=loc.localize("viz-tab-sites-info"),
                            interactive=True,
                            elem_classes="custom-checkbox-group",
                        )

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
                     class_thresholds_df = gr.DataFrame(
                         headers=[loc.localize("viz-tab-threshold-class-header"), loc.localize("viz-tab-threshold-value-header")],
                         datatype=["str", "number"],
                         label="Class Confidence Thresholds (Read-only)",
                         interactive=False,
                         visible=False,
                         col_count=(2, "fixed")
                     )
                with gr.Row():
                    threshold_json_select_btn = gr.Button(
                        loc.localize("viz-tab-select-threshold-json-button"),
                        variant="secondary",
                        visible=False,
                        scale=2
                    )
                    threshold_template_download_btn = gr.Button(
                        loc.localize("viz-tab-download-threshold-template-button"),
                        variant="secondary",
                        visible=False,
                        scale=2
                    )

                correctness_mode = gr.Radio(
                    choices=[
                        loc.localize("viz-tab-correctness-ignore"),
                        loc.localize("viz-tab-correctness-only-correct"),
                        loc.localize("viz-tab-correctness-only-incorrect"),
                        loc.localize("viz-tab-correctness-only-unspecified")
                    ],
                    value=loc.localize("viz-tab-correctness-ignore"),
                    label=loc.localize("viz-tab-correctness-mode-label"),
                    info=loc.localize("viz-tab-correctness-mode-info"),
                    interactive=True
                )

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

        plot_predictions_btn = gr.Button(
            loc.localize("viz-tab-plot-distributions-button-label"),
            variant="huggingface"
        )
        smooth_distribution_output = gr.Plot(label=loc.localize("viz-tab-distribution-plot-label"), visible=False)

        plot_map_btn = gr.Button(
            loc.localize("viz-tab-plot-map-button-label"),
            variant="huggingface"
        )
        map_output = gr.Plot(label=loc.localize("viz-tab-map-plot-label"), visible=False)

        plot_time_distribution_btn = gr.Button(
            loc.localize("viz-tab-plot-time-distribution-button-label"),
            variant="huggingface"
        )
        # Options for this plot in an Accordion
        with gr.Accordion(loc.localize("viz-tab-time-distribution-options-label"), open=False):
             with gr.Row():
                time_distribution_period = gr.Dropdown(
                    choices=["hour", "day", "month", "year"],
                    value="hour",
                    label=loc.localize("viz-tab-time-period-label"),
                    info=loc.localize("viz-tab-time-period-info")
                )
                use_boxplot = gr.Checkbox(
                    label=loc.localize("viz-tab-use-boxplot-label"),
                    info=loc.localize("viz-tab-use-boxplot-info"),
                    value=False
                )
        time_distribution_output = gr.Plot(
            label=loc.localize("viz-tab-time-distribution-plot-label"),
            visible=False
        )

        plot_temporal_scatter_btn = gr.Button(
            loc.localize("viz-tab-plot-temporal-scatter-button-label"),
            variant="huggingface"
        )
        temporal_scatter_output = gr.Plot(
            label=loc.localize("viz-tab-temporal-scatter-plot-label"),
            visible=False
        )

        calculate_detections_btn = gr.Button(
            loc.localize("viz-tab-calculate-detections-button-label"),
            variant="huggingface"
        )
        detections_table = gr.DataFrame(
            show_label=False,
            type="pandas",
            visible=False,
            interactive=False,
            wrap=True,
            column_widths=[200, 110, 80, 130, 90, 150, 110, 110, 80]
        )

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

        metadata_directory_input.change(
            lambda x: gr.update(visible=bool(x)),
            inputs=[metadata_files_state],
            outputs=[metadata_group],
        )

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
                    select_classes_checkboxgroup,
                    select_recordings_checkboxgroup,
                ],
                outputs=[
                    select_classes_checkboxgroup,
                    select_recordings_checkboxgroup,
                    processor_state,
                    class_thresholds_df,
                    threshold_json_select_btn,
                    threshold_template_download_btn,
                    classes_full_list_state,        # NEW
                    recordings_full_list_state      # NEW
                ],
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
            ).success(
                 fn=update_site_choices,
                 inputs=[processor_state],
                 outputs=[select_sites_checkboxgroup, sites_full_list_state] # 2 outputs now
            )

        threshold_json_select_btn.click(
            fn=select_threshold_json_file,
            inputs=[processor_state],
            outputs=[processor_state, class_thresholds_df],
            api_name="select_threshold_json"
        )

        threshold_template_download_btn.click(
            fn=download_threshold_template,
            inputs=[processor_state]
        )

        plot_predictions_btn.click(
            fn=plot_predictions_action,
            inputs=[
                processor_state,
                select_classes_checkboxgroup,
                select_recordings_checkboxgroup,
                select_sites_checkboxgroup,
                date_range_start,
                date_range_end,
                time_start_hour,
                time_start_minute,
                time_end_hour,
                time_end_minute,
                correctness_mode,
            ],
            outputs=[processor_state, smooth_distribution_output]
        )

        plot_map_btn.click(
            fn=plot_spatial_distribution,
            inputs=[
                processor_state,
                select_classes_checkboxgroup,
                select_recordings_checkboxgroup,
                select_sites_checkboxgroup,
                metadata_columns["X"],
                metadata_columns["Y"],
                metadata_columns["Site"],
                date_range_start,
                date_range_end,
                time_start_hour,
                time_start_minute,
                time_end_hour,
                time_end_minute,
                correctness_mode,
            ],
            outputs=[processor_state, map_output]
        )

        plot_time_distribution_btn.click(
            fn=plot_time_distribution,
            inputs=[
                processor_state,
                time_distribution_period,
                use_boxplot,
                select_classes_checkboxgroup,
                select_recordings_checkboxgroup,
                select_sites_checkboxgroup,
                date_range_start,
                date_range_end,
                time_start_hour,
                time_start_minute,
                time_end_hour,
                time_end_minute,
                correctness_mode,
            ],
            outputs=[time_distribution_output]
        )

        plot_temporal_scatter_btn.click(
            fn=plot_temporal_scatter,
            inputs=[
                processor_state,
                select_classes_checkboxgroup,
                select_recordings_checkboxgroup,
                select_sites_checkboxgroup,
                date_range_start,
                date_range_end,
                time_start_hour,
                time_start_minute,
                time_end_hour,
                time_end_minute,
                metadata_columns["X"],      # latitude-column dropdown
                metadata_columns["Y"],      # longitude-column dropdown
                correctness_mode,
            ],
            outputs=[processor_state, temporal_scatter_output]
        )

        calculate_detections_btn.click(
            fn=calculate_detections_action,
            inputs=[
                processor_state,
                select_classes_checkboxgroup,
                select_recordings_checkboxgroup,
                select_sites_checkboxgroup,
                date_range_start,
                date_range_end,
                time_start_hour,
                time_start_minute,
                time_end_hour,
                time_end_minute,
                correctness_mode,
            ],
            outputs=[detections_table]
        )

        # ───────────  CLASSES  ───────────
        classes_select_all_btn.click(
            fn=lambda full: gr.update(value=full),
            inputs=[classes_full_list_state],
            outputs=[select_classes_checkboxgroup],
            queue=False
        )
        classes_deselect_all_btn.click(
            fn=lambda: gr.update(value=[]),
            outputs=[select_classes_checkboxgroup],
            queue=False
        )

        # ───────────  RECORDINGS  ───────────
        recordings_select_all_btn.click(
            fn=lambda full: gr.update(value=full),
            inputs=[recordings_full_list_state],
            outputs=[select_recordings_checkboxgroup],
            queue=False
        )
        recordings_deselect_all_btn.click(
            fn=lambda: gr.update(value=[]),
            outputs=[select_recordings_checkboxgroup],
            queue=False
        )

        # ───────────  SITES  ───────────
        sites_select_all_btn.click(
            fn=lambda full: gr.update(value=full),
            inputs=[sites_full_list_state],
            outputs=[select_sites_checkboxgroup],
            queue=False
        )
        sites_deselect_all_btn.click(
            fn=lambda: gr.update(value=[]),
            outputs=[select_sites_checkboxgroup],
            queue=False
        )


if __name__ == "__main__":
    gu.open_window(build_visualization_tab)
