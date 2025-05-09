import json
import os
import shutil
import tempfile
import typing
import io
from pathlib import Path
import datetime
import traceback
import hashlib
import warnings

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

from birdnet_analyzer.visualization.common import ProcessorState, convert_timestamp_to_datetime, NO_SITE_LABEL
from birdnet_analyzer.visualization.caching import (
    fingerprint_prediction_dir, fingerprint_metadata_file,
)

_PROCESSOR_CACHE: dict[str, dp.DataProcessor] = {}
_METADATA_CACHE: dict[str, pd.DataFrame] = {}


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


def _process_selected_sites_for_filtering(selected_sites_list: typing.Optional[typing.List[typing.Any]]) -> typing.Optional[typing.List[typing.Any]]:
    if selected_sites_list is None:
        return None
    return [None if site == NO_SITE_LABEL else site for site in selected_sites_list]


def build_visualization_tab():
    """
    Builds a Gradio tab for loading and plotting prediction data only,
    using ConfidencePlotter. Annotation logic and metric calculations
    have been removed.
    """

    _prediction_file_cache = {"hash": None, "temp_dir": None}
    _metadata_file_cache = {"hash": None, "temp_dir": None}

    prediction_default_columns = {
        "Start Time": "File Offset (s)",
        "Class": "Common Name",
        "Recording": "Begin Path",
        "Confidence": "Confidence",
        "Correctness": "Correctness",
    }

    metadata_default_columns = {
        "Site": "SiteID",
        "X": "Latitude",
        "Y": "Longitude",
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

    def ensure_tmp_dir(file_paths, cache, prefix="birdnet_"):
        if not file_paths:
            if cache["temp_dir"] and Path(cache["temp_dir"]).exists():
                print(f"No files selected, removing old temp dir: {cache['temp_dir']}")
                shutil.rmtree(cache["temp_dir"], ignore_errors=True)
            cache["hash"] = None
            cache["temp_dir"] = None
            return None

        valid_paths = []
        errors = []
        for f in file_paths:
            try:
                p = Path(f)
                if p.is_file():
                    valid_paths.append(p)
                else:
                    errors.append(f"Path is not a file: {f}")
            except Exception as e:
                errors.append(f"Error accessing path {f}: {e}")

        if errors:
            print("Warnings accessing file paths:", errors)

        if not valid_paths:
            if cache["temp_dir"] and Path(cache["temp_dir"]).exists():
                print(f"No valid files found, removing old temp dir: {cache['temp_dir']}")
                shutil.rmtree(cache["temp_dir"], ignore_errors=True)
            cache["hash"] = None
            cache["temp_dir"] = None
            gr.Warning("No valid files found in selection.")
            return None

        try:
            sorted_paths = sorted(valid_paths, key=lambda p: str(p.resolve()))
            mtimes = tuple(p.stat().st_mtime for p in sorted_paths)
            path_strings = tuple(str(p.resolve()) for p in sorted_paths)
            hash_content = str((mtimes, path_strings)).encode()
            current_hash = hashlib.sha256(hash_content).hexdigest()
        except FileNotFoundError as e:
            print(f"Error accessing file stats for hashing: {e}. Clearing cache.")
            if cache["temp_dir"] and Path(cache["temp_dir"]).exists():
                shutil.rmtree(cache["temp_dir"], ignore_errors=True)
            cache["hash"] = None
            cache["temp_dir"] = None
            raise gr.Error(f"Error accessing file: {e}. Please check file paths.")

        if current_hash == cache["hash"] and cache["temp_dir"] and Path(cache["temp_dir"]).exists():
            print(f"Cache hit for {prefix[:-1]} files. Reusing temp dir: {cache['temp_dir']}")
            return cache["temp_dir"]

        if cache["temp_dir"] and Path(cache["temp_dir"]).exists():
            print(f"Cache miss or invalid temp dir. Removing old temp dir: {cache['temp_dir']}")
            shutil.rmtree(cache["temp_dir"], ignore_errors=True)

        new_temp_dir = tempfile.mkdtemp(prefix=prefix)
        print(f"Created new temp dir: {new_temp_dir}")

        try:
            for src_path in valid_paths:
                dest_path = Path(new_temp_dir) / src_path.name
                shutil.copy2(src_path, dest_path)
            print(f"Copied {len(valid_paths)} files to {new_temp_dir}")
        except Exception as e:
            print(f"Error copying files to temp dir {new_temp_dir}: {e}")
            shutil.rmtree(new_temp_dir, ignore_errors=True)
            cache["hash"] = None
            cache["temp_dir"] = None
            raise gr.Error(f"Failed to copy files to temporary location: {e}")

        cache["hash"] = current_hash
        cache["temp_dir"] = new_temp_dir

        return new_temp_dir

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
        """
        Return (classes, recordings, processor, prediction_dir).

        A new DataProcessor is only created when either
           – the directory contents      OR
           – the user column mapping
        changed, or if a cached processor's directory is no longer valid.
        """
        if not prediction_files:
            return [], [], None, None

        if prediction_dir is None:
            prediction_dir = ensure_tmp_dir(
                prediction_files, _prediction_file_cache, prefix="birdnet_pred_"
            )
        
        if prediction_dir is None:
            return [], [], None, None

        cols_pred = {
            "Start Time": pred_start_time  or prediction_default_columns["Start Time"],
            "Class"     : pred_class       or prediction_default_columns["Class"],
            "Confidence": pred_confidence  or prediction_default_columns["Confidence"],
            "Recording" : pred_recording   or prediction_default_columns["Recording"],
            "Correctness": pred_correctness or prediction_default_columns.get("Correctness"),
        }

        fp = fingerprint_prediction_dir(prediction_dir, cols_pred)
        proc = _PROCESSOR_CACHE.get(fp)

        if proc is not None:
            try:
                cached_proc_path = Path(proc.prediction_directory_path)
                if not cached_proc_path.exists() or not any(cached_proc_path.iterdir()):
                    print(f"⚠️ Cached DataProcessor's directory {cached_proc_path} is missing or empty. Rebuilding.")
                    proc = None
            except Exception as e:
                print(f"⚠️ Error validating cached DataProcessor's directory ({proc.prediction_directory_path if proc else 'N/A'}): {e}. Rebuilding.")
                proc = None

        if proc is None:
            print(f"⏳  Cache miss – building DataProcessor for {prediction_dir}")
            try:
                proc = dp.DataProcessor(
                    prediction_directory_path=prediction_dir,
                    columns_predictions=cols_pred,
                )
                _PROCESSOR_CACHE[fp] = proc
            except Exception as e:
                print(f"Error creating DataProcessor: {e}")
                traceback.print_exc()
                raise gr.Error(f"Error initializing processor: {str(e)}")

        else:
            print(f"✅  Using cached DataProcessor (key={fp[:8]}) for directory {proc.prediction_directory_path}")

        _ = proc.get_complete_data()

        classes = sorted(proc.predictions_df['Class'].dropna().astype(str).unique()) if not proc.predictions_df.empty and 'Class' in proc.predictions_df.columns else []
        recordings = sorted(proc_df['Recording'].dropna().astype(str).unique()) if not (proc_df := proc.predictions_df).empty and 'Recording' in proc_df.columns else []

        return classes, recordings, proc, prediction_dir

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
                print(f"Attempting to read metadata headers from: {file_path}")

                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message="Columns \\(.*\\) have mixed types.*",
                            category=pd.errors.DtypeWarning
                        )
                        warnings.filterwarnings(
                            "ignore",
                            message="Columns \\(.*\\) have mixed types.*",
                            category=UserWarning
                        )

                        if file_path.lower().endswith('.xlsx'):
                            try:
                                df = pd.read_excel(file_path, sheet_name=0, nrows=0)
                                print(f"Successfully read Excel headers: {list(df.columns)}")
                                cols.update(df.columns)
                                continue
                            except Exception as e_excel:
                                print(f"Reading as Excel failed: {e_excel}. Will try CSV.")

                        try:
                            df = pd.read_csv(file_path, sep=None, engine='python', nrows=0)
                            print(f"Successfully read CSV/TSV headers with auto-detect: {list(df.columns)}")
                            cols.update(df.columns)
                        except UnicodeDecodeError as e_unicode:
                            print(f"Auto-detect CSV failed with UnicodeDecodeError: {e_unicode}. Trying latin1.")
                            try:
                                df = pd.read_csv(file_path, sep=None, engine='python', encoding='latin1', nrows=0)
                                print(f"Successfully read CSV/TSV headers with latin1 fallback: {list(df.columns)}")
                                cols.update(df.columns)
                            except Exception as e_fallback:
                                print(f"CSV reading with latin1 fallback also failed: {e_fallback}")
                                gr.Warning(f"Could not read headers from {os.path.basename(file_path)} using common methods.")
                        except Exception as e_csv:
                            print(f"General CSV reading failed: {e_csv}")
                            if not file_path.lower().endswith('.xlsx'):
                                gr.Warning(f"Could not read headers from {os.path.basename(file_path)} as CSV/TSV.")

                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
                    gr.Warning(f"{loc.localize('eval-tab-warning-error-reading-file')} {os.path.basename(file_path)}")

        final_cols = sorted([col for col in cols if col and pd.notna(col)])
        print(f"Final unique columns found: {final_cols}")

        updates = []
        choices_with_empty = [""] + final_cols
        for label in ["Site", "X", "Y"]:
            default_val = metadata_default_columns.get(label)
            val = default_val if default_val in final_cols else None
            updates.append(gr.update(choices=choices_with_empty, value=val))
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
    ):
        prediction_dir = ensure_tmp_dir(prediction_files, _prediction_file_cache, prefix="birdnet_pred_")
        metadata_dir = ensure_tmp_dir(metadata_files, _metadata_file_cache, prefix="birdnet_meta_")

        avail_classes, avail_recordings, proc, prediction_dir_from_init = initialize_processor(
            prediction_files,
            pred_start_time,
            pred_class,
            pred_confidence,
            pred_recording,
            pred_correctness,
            prediction_dir,
        )
        
        prediction_dir = prediction_dir_from_init

        if proc and metadata_files and metadata_dir and meta_site and meta_x and meta_y:
            meta_path = Path(metadata_dir)
            meta_file = next(
                (p for p in meta_path.iterdir() if p.suffix.lower() in (".csv", ".xlsx")),
                None
            )
            if meta_file is not None:
                meta_fp = fingerprint_metadata_file(
                    str(meta_file),
                    {"Site": meta_site, "Latitude": meta_x, "Longitude": meta_y},
                )
                meta_df = _METADATA_CACHE.get(meta_fp)
                if meta_df is None:
                    print(f"⏳  Reading metadata {meta_file}")
                    try:
                        meta_df = (
                            pd.read_excel(meta_file) if meta_file.suffix.lower() == ".xlsx"
                            else pd.read_csv(meta_file, sep=None, engine="python")
                        )
                        _METADATA_CACHE[meta_fp] = meta_df
                    except Exception as e:
                        print(f"Error reading metadata file {meta_file}: {e}")
                        gr.Warning(f"Could not read metadata file: {meta_file.name}. Error: {e}")
                        meta_df = None
                else:
                    print(f"✅  Using cached metadata ({meta_fp[:8]})")

                if meta_df is not None and (proc.metadata_df is None or proc.metadata_df is not meta_df):
                    try:
                        proc.set_metadata(meta_df, site_col=meta_site, lat_col=meta_x, lon_col=meta_y)
                    except Exception as e:
                        print(f"Error setting metadata: {e}")
                        gr.Warning(f"Error processing metadata: {e}")

        class_thresholds_init_df = None
        threshold_df_update = gr.update(visible=False, value=None)
        threshold_json_btn_update = gr.update(visible=False)
        threshold_download_btn_update = gr.update(visible=False)

        if proc:
            default_threshold = 0.1
            if avail_classes:
                class_thresholds_init_df = pd.DataFrame({
                    'Class': sorted(list(avail_classes)),
                    'Threshold': [default_threshold] * len(avail_classes)
                })
                threshold_df_update = gr.update(visible=True, value=class_thresholds_init_df)
            else:
                class_thresholds_init_df = pd.DataFrame(columns=['Class', 'Threshold'])
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
            threshold_df_update = gr.update(visible=False, value=None)

        new_classes = []
        new_recordings = avail_recordings

        default_classes = avail_classes
        if len(avail_classes) > MAX_DEFAULT_CLASSES:
            default_classes = avail_classes[:MAX_DEFAULT_CLASSES]
            gr.Warning(
                f"Found {len(avail_classes)} classes. Selecting the first {MAX_DEFAULT_CLASSES} "
                f"by default for visualization clarity. You can change the selection in the Select classes and Recordings section."
            )

        if current_classes:
            new_classes = [c for c in current_classes if c in avail_classes]

        if not new_classes:
            new_classes = default_classes

        return (
            gr.update(choices=avail_classes, value=new_classes),
            state,
            threshold_df_update,
            threshold_json_btn_update,
            threshold_download_btn_update,
            avail_classes,          
            avail_recordings
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
            return (gr.update(choices=[], value=[]), [])

        if 'Site' not in proc_state.processor.complete_df.columns:
            return (gr.update(choices=[], value=[]), [])

        unique_sites_from_df = proc_state.processor.complete_df['Site'].unique()
        
        gradio_choices = []
        # Add actual sites, sorted alphabetically
        actual_sites = sorted(
            list(set(str(s).strip() for s in unique_sites_from_df if pd.notna(s) and str(s).strip()))
        )
        for site_str in actual_sites:
            gradio_choices.append((site_str, site_str)) # (value, label)

        # Check if there are any sites that are None/NaN
        has_null_equivalent_sites = any(pd.isna(s) for s in unique_sites_from_df)
        
        if has_null_equivalent_sites:
            # Add the "No Site Assigned" option, ensuring it's consistently placed (e.g., at the end)
            gradio_choices.append((NO_SITE_LABEL, NO_SITE_LABEL)) # value is NO_SITE_LABEL, label is the text

        # Default selection: select all, including the "No Site Assigned" if present
        default_selected_values = [choice[0] for choice in gradio_choices]
        
        # The sites_full_list_state should store the values that can be directly used for selection
        full_list_for_state = default_selected_values.copy()

        return (gr.update(choices=gradio_choices, value=default_selected_values), full_list_for_state)

    def combine_time_components(hour, minute) -> typing.Optional[datetime.time]:
        if hour is None or minute is None:
            return None
        try:
            return datetime.time(int(hour), int(minute))
        except ValueError:
            return None

    def map_correctness_mode_to_dp(ui_correctness_mode: str) -> str:
        if ui_correctness_mode == loc.localize("viz-tab-correctness-only-correct"):
            return "Correct"
        elif ui_correctness_mode == loc.localize("viz-tab-correctness-only-incorrect"):
            return "Incorrect"
        elif ui_correctness_mode == loc.localize("viz-tab-correctness-only-unspecified"):
            return "Unspecified"
        # Default, including loc.localize("viz-tab-correctness-ignore")
        return "All"

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
        dp_correctness_mode = map_correctness_mode_to_dp(correctness_mode)

        if not proc_state or not proc_state.processor:
            raise gr.Error(loc.localize("eval-tab-error-calc-metrics-first"))

        processor = proc_state.processor
        thresholds_df = proc_state.class_thresholds

        time_start = combine_time_components(time_start_hour, time_start_minute)
        time_end = combine_time_components(time_end_hour, time_end_minute)

        processed_selected_sites = _process_selected_sites_for_filtering(selected_sites_list)

        try:
            filtered_df = processor.get_filtered_data(
                selected_classes=selected_classes_list,
                selected_recordings=selected_recordings_list,
                selected_sites=processed_selected_sites,
                date_range_start=date_range_start,
                date_range_end=date_range_end,
                time_start=time_start,
                time_end=time_end,
                correctness_mode=dp_correctness_mode,
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
        correctness_mode="Ignore correctness flags"
    ):
        dp_correctness_mode = map_correctness_mode_to_dp(correctness_mode)
        if not proc_state or not proc_state.processor:
            raise gr.Error("Please load predictions first")

        processor = proc_state.processor
        thresholds_df = proc_state.class_thresholds

        time_start = combine_time_components(time_start_hour, time_start_minute)
        time_end = combine_time_components(time_end_hour, time_end_minute)

        processed_selected_sites = _process_selected_sites_for_filtering(selected_sites_list)

        try:
            filtered_df = processor.get_filtered_data(
                selected_classes=selected_classes_list,
                selected_recordings=selected_recordings_list,
                selected_sites=processed_selected_sites,
                date_range_start=date_range_start,
                date_range_end=date_range_end,
                time_start=time_start,
                time_end=time_end,
                correctness_mode=dp_correctness_mode,
                class_thresholds=thresholds_df,
            )
        except Exception as e:
            print(f"Error during filtering: {e}")
            traceback.print_exc()
            raise gr.Error(f"Error filtering data: {str(e)}")

        if filtered_df.empty:
            gr.Warning("No data matches the selected filters. Plot will be empty.")
            empty_fig = go.Figure()
            empty_fig.update_layout(title="Temporal Distribution of Detections (No Data)")
            return [proc_state, gr.update(value=empty_fig, visible=True)]

        col_class = "Class"
        conf_col = "Confidence"

        try:
            plotter = TemporalScatterPlotter(
                data=filtered_df,
                class_col=col_class,
                conf_col=conf_col
            )

            fig = plotter.plot(title="Temporal Distribution of Detections")

            if processor.metadata_centroid:
                try:
                    latitude, longitude = processor.metadata_centroid
                    fig = plotter.add_sunrise_sunset(fig, latitude, longitude)
                    print(f"Added sunrise/sunset lines using cached centroid: ({latitude:.4f}, {longitude:.4f})")
                except ImportError as e:
                     gr.Warning(f"Optional dependency missing for sunrise/sunset: {e}. Install with: pip install astral timezonefinder")
                except Exception as e:
                    gr.Warning(f"Could not add sunrise/sunset lines: {e}")
            elif proc_state.metadata_dir:
                 gr.Warning("Metadata loaded, but centroid could not be calculated or is invalid. Sunrise/sunset lines omitted.")

            if not proc_state.color_map and plotter.color_map:
                 proc_state.color_map = plotter.color_map
            elif plotter.color_map:
                 proc_state.color_map = plotter.color_map

            return [proc_state, gr.update(value=fig, visible=True)]

        except Exception as e:
            print(f"Error creating temporal scatter plot: {str(e)}")
            traceback.print_exc()
            raise gr.Error(f"Error creating temporal scatter plot: {str(e)}")

    def plot_spatial_distribution(
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
        dp_correctness_mode = map_correctness_mode_to_dp(correctness_mode)
        if not proc_state or not proc_state.processor:
            raise gr.Error("Please load predictions first")
        if proc_state.processor.metadata_df is None or proc_state.processor.metadata_df.empty:
             raise gr.Error("Metadata has not been loaded or is empty. Ensure metadata files and columns are selected correctly.")

        processor = proc_state.processor
        thresholds_df = proc_state.class_thresholds

        if thresholds_df is None:
            raise gr.Error("Class thresholds not initialized. Load data first.")

        try:
            if processor.metadata_df is None or 'latitude' not in processor.metadata_df.columns or 'longitude' not in processor.metadata_df.columns:
                 raise gr.Error("Processed metadata (with Latitude/Longitude) not found on processor. Check metadata loading steps.")

            all_locations_df = processor.metadata_df[['site_name', 'latitude', 'longitude']].drop_duplicates().copy()
            print(f"Found {len(all_locations_df)} unique locations in metadata for spatial plot.")

            time_start = combine_time_components(time_start_hour, time_start_minute)
            time_end = combine_time_components(time_end_hour, time_end_minute)

            processed_selected_sites = _process_selected_sites_for_filtering(selected_sites_list)

            filtered_df = processor.get_filtered_data(
                selected_classes=selected_classes_list,
                selected_recordings=selected_recordings_list,
                selected_sites=processed_selected_sites,
                date_range_start=date_range_start,
                date_range_end=date_range_end,
                time_start=time_start,
                time_end=time_end,
                correctness_mode=dp_correctness_mode,
                class_thresholds=thresholds_df,
            )
            print(f"Filtered data shape for spatial plot: {filtered_df.shape}")

            if filtered_df.empty:
                gr.Warning("No data matches the selected filters for the spatial plot.")
                empty_fig = go.Figure()
                empty_fig.update_layout(title="Spatial Distribution (No Data)")
                return [proc_state, gr.update(value=empty_fig, visible=True)]

            if 'Latitude' not in filtered_df.columns or 'Longitude' not in filtered_df.columns:
                 raise gr.Error("Latitude/Longitude columns missing after filtering. Check metadata processing and merging.")
            if filtered_df['Latitude'].isna().all() or filtered_df['Longitude'].isna().all():
                 if all_locations_df.empty:
                      raise gr.Error("Metadata loaded, but no valid coordinates found for any site.")
                 else:
                      raise gr.Error("All Latitude/Longitude values are missing in the filtered data. Check site ID matching in predictions or filter criteria.")

            class_col = "Class"
            site_col = "Site"
            lat_col = "Latitude"
            lon_col = "Longitude"

            agg_df = filtered_df.groupby([site_col, lat_col, lon_col, class_col]).size().reset_index(name='count')
            print(f"Aggregated data shape for plotting: {agg_df.shape}")

            plotter = SpatialDistributionPlotter(class_col=class_col)

            if proc_state.color_map:
                print("Using existing color map from state for spatial plot.")
                plotter.color_map = proc_state.color_map.copy()
            else:
                filtered_classes = sorted(filtered_df[class_col].unique())
                temp_plotter = TimeDistributionPlotter(data=filtered_df, class_col=class_col)
                plotter.color_map = temp_plotter._get_color_map(filtered_classes)
                print("Created new color map for spatial plot based on filtered data.")
                proc_state.color_map = plotter.color_map

            fig = plotter.plot(
                agg_df=agg_df,
                all_locations_df=all_locations_df,
                title="Spatial Distribution of Predictions by Class"
            )

            return [proc_state, gr.update(value=fig, visible=True)]
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
        dp_correctness_mode = map_correctness_mode_to_dp(correctness_mode)
        if not proc_state or not proc_state.processor:
            raise gr.Error("Please load predictions first")

        processor = proc_state.processor
        thresholds_df = proc_state.class_thresholds

        if thresholds_df is None:
            raise gr.Error("Class thresholds not initialized. Load data first.")

        time_start = combine_time_components(time_start_hour, time_start_minute)
        time_end = combine_time_components(time_end_hour, time_end_minute)

        processed_selected_sites = _process_selected_sites_for_filtering(selected_sites_list)

        try:
            filtered_df = processor.get_filtered_data(
                selected_classes=selected_classes_list,
                selected_recordings=selected_recordings_list,
                selected_sites=processed_selected_sites,
                date_range_start=date_range_start,
                date_range_end=date_range_end,
                time_start=time_start,
                time_end=time_end,
                correctness_mode=dp_correctness_mode,
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
        dp_correctness_mode = map_correctness_mode_to_dp(correctness_mode)
        if not proc_state or not proc_state.processor:
            raise gr.Error("Please load predictions first")

        processor = proc_state.processor
        thresholds_df = proc_state.class_thresholds

        if thresholds_df is None:
            raise gr.Error("Class thresholds not initialized. Load data first.")

        time_start = combine_time_components(time_start_hour, time_start_minute)
        time_end = combine_time_components(time_end_hour, time_end_minute)

        processed_selected_sites = _process_selected_sites_for_filtering(selected_sites_list)

        try:
            filtered_df = processor.get_filtered_data(
                selected_classes=selected_classes_list,
                selected_recordings=selected_recordings_list,
                selected_sites=processed_selected_sites,
                date_range_start=date_range_start,
                date_range_end=date_range_end,
                time_start=time_start,
                time_end=time_end,
                correctness_mode=dp_correctness_mode,
                class_thresholds=thresholds_df,
            )
        except Exception as e:
            print(f"Error during filtering: {e}")
            traceback.print_exc()
            raise gr.Error(f"Error filtering data: {str(e)}")

        if filtered_df.empty:
            return gr.update(value=pd.DataFrame(columns=["Species", "Count", "Percentage"]), visible=True)

        try:
            counts_df = calculate_detection_counts(filtered_df)
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
        root = Path(directory).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(root)

        gather = root.rglob if recursive else root.glob

        files: list[Path] = []
        for pattern in patterns:
            files.extend(gather(pattern))

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

            proc_state.class_thresholds = updated_thresholds_df

            gr.Info(f"Successfully loaded thresholds for {loaded_count} classes from JSON.")
            return proc_state, gr.update(value=updated_thresholds_df)

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

        load_data_btn = gr.Button("Load Data", variant="huggingface", elem_id="viz_load_refresh_data_btn") # Changed variant to huggingface

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

        load_data_btn.click(
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
            ],
            outputs=[
                select_classes_checkboxgroup,
                processor_state,
                class_thresholds_df,
                threshold_json_select_btn,
                threshold_template_download_btn,
                classes_full_list_state,        
                recordings_full_list_state      
            ],
            show_progress=True
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
            outputs=[select_sites_checkboxgroup, sites_full_list_state]
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
                recordings_full_list_state,
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
                recordings_full_list_state,
                select_sites_checkboxgroup,
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
                recordings_full_list_state,
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
                recordings_full_list_state,
                select_sites_checkboxgroup,
                date_range_start,
                date_range_end,
                time_start_hour,
                time_start_minute,
                time_end_hour,
                time_end_minute,
                correctness_mode
            ],
            outputs=[processor_state, temporal_scatter_output]
        )

        calculate_detections_btn.click(
            fn=calculate_detections_action,
            inputs=[
                processor_state,
                select_classes_checkboxgroup,
                recordings_full_list_state,
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
