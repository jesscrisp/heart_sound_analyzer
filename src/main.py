"""Main entry point for the Heart Sound Analyzer application."""

# Standard library imports
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Third-party imports
from dotenv import load_dotenv
import numpy as np
import pandas as pd

# Local application imports
from .config import load_config
from .processor import HeartSoundProcessor
from .io import AudioLoader # AudioLoader has the save_audio method
from .visualization import HeartSoundVisualizer
from .reporting import ResultsReporter # For saving segmentation results to CSV

# Configure logging as early as possible
logger = logging.getLogger(__name__) # Define module-level logger for main.py

# Prevent basicConfig from disabling other loggers that might have been configured by libraries
logging.disable_existing_loggers = False
# Set global logging level. Child loggers will inherit this unless overridden.
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(filename)s:%(lineno)d:%(message)s')


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze heart sound recordings.')
    parser.add_argument('input_file', type=str, nargs='?', default=None, help='Path to input WAV file. If not provided, tries to load from .env file.')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML configuration file.')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results (plots, processed audio).')
    parser.add_argument('--segmentation-method', type=str, default=None,
                        choices=['peak_detection'], # Removed 'lr_hsmm'
                        help='Segmentation method to use. Overrides config if set.')
    return parser.parse_args()


def _generate_and_save_html_report(
    results: dict,
    visualizer: HeartSoundVisualizer,
    sample_rate: Optional[float],
    output_dir: Path,
    base_filename: str,
    original_input_path: str
) -> None:
    """Generates an HTML report with visualizations of processing stages."""
    if not sample_rate:
        logger.warning("Sample rate not available, cannot generate HTML report with plots.")
        return

    report_parts = []
    report_parts.append(f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <title>Heart Sound Analysis Report - {base_filename}</title>
    <style>
        body {{ font-family: sans-serif; margin: 20px; background-color: #f4f4f4; color: #333; }}
        .container {{ max-width: 900px; margin: auto; background: white; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        .plot-container {{ margin-bottom: 30px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; background-color: #fff; }}
        h1, h2 {{ color: #333; }}
        h1 {{ text-align: center; border-bottom: 2px solid #007bff; padding-bottom: 10px; margin-bottom:20px; }}
        h2 {{ border-bottom: 1px solid #eee; padding-bottom: 5px; margin-top:0; }}
        img {{ max-width: 100%; height: auto; display: block; margin-left: auto; margin-right: auto; border-radius: 4px; }}
        p {{ line-height: 1.6; }}
        .error-message {{ color: red; font-weight: bold; border: 1px solid red; padding: 10px; border-radius: 4px; }}
    </style>
</head>
<body>
    <div class=\"container\">
    <h1>Heart Sound Analysis Report</h1>
    <p><strong>Input File:</strong> {original_input_path}</p>
""")

    signals_to_plot = [
        ('raw_audio_data', 'Raw Audio Data (Initial Load)'),
        ('raw_for_preprocessing', 'Signal Sent to pyPCG Preprocessing'),
        ('after_highpass', 'Signal After High-pass Filter'),
        ('after_lowpass', 'Signal After Low-pass Filter'),
        ('after_denoise', 'Signal After Denoising'),
        ('processed_audio_data', 'Final Processed Signal (After Normalization)'),
        ('envelope', 'Extracted Envelope')
    ]

    for key, title in signals_to_plot:
        signal_data = results.get(key)
        report_parts.append('<div class="plot-container">')
        report_parts.append(f'<h2>{title}</h2>')
        if signal_data is not None:
            # For envelope, sample rate might be different or effectively 1 if it's just points.
            # However, plot_signal_to_html_embeddable expects a sample_rate to compute time axis.
            # We'll use the main signal's sample_rate; this is okay if envelope has same length or is for visualization.
            plot_html = visualizer.plot_signal_to_html_embeddable(signal_data, sample_rate, title)
            report_parts.append(plot_html)
        else:
            report_parts.append(f'<p class="error-message">Data for \"{title}\" not available in results.</p>')
        report_parts.append('</div>')

    report_parts.append("""
    </div>
</body>
</html>
""")

    html_content = "\n".join(report_parts)
    report_file_path = output_dir / f"{base_filename}_visualization_report.html"
    try:
        with open(report_file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logger.info(f"HTML visualization report saved to: {report_file_path}")
    except IOError as e:
        logger.error(f"Failed to save HTML report to {report_file_path}: {e}")

def main() -> None:
    """Main function to run the heart sound analysis."""
    load_dotenv(override=True) # Load environment variables from .env file, overriding existing ones
    PROJECT_ROOT = Path(__file__).resolve().parent.parent # Get to PyPCG_first_attempt
    args = parse_arguments()

    input_file_path = args.input_file
    if input_file_path is None:
        env_input_file = os.getenv('DEFAULT_INPUT_FILE')
        if env_input_file:
            # Check if the path from .env is absolute or relative
            potential_path = Path(env_input_file)
            if potential_path.is_absolute():
                input_file_path = str(potential_path)
                logger.info(f"Input file not provided via CLI, using absolute DEFAULT_INPUT_FILE from .env: {input_file_path}")
            else:
                input_file_path = str(PROJECT_ROOT / env_input_file)
                logger.info(f"Input file not provided via CLI, using relative DEFAULT_INPUT_FILE from .env, resolved to: {input_file_path}")
        else:
            logger.error("Error: Input file not provided via CLI and DEFAULT_INPUT_FILE not set in .env. Please provide an input file.")
            sys.exit(1)
    
    # Update args.input_file to ensure it's correctly populated for the rest of the script
    # This is a bit redundant if we use input_file_path throughout, but safer if other parts of args are passed around.
    args.input_file = input_file_path
    
    try:
        # Load configuration. load_config provides a dictionary with a default structure.
        config = load_config(args.config)
        logger.debug(f"Initial configuration loaded: {config}")

        # Override segmentation method from command line if provided.
        # Assumes `config` has a 'segmentation' key from `load_config`.
        if args.segmentation_method:
            # Ensure 'segmentation' key exists (should be guaranteed by load_config's defaults)
            if 'segmentation' not in config:
                config['segmentation'] = {} # Fallback, though load_config should prevent this
            config['segmentation']['method'] = args.segmentation_method
            logger.info(f"Overridden segmentation method from command line: {config['segmentation']['method']}")

        # Initialize components
        processor = HeartSoundProcessor(config)
        audio_loader = AudioLoader(config) # Used for saving audio
        visualizer = HeartSoundVisualizer() # Visualizer might take config for styling in future

        logger.info(f"Starting processing for file: {input_file_path}")
        results = processor.process_file(input_file_path)
        logger.debug(f"Processing results: {results}")

        # Determine if plots should be shown interactively from config
        # Default to False if 'output' or 'show_plots' keys are missing
        show_plots_config = config.get('output', {}).get('show_plots', False)

        output_dir_path: Optional[Path] = None
        print(f"[MAIN.PY-DEBUG] Initializing output_dir_path. CLI args.output_dir: {args.output_dir}") # CASCADE DEBUG

        if args.output_dir:
            output_dir_path = Path(args.output_dir).resolve()
            print(f"[MAIN.PY-DEBUG] Using CLI --output-dir: {args.output_dir}, resolved to: {output_dir_path}") # CASCADE DEBUG
        else:
            config_output_settings = config.get('output', {})
            default_dir_name = config_output_settings.get('default_output_dir')
            print(f"[MAIN.PY-DEBUG] Config 'output' settings: {config_output_settings}") # CASCADE DEBUG
            print(f"[MAIN.PY-DEBUG] Config 'default_output_dir': {default_dir_name}") # CASCADE DEBUG
            if default_dir_name:
                output_dir_path = (PROJECT_ROOT / default_dir_name).resolve()
                print(f"[MAIN.PY-DEBUG] Using config default_output_dir: {default_dir_name}, resolved to: {output_dir_path}") # CASCADE DEBUG
            # If no CLI arg and no default_dir_name, output_dir_path remains None

        # Attempt to create the output directory if a path has been determined
        if output_dir_path:
            try:
                output_dir_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Output directory confirmed/created: {output_dir_path}")
                print(f"[MAIN.PY-DEBUG] Output directory confirmed/created: {output_dir_path}") # CASCADE DEBUG
            except OSError as e:
                logger.error(f"Error creating output directory {output_dir_path}: {e}. No files will be saved.")
                output_dir_path = None # Nullify path to prevent save attempts

        # Proceed with saving files only if output_dir_path is valid and directory creation succeeded
        if output_dir_path:
            output_file_stem = Path(input_file_path).stem
            output_base_path = output_dir_path / output_file_stem

            # Save processed audio
            processed_audio_to_save = results.get('processed_audio_data', results.get('raw_audio_data'))
            sample_rate = results.get('sample_rate')
            if processed_audio_to_save is not None and sample_rate is not None:
                save_audio_path = output_base_path.with_name(output_base_path.name + "_processed.wav")
                try:
                    audio_loader.save_audio(processed_audio_to_save, sample_rate, str(save_audio_path))
                    logger.info(f"Processed audio saved to: {save_audio_path}")
                except Exception as e:
                    logger.error(f"Error saving processed audio to {save_audio_path}: {e}")
            else:
                logger.warning("Processed audio or sample rate not available. Skipping saving processed audio.")

            # Save segmentation plot if configured
            if config.get('output', {}).get('save_plots', True):
                signal_for_plot = results.get('processed_audio_data', results.get('raw_audio_data'))
                segments = results.get('segments')
                envelope_for_plot = results.get('envelope') # Get envelope for plotting
                if signal_for_plot is not None and segments is not None and sample_rate is not None:
                    plot_format = config.get('output', {}).get('plot_format', 'png')
                    plot_save_path = output_base_path.with_name(output_base_path.name + f"_segmentation.{plot_format}")
                    try:
                        visualizer.plot_segmentation(
                            signal_for_plot,
                            sample_rate,
                            segments, # Envelope data is expected within the 'segments' dict
                            output_path=str(plot_save_path),
                            show=False # Do not show when saving to file
                        )
                        logger.info(f"Segmentation plot saved to: {plot_save_path}")
                    except Exception as e:
                        logger.error(f"Error saving segmentation plot to {plot_save_path}: {e}")
                else:
                    logger.warning("Signal, segments, or sample rate not available. Skipping saving segmentation plot.")
            else:
                logger.info("Configuration 'save_plots' is False. Skipping saving segmentation plot.")

            # Save segmentation results to CSV if configured
            if config.get('output', {}).get('save_csv_results', True):
                segmentation_data_for_csv = results.get('segments') # Correct key from processor
                logger.debug(f"[MAIN.PY-DEBUG] Attempting to save CSV. segmentation_data_for_csv available: {segmentation_data_for_csv is not None}")

                if segmentation_data_for_csv and isinstance(segmentation_data_for_csv, dict):
                    logger.debug("[MAIN.PY-DEBUG] Detailed types in segmentation_data_for_csv:")
                    for key, value in segmentation_data_for_csv.items():
                        logger.debug(f"  Key: '{key}', Value Type: {type(value)}")
                        if isinstance(value, np.ndarray):
                            logger.debug(f"    Value Shape: {value.shape}, Value Dtype: {value.dtype}")
                        elif isinstance(value, list) and value: # Check if list is not empty
                            logger.debug(f"    List item 0 Type: {type(value[0]) if value else 'N/A'}")
                    
                    sample_rate_csv = results.get('sample_rate')
                    if sample_rate_csv is not None:
                        try:
                            s1_segments_data = segmentation_data_for_csv.get('s1_segments')
                            s2_segments_data = segmentation_data_for_csv.get('s2_segments')
                            
                            # Ensure segments are numpy arrays for consistent processing
                            if s1_segments_data is not None and not isinstance(s1_segments_data, np.ndarray):
                                s1_segments_data = np.array(s1_segments_data)
                            if s2_segments_data is not None and not isinstance(s2_segments_data, np.ndarray):
                                s2_segments_data = np.array(s2_segments_data)

                            valid_s1 = s1_segments_data is not None and isinstance(s1_segments_data, np.ndarray) and s1_segments_data.ndim == 2 and s1_segments_data.shape[1] == 2
                            valid_s2 = s2_segments_data is not None and isinstance(s2_segments_data, np.ndarray) and s2_segments_data.ndim == 2 and s2_segments_data.shape[1] == 2

                            max_len = 0
                            if valid_s1: max_len = max(max_len, len(s1_segments_data))
                            if valid_s2: max_len = max(max_len, len(s2_segments_data))
                            
                            data_for_csv_output = {}
                            # S1 data handling
                            if valid_s1:
                                data_for_csv_output['S1_start_seconds'] = [seg[0] / sample_rate_csv for seg in s1_segments_data] + [float('nan')] * (max_len - len(s1_segments_data))
                                data_for_csv_output['S1_end_seconds'] = [seg[1] / sample_rate_csv for seg in s1_segments_data] + [float('nan')] * (max_len - len(s1_segments_data))
                            else:
                                data_for_csv_output['S1_start_seconds'] = [float('nan')] * max_len if max_len > 0 else []
                                data_for_csv_output['S1_end_seconds'] = [float('nan')] * max_len if max_len > 0 else []
                                if s1_segments_data is not None: # Log if data was present but not valid
                                    logger.warning(f"S1 segments data not in expected format (2D numpy array) for CSV. Shape: {s1_segments_data.shape if isinstance(s1_segments_data, np.ndarray) else type(s1_segments_data)}. Skipping S1 data for CSV.")

                            # S2 data handling
                            if valid_s2:
                                data_for_csv_output['S2_start_seconds'] = [seg[0] / sample_rate_csv for seg in s2_segments_data] + [float('nan')] * (max_len - len(s2_segments_data))
                                data_for_csv_output['S2_end_seconds'] = [seg[1] / sample_rate_csv for seg in s2_segments_data] + [float('nan')] * (max_len - len(s2_segments_data))
                            else:
                                data_for_csv_output['S2_start_seconds'] = [float('nan')] * max_len if max_len > 0 else []
                                data_for_csv_output['S2_end_seconds'] = [float('nan')] * max_len if max_len > 0 else []
                                if s2_segments_data is not None: # Log if data was present but not valid
                                    logger.warning(f"S2 segments data not in expected format (2D numpy array) for CSV. Shape: {s2_segments_data.shape if isinstance(s2_segments_data, np.ndarray) else type(s2_segments_data)}. Skipping S2 data for CSV.")
                            
                            if data_for_csv_output and max_len > 0:
                                df = pd.DataFrame(data_for_csv_output)
                                csv_save_path = output_base_path.with_name(output_base_path.name + "_segmentation_results.csv")
                                df.to_csv(csv_save_path, index=False)
                                logger.info(f"Segmentation results CSV saved to: {csv_save_path}")
                            elif max_len == 0:
                                logger.info("No S1 or S2 segments found. CSV file will not be created.")
                            else:
                                logger.info("No valid S1 or S2 segment data to save to CSV, or max_len was 0.")
                        except ImportError:
                            logger.warning("Pandas library not found. Cannot save segmentation results to CSV. Please install pandas.")
                        except Exception as e:
                            logger.error(f"Error saving segmentation results to CSV: {e}", exc_info=True)
                    elif sample_rate_csv is None:
                        logger.warning("Sample rate not available. Skipping saving CSV results.")
                else: # segmentation_data_for_csv is None or not a dict
                    logger.warning("Segmentation results for CSV are not available or not in dictionary format. Skipping saving CSV.")
            else: # save_csv_results is False
                logger.info("Configuration 'save_csv_results' is False. Skipping saving CSV.")

            # Generate and save HTML report if configured
            if config.get('output', {}).get('save_html_report', True):
                required_keys_for_html = ['raw_audio_data', 'processed_audio_data', 'envelope', 'segments', 'sample_rate']
                # Check if all necessary data is available in results
                if all(results.get(key) is not None for key in required_keys_for_html):
                    # Ensure 'segments' itself is a dictionary as expected
                    if not isinstance(results.get('segments'), dict):
                        logger.warning(f"HTML Report: 'segments' data is not a dictionary (type: {type(results.get('segments'))}). Skipping HTML report.")
                    else:
                        html_report_path = output_base_path.with_name(output_base_path.name + "_report.html")
                        try:
                            logger.info(f"Attempting to generate HTML report to: {html_report_path}")
                            visualizer.generate_html_report(
                                raw_audio=results['raw_audio_data'],
                                processed_audio=results['processed_audio_data'],
                                envelope=results['envelope'],
                                segments=results['segments'], # This is the dict from segmentation_results
                                sample_rate=results['sample_rate'],
                                output_html_path=str(html_report_path),
                                config=config # Pass the full config if needed by the report
                            )
                            logger.info(f"HTML report saved to: {html_report_path}")
                        except AttributeError:
                            logger.error(f"HTML report generation failed: 'generate_html_report' method not found in HeartSoundVisualizer. Please implement it.", exc_info=True)
                        except Exception as e:
                            logger.error(f"Error generating HTML report to {html_report_path}: {e}", exc_info=True)
                else:
                    missing_keys = [key for key in required_keys_for_html if results.get(key) is None]
                    logger.warning(f"One or more required data items for HTML report are missing ({missing_keys}). Skipping HTML report generation.")
            else:
                logger.info("Configuration 'save_html_report' is False or not set. Skipping HTML report.")
        
        elif not args.output_dir and not config.get('output', {}).get('default_output_dir'):
             logger.info("No output directory specified via CLI or in config default_output_dir. No files will be saved.")

        # Handle interactive plot display if configured, irrespective of saving files
        if show_plots_config:
            logger.info("Configuration 'show_plots' is True. Attempting to display segmentation plot interactively.")
            signal_for_plot = results.get('processed_audio_data', results.get('raw_audio_data'))
            segments = results.get('segments')
            sample_rate = results.get('sample_rate')
            envelope_for_plot = results.get('envelope') # Get envelope for plotting

            if signal_for_plot is not None and segments is not None and sample_rate is not None:
                try:
                    visualizer.plot_segmentation(
                        signal_for_plot,
                        sample_rate,
                        segments, # Envelope data is expected within the 'segments' dict
                        output_path=None, # No save path for interactive display
                        show=True # Force show
                    )
                except Exception as e:
                    logger.error(f"Error displaying segmentation plot interactively: {e}")
            else:
                logger.warning("Signal, segments, or sample rate not available for interactive plot. Skipping display.")
        elif not output_dir_path: # If no files were saved and not showing plots
            logger.info("No output directory specified and 'show_plots' is False. No output generated.")

        logger.info(f"Successfully processed '{input_file_path}'. Status: {results.get('status', 'Status not available')}")

    except FileNotFoundError:
        logger.error(f"Error: Input file not found at '{input_file_path}'. Please check the path.")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"An unexpected error occurred during processing of {input_file_path}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Setup basic logging configuration if no handlers are configured yet
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    main()
