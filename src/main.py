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
        if args.output_dir:
            output_dir_path = Path(args.output_dir)
            output_dir_path.mkdir(parents=True, exist_ok=True) # Create output directory
            logger.info(f"Output will be saved to: {output_dir_path.resolve()}")
            
            output_file_stem = Path(input_file_path).stem
            output_base_path = output_dir_path / output_file_stem

            # Save processed audio
            processed_audio_to_save = results.get('processed_audio_data', results.get('raw_audio_data'))
            sample_rate = results.get('sample_rate')

            if processed_audio_to_save is not None and sample_rate is not None:
                save_audio_path = output_base_path.with_name(output_base_path.name + "_processed.wav")
                audio_loader.save_audio(processed_audio_to_save, sample_rate, str(save_audio_path))
                logger.info(f"Processed audio saved to: {save_audio_path}")
            else:
                logger.warning("Processed audio or sample rate not available in results. Skipping saving processed audio.")

            # Save segmentation plot
            signal_for_plot = results.get('processed_audio_data', results.get('raw_audio_data'))
            segments = results.get('segments')
            
            if signal_for_plot is not None and segments and sample_rate is not None:
                plot_format = config.get('output', {}).get('plot_format', 'png')
                plot_save_path = output_base_path.with_name(output_base_path.name + f"_segmentation.{plot_format}")
                visualizer.plot_segmentation(
                    signal_for_plot,
                    sample_rate,
                    segments,
                    output_path=str(plot_save_path),
                    show=show_plots_config # Show plot if configured, even if saving
                )
                logger.info(f"Segmentation plot saved to: {plot_save_path}")

                # Save segmentation results to CSV
                try:
                    ResultsReporter.save_segmentation_results_csv(
                        results,
                        str(output_dir_path),
                        base_name=output_file_stem
                    )
                    # The method logs its own success/failure
                except Exception as e_report:
                    logger.error(f"Failed to save segmentation CSV report for {output_file_stem}: {e_report}")

            else:
                logger.warning("Signal for plotting, segments, or sample rate not available. Skipping saving segmentation plot and CSV report.")
        
        # If not saving to a directory, but config says to show plots
        elif show_plots_config:
            logger.info("No output directory specified, but 'show_plots' is true in config. Displaying plot.")
            signal_for_plot = results.get('processed_audio_data', results.get('raw_audio_data'))
            segments = results.get('segments')
            sample_rate = results.get('sample_rate')

            if signal_for_plot is not None and segments and sample_rate is not None:
                visualizer.plot_segmentation(
                    signal_for_plot,
                    sample_rate,
                    segments,
                    envelope_data=results.get('envelope'),
                    output_path=None, # No path to save, just show
                    show=True # Force show since this block is for showing only
                )
            else:
                logger.warning("Signal for plotting, segments, or sample rate not available. Cannot show plot.")
        
        # Log final status
        status_msg = results.get('status_message', 'Processing completed.')
        # Check all relevant success flags from results
        all_successful = all([
            results.get('processing_pipeline_successful', False),
            results.get('envelope_extraction_successful', False),
            results.get('segmentation_successful', False)
        ])

        if all_successful:
            logger.info(f"Successfully processed '{input_file_path}'. Status: {status_msg}")
        else:
            logger.error(f"Processing '{input_file_path}' encountered issues. Status: {status_msg}")

    except FileNotFoundError:
        logger.error(f"Error: Input file not found at '{input_file_path}'. Please check the path.")
    except Exception as e:
        logger.critical(f"An critical unexpected error occurred in main: {e}", exc_info=True)
        # import sys
        # sys.exit(1) # Consider exiting with a non-zero status code for scriptability

if __name__ == "__main__":
    main()
