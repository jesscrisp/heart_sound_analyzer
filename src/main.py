"""Main entry point for the Heart Sound Analyzer application."""
import argparse
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__) # Define module-level logger for main.py

# Configure logging as early as possible
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s') # Set global to DEBUG for testing
logging.getLogger('heart_sound_analyzer.src.io').setLevel(logging.INFO)
logging.getLogger('heart_sound_analyzer.src.segmentation').setLevel(logging.DEBUG) # Explicitly set segmentation to DEBUG


from .config import load_config
from .processor import HeartSoundProcessor
from .io import AudioLoader
from .visualization import HeartSoundVisualizer

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze heart sound recordings.')
    parser.add_argument('input_file', type=str, help='Path to input WAV file')
    parser.add_argument('--config', type=str, default=None, 
                        help='Path to config file')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results')
    parser.add_argument('--segmentation-method', type=str, default=None,
                        choices=['peak_detection', 'lr_hsmm'],
                        help='Segmentation method to use (peak_detection or lr_hsmm). Overrides config if set.')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to the LR-HSMM model file. Used if segmentation_method is lr_hsmm. Overrides config if set.')
    return parser.parse_args()

def main():
    """Main function to run the heart sound analysis."""
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)

    # Ensure segmentation_config and params exist
    if 'segmentation_config' not in config:
        config['segmentation_config'] = {}
    if 'params' not in config['segmentation_config']:
        config['segmentation_config']['params'] = {}

    # Override segmentation method from command line if provided
    if args.segmentation_method:
        config['segmentation_config']['method'] = args.segmentation_method
        logger.info(f"Using segmentation method from command line: {args.segmentation_method}")

    # Override model path from command line if provided and method is lr_hsmm
    # Ensure we check the effective method (either from config or overridden by CLI)
    effective_segmentation_method = config['segmentation_config'].get('method')
    if args.model_path and effective_segmentation_method == 'lr_hsmm':
        config['segmentation_config']['params']['model_path'] = args.model_path
        logger.info(f"Using LR-HSMM model path from command line: {args.model_path}")
    elif args.model_path and effective_segmentation_method != 'lr_hsmm':
        logger.warning(f"--model-path ('{args.model_path}') provided, but segmentation method is not 'lr_hsmm'. It will be ignored.")
    
    # Initialize components
    processor = HeartSoundProcessor(config)
    audio_loader = AudioLoader(config)
    visualizer = HeartSoundVisualizer()
    
    # Process the file
    results = processor.process_file(args.input_file)
    
    # Save and visualize results
    if args.output_dir:
        output_path = Path(args.output_dir) / Path(args.input_file).stem
        
        # Save processed audio
        if 'processed' in results and 'sample_rate' in results:
            audio_loader.save_audio(
                results['processed'],
                results['sample_rate'],
                f"{output_path}_processed.wav"
            )
        
        # Save segmentation results
        if 'segments' in results:
            visualizer.plot_segmentation(
                results.get('processed', results.get('signal')),
                results['sample_rate'],
                results['segments'],
                output_path=f"{output_path}_segmentation.png",
                show=True  # Show the plot interactively
            )

if __name__ == "__main__":
    main()
