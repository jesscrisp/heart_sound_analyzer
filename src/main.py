"""Main entry point for the Heart Sound Analyzer application."""
import argparse
from pathlib import Path
from typing import Optional

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
    return parser.parse_args()

def main():
    """Main function to run the heart sound analysis."""
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    
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
