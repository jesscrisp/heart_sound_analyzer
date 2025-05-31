"""
Debug script for visualizing heart sound segmentation.

This script processes a heart sound audio file, performs segmentation,
and generates visualizations of the results.
"""
import sys
import logging
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our modules
from src.processor import HeartSoundProcessor
from src.visualization import HeartSoundVisualizer

def main(audio_path: str, output_dir: str = None, method: str = 'peak_detection'):
    """Run segmentation and visualize results.
    
    Args:
        audio_path: Path to the audio file to analyze
        output_dir: Directory to save debug outputs (defaults to tests/debug_output)
        method: Segmentation method ('peak_detection' or 'lr_hsmm')
    """
    # Set up output directory
    if output_dir is None:
        output_dir = project_root / "tests" / "debug_output"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configuration for heart sound processing
    config = {
        'audio': {
            'sample_rate': 1000  # Target sample rate for resampling
        },
        'preprocessing': {
            'lowcut': 25,       # Lower frequency cutoff (Hz)
            'highcut': 400,     # Upper frequency cutoff (Hz)
            'normalize': True,  # Enable signal normalization
        },
        'segmentation': {
            'method': method,
            'params': {
                'min_peak_distance': 0.2,  # Minimum distance between peaks (in seconds)
                'peak_prominence': 0.1,    # Minimum prominence of peaks (as fraction of max)
                'fs': 1000                 # Sampling frequency
            }
        }
    }
    
    try:
        # Initialize processor
        processor = HeartSoundProcessor(config)
        logger.info(f"Processing file: {audio_path}")
        
        # Process the file
        results = processor.process_file(audio_path)
        
        # Get the processed signal
        signal_obj = results['signal_object']
        processed_obj = results['processed_object']
        
        # Extract signal data
        signal = signal_obj.signal if hasattr(signal_obj, 'signal') else signal_obj
        fs = signal_obj.fs if hasattr(signal_obj, 'fs') else 1000
        
        # Perform segmentation directly
        segments = segment_heart_sounds(
            signal=signal,
            fs=fs,
            method=method,
            **config['segmentation']['params']
        )
        
        # Generate output filename
        audio_name = Path(audio_path).stem
        output_path = output_dir / f"{audio_name}_{method}_analysis.png"
        
        # Create visualization
        plt.figure(figsize=(15, 8))
        
        # Plot the signal and envelope
        time = np.arange(len(signal)) / fs
        plt.plot(time, signal, 'b-', alpha=0.5, label='Signal')
        
        # Plot envelope if available
        if 'envelope' in segments:
            plt.plot(time, segments['envelope'], 'g-', alpha=0.7, label='Envelope')
        
        # Plot S1 segments
        s1_starts = segments.get('s1_starts', [])
        s1_ends = segments.get('s1_ends', [])
        if s1_starts and s1_ends:
            for i, (start, end) in enumerate(zip(s1_starts, s1_ends)):
                plt.axvspan(
                    start/fs, 
                    end/fs, 
                    color='r', 
                    alpha=0.2, 
                    label='S1' if i == 0 else ''
                )
        
        # Plot S2 segments
        s2_starts = segments.get('s2_starts', [])
        s2_ends = segments.get('s2_ends', [])
        if s2_starts and s2_ends:
            for i, (start, end) in enumerate(zip(s2_starts, s2_ends)):
                plt.axvspan(
                    start/fs, 
                    end/fs, 
                    color='g', 
                    alpha=0.2, 
                    label='S2' if i == 0 else ''
                )
        
        plt.title(f'Heart Sound Segmentation - {audio_name}')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        
        # Save the figure with high DPI
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved to {output_path}")
        logger.info(f"Found {len(s1_starts)} S1 segments")
        logger.info(f"Found {len(s2_starts)} S2 segments")
        
        if show:
            plt.show()
        plt.close()
        
        # Print summary of results
        segments = results['segments']
        print("\n=== Segmentation Results ===")
        print(f"File: {audio_path}")
        print(f"Method: {segments.get('method', 'unknown')}")
        print(f"Sample rate: {results['sample_rate']} Hz")
        print(f"Duration: {len(results['signal'])/results['sample_rate']:.2f} seconds")
        print(f"S1 sounds detected: {len(segments['s1_starts'])}")
        print(f"S2 sounds detected: {len(segments['s2_starts'])}")
        
        # Print first few segment times if available
        if len(segments['s1_starts']) > 0:
            print("\nFirst few S1 segments (seconds):")
            for i in range(min(3, len(segments['s1_starts']))):
                start = segments['s1_starts'][i] / results['sample_rate']
                end = segments['s1_ends'][i] / results['sample_rate']
                print(f"  S1 {i+1}: {start:.3f}s - {end:.3f}s (duration: {(end-start)*1000:.1f}ms)")
        
        if len(segments['s2_starts']) > 0:
            print("\nFirst few S2 segments (seconds):")
            for i in range(min(3, len(segments['s2_starts']))):
                start = segments['s2_starts'][i] / results['sample_rate']
                end = segments['s2_ends'][i] / results['sample_rate']
                print(f"  S2 {i+1}: {start:.3f}s - {end:.3f}s (duration: {(end-start)*1000:.1f}ms)")
        
        logger.info(f"Analysis complete. Results saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error processing {audio_path}", exc_info=True)
        raise

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Analyze heart sound recordings.')
    parser.add_argument('audio_path', nargs='?', 
                      default=str(project_root / "data" / "raw" / "normal" / "125_1306332456645_B.wav"),
                      help='Path to the audio file to analyze')
    parser.add_argument('-o', '--output-dir', 
                      help='Directory to save output files (default: tests/debug_output)')
    parser.add_argument('-m', '--method', 
                      choices=['peak_detection', 'lr_hsmm'], 
                      default='peak_detection',
                      help='Segmentation method to use')
    parser.add_argument('-v', '--verbose', action='store_true',
                      help='Enable verbose logging')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run the analysis
    try:
        main(
            audio_path=args.audio_path,
            output_dir=args.output_dir,
            method=args.method
        )
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        sys.exit(1)
