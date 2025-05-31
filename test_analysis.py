#!/usr/bin/env python3
"""
Test script for Heart Sound Analyzer

This script processes both normal and noisy heart sound recordings.
It handles the complete workflow from loading audio to generating visualizations.
"""
import sys
import logging
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional, Tuple

# Configure logging
def setup_logging():
    """Set up logging configuration."""
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler
    file_handler = logging.FileHandler('heart_sound_analysis.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Set log levels for specific loggers
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('numexpr').setLevel(logging.INFO)
    
    return logging.getLogger(__name__)

# Set up logging
logger = setup_logging()

def setup_directories() -> Dict[str, Path]:
    """
    Create necessary directories if they don't exist.
    
    Returns:
        Dictionary of directory paths
    """
    dirs = {
        'data': Path('data'),
        'raw': {
            'normal': Path('data/raw/normal'),
            'noisy': Path('data/raw/noisy')
        },
        'processed': Path('data/processed'),
        'results': {
            'normal': Path('data/results/normal'),
            'noisy': Path('data/results/noisy'),
            'comparison': Path('data/results/comparison')
        }
    }
    
    # Create all directories
    for dir_group in dirs.values():
        if isinstance(dir_group, dict):
            for path in dir_group.values():
                path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Directory ready: {path.absolute()}")
        else:
            dir_group.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory ready: {dir_group.absolute()}")
    
    return dirs

def get_audio_files(dirs: Dict) -> Dict[str, List[Path]]:
    """
    Get lists of normal and noisy audio files.
    
    Args:
        dirs: Dictionary of directory paths
        
    Returns:
        Dictionary with 'normal' and 'noisy' file lists
    """
    print(f"DEBUG: Starting get_audio_files with dirs={dirs}")
    result = {
        'normal': list(dirs['raw']['normal'].glob('*.wav')),
        'noisy': list(dirs['raw']['noisy'].glob('*.wav'))
    }
    print(f"DEBUG: Finished get_audio_files with result={result}")
    return result

def analyze_heart_sound(audio_path: Path, output_dir: Path, config: Optional[Dict] = None) -> Optional[Dict]:
    """Analyze a heart sound recording.
    
    Args:
        audio_path: Path to the audio file
        output_dir: Directory to save results
        config: Optional configuration dictionary
        
    Returns:
        Dictionary containing analysis results or None if analysis failed
    """
    print(f"DEBUG: Starting analyze_heart_sound with audio_path={audio_path}, output_dir={output_dir}")
    
    # Import here to avoid circular imports
    try:
        from src.processor import HeartSoundProcessor
        from src.visualization import HeartSoundVisualizer
        print("DEBUG: Successfully imported HeartSoundProcessor and HeartSoundVisualizer")
    except ImportError as e:
        print(f"DEBUG: Failed to import required modules: {e}")
        raise
    
    # Default configuration
    if config is None:
        config = {
            'audio': {
                'sample_rate': 4000,
                'normalize': True,
                'denoise': 'noisy' in str(audio_path)  # Auto-denoise noisy recordings
            },
            'segmentation': {
                'method': 'envelope',  # 'envelope' or 'hsmm'
                'min_heart_rate': 40,
                'max_heart_rate': 200,
                'hsmm_model_path': None
            },
            'output': {
                'save_plots': True,
                'save_audio': False
            }
        }
    
    # Initialize components
    processor = HeartSoundProcessor(config)
    visualizer = HeartSoundVisualizer()
    
    try:
        # Process the file
        logger.info(f"Processing file: {audio_path}")
        results = processor.process_file(str(audio_path))
        
        if not results or 'signal' not in results:
            logger.error(f"No results returned for {audio_path}")
            return None
            
        # Create output filename base
        output_base = output_dir / audio_path.stem
        
        # Generate visualizations if enabled
        if config['output']['save_plots']:
            logger.info("Generating visualizations...")
            
            # 1. Plot segmentation
            visualizer.plot_segmentation(
                signal=results['signal'].data,
                sample_rate=results['sample_rate'],
                segments=results['segments'],
                output_path=f"{output_base}_segmentation.png"
            )
            
            # 2. Plot envelope with peaks if available
            if 'envelope' in results['processed'].history:
                visualizer.plot_envelope(
                    signal=results['signal'].data,
                    envelope=results['processed'].data,
                    sample_rate=results['sample_rate'],
                    peaks=results['segments'],
                    output_path=f"{output_base}_envelope.png"
                )
            
            # 3. Plot individual heart cycles
            visualizer.plot_heart_cycles(
                signal=results['signal'].data,
                sample_rate=results['sample_rate'],
                segments=results['segments'],
                output_dir=f"{output_base}_cycles",
                max_plots=3  # Limit to first 3 cycles
            )
        
        # Create a timestamped subdirectory for this analysis
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_dir = output_dir / f"{audio_path.stem}_{timestamp}"
        
        logger.debug(f"Attempting to create directory: {analysis_dir.absolute()}")
        try:
            analysis_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Successfully created directory: {analysis_dir.absolute()}")
            
            # Save numerical results
            print(f"DEBUG: About to call save_segmentation_results with output_dir: {analysis_dir}")
            try:
                save_segmentation_results(results, analysis_dir)
                print(f"DEBUG: Successfully called save_segmentation_results")
            except Exception as e:
                print(f"DEBUG: Error in save_segmentation_results: {e}")
                raise
            
            # Create a symbolic link to the latest results for easy access
            latest_link = output_dir / f"{audio_path.stem}_latest"
            logger.debug(f"Creating/updating latest symlink: {latest_link} -> {analysis_dir.name}")
            
            try:
                if latest_link.exists() or latest_link.is_symlink():
                    latest_link.unlink()
                latest_link.symlink_to(analysis_dir.name, target_is_directory=True)
                logger.debug(f"Successfully created symlink: {latest_link} -> {analysis_dir.name}")
            except Exception as e:
                logger.warning(f"Could not create symlink: {e}")
            
            logger.info(f"Analysis complete! Results saved to {analysis_dir.absolute()}")
            if latest_link.exists():
                logger.info(f"Latest results symlink: {latest_link.absolute()}")
                
        except Exception as e:
            logger.error(f"Error creating output directory or saving results: {e}", exc_info=True)
            raise
        return results
        
    except Exception as e:
        logger.error(f"Error processing {audio_path}: {e}", exc_info=True)
        return None

def compare_recordings(normal_results: Dict, noisy_results: Dict, output_dir: Path) -> None:
    """
    Compare normal and noisy recordings.
    
    Args:
        normal_results: Results from normal recording
        noisy_results: Results from noisy recording
        output_dir: Directory to save comparison results
    """
    try:
        from src.visualization import HeartSoundVisualizer
        visualizer = HeartSoundVisualizer()
        
        # Generate comparison plots here
        # Example: SNR comparison, feature differences, etc.
        logger.info("Generating comparison plots...")
        
        # Placeholder for comparison logic
        comparison_metrics = {
            'normal_s1_count': len(normal_results['segments'].get('s1_starts', [])),
            'noisy_s1_count': len(noisy_results['segments'].get('s1_starts', [])),
            'normal_s2_count': len(normal_results['segments'].get('s2_starts', [])),
            'noisy_s2_count': len(noisy_results['segments'].get('s2_starts', [])),
        }
        
        # Save comparison metrics
        import json
        with open(output_dir / 'comparison_metrics.json', 'w') as f:
            json.dump(comparison_metrics, f, indent=2)
            
        logger.info(f"Comparison complete! Results saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error in comparison: {e}", exc_info=True)

def save_segmentation_results(results: Dict, output_path: Path) -> None:
    """Save segmentation results to disk.
    
    Args:
        results: Dictionary containing analysis results
        output_path: Directory to save results
    """
    try:
        import json
        import numpy as np
        import pandas as pd
        from pathlib import Path
        
        logger.debug(f"Starting to save results to: {output_path.absolute()}")
        
        # Create output directory if it doesn't exist
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Successfully created output directory: {output_path.absolute()}")
        except Exception as e:
            logger.error(f"Failed to create output directory {output_path.absolute()}: {e}")
            raise
        
        # Save configuration
        config_path = output_path / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(results.get('config', {}), f, indent=2)
        logger.info(f"Saved configuration to {config_path}")
        
        # Save segmentation results
        segments = results.get('segments', {})
        if segments:
            # Convert numpy arrays to lists for JSON serialization
            segments_serializable = {
                k: v.tolist() if isinstance(v, np.ndarray) else v 
                for k, v in segments.items()
            }
            
            # Save as JSON
            segments_json_path = output_path / 'segments.json'
            with open(segments_json_path, 'w') as f:
                json.dump(segments_serializable, f, indent=2)
            logger.info(f"Saved segmentation results to {segments_json_path}")
            
            # Save as CSV if we have valid segments
            if all(k in segments for k in ['s1_starts', 's1_ends', 's2_starts', 's2_ends']):
                # Create a DataFrame for S1 sounds
                s1_data = {
                    'type': ['S1'] * len(segments['s1_starts']),
                    'start_sample': segments['s1_starts'],
                    'end_sample': segments['s1_ends'],
                    'duration_samples': segments['s1_ends'] - segments['s1_starts'],
                    'start_time': segments['s1_starts'] / results['sample_rate'],
                    'end_time': segments['s1_ends'] / results['sample_rate'],
                    'duration_seconds': (segments['s1_ends'] - segments['s1_starts']) / results['sample_rate']
                }
                
                # Create a DataFrame for S2 sounds
                s2_data = {
                    'type': ['S2'] * len(segments['s2_starts']),
                    'start_sample': segments['s2_starts'],
                    'end_sample': segments['s2_ends'],
                    'duration_samples': segments['s2_ends'] - segments['s2_starts'],
                    'start_time': segments['s2_starts'] / results['sample_rate'],
                    'end_time': segments['s2_ends'] / results['sample_rate'],
                    'duration_seconds': (segments['s2_ends'] - segments['s2_starts']) / results['sample_rate']
                }
                
                # Combine and sort by start time
                df = pd.concat([
                    pd.DataFrame(s1_data),
                    pd.DataFrame(s2_data)
                ]).sort_values('start_time').reset_index(drop=True)
                
                # Save to CSV
                csv_path = output_path / 'segments.csv'
                df.to_csv(csv_path, index=False)
                logger.info(f"Saved segmentation data to {csv_path}")
        
        # Save signal metadata
        signal_meta = {
            'file_path': str(results.get('file_path', '')),
            'sample_rate': results.get('sample_rate', 0),
            'duration_samples': len(results.get('signal', {}).get('data', [])),
            'duration_seconds': len(results.get('signal', {}).get('data', [])) / results.get('sample_rate', 1)
        }
        with open(output_path / 'signal_metadata.json', 'w') as f:
            json.dump(signal_meta, f, indent=2)
        logger.info(f"Saved signal metadata to {output_path}/signal_metadata.json")
        
    except Exception as e:
        logger.error(f"Error saving results: {e}", exc_info=True)

def main():
    """Main function to run the analysis."""
    # Setup directory structure
    dirs = setup_directories()
    
    # Get audio files
    audio_files = get_audio_files(dirs)
    
    # Check if we have files to process
    if not any(audio_files.values()):
        logger.warning("No WAV files found in data/raw/normal/ or data/raw/noisy/")
        logger.info("Please place your WAV files in the appropriate directories")
        return
    
    # Process normal recordings
    normal_results = {}
    for audio_file in audio_files['normal']:
        try:
            logger.info(f"\n{'='*50}")
            logger.info(f"Analyzing normal recording: {audio_file.name}")
            logger.info("="*50)
            
            result = analyze_heart_sound(
                audio_file,
                dirs['results']['normal']
            )
            
            if result:
                normal_results[audio_file.stem] = result
                
        except Exception as e:
            logger.error(f"Error processing {audio_file}: {e}", exc_info=True)
    
    # Process noisy recordings
    noisy_results = {}
    for audio_file in audio_files['noisy']:
        try:
            logger.info(f"\n{'='*50}")
            logger.info(f"Analyzing noisy recording: {audio_file.name}")
            logger.info("="*50)
            
            result = analyze_heart_sound(
                audio_file,
                dirs['results']['noisy']
            )
            
            if result:
                noisy_results[audio_file.stem] = result
                
                # If we have a matching normal recording, run comparison
                if audio_file.stem in normal_results:
                    compare_dir = dirs['results']['comparison'] / audio_file.stem
                    compare_dir.mkdir(parents=True, exist_ok=True)
                    compare_recordings(
                        normal_results[audio_file.stem],
                        result,
                        compare_dir
                    )
                    
        except Exception as e:
            logger.error(f"Error processing {audio_file}: {e}", exc_info=True)
    
    logger.info("\nAnalysis complete!")
    logger.info(f"Processed {len(normal_results)} normal and {len(noisy_results)} noisy recordings")
    logger.info(f"Results saved to: {dirs['results']['normal']} and {dirs['results']['noisy']}")

if __name__ == "__main__":
    main()
