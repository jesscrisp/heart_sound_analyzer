"""
Heart Sound Processor module for coordinating the heart sound analysis pipeline.
"""
import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

# Import internal modules
from .segmentation import segment_heart_sounds, create_heart_cycle_segments
from .io import AudioLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HeartSoundProcessor:
    """Coordinates the heart sound analysis pipeline using pyPCG."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the processor with configuration.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.audio_loader = AudioLoader(config)
        self.segmentation_config = config.get('segmentation', {})
    
    def _preprocess_signal(self, signal: np.ndarray, sample_rate: float) -> np.ndarray:
        """Apply preprocessing to the signal.
        
        Args:
            signal: Raw audio signal
            sample_rate: Sample rate in Hz
            
        Returns:
            Preprocessed signal
        """
        # Simple bandpass filtering
        from scipy import signal as sp_signal
        
        # Get filter parameters from config or use defaults
        lowcut = self.config.get('preprocessing', {}).get('lowcut', 25)
        highcut = self.config.get('preprocessing', {}).get('highcut', 400)
        fs = sample_rate
        
        # Design bandpass filter
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = sp_signal.butter(4, [low, high], btype='band')
        
        # Apply filter
        filtered = sp_signal.filtfilt(b, a, signal)
        
        # Normalize
        if self.config.get('preprocessing', {}).get('normalize', True):
            filtered = filtered / np.max(np.abs(filtered))
            
        return filtered
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """Process a heart sound audio file end-to-end.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Dictionary containing processing results with keys:
            - 'file_path': Path to the input file
            - 'signal': Raw audio signal data (numpy array)
            - 'processed': Preprocessed signal data (numpy array)
            - 'sample_rate': Sample rate of the audio
            - 'segments': Segmentation results
            - 'heart_cycles': List of heart cycles with timing information
        """
        try:
            logger.info(f"Processing file: {file_path}")
            
            # Load the audio file (returns tuple of (signal_data, sample_rate))
            signal_data, sample_rate = self.audio_loader.load_audio(file_path)
            
            # Preprocess the signal
            processed_data = self.audio_loader.preprocess(signal_data, sample_rate)
            
            # Perform segmentation
            seg_method = self.segmentation_config.get('method', 'peak_detection')
            seg_params = self.segmentation_config.get('params', {})
            seg_params['fs'] = sample_rate  # Ensure sample rate is included
            
            segments = segment_heart_sounds(
                processed_data,  # Pass the processed signal data
                method=seg_method,
                **seg_params  # Includes sample_rate as 'fs'
            )
            
            # Create heart cycles from segments
            heart_cycles = create_heart_cycle_segments(segments)
            
            return {
                'file_path': file_path,
                'signal': signal_data,  # Raw signal as numpy array
                'processed': processed_data,  # Processed signal as numpy array
                'sample_rate': sample_rate,
                'segments': segments,
                'heart_cycles': heart_cycles,
                'signal_object': signal,  # Keep the full signal object for reference
                'processed_object': processed_signal  # Keep the processed signal object
            }
        
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            raise
    
    def _extract_features(self, signal, segments):
        """Extract features from the segmented signal.
        
        Args:
            signal: Preprocessed PCG signal
            segments: Dictionary containing segmentation results
            
        Returns:
            Dictionary of extracted features
        """
        # Placeholder for feature extraction logic
        return {}
        
    def batch_process(self, file_paths):
        """Process multiple audio files.
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            List of processing results
        """
        return [self.process_file(fp) for fp in file_paths]
