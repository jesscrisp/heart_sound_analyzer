"""
Unit tests for the processor module.
"""
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the processor module
from src.processor import HeartSoundProcessor

class TestHeartSoundProcessor:
    """Test cases for HeartSoundProcessor class."""
    
    @pytest.fixture(autouse=True)
    def setup_processor(self):
        """Set up a test processor instance."""
        self.config = {
            'audio': {
                'sample_rate': 4000,
                'normalize': True
            },
            'segmentation': {
                'method': 'envelope',
                'min_heart_rate': 40,
                'max_heart_rate': 200
            }
        }
        self.processor = HeartSoundProcessor(self.config)
    
    def test_init(self):
        """Test processor initialization with config."""
        assert self.processor.config == self.config
        assert self.processor.audio_loader is not None
        assert self.processor.segmenter is not None
    
    @patch('src.processor.AudioLoader')
    def test_process_file(self, mock_audio_loader, sample_signal):
        """Test the process_file method with a mock audio loader."""
        # Setup mock
        mock_audio_loader.return_value.load_audio.return_value = MagicMock(
            data=sample_signal[0],
            fs=sample_signal[1],
            history=[]
        )
        mock_audio_loader.return_value.preprocess.return_value = MagicMock(
            data=sample_signal[0] * 0.8,  # Simulate processed signal
            fs=sample_signal[1],
            history=['normalize', 'filter', 'envelope']
        )
        
        # Mock segmenter
        self.processor.segmenter.segment.return_value = {
            's1_starts': [100],
            's1_ends': [200],
            's2_starts': [300],
            's2_ends': [400],
            'method': 'envelope'
        }
        
        # Test
        test_file = 'test_audio.wav'
        results = self.processor.process_file(test_file)
        
        # Verify
        assert 'file_path' in results
        assert 'signal' in results
        assert 'processed' in results
        assert 'segments' in results
        assert 'features' in results
        assert 'sample_rate' in results
        assert 'config' in results
        
        # Verify audio loader was called correctly
        mock_audio_loader.return_value.load_audio.assert_called_once_with(test_file)
        
        # Verify segmenter was called
        self.processor.segmenter.segment.assert_called_once()
    
    def test_batch_process(self):
        """Test batch processing of multiple files."""
        # Mock process_file to return predictable results
        self.processor.process_file = MagicMock(side_effect=[
            {'file_path': 'file1.wav', 'sample_rate': 4000},
            {'file_path': 'file2.wav', 'sample_rate': 4000}
        ])
        
        # Test
        file_paths = ['file1.wav', 'file2.wav']
        results = self.processor.batch_process(file_paths)
        
        # Verify
        assert len(results) == 2
        assert results[0]['file_path'] == 'file1.wav'
        assert results[1]['file_path'] == 'file2.wav'
        assert self.processor.process_file.call_count == 2

    def test_extract_features(self):
        """Test feature extraction placeholder."""
        # Create a mock signal and segments
        signal = MagicMock()
        segments = {
            's1_starts': [100],
            's1_ends': [200],
            's2_starts': [300],
            's2_ends': [400]
        }
        
        # Test
        features = self.processor._extract_features(signal, segments)
        
        # Verify features is a dictionary (placeholder implementation)
        assert isinstance(features, dict)
