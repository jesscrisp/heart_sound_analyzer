"""
Unit tests for the segmentation module.
"""
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock
from src.segmentation import segment_heart_sounds, create_heart_cycle_segments

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / 'test_data'

class TestSegmentation:
    """Test cases for the segmentation functions."""
    
    def test_segment_peak_detection(self, sample_signal):
        """Test peak detection segmentation with synthetic signal."""
        signal_data, fs = sample_signal
        
        # Create a mock signal object with required attributes
        pcg_signal = MagicMock()
        pcg_signal.signal = signal_data
        pcg_signal.fs = fs
        pcg_signal.envelope = np.abs(signal_data)
        
        # Mock the adv_peak function to return test peaks
        with pytest.MonkeyPatch().context() as m:
            m.setattr('pyPCG.segment.adv_peak', 
                     lambda sig, percent_th: (np.array([0.5, 0.6]), np.array([100, 300])))
            m.setattr('pyPCG.segment.peak_sort_diff', 
                     lambda peaks: (np.array([100]), np.array([300])))
            m.setattr('pyPCG.segment.segment_peaks', 
                     lambda peaks, sig, start_drop, end_drop: (np.array([90, 290]), np.array([110, 310])))
            
            # Run segmentation
            segments = segment_heart_sounds(
                pcg_signal,
                method='peak_detection',
                peak_threshold=0.1,
                start_drop=0.8,
                end_drop=0.7
            )
        
        # Basic validation of results
        assert 's1_starts' in segments
        assert 's1_ends' in segments
        assert 's2_starts' in segments
        assert 's2_ends' in segments
        assert segments['method'] == 'peak_detection'
    
    def test_segment_lr_hsmm(self, sample_signal, mocker):
        """Test LR-HSMM segmentation with mock model."""
        signal_data, fs = sample_signal
        
        # Create a mock signal object
        pcg_signal = MagicMock()
        pcg_signal.signal = signal_data
        pcg_signal.fs = fs
        
        # Mock the LR-HSMM segmentation
        mock_states = [
            (100, 200, 'S1'),
            (200, 400, 'systole'),
            (400, 500, 'S2'),
            (500, 1000, 'diastole')
        ]
        mocker.patch('src.segmentation._segment_with_lr_hsmm', 
                    return_value={
                        's1_starts': np.array([100]),
                        's1_ends': np.array([200]),
                        's2_starts': np.array([400]),
                        's2_ends': np.array([500]),
                        'method': 'lr_hsmm'
                    })
        
        # Run segmentation with a dummy model path
        segments = segment_heart_sounds(
            pcg_signal,
            method='lr_hsmm',
            model_path='dummy_model.json'
        )
        
        # Basic validation of results
        assert 's1_starts' in segments
        assert 's1_ends' in segments
        assert 's2_starts' in segments
        assert 's2_ends' in segments
        assert segments['method'] == 'lr_hsmm'
    
    def test_invalid_method(self, sample_signal):
        """Test that an invalid method raises a ValueError."""
        signal_data, fs = sample_signal
        pcg_signal = MagicMock()
        pcg_signal.signal = signal_data
        pcg_signal.fs = fs
        
        with pytest.raises(ValueError, match='Unsupported segmentation method'):
            segment_heart_sounds(pcg_signal, method='invalid_method')

def test_create_heart_cycle_segments():
    """Test creating heart cycle segments from segmentation results."""
    # Sample segmentation results
    segments = {
        's1_starts': np.array([100, 1000]),
        's1_ends': np.array([200, 1100]),
        's2_starts': np.array([400, 1300]),
        's2_ends': np.array([500, 1400]),
        'method': 'test'
    }
    
    # Create heart cycles
    cycles = create_heart_cycle_segments(segments)
    
    # Validate results
    assert len(cycles) == 2
    assert cycles[0]['s1_start'] == 100
    assert cycles[0]['s1_end'] == 200
    assert cycles[0]['s2_start'] == 400
    assert cycles[0]['s2_end'] == 500
    assert cycles[0]['s1_duration'] == 100
    assert cycles[0]['s2_duration'] == 100
    assert cycles[0]['s1s2_interval'] == 300
    assert cycles[0]['cycle_duration'] == 400
