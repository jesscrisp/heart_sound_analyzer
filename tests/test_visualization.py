"""
Unit tests for the visualization module.
"""
import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the visualization module
from src.visualization import HeartSoundVisualizer

class TestHeartSoundVisualizer:
    """Test cases for HeartSoundVisualizer class."""
    
    @pytest.fixture(autouse=True)
    def setup_visualizer(self):
        """Set up a test visualizer instance."""
        self.config = {
            'plot': {
                'figsize': (12, 6),
                'dpi': 100,
                'fontsize': 10
            }
        }
        self.visualizer = HeartSoundVisualizer(self.config)
    
    def test_init(self):
        """Test visualizer initialization with config."""
        assert self.visualizer.config == self.config
        
        # Test default config
        visualizer = HeartSoundVisualizer()
        assert isinstance(visualizer.config, dict)
    
    def test_plot_segmentation(self, sample_signal, sample_segments, tmp_path):
        """Test plotting segmentation results."""
        signal, fs = sample_signal
        output_path = tmp_path / 'segmentation_test.png'
        
        # Test
        self.visualizer.plot_segmentation(
            signal=signal,
            sample_rate=fs,
            segments=sample_segments,
            output_path=output_path
        )
        
        # Verify file was created
        assert output_path.exists()
        
        # Clean up
        plt.close('all')
    
    def test_plot_envelope(self, sample_signal, sample_segments, tmp_path):
        """Test plotting signal envelope with peaks."""
        signal, fs = sample_signal
        envelope = np.abs(signal)  # Simple envelope for testing
        output_path = tmp_path / 'envelope_test.png'
        
        # Test
        self.visualizer.plot_envelope(
            signal=signal,
            envelope=envelope,
            sample_rate=fs,
            peaks=sample_segments,
            output_path=output_path
        )
        
        # Verify file was created
        assert output_path.exists()
        
        # Clean up
        plt.close('all')
    
    def test_plot_heart_cycles(self, sample_signal, sample_segments, tmp_path):
        """Test plotting individual heart cycles."""
        signal, fs = sample_signal
        output_dir = tmp_path / 'cycles'
        output_dir.mkdir()
        
        # Test
        self.visualizer.plot_heart_cycles(
            signal=signal,
            sample_rate=fs,
            segments=sample_segments,
            output_dir=output_dir,
            max_plots=2
        )
        
        # Verify files were created
        assert (output_dir / 'cycle_1.png').exists()
        
        # Clean up
        plt.close('all')
    
    def test_create_time_axis(self):
        """Test time axis creation."""
        signal = np.random.randn(1000)  # 1 second at 1000 Hz
        fs = 1000
        
        # Test
        time_axis = self.visualizer.create_time_axis(signal, fs)
        
        # Verify
        assert len(time_axis) == len(signal)
        assert np.isclose(time_axis[-1], 0.999)  # 0 to 0.999s for 1000 samples at 1000 Hz
    
    @patch('matplotlib.pyplot.show')
    def test_show_plot(self, mock_show):
        """Test showing a plot."""
        # Create a simple plot
        plt.plot([1, 2, 3], [4, 5, 6])
        
        # Test
        self.visualizer._show_plot()
        
        # Verify
        mock_show.assert_called_once()
        
        # Clean up
        plt.close('all')
