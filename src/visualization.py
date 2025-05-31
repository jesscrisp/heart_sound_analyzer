"""
Visualization module for heart sound analysis results.
"""
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from matplotlib.patches import Rectangle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HeartSoundVisualizer:
    """Handles visualization of heart sound analysis results."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the visualizer with optional configuration.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self._setup_style()
    
    def _setup_style(self):
        """Configure the plotting style."""
        plt.style.use('seaborn-v0_8')
        plt.rcParams.update({
            'figure.figsize': (14, 8),
            'font.size': 10,
            'lines.linewidth': 1.0,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'figure.autolayout': True
        })
    
    def create_time_axis(self, signal: np.ndarray, sample_rate: float) -> np.ndarray:
        """Create a time axis for a signal.
        
        Args:
            signal: The signal array
            sample_rate: Sample rate in Hz
            
        Returns:
            Time axis array in seconds
        """
        return np.arange(len(signal)) / sample_rate
    
    def plot_heart_sound_analysis(
        self,
        signal: np.ndarray,
        processed: np.ndarray,
        sample_rate: float,
        segments: Dict[str, np.ndarray],
        heart_cycles: List[Dict[str, float]],
        title: str = "Heart Sound Analysis",
        output_path: Optional[Union[str, Path]] = None,
        show: bool = True
    ) -> plt.Figure:
        """Create a comprehensive visualization of heart sound analysis.
        
        Args:
            signal: Raw audio signal
            processed: Processed audio signal
            sample_rate: Sample rate in Hz
            segments: Dictionary containing segment information
            heart_cycles: List of heart cycle dictionaries
            title: Plot title
            output_path: Path to save the figure (optional)
            show: Whether to display the plot
            
        Returns:
            The created matplotlib Figure
        """
        time = self.create_time_axis(signal, sample_rate)
        
        # Create figure with multiple subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        
        # Plot raw signal
        ax1.plot(time, signal, label='Raw Signal', color='#1f77b4', alpha=0.7)
        ax1.set_ylabel('Amplitude')
        ax1.set_title(f"{title} - Raw Signal")
        
        # Plot processed signal with segments
        ax2.plot(time, processed, label='Processed', color='#2ca02c')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Amplitude')
        ax2.set_title('Processed Signal with Detected Segments')
        
        # Add S1 and S2 segments to the plot
        self._plot_segments(ax2, time, segments, heart_cycles)
        
        # Add legend and adjust layout
        ax1.legend(loc='upper right')
        ax2.legend(loc='upper right')
        plt.tight_layout()
        
        # Save figure if output path is provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved visualization to {output_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def _plot_segments(
        self,
        ax: plt.Axes,
        time: np.ndarray,
        segments: Dict[str, np.ndarray],
        heart_cycles: List[Dict[str, float]]
    ) -> None:
        """Add segment annotations to the plot.
        
        Args:
            ax: Matplotlib Axes to plot on
            time: Time axis array
            segments: Dictionary containing segment information
            heart_cycles: List of heart cycle dictionaries
        """
        # Plot S1 segments
        for start, end in zip(segments['s1_starts'], segments['s1_ends']):
            if start < len(time) and end < len(time):
                ax.axvspan(time[start], time[end], color='red', alpha=0.2, label='S1')
        
        # Plot S2 segments
        for start, end in zip(segments['s2_starts'], segments['s2_ends']):
            if start < len(time) and end < len(time):
                ax.axvspan(time[start], time[end], color='blue', alpha=0.2, label='S2')
        
        # Add legend entries (without duplicates)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
    
    def plot_signal(
        self,
        signal: np.ndarray,
        sample_rate: float,
        ax: Optional[plt.Axes] = None,
        label: str = 'Signal',
        **kwargs
    ) -> plt.Axes:
        """Plot a signal with proper time axis.
        
        Args:
            signal: The signal to plot
            sample_rate: Sample rate in Hz
            ax: Optional matplotlib axes to plot on
            label: Label for the plot legend
            **kwargs: Additional arguments to pass to plot()
            
        Returns:
            The axes object used for plotting
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 4))
        
        time = self.create_time_axis(signal, sample_rate)
        ax.plot(time, signal, label=label, **kwargs)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        return ax
    
    def plot_segmentation(
        self,
        signal: np.ndarray,
        sample_rate: float,
        segments: Dict[str, np.ndarray],
        output_path: Optional[Union[str, Path]] = None,
        show: bool = True,
        **kwargs
    ) -> None:
        """Plot the signal with segmentation boundaries.
        
        Args:
            signal: The audio signal
            sample_rate: Sample rate in Hz
            segments: Dictionary containing segmentation results
            output_path: If provided, save the plot to this path
            show: If True, display the plot
            **kwargs: Additional arguments for plot_signal()
        """
        try:
            fig, ax = plt.subplots(figsize=(14, 6))
            
            # Plot the signal
            self.plot_signal(signal, sample_rate, ax=ax, **kwargs)
            
            # Plot S1 segments
            s1_starts = segments.get('s1_starts', [])
            s1_ends = segments.get('s1_ends', [])
            for start, end in zip(s1_starts, s1_ends):
                ax.axvspan(
                    start/sample_rate, end/sample_rate,
                    color='red', alpha=0.2, label='S1'
                )
            
            # Plot S2 segments
            s2_starts = segments.get('s2_starts', [])
            s2_ends = segments.get('s2_ends', [])
            for start, end in zip(s2_starts, s2_ends):
                ax.axvspan(
                    start/sample_rate, end/sample_rate,
                    color='green', alpha=0.2, label='S2'
                )
            
            # Customize the plot
            ax.set_title('Heart Sound Segmentation')
            
            # Create custom legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='red', alpha=0.2, label='S1'),
                Patch(facecolor='green', alpha=0.2, label='S2')
            ]
            ax.legend(handles=legend_elements)
            
            plt.tight_layout()
            
            # Save or show the plot
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
                logger.info(f"Saved segmentation plot to {output_path}")
                
            if show:
                plt.show()
                
            plt.close()
            
        except Exception as e:
            logger.error(f"Error generating segmentation plot: {e}")
            raise
    
    def plot_envelope(
        self,
        signal: np.ndarray,
        envelope: np.ndarray,
        sample_rate: float,
        peaks: Optional[Dict[str, np.ndarray]] = None,
        output_path: Optional[Union[str, Path]] = None,
        show: bool = True,
        **kwargs
    ) -> None:
        """Plot the signal with its envelope and optional peaks.
        
        Args:
            signal: The original signal
            envelope: The envelope of the signal
            sample_rate: Sample rate in Hz
            peaks: Dictionary containing peak information
            output_path: If provided, save the plot to this path
            show: If True, display the plot
            **kwargs: Additional arguments for plotting
        """
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
            
            # Plot the original signal
            self.plot_signal(signal, sample_rate, ax=ax1, label='Original Signal')
            ax1.set_title('Original Signal')
            
            # Plot the envelope
            time = self.create_time_axis(envelope, sample_rate)
            ax2.plot(time, envelope, label='Envelope', **kwargs)
            
            # Plot peaks if provided
            if peaks:
                peak_indices = peaks.get('peaks', [])
                peak_values = [envelope[i] for i in peak_indices]
                ax2.plot(
                    peak_indices/sample_rate, peak_values,
                    'rx', label='Peaks', markersize=8
                )
                
                # Plot S1 and S2 peaks if available
                s1_peaks = peaks.get('s1_peaks', [])
                if len(s1_peaks) > 0:
                    s1_values = [envelope[i] for i in s1_peaks]
                    ax2.plot(
                        np.array(s1_peaks)/sample_rate, s1_values,
                        'go', label='S1 Peaks', markersize=8, alpha=0.7
                    )
                
                s2_peaks = peaks.get('s2_peaks', [])
                if len(s2_peaks) > 0:
                    s2_values = [envelope[i] for i in s2_peaks]
                    ax2.plot(
                        np.array(s2_peaks)/sample_rate, s2_values,
                        'mo', label='S2 Peaks', markersize=8, alpha=0.7
                    )
            
            ax2.set_title('Envelope with Detected Peaks')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Amplitude')
            ax2.legend()
            
            plt.tight_layout()
            
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
                logger.info(f"Saved envelope plot to {output_path}")
                
            if show:
                plt.show()
                
            plt.close()
            
        except Exception as e:
            logger.error(f"Error generating envelope plot: {e}")
            raise
    
    def plot_heart_cycles(
        self,
        signal: np.ndarray,
        sample_rate: float,
        segments: Dict[str, np.ndarray],
        output_dir: Optional[Union[str, Path]] = None,
        max_plots: int = 5,
        **kwargs
    ) -> None:
        """Plot individual heart cycles with their segments.
        
        Args:
            signal: The audio signal
            sample_rate: Sample rate in Hz
            segments: Dictionary containing segmentation results
            output_dir: If provided, save plots to this directory
            max_plots: Maximum number of heart cycles to plot
            **kwargs: Additional arguments for plotting
        """
        try:
            s1_starts = segments.get('s1_starts', [])
            s1_ends = segments.get('s1_ends', [])
            s2_starts = segments.get('s2_starts', [])
            s2_ends = segments.get('s2_ends', [])
            
            # Determine number of plots to generate
            num_plots = min(len(s1_starts), max_plots) if s1_starts else 0
            
            for i in range(num_plots):
                # Get the current heart cycle (S1 to next S1)
                if i < len(s1_starts) - 1:
                    start_idx = s1_starts[i]
                    end_idx = s1_starts[i+1] if i+1 < len(s1_starts) else len(signal)
                else:
                    # For the last cycle, use S2 end + some padding
                    start_idx = s1_starts[i]
                    cycle_duration = s2_ends[i] - s1_starts[i]
                    end_idx = min(s1_starts[i] + int(1.5 * cycle_duration), len(signal))
                
                # Extract the cycle
                cycle_signal = signal[start_idx:end_idx]
                time = self.create_time_axis(cycle_signal, sample_rate)
                
                # Create the plot
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.plot(time, cycle_signal, label='Signal')
                
                # Mark S1
                if i < len(s1_starts) and i < len(s1_ends):
                    s1_start = (s1_starts[i] - start_idx) / sample_rate
                    s1_end = (s1_ends[i] - start_idx) / sample_rate
                    ax.axvspan(s1_start, s1_end, color='red', alpha=0.2, label='S1')
                
                # Mark S2
                if i < len(s2_starts) and i < len(s2_ends):
                    s2_start = (s2_starts[i] - start_idx) / sample_rate
                    s2_end = (s2_ends[i] - start_idx) / sample_rate
                    ax.axvspan(s2_start, s2_end, color='green', alpha=0.2, label='S2')
                
                ax.set_title(f'Heart Cycle {i+1}')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Amplitude')
                ax.legend()
                
                # Save or show the plot
                if output_dir:
                    output_dir = Path(output_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = output_dir / f'heart_cycle_{i+1}.png'
                    plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
                    logger.info(f"Saved heart cycle plot to {output_path}")
                
                plt.close()
                
        except Exception as e:
            logger.error(f"Error generating heart cycle plots: {e}")
            raise
