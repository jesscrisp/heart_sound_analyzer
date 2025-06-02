"""
Visualization module for heart sound analysis results.
"""
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from matplotlib.patches import Rectangle
import io
import base64
import datetime

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
            plt.show(block=True)
        
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
        fig = None  # Initialize fig to None for the finally block
        try:
            fig, ax = plt.subplots(figsize=(14, 6))
            
            # Plot the signal
            self.plot_signal(signal, sample_rate, ax=ax, **kwargs)

            time_axis_signal = np.linspace(0, len(signal) / sample_rate, num=len(signal))
            envelope_data = segments.get('envelope')
            plotted_envelope = False # Flag to track if envelope was actually plotted

            if envelope_data is not None and len(np.asarray(envelope_data)) > 0:
                current_envelope_data = np.asarray(envelope_data).copy() # Work with a copy
                
                # Adjust envelope length if it mismatches signal length
                if len(current_envelope_data) != len(time_axis_signal):
                    logger.debug(f"Original envelope length ({len(current_envelope_data)}) differs from signal length ({len(time_axis_signal)}).")
                    if len(current_envelope_data) > len(time_axis_signal):
                        logger.debug(f"Truncating envelope to match signal length: {len(time_axis_signal)}.")
                        current_envelope_data = current_envelope_data[:len(time_axis_signal)]
                    # If envelope is shorter, it won't be plotted by the next check.

                # Ensure current_envelope_data has the same length as time_axis_signal for plotting
                if len(current_envelope_data) == len(time_axis_signal):
                    ax.plot(time_axis_signal, current_envelope_data, color='purple', linestyle='--', alpha=0.7, label='Envelope')
                    plotted_envelope = True
                else:
                    logger.warning(f"Envelope length ({len(current_envelope_data)}) still does not match signal length ({len(time_axis_signal)}) after attempted adjustment. Skipping envelope plot.")
        
            # Plot S1 segments
            s1_starts = segments.get('s1_starts', [])
            s1_ends = segments.get('s1_ends', [])
            s1_label_added = False
            for start, end in zip(s1_starts, s1_ends):
                ax.axvspan(
                    start/sample_rate, end/sample_rate,
                    color='red', alpha=0.2, label='S1' if not s1_label_added else None
                )
                s1_label_added = True
            
            # Plot S2 segments
            s2_starts = segments.get('s2_starts', [])
            s2_ends = segments.get('s2_ends', [])
            s2_label_added = False
            for start, end in zip(s2_starts, s2_ends):
                ax.axvspan(
                    start/sample_rate, end/sample_rate,
                    color='green', alpha=0.2, label='S2' if not s2_label_added else None
                )
                s2_label_added = True
            
            # Customize the plot
            ax.set_title('Heart Sound Segmentation')
            
            # Create custom legend
            from matplotlib.patches import Patch
            from matplotlib.lines import Line2D
            legend_elements = []
            if s1_label_added:
                 legend_elements.append(Patch(facecolor='red', alpha=0.2, label='S1'))
            if s2_label_added:
                 legend_elements.append(Patch(facecolor='green', alpha=0.2, label='S2'))
            
            if plotted_envelope:
                legend_elements.append(Line2D([0], [0], color='purple', linestyle='--', lw=2, label='Envelope'))
        
            if legend_elements:
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
            
        except Exception as e:
            logger.error(f"Error generating segmentation plot: {e}")
            raise # Re-raise the exception after logging
        finally:
            if fig is not None:
                plt.close(fig) # Ensure figure is closed
    
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
    
    def plot_signal_to_html_embeddable(self, signal: np.ndarray, sample_rate: float, title: str) -> str:
        """
        Generates a plot of the signal and returns it as an HTML embeddable img tag.

        Args:
            signal: The signal array (NumPy array).
            sample_rate: Sample rate in Hz.
            title: Title for the plot.

        Returns:
            An HTML string for an <img> tag with the plot embedded as base64,
            or an error message if the signal is empty.
        """
        if signal is None or signal.size == 0:
            logger.warning(f"Cannot plot empty or None signal for title: {title}")
            # Return a placeholder or error message as HTML
            return (f'<div style="border: 1px solid red; padding: 10px; margin: 5px;">'
                    f'<p style="color: red; font-weight: bold;">Error: No data to plot for \"{title}\”.</p>'
                    f'</div>')

        fig, ax = plt.subplots()
        time_axis = self.create_time_axis(signal, sample_rate)
        
        ax.plot(time_axis, signal, label=title) # Removed color for default cycling
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title(title)
        if title: # Only add legend if title is not empty, as it's used as label
            ax.legend()
        # ax.grid(True) # Grid is typically handled by _setup_style

        # Save plot to an in-memory buffer
        buf = io.BytesIO()
        try:
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            # Encode image to base64
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')
            # Create HTML img tag
            html_img = f'<img src="data:image/png;base64,{image_base64}" alt="{title}" style="max-width:100%; height:auto;">'
        except Exception as e:
            logger.error(f"Error generating plot for {title}: {e}")
            html_img = (f'<div style="border: 1px solid red; padding: 10px; margin: 5px;">'
                        f'<p style="color: red; font-weight: bold;">Error generating plot for \"{title}\”: {e}</p>'
                        f'</div>')
        finally:
            plt.close(fig) # Close the figure to free memory
            buf.close()
        
        return html_img

    def generate_html_report(self, raw_audio: np.ndarray, processed_audio: np.ndarray,
                             envelope: np.ndarray, segments: Dict[str, np.ndarray],
                             sample_rate: float, output_html_path: str,
                             config: Optional[Dict[str, Any]] = None) -> None:
        """Generates a comprehensive HTML report with various pipeline visualizations."""
        html_parts = []
        html_parts.append("<html><head><title>Heart Sound Analysis Report</title>")
        html_parts.append("<style>body { font-family: Arial, sans-serif; margin: 20px; } h1 { text-align: center; color: #333; } h2 { margin-top: 30px; color: #444; border-bottom: 1px solid #eee; padding-bottom: 5px;} hr { display: none; } img { display: block; margin-left: auto; margin-right: auto; max-width:90%; height:auto; border: 1px solid #ccc; margin-bottom: 10px; box-shadow: 2px 2px 5px #ddd; margin-top:10px; } p.error { color:red; font-weight:bold; text-align:center; } p.warning { color:orange; text-align:center; }</style>")
        html_parts.append("</head><body>")
        html_parts.append("<h1>Heart Sound Analysis Report</h1>")

        # 1. Raw Audio
        try:
            html_parts.append("<h2>Raw Audio Signal</h2>")
            html_parts.append(self.plot_signal_to_html_embeddable(raw_audio, sample_rate, "Raw Audio Signal Plot"))
        except Exception as e:
            logger.error(f"Error generating raw audio plot for HTML report: {e}", exc_info=True)
            html_parts.append(f'<p class="error">Error generating raw audio plot: {e}</p>')
        html_parts.append("<hr>")

        # 2. Processed Audio
        try:
            html_parts.append("<h2>Processed Audio Signal</h2>")
            html_parts.append(self.plot_signal_to_html_embeddable(processed_audio, sample_rate, "Processed Audio Signal Plot"))
        except Exception as e:
            logger.error(f"Error generating processed audio plot for HTML report: {e}", exc_info=True)
            html_parts.append(f'<p class="error">Error generating processed audio plot: {e}</p>')
        html_parts.append("<hr>")

        # 3. Envelope Plot (custom plot for HTML report)
        html_parts.append("<h2>Signal with Envelope</h2>")
        fig_env_plot = None # Ensure fig_env_plot is defined for finally block
        buf_env = io.BytesIO()
        try:
            fig_env_plot, ax_env_plot = plt.subplots(figsize=self.config.get('plot_envelope_figsize', (12,4)))
            time_axis_env = self.create_time_axis(processed_audio, sample_rate)
            ax_env_plot.plot(time_axis_env, processed_audio, label='Processed Signal', alpha=0.5, linewidth=0.8)
            ax_env_plot.plot(time_axis_env, envelope, label='Envelope', color='red', linewidth=1.0)
            ax_env_plot.set_title("Signal with Envelope Plot")
            ax_env_plot.set_xlabel("Time (s)")
            ax_env_plot.set_ylabel("Amplitude")
            ax_env_plot.legend()
            ax_env_plot.grid(True, linestyle=':', alpha=0.7)
            plt.tight_layout()
            fig_env_plot.savefig(buf_env, format='png', bbox_inches='tight')
            buf_env.seek(0)
            image_base64_env = base64.b64encode(buf_env.read()).decode('utf-8')
            html_parts.append(f'<img src="data:image/png;base64,{image_base64_env}" alt="Signal with Envelope Plot">')
        except Exception as e_env:
            logger.error(f"Error generating envelope plot for HTML report: {e_env}", exc_info=True)
            html_parts.append(f'<p class="error">Error generating envelope plot: {e_env}</p>')
        finally:
            if fig_env_plot:
                 plt.close(fig_env_plot)
            buf_env.close()
        html_parts.append("<hr>")

        # 4. Segmentation Plot (using temp file method)
        html_parts.append("<h2>Signal Segmentation</h2>")
        output_dir_report = Path(output_html_path).parent
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
        temp_seg_filename = f"temp_seg_plot_{Path(output_html_path).stem}_{timestamp_str}.png"
        temp_seg_path = output_dir_report / temp_seg_filename
        
        try:
            self.plot_segmentation(
                signal=processed_audio, 
                sample_rate=sample_rate, 
                segments=segments, 
                output_path=str(temp_seg_path), 
                show=False
            )
            if temp_seg_path.exists():
                with open(temp_seg_path, "rb") as image_file:
                    image_base64_seg = base64.b64encode(image_file.read()).decode('utf-8')
                html_parts.append(f'<img src="data:image/png;base64,{image_base64_seg}" alt="Signal Segmentation Plot">')
            else:
                logger.warning(f"Segmentation plot temp file not found at {temp_seg_path} after attempting to save.")
                html_parts.append(f'<p class="warning">Could not generate segmentation plot (temp file not found at {temp_seg_path}).</p>')
        except Exception as e_seg:
            logger.error(f"Error generating or processing segmentation plot for HTML report: {e_seg}", exc_info=True)
            html_parts.append(f'<p class="error">Error generating segmentation plot: {e_seg}</p>')
        finally:
            if temp_seg_path.exists():
                 try:
                     temp_seg_path.unlink()
                 except OSError as e_unlink:
                     logger.warning(f"Could not delete temp segmentation plot {temp_seg_path}: {e_unlink}")

        html_parts.append("</body></html>")
        final_html = "\n".join(html_parts)

        try:
            with open(output_html_path, 'w', encoding='utf-8') as f:
                f.write(final_html)
            logger.info(f"HTML report successfully generated: {output_html_path}")
        except Exception as e:
            logger.error(f"Error writing HTML report to {output_html_path}: {e}", exc_info=True)

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
