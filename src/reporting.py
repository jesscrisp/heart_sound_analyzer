"""
Module for reporting and saving analysis results.
"""
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)

class ResultsReporter:
    """Handles saving of analysis results to various formats."""

    @staticmethod
    def save_segmentation_results_csv(
        results: Dict[str, Any],
        output_dir: str,
        base_name: Optional[str] = None
    ) -> None:
        """Save segmentation S1/S2 start/end times to a CSV file.
        
        Args:
            results: Dictionary containing segmentation results. Expected to have
                     a 'segments' key with 's1_starts', 's1_ends', etc.,
                     and optionally a 'file_path' key for deriving base_name.
            output_dir: Directory to save the CSV file.
            base_name: Base name for the output CSV file. If None, it's derived
                       from 'file_path' in results or defaults to 'output'.
        """
        try:
            output_dir_path = Path(output_dir)
            output_dir_path.mkdir(parents=True, exist_ok=True)
            
            if base_name is None:
                input_file_path = results.get('file_path', 'output')
                base_name = Path(input_file_path).stem
            
            segments_data = results.get('segments', {})
            if not segments_data:
                logger.warning(f"No segments found in results for {base_name}. Skipping CSV report.")
                return

            # Prepare data for DataFrame ensuring all arrays are of equal length if possible
            # or handle cases where one sound type might be missing/have different counts.
            # For simplicity, we'll use the existing structure which pandas handles by NaN padding.
            s1_starts = segments_data.get('s1_starts', [])
            s1_ends = segments_data.get('s1_ends', [])
            s2_starts = segments_data.get('s2_starts', [])
            s2_ends = segments_data.get('s2_ends', [])

            # Create a list of dictionaries for more robust DataFrame creation with unequal columns
            max_len = max(len(s1_starts), len(s1_ends), len(s2_starts), len(s2_ends), 0)
            report_data = []
            for i in range(max_len):
                row = {}
                if i < len(s1_starts): row['s1_start_sample'] = s1_starts[i]
                if i < len(s1_ends): row['s1_end_sample'] = s1_ends[i]
                if i < len(s2_starts): row['s2_start_sample'] = s2_starts[i]
                if i < len(s2_ends): row['s2_end_sample'] = s2_ends[i]
                report_data.append(row)

            if not report_data:
                logger.info(f"No segment data to save for {base_name}.")
                return
                
            df = pd.DataFrame(report_data)
            
            csv_filename = f"{base_name}_segmentation_report.csv"
            csv_path = output_dir_path / csv_filename
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved segmentation report to: {csv_path}")
                
        except Exception as e:
            logger.error(f"Error saving segmentation report for {base_name}: {e}")
            # Decide if to raise, or just log and continue if reporting is non-critical
            raise RuntimeError(f"Failed to save segmentation report for {base_name}.") from e
