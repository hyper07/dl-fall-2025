"""
File processing utilities for handling image files.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Union, Any, Tuple
import logging
from datetime import datetime
import tempfile
import shutil

logger = logging.getLogger(__name__)


class FileProcessor:
    """Base class for file processing operations."""

    SUPPORTED_EXTENSIONS = []

    @classmethod
    def can_process(cls, filepath: Union[str, Path]) -> bool:
        """Check if file can be processed by this processor."""
        filepath = Path(filepath)
        return filepath.suffix.lower() in cls.SUPPORTED_EXTENSIONS

    def process_file(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """Process a single file. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement process_file method")


class ImageProcessor(FileProcessor):
    """Image file processing using OpenCV and PIL."""

    SUPPORTED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

    def __init__(self):
        try:
            # Prefer PIL + numpy implementation; cv2 is optional
            from PIL import Image
            import numpy as _np
            self.pil_available = True
        except Exception:
            logger.warning("PIL or numpy not available. Image processing will be limited.")
            self.pil_available = False

    def process_file(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """Extract image metadata and basic features."""
        filepath = Path(filepath)

        if not self.pil_available:
            return {'error': 'PIL or numpy not available'}

        try:
            from PIL import Image
            import numpy as np

            img_pil = Image.open(filepath)
            img_arr = np.array(img_pil)

            # Handle grayscale images by adding channel dimension
            if img_arr.ndim == 2:
                height, width = img_arr.shape
                channels = 1
            else:
                height, width, channels = img_arr.shape

            metadata = {
                'width': int(width),
                'height': int(height),
                'channels': int(channels),
                'mode': img_pil.mode,
                'format': img_pil.format,
                'file_size': filepath.stat().st_size,
                'processed_at': datetime.now().isoformat()
            }

            # Basic image statistics
            if channels >= 3:
                # Convert to RGB order if PIL gave RGBA, take first 3 channels
                arr3 = img_arr[:, :, :3].astype(np.float32)
                means = arr3.mean(axis=(0, 1))
                stds = arr3.std(axis=(0, 1))
                metadata.update({
                    'mean_r': float(means[0]),
                    'mean_g': float(means[1]),
                    'mean_b': float(means[2]),
                    'std_r': float(stds[0]),
                    'std_g': float(stds[1]),
                    'std_b': float(stds[2])
                })

            return metadata

        except Exception as e:
            logger.error(f"Failed to process image {filepath}: {e}")
            return {'error': str(e)}


class FileManager:
    """File management utilities."""

    def __init__(self, base_path: Union[str, Path]):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save_uploaded_file(self, uploaded_file, filename: str) -> Path:
        """Save uploaded file to managed directory."""
        filepath = self.base_path / filename

        with open(filepath, 'wb') as f:
            if hasattr(uploaded_file, 'read'):
                # Streamlit uploaded file
                f.write(uploaded_file.read())
            else:
                # Regular file
                f.write(uploaded_file)

        logger.info(f"Saved file to {filepath}")
        return filepath

    def get_file_info(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """Get file information."""
        filepath = Path(filepath)

        if not filepath.exists():
            return {'error': 'File does not exist'}

        stat = filepath.stat()

        return {
            'name': filepath.name,
            'size': stat.st_size,
            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'extension': filepath.suffix,
            'path': str(filepath)
        }

    def list_files(self, pattern: str = "*") -> List[Dict[str, Any]]:
        """List files matching pattern."""
        files = []
        for filepath in self.base_path.glob(pattern):
            if filepath.is_file():
                files.append(self.get_file_info(filepath))
        return files

    def cleanup_old_files(self, days_old: int = 7):
        """Remove files older than specified days."""
        cutoff_time = datetime.now().timestamp() - (days_old * 24 * 60 * 60)

        removed_count = 0
        for filepath in self.base_path.glob("*"):
            if filepath.is_file() and filepath.stat().st_mtime < cutoff_time:
                filepath.unlink()
                removed_count += 1

        logger.info(f"Cleaned up {removed_count} old files")
        return removed_count


def get_file_processor(filepath: Union[str, Path]) -> Optional[FileProcessor]:
    """Factory function to get appropriate file processor."""
    processors = [ImageProcessor()]

    for processor in processors:
        if processor.can_process(filepath):
            return processor

    return None


def process_file(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Process a file using the appropriate processor."""
    processor = get_file_processor(filepath)

    if processor is None:
        return {'error': f'No processor available for {Path(filepath).suffix}'}

    return processor.process_file(filepath)


def batch_process_files(filepaths: List[Union[str, Path]],
                       max_workers: int = 4) -> List[Dict[str, Any]]:
    """Process multiple files in parallel."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(process_file, fp): fp for fp in filepaths}

        for future in as_completed(future_to_file):
            filepath = future_to_file[future]
            try:
                result = future.result()
                result['filepath'] = str(filepath)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {filepath}: {e}")
                results.append({'filepath': str(filepath), 'error': str(e)})

    return results