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
            import cv2
            from PIL import Image
            self.cv2_available = True
        except ImportError:
            logger.warning("OpenCV or PIL not available. Image processing will be limited.")
            self.cv2_available = False

    def process_file(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """Extract image metadata and basic features."""
        filepath = Path(filepath)

        if not self.cv2_available:
            return {'error': 'Image libraries not available'}

        try:
            import cv2
            from PIL import Image

            # Read image with OpenCV
            img_cv = cv2.imread(str(filepath))
            height, width, channels = img_cv.shape

            # Read with PIL for additional metadata
            img_pil = Image.open(filepath)

            metadata = {
                'width': width,
                'height': height,
                'channels': channels,
                'mode': img_pil.mode,
                'format': img_pil.format,
                'file_size': filepath.stat().st_size,
                'processed_at': datetime.now().isoformat()
            }

            # Basic image statistics
            if channels == 3:
                b, g, r = cv2.split(img_cv)
                metadata.update({
                    'mean_b': float(b.mean()),
                    'mean_g': float(g.mean()),
                    'mean_r': float(r.mean()),
                    'std_b': float(b.std()),
                    'std_g': float(g.std()),
                    'std_r': float(r.std())
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