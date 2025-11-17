import tempfile
from pathlib import Path
from PIL import Image
import numpy as np

from core.file_processing import process_file


def test_process_file_basic(tmp_path: Path):
    # create a small RGB image
    img_path = tmp_path / "test.jpg"
    arr = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
    Image.fromarray(arr).save(img_path)

    result = process_file(str(img_path))
    assert isinstance(result, dict)
    assert result.get('width') == 32
    assert result.get('height') == 32
    assert result.get('channels') >= 3
    assert 'mean_r' in result and 'std_r' in result
