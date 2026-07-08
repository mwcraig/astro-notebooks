import numpy as np
import pytest
from astropy.io import fits

N_IMAGES = 5
IMAGE_SHAPE = (128, 128)


@pytest.fixture
def fits_dir(tmp_path, monkeypatch):
    """Directory of small synthetic FITS images; cwd is moved to tmp_path
    because ImageSelect creates its thumbs/ cache relative to cwd."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rng = np.random.default_rng(42)
    for i in range(N_IMAGES):
        data = rng.uniform(100.0, 1000.0, size=IMAGE_SHAPE)
        # a few NaNs and one very large value to exercise cleaning/clamping
        data[0:2, 0:2] = np.nan
        data[5, 5] = 2e5
        hdu = fits.PrimaryHDU(data)
        hdu.header["IMAGETYP"] = "LIGHT"
        hdu.writeto(data_dir / f"image-{i:03d}.fit")
    monkeypatch.chdir(tmp_path)
    return data_dir
