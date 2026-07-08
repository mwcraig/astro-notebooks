import shutil
from pathlib import Path

import ipywidgets as ipw
import numpy as np
from PIL import Image

from astro_notebooks.image_selector import (
    ImageSelect,
    ImageWithSelector,
    _make_one_thumbnail,
    _scale_and_downsample,
)

from .conftest import IMAGE_SHAPE, N_IMAGES


def _walk_widgets(widget):
    """Yield widget and all its descendants."""
    yield widget
    for child in getattr(widget, "children", ()):
        yield from _walk_widgets(child)


def _thumb_arrays(thumb_dir):
    """Map png name -> pixel array for every thumbnail."""
    return {
        p.name: np.asarray(Image.open(p))
        for p in sorted(Path(thumb_dir).glob("*.png"))
    }


def test_scale_and_downsample_output_range_and_nans():
    data = np.random.default_rng(7).uniform(100, 1000, (64, 64))
    data[0, 0] = np.nan
    data[1, 1] = 2e5
    out = _scale_and_downsample(data, downsample=8)
    assert not np.isnan(out).any()
    assert out.min() >= 0
    assert out.max() <= 1


def test_scale_and_downsample_shape_reduced():
    data = np.random.default_rng(7).uniform(100, 1000, (64, 64))
    out_downsampled = _scale_and_downsample(data, downsample=8)
    assert out_downsampled.shape == (8, 8)
    out_no_downsample = _scale_and_downsample(data, downsample=1)
    assert out_no_downsample.shape == (64, 64)


def test_scale_and_downsample_does_not_mutate_input():
    data = np.random.default_rng(7).uniform(100, 1000, (64, 64))
    data[0, 0] = np.nan
    data[1, 1] = 2e5
    original = data.copy()
    _scale_and_downsample(data, downsample=8)
    np.testing.assert_array_equal(data, original)


def test_make_one_thumbnail(fits_dir, tmp_path):
    src = sorted(fits_dir.glob("*.fit"))[0]
    dest = tmp_path / "thumb.png"
    _make_one_thumbnail(src, dest, 8)
    assert dest.exists()
    img = Image.open(dest)
    assert img.mode == "L"
    assert img.size == (IMAGE_SHAPE[1] // 8, IMAGE_SHAPE[0] // 8)
    assert np.asarray(img).std() > 0


def test_thumbnails_one_per_fits_grayscale(fits_dir):
    ImageSelect(directory=fits_dir)
    thumbs_dir = Path("thumbs")
    png_names = {p.name for p in thumbs_dir.glob("*.png")}
    expected_names = {f"image-{i:03d}.png" for i in range(N_IMAGES)}
    assert png_names == expected_names
    for p in thumbs_dir.glob("*.png"):
        img = Image.open(p)
        assert img.mode == "L"
        assert img.size == (IMAGE_SHAPE[1] // 8, IMAGE_SHAPE[0] // 8)


def test_existing_thumbnails_not_regenerated(fits_dir):
    ImageSelect(directory=fits_dir)
    mtimes = {p.name: p.stat().st_mtime_ns for p in Path("thumbs").glob("*.png")}
    ImageSelect(directory=fits_dir)
    mtimes2 = {p.name: p.stat().st_mtime_ns for p in Path("thumbs").glob("*.png")}
    assert mtimes == mtimes2


def test_stale_thumbnails_cleaned_up(fits_dir):
    ImageSelect(directory=fits_dir)
    (fits_dir / "image-000.fit").unlink()
    ImageSelect(directory=fits_dir)
    assert not (Path("thumbs") / "image-000.png").exists()
    for i in range(1, N_IMAGES):
        assert (Path("thumbs") / f"image-{i:03d}.png").exists()


def test_parallel_matches_serial(fits_dir):
    ImageSelect(directory=fits_dir, max_workers=1)
    serial = _thumb_arrays("thumbs")
    shutil.rmtree("thumbs")
    ImageSelect(directory=fits_dir, max_workers=2)
    parallel = _thumb_arrays("thumbs")
    assert serial.keys() == parallel.keys()
    for k in serial:
        assert serial[k].size > 0
        assert np.array_equal(serial[k], parallel[k])


def test_progress_ui_shown_and_hidden(fits_dir, monkeypatch):
    displayed = []
    monkeypatch.setattr("astro_notebooks.image_selector.display", lambda *a, **k: displayed.extend(a))
    ImageSelect(directory=fits_dir)
    assert len(displayed) == 1
    progress_widgets = [w for w in _walk_widgets(displayed[0]) if isinstance(w, ipw.IntProgress)]
    assert len(progress_widgets) == 1
    assert progress_widgets[0].max == N_IMAGES
    assert progress_widgets[0].value == N_IMAGES
    assert displayed[0].layout.display == "none"


def test_no_progress_display_when_cached(fits_dir, monkeypatch):
    ImageSelect(directory=fits_dir)
    displayed = []
    monkeypatch.setattr("astro_notebooks.image_selector.display", lambda *a, **k: displayed.extend(a))
    ImageSelect(directory=fits_dir)
    assert displayed == []


def test_image_select_structure(fits_dir):
    w = ImageSelect(directory=fits_dir)
    assert len(w.children) == 2
    assert isinstance(w.children[0], ipw.GridspecLayout)
    assert isinstance(w.children[1], ipw.Button)
    assert w._im_base_bames == [f"image-{i:03d}" for i in range(N_IMAGES)]
    assert len(w._selectors) == N_IMAGES
    for sel in w._selectors:
        assert isinstance(sel, ImageWithSelector)
    for sel in w._selectors:
        # ipywidgets may hand the value back as a memoryview
        png_bytes = bytes(sel.image_display.value)
        assert len(png_bytes) > 0
        assert png_bytes.startswith(b'\x89PNG')


def test_downsample_kwarg_flows_through(fits_dir):
    ImageSelect(directory=fits_dir, downsample=4)
    for p in Path("thumbs").glob("*.png"):
        img = Image.open(p)
        assert img.size == (IMAGE_SHAPE[1] // 4, IMAGE_SHAPE[0] // 4)


def test_move_rejects(fits_dir):
    w = ImageSelect(directory=fits_dir)
    w._selectors[0]._selector.value = False
    w._move_rejects_clicked(None)
    assert (fits_dir / "rejects" / "image-000.fit").exists()
    assert not (fits_dir / "image-000.fit").exists()
    assert len(w._selectors) == N_IMAGES - 1
    assert "image-000" not in w._im_base_bames
    assert len(w.children) == 2
    assert isinstance(w.children[0], ipw.GridspecLayout)
