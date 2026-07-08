from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ipywidgets as ipw
import numpy as np

from astropy.io import fits
from astropy.nddata import block_reduce
from astropy.visualization import simple_norm
from ccdproc import ImageFileCollection
from IPython.display import display
from PIL import Image

try:
    from stellarphot.gui.custom_widgets import Spinner
except Exception:
    # stellarphot's GUI extras may be missing or incompatible; fall back to
    # a message-only stand-in with the same start/stop interface.
    Spinner = None


class _MessageSpinner(ipw.VBox):
    """Fallback for stellarphot's Spinner when it cannot be imported."""

    def __init__(self, *args, message="", **kwargs):
        super().__init__(*args, **kwargs)
        self._message = ipw.HTML(message)
        self.children = [self._message]
        self.layout.display = "none"

    def start(self):
        self.layout.display = "flex"

    def stop(self):
        self.layout.display = "none"


def _scale_and_downsample(data, downsample=8,
                         min_percent=20,
                         max_percent=99.5):

    scaled_data = data.copy()
    scaled_data[scaled_data > 1e5] = 1e5
    if downsample > 1:
        scaled_data = block_reduce(scaled_data,
                                   block_size=(downsample, downsample))
    norm = simple_norm(scaled_data,
                       min_percent=min_percent,
                       max_percent=max_percent,
                       clip=True)

    # Replace all NaNs with 0
    normed_data = norm(scaled_data)
    normed_data[np.isnan(normed_data)] = 0

    return normed_data


def _make_one_thumbnail(fits_path, dest_path, downsample):
    """Make a single uint8 grayscale PNG thumbnail for a FITS image.

    Runs in a worker thread; the FITS read, numpy work and PNG encode all
    release the GIL for most of their run time.
    """
    data = fits.getdata(fits_path)
    scaled = _scale_and_downsample(data, downsample=downsample)
    Image.fromarray((scaled * 255).astype(np.uint8), mode="L").save(dest_path)


class ImageWithSelector(ipw.VBox):
    # value = tr.Bool(default_value=True).tag(sync=True)

    def __init__(self, image_png, *args, width="200px", fname="", **kwargs):
        super().__init__(*args, **kwargs)
        self._fname = fname
        img_layout = dict(
            object_fit='contain',
            width='100%'
        )
        self.image_display = ipw.Image(
            value=image_png,
            format='png',
            layout=img_layout
        )
        self._selector = ipw.Checkbox(
            description='Use image',
            value=True
        )
        self._valid_mark = ipw.Valid(
            description='',
            value=True
        )

        self._name = ipw.HTML(value=fname)

        ipw.link((self._selector, 'value'), (self._valid_mark, 'value'))
        # ipw.link((self, 'value'), (self._selector, 'value'))

        self.select_box = ipw.HBox(children=[self._selector, self._valid_mark])
        self.mobox = ipw.VBox(children=[self._name, self.select_box])
        self.children = [self.image_display, self.mobox]
        self.layout.width = width


class ImageSelect(ipw.VBox):
    def __init__(self, *args, directory=".", downsample=8, max_workers=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.path = Path(directory)
        self._downsample = downsample
        self._max_workers = max_workers
        self._collection = ImageFileCollection(self.path)
        self._move_rejects = ipw.Button(description='Move rejects')

        self.thumbs = Path('thumbs')
        self.make_thumbnails(thumb_dir=self.thumbs)
        self.make_selectors(thumb_dir=self.thumbs)
        self.n_cols = 4
        gs = self._make_grid()
        self.children = [gs, self._move_rejects]
        # self.layout.max_height = "400px"
        # self.layout.overflow = "scroll hidden"
        self._move_rejects.on_click(self._move_rejects_clicked)

    def make_thumbnails(self, thumb_dir="thumbs"):
        self._images = []
        thumby = Path(thumb_dir)
        thumby.mkdir(exist_ok=True)
        self._collection.refresh()
        self._im_base_bames = []
        todo = []
        for fname in self._collection.files_filtered(include_path=True):
            base = Path(fname).stem
            self._im_base_bames.append(base)
            dest_path = thumby / (base + '.png')
            if dest_path.exists():
                continue
            todo.append((Path(fname), dest_path))

        if not todo:
            return

        spinner_cls = Spinner if Spinner is not None else _MessageSpinner
        spinner = spinner_cls(message="Generating image thumbnails...")
        progress = ipw.IntProgress(
            value=0, min=0, max=len(todo), description="Thumbnails"
        )
        progress_box = ipw.VBox(children=[spinner, progress])
        # Display right away so the user sees activity while the rest of
        # the widget is still being constructed.
        display(progress_box)
        spinner.start()

        try:
            with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                futures = [
                    executor.submit(_make_one_thumbnail, src, dest, self._downsample)
                    for src, dest in todo
                ]
                for future in as_completed(futures):
                    future.result()
                    progress.value += 1
        finally:
            spinner.stop()
            progress_box.layout.display = "none"

    def make_selectors(self, thumb_dir="thumbs"):
        kiddos = []
        pngs = list(Path(thumb_dir).glob('*.png'))

        for thumb in pngs:
            if thumb.stem not in self._im_base_bames:
                thumb.unlink()
        pngs = list(Path(thumb_dir).glob('*.png'))

        png_dict = {p.stem: p for p in pngs}

        kiddos = {}
        for ims in self._im_base_bames:
            image_png = png_dict[ims].read_bytes()
            iws = ImageWithSelector(image_png, fname=ims)
            kiddos[ims] = iws

        self._selectors = [kiddos[ims] for ims in self._im_base_bames]

    def _move_rejects_clicked(self, _):
        reject_land = Path(self.path / 'rejects')
        for f, selector in zip(self._im_base_bames, self._selectors):
            if not selector._valid_mark.value:
                reject_land.mkdir(exist_ok=True)
                source = self.path / Path(f + ".fit")
                dest = reject_land / source.name
                source.rename(dest)

        self.make_thumbnails()
        self.make_selectors()
        gs = self._make_grid()
        self.children = [gs, self._move_rejects]

    def _make_grid(self):
        rows = len(self._selectors) // self.n_cols
        if len(self._selectors) % self.n_cols:
            rows += 1
        gs = ipw.GridspecLayout(rows, self.n_cols)
        for i in range(self.n_cols):
            for j in range(rows):
                index = i + j * self.n_cols
                if index >= len(self._selectors):
                    break
                gs[j, i] = self._selectors[index]

        return gs
