import glob
from pathlib import Path
import numpy as np
import scipy.ndimage as ndimage
from data.cell_crop import CellCrop
from PIL import Image
from skimage import io
import re


def read_channels(path):
    with open(path, 'r') as f:
        channels = f.read().strip().split('\n')
    return channels


def filter_channels(channels, blacklist=None):
    return [(i, c) for i, c in enumerate(channels) if c not in blacklist]


def load_data(fname, channels=[]) -> np.ndarray:
    fname = Path(fname)
    if fname.is_dir():
        if not channels:
            raise ValueError("Channels must be provided when loading from a directory.")

        images = []
        for idx, channel_name in channels:
            # Build a pattern to match files containing the channel name
            pattern = f"*{channel_name}*"
            # Find matching files in the directory
            matching_files = sorted([
                f for f in fname.glob(pattern)
                if f.suffix.lower() in ['.npz', '.tif', '.tiff'] and not f.name.startswith("._")
            ])
            if not matching_files:
                raise ValueError(f"No files found for channel '{channel_name}' in directory: {fname}")
            if len(matching_files) > 1:
                raise ValueError(f"Multiple files found for channel '{channel_name}' in directory: {fname}. Files: {matching_files}")
            file_path = matching_files[0]
            # Load the image from the file
            if file_path.suffix.lower() == ".npz":
                image = np.load(file_path, allow_pickle=True)['data']
            elif file_path.suffix.lower() in [".tif", ".tiff"]:
                image = io.imread(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            images.append(image)
        # Stack images along the last dimension
        image = np.stack(images, axis=-1)
        return image

    else:
        # Handle single file case
        if fname.suffix.lower() == ".npz":
            image = np.load(fname, allow_pickle=True)['data']
        elif fname.suffix.lower() in [".tif", ".tiff"]:
            image = io.imread(fname)
        else:
            raise ValueError(f"Unsupported file format: {fname}")

        if channels:
            # Extract channel indices
            channels_indices = [idx for idx, _ in channels]
            image = image[..., channels_indices]

        return image



def load_image(image_path, cells_path, cells2labels_path, channels=[], to_pad=False, crop_size=0):
    """
    Load an image, segmentation, and cell-to-label mapping from the provided paths.
    Handles single files or directories for `image_path` and `cells_path`.
    """

    # Load image and segmentation data
    image = load_data(image_path,channels)
    cells = load_data(cells_path).astype(np.int64)
    
    # Load cells-to-labels mapping
    if cells2labels_path.endswith(".npz"):
        cells2labels = np.load(cells2labels_path, allow_pickle=True)['data'].astype(np.int32)
    elif cells2labels_path.endswith(".txt"):
        with open(cells2labels_path, "r") as f:
            cells2labels = np.array(f.read().strip().split('\n')).astype(float).astype(int)
    else:
        raise ValueError(f"Unsupported file format: {cells2labels_path}")
    
    # Pad image and cells if required
    if to_pad:
        image = np.pad(image, ((crop_size // 2, crop_size // 2), (crop_size // 2, crop_size // 2), (0, 0)), 'constant')
        cells = np.pad(cells, ((crop_size // 2, crop_size // 2), (crop_size // 2, crop_size // 2)), 'constant')
    return image, cells, cells2labels


def _extend_slices_1d(slc, crop_size, max_size):
    """
    Extend a slice to be the size of crop size
    """
    d = crop_size - (slc.stop - slc.start)
    start = slc.start - (d // 2)
    stop = slc.stop + (d + 1) // 2
    if start < 0 or stop > max_size:
        raise Exception("Cell crop is out of bound of the image")
    return slice(start, stop)


def create_slices(slices, crop_size, bounds):
    """

    Args:
        slices: slices that bound the cell
        crop_size: the needed size of the crop
        bounds: shape of the image containing the cell

    Returns:
        new slices that bound the cell the size of crop_size
    """
    all_dim_slices = []
    for slc, cs, max_size in zip(slices, crop_size, bounds):
        all_dim_slices += [_extend_slices_1d(slc, cs, max_size)]
    return tuple(all_dim_slices)

def find_matching_files(directory, pattern):
    """Find files and directories in a directory matching a pattern using regex."""
    return [str(f) for f in Path(directory).glob("*") if re.match(pattern, f.name, re.IGNORECASE)]


def load_samples(images_dir, cells_dir, cells2labels_dir, images_names, crop_size, to_pad=False, channels=None):
    """

    Args:
        images_dir: path to the images
        cells_dir: path to the segmentation
        cells2labels_dir: path to mapping cells to labels
        images_names: names of images to load from the images_dir
        crop_size: the size of the crop of the cell
        channels: indices of channels to load from each image
    Returns:
        Array of CellCrop per cell in the dataset
    """
    images_dir = Path(images_dir)
    cells_dir = Path(cells_dir)
    cells2labels_dir = Path(cells2labels_dir)
    crops = []
    
    if not images_names:
        images_files = [f.stem for f in images_dir.glob("*.npz")] + \
                       [f.stem for f in images_dir.glob("*.tiff")] + \
                       [f.stem for f in images_dir.glob("*.tif")]
        images_dirs = [f.name for f in images_dir.glob("*") if f.is_dir()]
        images_names = list(set(images_files + images_dirs))  # Combine and deduplicate

        images_names = list(set(images_names))  # Remove duplicates in case of multiple extensions

    for image_id in images_names:
        # Regex pattern to match files starting with image_id (case-insensitive)
        image_pattern = rf"{re.escape(image_id)}.*(\.(npz|tiff|tif)|/)?$"
        cells_pattern = rf"{re.escape(image_id)}.*\.(npz|tiff|tif)$"
        cells2labels_pattern = rf"{re.escape(image_id)}.*\.(npz|txt)$"
        
        # Match files using regex
        image_path = find_matching_files(images_dir, image_pattern)
        cells_path = find_matching_files(cells_dir, cells_pattern)
        cells2labels_path = find_matching_files(cells2labels_dir, cells2labels_pattern)
    
        if not image_path or not cells_path or not cells2labels_path:
            continue  # Skip if no match is found
    
        image, cells, cl2lbl = load_image(image_path=image_path[0],
                                          cells_path=cells_path[0],
                                          cells2labels_path=cells2labels_path[0],
                                          channels=channels,
                                          to_pad=to_pad,
                                          crop_size=crop_size)

        objs = ndimage.find_objects(cells)
        for cell_id, obj in enumerate(objs, 1):
            try:
                slices = create_slices(obj, crop_size, cells.shape)
                label = cl2lbl[cell_id]
                crops.append(
                    CellCrop(cell_id=cell_id,
                             image_id=image_id,
                             label=label,
                             slices=slices,
                             cells=cells,
                             image=image))
            except Exception as e:
                pass
    return np.array(crops)


def load_crops(root_dir,
               channels_path,
               crop_size,
               train_set,
               val_set,
               to_pad=False,
               blacklist_channels=[]):
    """
    Given paths to the data, generate crops for all the cells in the data
    Args:
        root_dir:
        channels_path:
        crop_size: size of the environment to keep for each cell
        train_set: name of images to train on
        val_set: name of images to validate on
        to_pad: whether to pad the image with zeros in order to work on cell on the border
        blacklist_channels: channels to not use in the training/validation
    Returns:
        train_crops - list of crops from the train set
        val_crops - list of crops from the validation set
    """
    cell_types_dir = Path(root_dir) / "CellTypes"
    data_dir = cell_types_dir / "data" / "images"
    cells_dir = cell_types_dir / "cells"
    cells2labels_dir = cell_types_dir / "cells2labels"

    channels = read_channels(channels_path)
    channels_filtered = filter_channels(channels, blacklist_channels)
    # channels = [i for i, c in channels_filtered]
    print(channels)
    print('Load training data...')
    train_crops = load_samples(images_dir=data_dir, cells_dir=cells_dir, cells2labels_dir=cells2labels_dir,
                               images_names=train_set, crop_size=crop_size, to_pad=to_pad,
                               channels=channels_filtered)

    print('Load validation data...')
    val_crops = load_samples(images_dir=data_dir, cells_dir=cells_dir, cells2labels_dir=cells2labels_dir,
                             images_names=val_set, crop_size=crop_size, to_pad=to_pad,
                             channels=channels_filtered)

    return train_crops, val_crops
