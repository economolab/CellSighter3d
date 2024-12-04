import cv2
import numpy as np
import torchvision
from torchvision.transforms import Lambda
from data.shift_augmentation import ShiftAugmentation
import scipy.ndimage
import torchvision.transforms as transforms

import torch

class ToTensor3D:
    def __call__(self, x):
        # Convert a 4D array (Z, H, W, C) to a PyTorch tensor (C, Z, H, W).
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input, got {x.ndim}D input")
        
        x = torch.tensor(x).permute(3,0,1,2)  # Move channels to the first dimension
        return x

def poisson_sampling(x):
    """
    Apply Gaussian blur and Poisson sampling for 3D volumetric data.

    Args:
        x: A 4D array of shape (Z, H, W, C) where C includes markers and masks.
    Returns:
        Augmented tensor with Poisson noise applied to blurred data.
    """
    for z in range(x.shape[0]):  # Iterate over the Z dimension
        blur = cv2.GaussianBlur(x[z, :, :, :-2], (5, 5), 0)  # Apply 2D blur to each slice
        x[z, :, :, :-2] = np.random.poisson(lam=blur, size=blur.shape)  # Apply Poisson sampling
    return x



def cell_shape_aug(x):
    """
    Augment the mask of the cell size by dilating the size of the cell with random kernel
    """
    if np.random.random() < 0.5:
        cell_mask = x[:, :, -1]
        kernel_size = np.random.choice([2, 3, 5])
        kernel = np.ones(kernel_size, np.uint8)
        img_dilation = cv2.dilate(cell_mask, kernel, iterations=1)
        x[:, :, -1] = img_dilation
    return x

class CenterCrop3D:
    def __init__(self, crop_size):
        """
        Args:
            crop_size (tuple): (depth, height, width) or single int for all dimensions.
        """
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size, crop_size)
        elif isinstance(crop_size, tuple) and len(crop_size) == 3:
            self.crop_size = crop_size
        elif isinstance(crop_size, list) and len(crop_size) == 3:
            self.crop_size = tuple(crop_size)
        else:
            raise ValueError(f"Invalid crop_size: {crop_size}")

    def __call__(self, x):
        """
        Args:
            x: Tensor of shape (C, D, H, W)
        Returns:
            Tensor of cropped shape.
        """
        _, d, h, w = x.shape
        crop_d, crop_h, crop_w = self.crop_size

        start_d = (d - crop_d) // 2
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2

        return x[:, start_d:start_d + crop_d, start_h:start_h + crop_h, start_w:start_w + crop_w]


class RandomRotation3D:
    def __init__(self, degrees):
        self.degrees = degrees  # (min_degree, max_degree) for rotation range

    def __call__(self, x):
        """
        Args:
            x: Tensor of shape (C, D, H, W)
        Returns:
            Rotated tensor.
        """
        angle = np.random.uniform(*self.degrees)
        rotated = torch.zeros_like(x)
        for c in range(x.shape[0]):  # Rotate each channel independently
            rotated[c] = torch.tensor(
                scipy.ndimage.rotate(x[c].numpy(), angle, axes=(1, 2), reshape=False, mode='nearest')
            )
        return rotated



def env_shape_aug(x):
    """
        Augment the size of the cells mask in the environment,
        by dilating the size of the cell with random kernel
    """
    if np.random.random() < 0.5:
        cell_mask = x[:, :, -2]
        kernel_size = np.random.choice([2, 3, 5])
        kernel = np.ones(kernel_size, np.uint8)
        img_dilation = cv2.dilate(cell_mask, kernel, iterations=1)
        x[:, :, -2] = img_dilation
    return x


class RandomFlip3D:
    def __init__(self, flip_axes=(0, 1, 2), flip_prob=0.5):
        """
        Args:
            flip_axes: Axes to consider for flipping. Default is all 3 axes.
            flip_prob: Probability of flipping along each axis.
        """
        self.flip_axes = flip_axes
        self.flip_prob = flip_prob

    def __call__(self, x):
        """
        Args:
            x: Tensor of shape (C, D, H, W)
        Returns:
            Flipped tensor.
        """
        for axis in self.flip_axes:
            if np.random.rand() < self.flip_prob:
                x = torch.flip(x, dims=(axis,))  # +1 because torch.flip expects channel-first tensors
        return x


def create_val_transform(crop_size):
    return transforms.Compose([
        ToTensor3D(),
        CenterCrop3D(crop_size=crop_size),
    ])

# Define this function globally
def shift_augmentation_with_probability(x, shift, crop_size):
    if np.random.random() < 0.5:
        return ShiftAugmentation(shift_max=shift, n_size=crop_size)(x)
    return x

class ShiftAugmentationWithProbability:
    def __init__(self, shift, crop_size):
        self.shift = shift
        self.crop_size = crop_size

    def __call__(self, x):
        if np.random.random() < 0.5:
            return ShiftAugmentation(shift_max=self.shift, n_size=self.crop_size)(x)
        return x


def create_train_transform(crop_size, shift):
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size, crop_size)  # Ensure 3D crop size

    return torchvision.transforms.Compose([
        torchvision.transforms.Lambda(poisson_sampling),        # Custom 3D poisson sampling
        # torchvision.transforms.Lambda(cell_shape_aug),          # Custom cell augmentation
        # torchvision.transforms.Lambda(env_shape_aug),           # Custom environment augmentation
        ToTensor3D(),                                           # Convert 4D numpy array to PyTorch tensor
        RandomRotation3D(degrees=(0, 360)),                     # Random 3D rotations
        ShiftAugmentationWithProbability(shift=shift, crop_size=crop_size),  # Shift augmentation
        CenterCrop3D(crop_size=crop_size),                      # Center crop for 3D
        RandomFlip3D(flip_axes=(1, 2,3), flip_prob=0.75),         # Random horizontal/vertical flips
    ])


