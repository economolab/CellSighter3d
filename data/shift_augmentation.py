from torchvision.transforms import Lambda, RandomCrop, CenterCrop
import numpy as np
import torch
import torch
import numpy as np
import torch
import numpy as np

class ShiftAugmentation(torch.nn.Module):
    """
    Augmentation that shifts each marker channel a few pixels in random directions for 3D data.
    """
    def __init__(self, n_size, shift_max=0):
        """
        Args:
            n_size (tuple or int): Target size (D, H, W) for the cropped output.
            shift_max (int): Maximum shift in any direction.
        """
        super(ShiftAugmentation, self).__init__()
        self.shift_max = shift_max
        self.p = 0.3  # Probability of applying a shift

        # Ensure n_size is a tuple (D, H, W)
        if isinstance(n_size, int):
            self.n_size = (n_size, n_size, n_size)
        elif isinstance(n_size, (list, tuple)) and len(n_size) == 3:
            self.n_size = tuple(n_size)
        else:
            raise ValueError(f"Invalid n_size: {n_size}. Must be int or tuple of length 3.")

    def center_crop_3d(self, x, crop_size):
        """
        Center crop a 3D tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (D, H, W).
            crop_size (tuple): Target size (crop_d, crop_h, crop_w).
        
        Returns:
            torch.Tensor: Center-cropped tensor.
        """
        d, h, w = x.shape
        crop_d, crop_h, crop_w = crop_size
        start_d = (d - crop_d) // 2
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2

        return x[start_d:start_d + crop_d, start_h:start_h + crop_h, start_w:start_w + crop_w]

    def random_crop_3d(self, x, crop_size):
        """
        Random crop a 3D tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (D, H, W).
            crop_size (tuple): Target size (crop_d, crop_h, crop_w).
        
        Returns:
            torch.Tensor: Randomly cropped tensor.
        """
        d, h, w = x.shape
        crop_d, crop_h, crop_w = crop_size

        start_d = np.random.randint(0, max(1, d - crop_d + 1))
        start_h = np.random.randint(0, max(1, h - crop_h + 1))
        start_w = np.random.randint(0, max(1, w - crop_w + 1))

        return x[start_d:start_d + crop_d, start_h:start_h + crop_h, start_w:start_w + crop_w]

    def forward(self, x):
        """
        Apply shift augmentation to a 3D tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (C, D, H, W).
        
        Returns:
            torch.Tensor: Augmented tensor of shape (C, n_size[0], n_size[1], n_size[2]).
        """
        # Create output tensor on the same device as input
        aug_x = torch.zeros((x.shape[0], *self.n_size), device=x.device)

        for i in range(x.shape[0]):
            # Determine dynamic crop size with optional shift
            crop_size = (
                self.n_size[0] + (self.shift_max if np.random.random() < self.p else 0),
                self.n_size[1] + (self.shift_max if np.random.random() < self.p else 0),
                self.n_size[2] + (self.shift_max if np.random.random() < self.p else 0),
            )

            # Apply center crop followed by random crop
            cropped = self.center_crop_3d(x[i], crop_size)
            shifted = self.random_crop_3d(cropped, self.n_size)

            # Assign to output
            aug_x[i,...] = shifted

        return aug_x
