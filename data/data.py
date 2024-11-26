import numpy as np
from torch.utils.data import Dataset


class CellCropsDataset(Dataset):
    def __init__(self,
                 crops,
                 mask=False,
                 transform=None):
        super().__init__()
        self._crops = crops
        self._transform = transform
        self._mask = mask

    def __len__(self):
        return len(self._crops)

    def __getitem__(self, idx):
        # Get the sample from the dataset
        sample = self._crops[idx].sample(self._mask)
        
        # Extract components
        image = sample['image']  # Shape: (Z, H, W, C)
        all_cells_mask = sample['all_cells_mask']  # Shape: (Z, H, W)
        mask = sample['mask']  # Shape: (Z, H, W)
        
        # Check for shape compatibility
        if image.shape[:3] != all_cells_mask.shape or image.shape[:3] != mask.shape:
            raise ValueError(f"Shape mismatch: image: {image.shape[:3]}, "
                             f"all_cells_mask: {all_cells_mask.shape}, mask: {mask.shape}")
        
        # Add a channel dimension to masks to make them (Z, H, W, 1)
        all_cells_mask = all_cells_mask[..., np.newaxis]
        mask = mask[..., np.newaxis]
        
        # Stack along the channel dimension
        combined = np.concatenate([image, all_cells_mask, mask], axis=-1)  # Shape: (Z, H, W, C + 2)
        
        # Apply transformation
        aug = self._transform(combined).float()  # Ensure _transform handles (Z, H, W, C)
    
        # Split augmented data back into components
        sample['image'] = aug[:-1,...]  # All original channels except last two
        sample['mask'] = aug[-1:,...]  # Last channel is the mask
    
        return sample

