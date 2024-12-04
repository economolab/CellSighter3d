import argparse
import sys

sys.path.append(".")
from torch.utils.tensorboard import SummaryWriter
import torch
import os
import numpy as np
from model import Model
from data.data import CellCropsDataset
from data.utils import load_crops,read_channels
from data.transform import create_val_transform
from torch.utils.data import DataLoader
from metrics.metrics import Metrics
import json
import napari
from pathlib import Path
import glob
import numpy as np
import scipy.ndimage as ndimage
from data.cell_crop import CellCrop
from PIL import Image
from skimage import io
import re
import pandas


def view_cell(cell,root_dir,channels):
    cell_types_dir = Path(root_dir) / "CellTypes"
    data_dir = cell_types_dir / "data" / "images"
    cells_dir = cell_types_dir / "cells"
    cells2labels_dir = cell_types_dir / "cells2labels"
    geneAnnotations = cell_types_dir / "genes"
    image_id = cell['image_id'][0]
    
    for idx, channel_name in channels:
        # Build a pattern to match files containing the channel name
        pattern = f"*{image_id}*{channel_name}*"
        # Find matching files in the directory
        matching_files = sorted([
            f for f in geneAnnotations.glob(pattern)
            if f.suffix.lower() in ['.csv'] and not f.name.startswith("._")
        ])
        if not matching_files:
            raise ValueError(f"No files found for channel '{channel_name}' in directory: {geneAnnotations}")
        if len(matching_files) > 1:
            raise ValueError(f"Multiple files found for channel '{channel_name}' in directory: {geneAnnotations}. Files: {matching_files}")
        file_path = matching_files[0]
        # Load the image from the file
        if file_path.suffix.lower() == ".csv":
            geneInfo = pandas.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
    

def val_epoch(model, dataloader, device=None):
    with torch.no_grad():
        model.eval()
        results = []
        cells = []
        for i, batch in enumerate(dataloader):
            x = batch['image']
            m = batch.get('mask', None)
            if m is not None:
                x = torch.cat([x, m], dim=1)
            x = x.to(device=device)
            m = m.to(device=device)
            y_pred = model(x)
            results += y_pred.detach().cpu().numpy().tolist()

            # del batch["image"]
            cells += [batch]
            if i % 500 == 0:
                print(f"Eval {i} / {len(dataloader)}")
        return cells, np.array(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('--base_path', type=str,
                        help='configuration_path')

    args = parser.parse_args()
    writer = SummaryWriter(log_dir=args.base_path)

    config_path = os.path.join(args.base_path, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    _, val_crops = load_crops(config["root_dir"],
                            config["channels_path"],
                            config["crop_size"],
                            ["zxcv"],
                            config["val_set"],
                            config["to_pad"],
                            blacklist_channels=config["blacklist"])
    crop_input_size = config["crop_input_size"] if "crop_input_size" in config else 100
    val_transform_fn = create_val_transform(crop_input_size)
    val_dataset = CellCropsDataset(val_crops, transform=val_transform_fn, mask=True)
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        
    num_channels = sum(1 for line in open(config["channels_path"])) + 1 - len(config["blacklist"])
    class_num = config["num_classes"]

    model = Model(num_channels+1, class_num)
    weight_files = [f for f in os.listdir(args.base_path) if f.startswith("weights_") and f.endswith("_count.pth")]
    if weight_files:
        latest_weight_file = max(weight_files, key=lambda x: int(x.split("_")[1]))
        latestDex = int(latest_weight_file.split("_")[1])
        model.load_state_dict(torch.load(os.path.join(args.base_path, latest_weight_file), map_location=device))
        print(f"Loaded model weights from {latest_weight_file}")
    else:
        print("No pre-trained model found. Starting from scratch.")
        eval_weights = config["weight_to_eval"]
        model.load_state_dict(torch.load(eval_weights))
    model = model.to(device=device)

    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"],
                            num_workers=config["num_workers"], shuffle=False, pin_memory=True)
    cells, results = val_epoch(model, val_loader, device=device)
    channels = read_channels(config['channels_path'])
    
    metrics = Metrics(
        [],
        writer,
        prefix="val")
    metrics(cells, results, 0)
    metrics.save_results(os.path.join(args.base_path, f"val_results.csv"), cells, results)
