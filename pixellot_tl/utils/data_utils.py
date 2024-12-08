import json
import os

import numpy as np

from pixellot_tl.config import CONFIG
from pixellot_tl.utils.aws_utils import download


def load_dataset(subset_dir: str):
    if subset_dir not in ["train_dir", "val_dir", "test_dir"]:
        raise ValueError("data_dir must be one of 'train_dir', 'val_dir', 'test_dir'")

    subset_s3_path = os.path.join(CONFIG["data_dir"], CONFIG[subset_dir])

    inputs = np.load(download(f"{subset_s3_path}/inputs.npy")).astype(np.float32)

    gt_state = np.load(download(f"{subset_s3_path}/states.npy")).astype(np.int32)
    gt_focus = np.load(download(f"{subset_s3_path}/focus.npy")).astype(np.float32)

    # Masks tell which rows in gt_state and gt_focus are valid
    state_mask = np.load(download(f"{subset_s3_path}/state_mask.npy")).astype(bool)
    focus_mask = np.load(download(f"{subset_s3_path}/focus_mask.npy")).astype(bool)

    metadata_local_path = download(f"{subset_s3_path}/clip_name2id_map.json")
    with open(metadata_local_path) as f:
        clip_name2id_map = json.load(f)

    # Stores clip id for each data sample
    sample_clip_ids = np.load(download(f"{subset_s3_path}/sample_clip_ids.npy"))

    return (
        inputs,
        gt_state,
        gt_focus,
        state_mask,
        focus_mask,
        clip_name2id_map,
        sample_clip_ids,
    )
