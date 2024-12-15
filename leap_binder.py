from typing import Dict, List, Union

import numpy as np

# Tensorleap imports
from code_loader import leap_binder
from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.contract.enums import Metric, LeapDataType
from code_loader.inner_leap_binder.leapbinder_decorators import (
    tensorleap_preprocess,
    tensorleap_input_encoder,
    tensorleap_gt_encoder,
    tensorleap_metadata,
)

from pixellot_tl.utils.data_utils import load_dataset
from pixellot_tl.metrics import get_state_metrics, get_focus_metrics, dummy_loss


@tensorleap_preprocess()
def preprocess_func() -> List[PreprocessResponse]:
    (
        train_inputs,
        train_gt_state,
        train_gt_focus,
        train_state_mask,
        train_focus_mask,
        train_clip_name2id_map,
        train_sample_clip_ids,
    ) = load_dataset("train_dir")

    train = PreprocessResponse(
        length=16,
        data={
            "inputs": train_inputs,
            "gt_state": train_gt_state,
            "gt_focus": train_gt_focus,
            "state_mask": train_state_mask,
            "focus_mask": train_focus_mask,
            "clip_name2id_map": train_clip_name2id_map,
            "sample_clip_ids": train_sample_clip_ids,
        },
    )
    val = PreprocessResponse(
        length=16,
        data={
            "inputs": train_inputs,
            "gt_state": train_gt_state,
            "gt_focus": train_gt_focus,
            "state_mask": train_state_mask,
            "focus_mask": train_focus_mask,
            "clip_name2id_map": train_clip_name2id_map,
            "sample_clip_ids": train_sample_clip_ids,
        },
    )
    responses = [train, val]
    return responses


@tensorleap_input_encoder("input")
def input_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    return preprocess.data["inputs"][idx].astype(np.float32)


@tensorleap_gt_encoder("state")
def gt_state_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    return preprocess.data["gt_state"][idx].astype(np.float32)


@tensorleap_gt_encoder("focus")
def gt_focus_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    return preprocess.data["gt_focus"][idx].astype(np.float32)


@tensorleap_gt_encoder("is_valid_state")
def gt_state_mask_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    return np.asarray(preprocess.data["state_mask"][idx].astype(np.float32))


@tensorleap_gt_encoder("is_valid_focus")
def gt_focus_mask_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    return np.asarray(preprocess.data["focus_mask"][idx].astype(np.float32))


@tensorleap_metadata("clip_metadta")
def get_clip_metadtata(
    idx: int, preprocess: PreprocessResponse
) -> Dict[str, Union[str, int, float]]:
    clip_name2id_map = preprocess.data["clip_name2id_map"]
    clip_id2name = {v: k for k, v in clip_name2id_map.items()}
    sample_clip_id = preprocess.data["sample_clip_ids"][idx]
    sample_clip_name = clip_id2name[sample_clip_id]
    return {"clip_name": sample_clip_name, "clip_id": sample_clip_id}


if __name__ == "__main__":
    leap_binder.check()

# TODO: Implement the following functions
"""
- support focal loss
- visualization ?
"""
