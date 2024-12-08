import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import tensorflow as tf
from tensorflow import keras
from keras.losses import (
    CategoricalCrossentropy,
    # CategoricalFocalCrossentropy,
    Huber,
)
from keras.metrics import (
    CategoricalAccuracy,
    MeanAbsoluteError,
    MeanSquaredError,
)


def load_data(src_path: Path):
    inputs = np.load(f"{src_path}/inputs.npy").astype(np.float32)

    gt_state = np.load(f"{src_path}/states.npy").astype(np.int32)
    gt_focus = np.load(f"{src_path}/focus.npy").astype(np.float32)

    # Masks tell which rows in gt_state and gt_focus are valid
    state_mask = np.load(f"{src_path}/state_mask.npy").astype(bool)
    focus_mask = np.load(f"{src_path}/focus_mask.npy").astype(bool)

    return inputs, gt_state, gt_focus, state_mask, focus_mask


def load_metadata(src_path: Path):
    with open(f"{src_path}/clip_name2id_map.json") as f:
        clip_name2id_map = json.load(f)

    # Stores clip id for each data sample
    sample_clip_ids = np.load(f"{src_path}/sample_clip_ids.npy")

    return clip_name2id_map, sample_clip_ids


def metadata_usage_example(clip_name2id_map, sample_clip_ids, clip_id=1):
    print("===== Metadata usage example =====")

    # Look for clip name by clip_id
    clip_name = None
    for name, idx in clip_name2id_map.items():
        if idx == clip_id:
            clip_name = name
            break
    else:
        print(f"Clip with id {clip_id} not found")
        return

    # Get all samples that belong to the clip
    clip_samples = sample_clip_ids[sample_clip_ids == clip_id]

    print(f"Clip name: {clip_name}, Clip id: {clip_id}")
    print(f"Number of samples in the clip: {len(clip_samples)}")


def run_batches(inputs, ort_session, batch_size=512):
    print("===== Running batches =====")

    state_probs = np.zeros((len(inputs), 4), dtype=float)
    pr_focus = np.zeros((len(inputs), 2), dtype=float)

    def run_single_batch(batch, batch_slice):
        probs, focus = ort_session.run(None, {"input_name": batch})
        state_probs[batch_slice] = probs
        pr_focus[batch_slice] = focus

    # Split input into batches to avoid memory issues
    total_batches = len(inputs) // batch_size
    for batch_id in range(total_batches):
        print(f"Processing batch {batch_id + 1} of {total_batches}", end="\r")
        batch_slice = slice(batch_id * batch_size, (batch_id + 1) * batch_size)
        batch = inputs[batch_slice]
        run_single_batch(batch, batch_slice)

    # Process last batch
    # For simplicity ignore that it might be smaller than batch_size
    batch_slice = slice(-batch_size, None)
    batch = inputs[batch_slice]
    run_single_batch(batch, batch_slice)

    # State one hot vector
    states = np.zeros((len(inputs), 4), dtype=int)
    state_ids = np.argmax(state_probs, axis=-1)
    states[np.arange(len(inputs)), state_ids] = 1

    return states, state_probs, pr_focus


def get_state_metrics(gt_state, pr_state, state_probs, state_mask):
    # Filter out invalid states
    masked_state_probs = state_probs[state_mask]
    masked_gt_state = gt_state[state_mask]
    masked_pr_state = pr_state[state_mask]

    state_accuracy = CategoricalAccuracy()(masked_gt_state, masked_pr_state)
    cce_loss = CategoricalCrossentropy()(masked_gt_state, masked_state_probs)
    # Label smoothing was chosen during hyperparameter tuning
    # focal_loss = CategoricalFocalCrossentropy(label_smoothing=0.5)(
    #     masked_gt_state, masked_state_probs
    # )

    return state_accuracy, cce_loss  # , focal_loss


def get_focus_metrics(gt_focus, pr_focus, focus_mask):
    # Filter out invalid focus
    masked_pr_focus = pr_focus[focus_mask]
    masked_gt_focus = gt_focus[focus_mask]

    mse = MeanSquaredError()(masked_gt_focus, masked_pr_focus)
    mae = MeanAbsoluteError()(masked_gt_focus, masked_pr_focus)
    # Huber loss delta was chosen during hyperparameter tuning
    huber = Huber(0.25)(masked_gt_focus, masked_pr_focus)

    # Since MSE and MAE are computed on 2D, we need to use distance formula
    error_norm = np.linalg.norm(masked_gt_focus - masked_pr_focus, axis=-1)
    mse_dist = (error_norm**2).mean()
    mae_dist = error_norm.mean()

    return mse, mae, mse_dist, mae_dist, huber


def main():
    onnx_path = "models/model.onnx"
    data_root = "/Users/ranhomri/tensorleap/data/pixellot-assets-bucket-fgkhjab/american_football/train_data"

    ort_session = ort.InferenceSession(onnx_path)
    inputs, gt_state, gt_focus, state_mask, focus_mask = load_data(data_root)

    clip_name2id_map, sample_clip_ids = load_metadata(data_root)
    metadata_usage_example(clip_name2id_map, sample_clip_ids)

    pr_state, state_probs, pr_focus = run_batches(inputs, ort_session)

    print("===== Metrics computation =====")
    # state_accuracy, cce_loss, focal_loss = get_state_metrics(
    #     gt_state, pr_state, state_probs, state_mask
    # )
    state_accuracy, cce_loss = get_state_metrics(
        gt_state, pr_state, state_probs, state_mask
    )
    mse, mae, mse_np, mae_np, huber = get_focus_metrics(gt_focus, pr_focus, focus_mask)

    print(
        "===== State Metrics =====",
        f"State accuracy:    {state_accuracy}",
        f"State CCE loss:    {cce_loss}",
        # f"State Focal loss:  {focal_loss}",
        sep="\n",
        end="\n\n",
    )
    print(
        "===== Focus Metrics =====",
        f"Focus MSE loss:        {mse}",
        f"Focus MAE loss:        {mae}",
        f"Focus Huber loss:      {huber}",
        f"Focus MSE (distance):  {mse_np}",
        f"Focus MAE (distance):  {mae_np}",
        sep="\n",
        end="\n\n",
    )


if __name__ == "__main__":
    main()
