from typing import Dict
from numpy.typing import NDArray

import tensorflow as tf
import numpy as np

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

from code_loader.inner_leap_binder.leapbinder_decorators import (
    tensorleap_custom_metric,
    tensorleap_custom_loss,
)

from pixellot_tl.config import CONFIG


# TODO: add the focal loss as well (not supported in tf<2.12)
@tensorleap_custom_metric("state_metrics")
def get_state_metrics(
    gt_state: NDArray[np.float32],
    state_probs: NDArray[np.float32],
    state_mask: NDArray[np.float32],
) -> Dict[str, NDArray[np.float32]]:
    bs = gt_state.shape[0]
    valid_indices = state_mask.astype(bool).reshape((bs,))
    # If no valid entries, return default arrays
    if not np.any(valid_indices):
        # Return zeros or NaNs as a fallback.
        metrics_dict = {
            "state_accuracy": np.zeros((bs,), dtype=np.float32),
            "cce_loss": np.zeros((bs,), dtype=np.float32),
            # Add any other metrics here, also as zeros or NaNs
        }
        metrics_dict = {
            k: v + CONFIG["default_metric_value"] for k, v in metrics_dict.items()
        }

        return metrics_dict

    # Otherwise, compute metrics as normal
    pr_state = get_states_one_hot_vector(bs, state_probs)
    masked_state_probs = state_probs[valid_indices, ...]
    masked_gt_state = gt_state[valid_indices, ...]
    masked_pr_state = pr_state[valid_indices, ...]

    state_accuracy = CategoricalAccuracy()(masked_gt_state, masked_pr_state)
    cce_loss = CategoricalCrossentropy()(masked_gt_state, masked_state_probs)

    metrics_dict = {
        "state_accuracy": state_accuracy,
        "cce_loss": cce_loss,
    }
    metrics_dict = {k: np.asarray(v).reshape((bs,)) for k, v in metrics_dict.items()}
    return metrics_dict


@tensorleap_custom_metric("focus_metrics")
def get_focus_metrics(
    gt_focus: NDArray[np.float32],
    pr_focus: NDArray[np.float32],
    focus_mask: NDArray[np.float32],
) -> Dict[str, NDArray[np.float32]]:
    bs = gt_focus.shape[0]
    valid_indices = focus_mask.astype(bool).reshape((bs,))

    # If no valid entries, return default arrays
    if not np.any(valid_indices):
        metrics_dict = {
            "mse": np.zeros((bs,), dtype=np.float32),
            "mae": np.zeros((bs,), dtype=np.float32),
            "mse_dist": np.zeros((bs,), dtype=np.float32),
            "mae_dist": np.zeros((bs,), dtype=np.float32),
            "huber": np.zeros((bs,), dtype=np.float32),
        }
        metrics_dict = {
            k: v + CONFIG["default_metric_value"] for k, v in metrics_dict.items()
        }
        return metrics_dict

    # Otherwise, compute metrics as normal
    masked_pr_focus = pr_focus[valid_indices, ...]
    masked_gt_focus = gt_focus[valid_indices, ...]

    mse = MeanSquaredError()(masked_gt_focus, masked_pr_focus)
    mae = MeanAbsoluteError()(masked_gt_focus, masked_pr_focus)
    huber = Huber(delta=0.25)(masked_gt_focus, masked_pr_focus)

    error_norm = np.linalg.norm(masked_gt_focus - masked_pr_focus, axis=-1)
    mse_dist = (error_norm**2).mean()
    mae_dist = error_norm.mean()

    metrics_dict = {
        "mse": mse,
        "mae": mae,
        "mse_dist": mse_dist,
        "mae_dist": mae_dist,
        "huber": huber,
    }
    metrics_dict = {k: np.asarray(v).reshape((bs,)) for k, v in metrics_dict.items()}
    return metrics_dict


def get_states_one_hot_vector(bs, state_probs):
    states = np.zeros((bs, 4), dtype=int)
    state_ids = np.argmax(state_probs, axis=-1)
    states[np.arange(bs), state_ids] = 1
    return states


@tensorleap_custom_loss("dummy_loss")
def dummy_loss(state_pred, ball_pred, state_gt, ball_gt):
    return tf.constant(0.0)
