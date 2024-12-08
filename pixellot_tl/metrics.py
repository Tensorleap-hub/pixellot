from typing import Dict
from numpy.typing import NDArray
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

from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_custom_metric


# TODO: add the focal loss as well (not supported in tf<2.12)
@tensorleap_custom_metric("state_metrics")
def get_state_metrics(
    gt_state: NDArray[np.float32],
    state_probs: NDArray[np.float32],
    state_mask: NDArray[np.float32],
) -> Dict[str, NDArray[np.float32]]:
    # Filter out invalid states
    bs = gt_state.shape[0]
    pr_state = get_states_one_hot_vector(bs, state_probs)
    masked_state_probs = state_probs[state_mask.astype(bool)]
    masked_gt_state = gt_state[state_mask.astype(bool)]
    masked_pr_state = pr_state[state_mask.astype(bool)]

    state_accuracy = CategoricalAccuracy()(masked_gt_state, masked_pr_state)
    cce_loss = CategoricalCrossentropy()(masked_gt_state, masked_state_probs)
    # Label smoothing was chosen during hyperparameter tuning
    # focal_loss = CategoricalFocalCrossentropy(label_smoothing=0.5)(
    #     masked_gt_state, masked_state_probs
    # )
    metrics_dict = {
        "state_accuracy": state_accuracy,
        "cce_loss": cce_loss,
        # "focal_loss": focal_loss.numpy(),
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
    # Filter out invalid focus
    masked_pr_focus = pr_focus[focus_mask.astype(bool)]
    masked_gt_focus = gt_focus[focus_mask.astype(bool)]

    mse = MeanSquaredError()(masked_gt_focus, masked_pr_focus)
    mae = MeanAbsoluteError()(masked_gt_focus, masked_pr_focus)
    # Huber loss delta was chosen during hyperparameter tuning
    huber = Huber(0.25)(masked_gt_focus, masked_pr_focus)

    # Since MSE and MAE are computed on 2D, we need to use distance formula
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
