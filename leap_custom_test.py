import onnxruntime as ort
import numpy as np

from leap_binder import (
    preprocess_func,
    input_encoder,
    gt_state_encoder,
    gt_focus_encoder,
    gt_state_mask_encoder,
    gt_focus_mask_encoder,
    get_clip_metadtata,
    get_state_metrics,
    get_focus_metrics,
)


def check_integration():
    # load onnx model (TODO: load h5 model)
    onnx_path = "models/model.onnx"
    ort_session = ort.InferenceSession(onnx_path)

    # load data
    responses = preprocess_func()
    train = responses[0]
    idx = 0  # set index
    inputs = input_encoder(idx, train)
    gt_state = gt_state_encoder(idx, train)
    gt_focus = gt_focus_encoder(idx, train)
    state_mask = gt_state_mask_encoder(idx, train)
    focus_mask = gt_focus_mask_encoder(idx, train)
    clip_metadata = get_clip_metadtata(idx, train)
    # add batch dimension
    batched_inputs = np.expand_dims(inputs, axis=0)
    batched_gt_state = np.expand_dims(gt_state, axis=0)
    batched_gt_focus = np.expand_dims(gt_focus, axis=0)
    batched_state_mask = np.expand_dims(state_mask, axis=0)
    batched_focus_mask = np.expand_dims(focus_mask, axis=0)
    # run inference
    state_probs, pr_focus = ort_session.run(None, {"input_name": batched_inputs})
    # compute metrics
    state_metrics = get_state_metrics(batched_gt_state, state_probs, state_mask)
    focus_metrics = get_focus_metrics(batched_gt_focus, pr_focus, focus_mask)

    print("Custom Test Done!")


if __name__ == "__main__":
    check_integration()
