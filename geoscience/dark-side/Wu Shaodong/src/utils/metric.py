import numpy as np


def get_dice(gt_mask, pred_mask):
    # masks should be binary
    # DICE Score = (2 * Intersection) / (Area of Set A + Area of Set B)
    intersect = np.sum(pred_mask * gt_mask)
    total_sum = np.sum(pred_mask) + np.sum(gt_mask)
    if total_sum == 0:  # both samples are without positive masks
        dice = 1.0
    else:
        dice = (2 * intersect) / total_sum
    return dice


def get_submission_score(
    gt_submission_path, prediction_submission_path, mask_shape=(300, 300, 1259)
):
    # load submissions
    gt_submission = dict(np.load(gt_submission_path))
    prediction_submission = dict(np.load(prediction_submission_path))

    # prepare place to store per sample score
    global_scores = []
    for sample_id in gt_submission.keys():
        # reconstruct gt mask
        gt_mask = np.zeros(mask_shape)
        gt_coordinates = gt_submission[sample_id]
        if gt_coordinates.shape[0] > 0:
            gt_mask[
                gt_coordinates[:, 0], gt_coordinates[:, 1], gt_coordinates[:, 2]
            ] = 1

        # reconstruct prediction mask
        pred_mask = np.zeros(mask_shape)
        pred_coordinates = prediction_submission[sample_id]
        if pred_coordinates.shape[0] > 0:
            pred_mask[
                pred_coordinates[:, 0], pred_coordinates[:, 1], pred_coordinates[:, 2]
            ] = 1

        global_scores.append(get_dice(gt_mask, pred_mask))

    sub_score = sum(global_scores) / len(global_scores)

    return sub_score