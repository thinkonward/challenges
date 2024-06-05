import numpy as np
import torch
import segmentation_models_pytorch as smp
from pytorch_toolbelt import losses
from torch import nn
from functools import partial


def get_model(name, params):
    if name == "unet":
        return smp.Unet(**params)
    else:
        raise ValueError("Invalid model name")


def get_loss(name: str, mode: str = None, params: dict = {}):
    """
    Constructs specified by name binary of multiclass loss
    with the defined params
    """
    losses_dict = {
        "focal": get_focal(mode=mode),
        "dice": partial(losses.DiceLoss, mode=mode),
        "bce": nn.BCEWithLogitsLoss,
        "cross_entropy": losses.SoftCrossEntropyLoss,
    }

    return losses_dict[name](**params)


def get_focal(mode):
    """
    Get the appropriate Focal Loss function based on the specified mode.

    Parameters:
    - mode (str): The mode specifying the type of Focal Loss function to retrieve.
                 It can be either 'binary' or 'multiclass'.

    Returns:
    - Focal Loss function: An instance of the Focal Loss function corresponding
                           to the specified mode.

    Raises:
    - ValueError: If the specified mode is not 'binary' or 'multiclass'.
                  In such cases, an appropriate error message is raised.

    Example:
    >>> focal_loss_func = get_focal('binary')
    >>> output = focal_loss_func(predictions, targets)
    """

    if mode == "binary":
        return losses.BinaryFocalLoss
    elif mode == "multiclass":
        return losses.FocalLoss
    else:
        raise ValueError("Mode {} is not supported".format(mode))


def get_metric(name: str, params: dict):
    return partial(multiclass_iou_dice_score, metric=name, **params)


def binary_iou_dice_score(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    metric="iou",
    apply_sigmoid: bool = True,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> float:
    """
    Compute the IoU (Intersection over Union) or Dice coefficient score for binary segmentation.

    Parameters:
    - y_pred (torch.Tensor): Predicted binary segmentation mask with shape matching y_true.
    - y_true (torch.Tensor): Ground truth binary segmentation mask with shape matching y_pred.
    - metric (str): Specifies the metric to compute, either 'iou' for Intersection over Union
                    or 'dice' for Dice coefficient. Default is 'iou'.
    - apply_sigmoid (bool): Whether to apply sigmoid activation to y_pred before thresholding.
                            Default is True.
    - threshold (float): Threshold value for binarization of y_pred. Default is 0.5.
    - eps (float): Small constant to avoid division by zero. Default is 1e-7.

    Returns:
    - float: The computed IoU or Dice coefficient score.

    Raises:
    - AssertionError: If the metric is not 'iou' or 'dice', or if y_pred and y_true have
                      different shapes.
    """

    assert metric in {"iou", "dice"}
    assert y_pred.shape == y_true.shape

    # Apply sigmoid if needed
    if apply_sigmoid:
        y_pred = torch.sigmoid(y_pred)

    # Make binary predictions
    y_pred = (y_pred > threshold).type(y_true.dtype)

    intersection = torch.sum(y_pred * y_true).item()
    cardinality = (torch.sum(y_pred) + torch.sum(y_true)).item()

    if metric == "iou":
        score = (intersection + eps) / (cardinality - intersection + eps)
    else:
        score = (2.0 * intersection + eps) / (cardinality + eps)

    return score


def multiclass_iou_dice_score(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    metric="iou",
    threshold: float = 0.35,
    eps=1e-7,
):
    """
    Compute the mean Intersection over Union (IoU) or Dice coefficient score for multiclass segmentation.

    Parameters:
    - y_pred (torch.Tensor): Predicted multiclass segmentation map with shape (batch_size, num_classes, height, width).
    - y_true (torch.Tensor): Ground truth multiclass segmentation map with shape (batch_size, height, width).
    - metric (str): Specifies the metric to compute, either 'iou' for Intersection over Union or 'dice' for Dice coefficient.
                    Default is 'iou'.
    - threshold (float): Threshold value for binarization of class predictions in y_pred. Default is 0.35.
    - eps (float): Small constant to avoid division by zero. Default is 1e-7.

    Returns:
    - float: The mean IoU or Dice coefficient score across all classes.

    Raises:
    - AssertionError: If the metric is not 'iou' or 'dice', or if y_pred and y_true have
                      incompatible shapes.

    """
    scores = []
    num_classes = y_pred.shape[1]
    y_pred_ = y_pred.log_softmax(dim=1).exp()

    for class_index in range(num_classes):
        y_pred_i = y_pred_[:, class_index, :, :]
        y_true_i = y_true == class_index

        score = binary_iou_dice_score(
            y_pred=y_pred_i,
            y_true=y_true_i,
            metric=metric,
            apply_sigmoid=False,
            threshold=threshold,
            eps=eps,
        )
        scores.append(score)

    return np.mean(scores)


def get_scheduler(
    optimizer: torch.optim, name: str, params: dict, additional_params: dict
):
    """
    Get the specified learning rate scheduler based on the given name and parameters.

    Parameters:
    - optimizer (torch.optim): The optimizer for which the scheduler will be applied.
    - name (str): The name of the scheduler to retrieve. Options: 'ReduceLROnPlateau', 'ExponentialLR'.
    - params (dict): Parameters for the scheduler initialization.
    - additional_params (dict): Additional parameters to be included in the scheduler dictionary.

    Returns:
    - dict or torch.optim.lr_scheduler: A dictionary containing the scheduler object along with additional parameters,
                                         if name is 'ReduceLROnPlateau', or directly returns the scheduler object
                                         if name is 'ExponentialLR'.

    Raises:
    - ValueError: If the specified scheduler name is not supported.
    """
    if name == "ReduceLROnPlateau":
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, **params
            ),
            **additional_params,
        }
        return scheduler
    if name == "ExponentialLR":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, **params)
