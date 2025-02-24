"""
smp_dropout.py

This module provides a custom PyTorch `nn.Module` that integrates dropout layers
into segmentation models from the `segmentation_models_pytorch` (SMP) library.
The `SMPWithDropout` class wraps an existing SMP model, adding dropout to its encoder
features to enhance regularization and prevent overfitting.

Usage:
    from segmentation_models_pytorch import Unet
    from smp_dropout import SMPWithDropout

    model = SMPWithDropout(
        model_class=Unet,
        encoder_name='resnet34',
        encoder_weights='imagenet',
        in_channels=3,
        classes=1,
        max_dropout=0.5,
        decoder_attention_type='scse',
        decoder_use_batchnorm=True
    )
"""

import torch
import torch.nn as nn
from typing import Any, Callable, Optional


class SMPWithDropout(nn.Module):
    """
    A wrapper for SMP (Segmentation Models PyTorch) models that adds dropout layers
    to the encoder features to improve model generalization and reduce overfitting.

    This class initializes a base SMP model and augments it by inserting dropout layers
    with progressively increasing dropout rates into each feature map produced by the encoder.
    The dropout rates are determined based on the `max_dropout` parameter and the number
    of feature maps.

    Attributes:
        base_model (nn.Module): The underlying SMP model without dropout.
        dropouts (nn.ModuleList): A list of dropout layers applied to each encoder feature.
    """

    def __init__(
        self,
        model_class: Callable[..., nn.Module],
        encoder_name: str,
        encoder_weights: Optional[str],
        in_channels: int,
        classes: int,
        max_dropout: float = 0.5,
        decoder_attention_type: str = 'scse',
        decoder_use_batchnorm: bool = True,
        **kwargs: Any
    ) -> None:
        """
        Initializes the SMPWithDropout module.

        Args:
            model_class (Callable[..., nn.Module]): The SMP model class to instantiate (e.g., Unet, Linknet).
            encoder_name (str): Name of the encoder to use (e.g., 'resnet34').
            encoder_weights (Optional[str]): Pretrained weights for the encoder (e.g., 'imagenet').
            in_channels (int): Number of input channels for the model.
            classes (int): Number of output classes for segmentation.
            max_dropout (float, optional): Maximum dropout rate to apply. Defaults to 0.5.
            decoder_attention_type (str, optional): Type of attention module to use in the decoder.
                Defaults to 'scse'.
            decoder_use_batchnorm (bool, optional): Whether to use batch normalization in the decoder.
                Defaults to True.
            **kwargs (Any): Additional keyword arguments for the SMP model.
        """
        super().__init__()
        self.base_model = model_class(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,
            decoder_attention_type=decoder_attention_type,
            decoder_use_batchnorm=decoder_use_batchnorm,
            **kwargs
        )

        # Determine the number of encoder feature maps by performing a forward pass with a dummy input
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, 64, 64)
            encoder_features = self.base_model.encoder(dummy_input)
            num_features = len(encoder_features)

        # Initialize dropout layers with progressively increasing dropout rates
        self.dropouts = nn.ModuleList([
            nn.Dropout(p=max_dropout * i / (num_features - 1)) for i in range(num_features)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SMPWithDropout model.

        Args:
            x (torch.Tensor): Input tensor with shape [B, C, H, W], where
                B = batch size,
                C = number of channels,
                H = height,
                W = width.

        Returns:
            torch.Tensor: Output segmentation map with shape [B, classes, H, W].
        """
        # Extract features from the encoder
        features = self.base_model.encoder(x)

        # Apply dropout to each encoder feature map
        modified_features = tuple(
            dropout(feat) for feat, dropout in zip(features, self.dropouts)
        )

        # Decode the modified features to obtain decoder output
        decoder_output = self.base_model.decoder(*modified_features)

        # Generate the final segmentation map
        segmentation_map = self.base_model.segmentation_head(decoder_output)

        return segmentation_map
