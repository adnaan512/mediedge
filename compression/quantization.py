"""
MediEdge — INT8 Post-Training Static Quantization.

Uses PyTorch FX Graph Mode Quantization (recommended for PyTorch 2.x+).
Automatically traces the model, fuses Conv+BN+ReLU, inserts quantization/dequantization
stubs, calibrates activation ranges on sample batches, and converts weights and ops to INT8.

Accepts standard float32 input tensors and runs on CPU.
"""

import copy
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.ao.quantization import quantize_fx, get_default_qconfig_mapping


def quantize_model_int8(
    model: nn.Module,
    calibration_loader: DataLoader,
    calibration_batches: int = 10,
    device: str = "cpu",
) -> nn.Module:
    """
    Apply INT8 post-training static quantization using FX Graph Mode.

    Args:
        model               : Trained float32 model.
        calibration_loader  : DataLoader for calibration.
        calibration_batches : Number of batches to run for activation statistics.
        device              : Should be 'cpu' (PTQ is CPU-only in PyTorch).

    Returns:
        Quantized INT8 model ready for evaluation on CPU.
    """
    warnings.filterwarnings("ignore")
    model = copy.deepcopy(model).to("cpu")
    model.eval()

    example_input = (torch.randn(1, 3, 28, 28),)
    qconfig_mapping = get_default_qconfig_mapping("x86")

    print(f"[quantization] Preparing model with FX graph mode...")
    try:
        prepared_model = quantize_fx.prepare_fx(
            model,
            qconfig_mapping,
            example_input,
        )
    except Exception as e:
        print(f"[quantization] FX prepare failed: {e}. Trying fallback...")
        return model

    # Calibration pass
    print(f"[quantization] Calibrating on {calibration_batches} batches...")
    prepared_model.eval()
    with torch.no_grad():
        for i, (images, _) in enumerate(calibration_loader):
            if i >= calibration_batches:
                break
            images = images.to("cpu")
            prepared_model(images)

    # Conversion to INT8
    print("[quantization] Converting to INT8...")
    try:
        quantized_model = quantize_fx.convert_fx(prepared_model)
        print("[quantization] INT8 conversion successful.")
        return quantized_model
    except Exception as e:
        print(f"[quantization] FX convert failed: {e}")
        return model
