# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import torch

from ultralytics.utils import LOGGER, YAML


def torch2litert(
    model: torch.nn.Module,
    im: torch.Tensor,
    file: Path | str,
    half: bool = False,
    int8: bool = False,
    metadata: dict | None = None,
    prefix: str = "",
) -> Path:
    """Export a PyTorch model to LiteRT format using litert_torch with optional FP16/INT8 quantization.

    Args:
        model (torch.nn.Module): The PyTorch model to export.
        im (torch.Tensor): Example input tensor for tracing.
        file (Path | str): Source model file path used to derive output directory.
        half (bool): Whether to apply FP16 quantization.
        int8 (bool): Whether to apply INT8 quantization.
        metadata (dict | None): Optional metadata saved as ``metadata.yaml``.
        prefix (str): Prefix for log messages.

    Returns:
        (Path): Path to the exported ``_litert_model`` directory.
    """
    from ultralytics.utils.checks import check_requirements

    check_requirements("litert-torch-nightly")
    import litert_torch

    LOGGER.info(f"\n{prefix} starting export with litert_torch {litert_torch.__version__}...")
    file = Path(file)
    f = Path(str(file).replace(file.suffix, "_litert_model"))
    f.mkdir(parents=True, exist_ok=True)

    sample_inputs = (im,)
    edge_model = litert_torch.convert(model, sample_inputs)

    if int8:
        tflite_file = f / f"{file.stem}_int8.tflite"
    elif half:
        tflite_file = f / f"{file.stem}_float16.tflite"
    else:
        tflite_file = f / f"{file.stem}_float32.tflite"
    edge_model.export(tflite_file)

    # Apply quantization using ai-edge-quantizer-nightly
    if int8 or half:
        check_requirements("ai-edge-quantizer-nightly")
        from ai_edge_quantizer import quantizer, recipe

        LOGGER.info(f"{prefix} applying {'INT8' if int8 else 'FP16'} quantization...")
        qt = quantizer.Quantizer(str(tflite_file))
        if int8:
            qt.load_quantization_recipe(recipe.dynamic_wi8_afp32())
        else:  # half (FP16) - use weight-only int8 for size reduction while keeping float compute
            qt.load_quantization_recipe(recipe.weight_only_wi8_afp32())
        qt.quantize().export_model(str(tflite_file), overwrite=True)

    YAML.save(f / "metadata.yaml", metadata or {})
    return f
