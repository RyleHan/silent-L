"""Reproducible OpenVLA loading and inference helpers for this project."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from experiments.robot.openvla_utils import crop_and_resize
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

DEFAULT_CHECKPOINT = "openvla/openvla-7b-finetuned-libero-object"
DEFAULT_REVISION = "287d6cfdf12d07b1449505f66d9bf3550257e9b3"
EMPTY_ACTION_TOKEN_ID = 29871


def register_openvla() -> None:
    AutoConfig.register("openvla", OpenVLAConfig, exist_ok=True)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor, exist_ok=True)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor, exist_ok=True)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction, exist_ok=True)


def load_openvla(
    checkpoint: str,
    revision: str,
    device: torch.device,
    *,
    dtype: torch.dtype = torch.float16,
    attention_implementation: str = "eager",
    local_files_only: bool = True,
) -> tuple[OpenVLAForActionPrediction, PrismaticProcessor]:
    register_openvla()
    common_kwargs = {
        "revision": revision,
        "local_files_only": local_files_only,
        "trust_remote_code": False,
    }
    processor = AutoProcessor.from_pretrained(checkpoint, **common_kwargs)
    model = AutoModelForVision2Seq.from_pretrained(
        checkpoint,
        attn_implementation=attention_implementation,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        **common_kwargs,
    ).to(device)
    model.eval()
    return model, processor


def build_openvla_prompt(task_description: str) -> str:
    return f"In: What action should the robot take to {task_description.lower()}?\nOut:"


def center_crop_openvla_image(image: np.ndarray) -> Image.Image:
    import tensorflow as tf

    tensor = tf.convert_to_tensor(image)
    original_dtype = tensor.dtype
    tensor = tf.image.convert_image_dtype(tensor, tf.float32)
    tensor = crop_and_resize(tensor, crop_scale=0.9, batch_size=1)
    tensor = tf.clip_by_value(tensor, 0, 1)
    tensor = tf.image.convert_image_dtype(tensor, original_dtype, saturate=True)
    return Image.fromarray(tensor.numpy()).convert("RGB")


def prepare_openvla_inputs(
    processor: PrismaticProcessor,
    prompt: str,
    image: np.ndarray | Image.Image,
    device: torch.device,
    *,
    dtype: torch.dtype = torch.float16,
    center_crop: bool = True,
) -> dict[str, Any]:
    if isinstance(image, np.ndarray):
        processed_image = (
            center_crop_openvla_image(image) if center_crop else Image.fromarray(image).convert("RGB")
        )
    else:
        processed_image = image.convert("RGB")

    inputs = processor(prompt, processed_image).to(device, dtype=dtype)
    if not torch.all(inputs["input_ids"][:, -1] == EMPTY_ACTION_TOKEN_ID):
        empty_token = torch.full(
            (inputs["input_ids"].shape[0], 1),
            EMPTY_ACTION_TOKEN_ID,
            dtype=inputs["input_ids"].dtype,
            device=device,
        )
        inputs["input_ids"] = torch.cat((inputs["input_ids"], empty_token), dim=1)
        if "attention_mask" in inputs:
            mask_extension = torch.ones(
                (inputs["attention_mask"].shape[0], 1),
                dtype=inputs["attention_mask"].dtype,
                device=device,
            )
            inputs["attention_mask"] = torch.cat((inputs["attention_mask"], mask_extension), dim=1)
    return inputs


def predict_openvla_action(
    model: OpenVLAForActionPrediction,
    inputs: dict[str, Any],
    *,
    unnorm_key: str = "libero_object",
) -> np.ndarray:
    with torch.inference_mode():
        action = model.predict_action(
            **inputs,
            unnorm_key=unnorm_key,
            do_sample=False,
        )
    return np.asarray(action)

