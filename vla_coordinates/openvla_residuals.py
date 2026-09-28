"""Residual-stream extraction helpers for pinned OpenVLA models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from vla_coordinates.openvla_runtime import EMPTY_ACTION_TOKEN_ID


@dataclass(frozen=True)
class ResidualReadout:
    """Selected post-block residuals from one multimodal forward pass."""

    prompt_positions: torch.Tensor
    visual_mean: torch.Tensor | None
    num_visual_tokens: int
    multimodal_sequence_length: int


class OpenVLAResidualExtractor:
    """Extract compact readouts from every Llama block with forward hooks.

    The convention matches Emergent OpenVLA's ``residual_output`` hooks: layer
    ``i`` is the output of decoder block ``i`` before the model's final norm.
    """

    def __init__(self, model: Any) -> None:
        self.model = model
        self.layers = model.language_model.model.layers
        self._handles = [
            layer.register_forward_hook(self._make_hook(layer_index))
            for layer_index, layer in enumerate(self.layers)
        ]
        self._layer_prompt_positions: list[torch.Tensor | None] = []
        self._layer_visual_means: list[torch.Tensor | None] = []
        self._text_length = 0
        self._num_visual_tokens: int | None = None
        self._include_visual = False

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    def _make_hook(self, layer_index: int):
        def hook(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.ndim != 3 or hidden.shape[0] != 1:
                raise ValueError(f"Expected hidden state [1, sequence, hidden], got {tuple(hidden.shape)}")

            num_visual_tokens = hidden.shape[1] - self._text_length
            if num_visual_tokens <= 0:
                raise ValueError(
                    f"Multimodal sequence {hidden.shape[1]} is not longer than text sequence {self._text_length}."
                )
            if self._num_visual_tokens is None:
                self._num_visual_tokens = num_visual_tokens
            elif self._num_visual_tokens != num_visual_tokens:
                raise ValueError("Visual token count changed between transformer blocks.")

            # The final two text positions are the last prompt token and the
            # empty action token appended by prepare_openvla_inputs().
            self._layer_prompt_positions[layer_index] = hidden[0, -2:, :].detach().clone()
            if self._include_visual:
                visual = hidden[0, 1 : 1 + num_visual_tokens, :]
                self._layer_visual_means[layer_index] = visual.float().mean(dim=0).to(hidden.dtype).detach()

        return hook

    def extract(self, inputs: dict[str, Any], *, include_visual: bool) -> ResidualReadout:
        input_ids = inputs["input_ids"]
        if input_ids.shape[0] != 1:
            raise ValueError("Residual extraction currently requires batch size one.")
        if input_ids.shape[1] < 2:
            raise ValueError("At least a prompt token and the empty action token are required.")
        if input_ids[0, -1].item() != EMPTY_ACTION_TOKEN_ID:
            raise ValueError("The final input token is not OpenVLA's empty action token.")

        self._text_length = input_ids.shape[1]
        self._num_visual_tokens = None
        self._include_visual = include_visual
        self._layer_prompt_positions = [None] * self.num_layers
        self._layer_visual_means = [None] * self.num_layers

        with torch.inference_mode():
            _ = self.model(
                **inputs,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )

        if any(value is None for value in self._layer_prompt_positions):
            raise RuntimeError("Not every transformer block produced a prompt-position residual.")
        prompt_positions = torch.stack(self._layer_prompt_positions).cpu()

        visual_mean = None
        if include_visual:
            if any(value is None for value in self._layer_visual_means):
                raise RuntimeError("Not every transformer block produced a visual residual.")
            visual_mean = torch.stack(self._layer_visual_means).cpu()

        assert self._num_visual_tokens is not None
        return ResidualReadout(
            prompt_positions=prompt_positions,
            visual_mean=visual_mean,
            num_visual_tokens=self._num_visual_tokens,
            multimodal_sequence_length=self._text_length + self._num_visual_tokens,
        )

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> "OpenVLAResidualExtractor":
        return self

    def __exit__(self, _exc_type: object, _exc_value: object, _traceback: object) -> None:
        self.close()
