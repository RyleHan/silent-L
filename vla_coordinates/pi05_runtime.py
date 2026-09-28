"""Official openpi loading and prefix-residual helpers for pi0.5-LIBERO."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import torch

from openpi.models import model as model_types
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
from openpi.policies import policy_config
from openpi.training import config as training_config


DEFAULT_CONFIG = "pi05_libero"
DEFAULT_OPENPI_REVISION = "15a9616a00943ada6c20a0f158e3adb39df2ccac"


def load_pi05_policy(checkpoint: Path | str, device: torch.device):
    config = training_config.get_config(DEFAULT_CONFIG)
    config = dataclasses.replace(
        config,
        model=dataclasses.replace(config.model, pytorch_compile_mode=None),
    )
    policy = policy_config.create_trained_policy(
        config,
        checkpoint,
        pytorch_device=str(device),
    )
    if not policy._is_pytorch_model:  # noqa: SLF001
        raise TypeError(f"Expected a PyTorch checkpoint at {checkpoint}.")
    return policy


def _stack_tree(values: list[Any], device: torch.device) -> Any:
    first = values[0]
    if isinstance(first, dict):
        return {key: _stack_tree([value[key] for value in values], device) for key in first}
    array = np.stack([np.asarray(value) for value in values], axis=0)
    return torch.from_numpy(array).to(device)


def prepare_pi05_observation_batch(
    policy,
    *,
    base_images: list[np.ndarray],
    wrist_images: list[np.ndarray],
    states: list[np.ndarray],
    prompts: list[str],
    device: torch.device,
) -> model_types.Observation:
    lengths = {len(base_images), len(wrist_images), len(states), len(prompts)}
    if len(lengths) != 1:
        raise ValueError("All pi0.5 observation fields must have the same batch length.")

    transformed = []
    for base_image, wrist_image, state, prompt in zip(
        base_images, wrist_images, states, prompts, strict=True
    ):
        raw = {
            "observation/image": np.asarray(base_image, dtype=np.uint8),
            "observation/wrist_image": np.asarray(wrist_image, dtype=np.uint8),
            "observation/state": np.asarray(state, dtype=np.float32),
            "prompt": prompt,
        }
        transformed.append(policy._input_transform(raw))  # noqa: SLF001

    tensor_inputs = _stack_tree(transformed, device)
    return model_types.Observation.from_dict(tensor_inputs)


class Pi05PrefixResidualExtractor:
    """Read post-block PaliGemma residuals without modifying official openpi."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.layers = model.paligemma_with_expert.paligemma.language_model.layers

    @staticmethod
    def _hidden_from_hook(output: Any) -> torch.Tensor:
        if isinstance(output, (tuple, list)):
            return output[0]
        return output

    @torch.no_grad()
    def extract(self, observation: model_types.Observation) -> dict[str, torch.Tensor | int]:
        images, image_masks, language_tokens, language_masks, _ = self.model._preprocess_observation(  # noqa: SLF001
            observation, train=False
        )
        prefix_embs, prefix_pad_masks, prefix_attention_masks = self.model.embed_prefix(
            images, image_masks, language_tokens, language_masks
        )
        prefix_attention_2d = make_att_2d_masks(prefix_pad_masks, prefix_attention_masks)
        attention_mask = self.model._prepare_attention_masks_4d(prefix_attention_2d)  # noqa: SLF001
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        language_token_capacity = language_tokens.shape[1]
        num_image_tokens = prefix_embs.shape[1] - language_token_capacity
        image_pad_masks = prefix_pad_masks[:, :num_image_tokens]
        visual_input_mean = (
            prefix_embs[:, :num_image_tokens] * image_pad_masks.unsqueeze(-1)
        ).sum(dim=1) / image_pad_masks.sum(dim=1, keepdim=True)

        prompt_end_indices = num_image_tokens + language_masks.sum(dim=1) - 1
        captured: list[torch.Tensor | None] = [None] * len(self.layers)
        handles = []
        for layer_index, layer in enumerate(self.layers):
            handles.append(
                layer.register_forward_hook(
                    lambda _module, _inputs, output, index=layer_index: captured.__setitem__(
                        index, self._hidden_from_hook(output).detach()
                    )
                )
            )

        language_model = self.model.paligemma_with_expert.paligemma.language_model
        language_model.config._attn_implementation = "eager"  # noqa: SLF001
        try:
            self.model.paligemma_with_expert.forward(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=False,
            )
        finally:
            for handle in handles:
                handle.remove()

        if any(hidden is None for hidden in captured):
            missing = [index for index, hidden in enumerate(captured) if hidden is None]
            raise RuntimeError(f"No residual captured for PaliGemma layers {missing}.")

        batch_indices = torch.arange(prefix_embs.shape[0], device=prefix_embs.device)
        prompt_end = torch.stack(
            [hidden[batch_indices, prompt_end_indices] for hidden in captured if hidden is not None],
            dim=1,
        )
        return {
            "prompt_end": prompt_end.cpu(),
            "visual_input_mean": visual_input_mean.cpu(),
            "language_tokens": language_tokens.cpu(),
            "language_masks": language_masks.cpu(),
            "prompt_end_indices": prompt_end_indices.cpu(),
            "num_image_tokens": num_image_tokens,
        }


class Pi05ActionExpertResidualExtractor:
    """Read action-token residuals at one deterministic denoising step."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.layers = model.paligemma_with_expert.gemma_expert.model.layers

    @staticmethod
    def _hidden_from_hook(output: Any) -> torch.Tensor:
        if isinstance(output, (tuple, list)):
            return output[0]
        return output

    @staticmethod
    def _replace_hidden_in_hook_output(output: Any, hidden: torch.Tensor) -> Any:
        if isinstance(output, tuple):
            return (hidden, *output[1:])
        if isinstance(output, list):
            return [hidden, *output[1:]]
        return hidden

    def _prepare_denoise_context(
        self,
        observation: model_types.Observation,
        noise: torch.Tensor,
        *,
        capture_prefix: bool = False,
    ) -> dict[str, Any]:
        images, image_masks, language_tokens, language_masks, state = (  # noqa: SLF001
            self.model._preprocess_observation(observation, train=False)
        )
        batch_size = language_tokens.shape[0]
        expected_noise_shape = (
            batch_size,
            self.model.config.action_horizon,
            self.model.config.action_dim,
        )
        if tuple(noise.shape) != expected_noise_shape:
            raise ValueError(f"Noise shape {tuple(noise.shape)} != {expected_noise_shape}.")

        prefix_embs, prefix_pad_masks, prefix_attention_masks = self.model.embed_prefix(
            images, image_masks, language_tokens, language_masks
        )
        language_token_capacity = language_tokens.shape[1]
        num_image_tokens = prefix_embs.shape[1] - language_token_capacity
        image_pad_masks = prefix_pad_masks[:, :num_image_tokens]
        visual_input_mean = (
            prefix_embs[:, :num_image_tokens] * image_pad_masks.unsqueeze(-1)
        ).sum(dim=1) / image_pad_masks.sum(dim=1, keepdim=True)

        prefix_attention_2d = make_att_2d_masks(prefix_pad_masks, prefix_attention_masks)
        prefix_attention_mask = self.model._prepare_attention_masks_4d(  # noqa: SLF001
            prefix_attention_2d
        )
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_model = self.model.paligemma_with_expert.paligemma.language_model
        prefix_model.config._attn_implementation = "eager"  # noqa: SLF001
        prefix_captured: list[torch.Tensor | None] | None = None
        prefix_handles = []
        if capture_prefix:
            prefix_captured = [None] * len(prefix_model.layers)
            for layer_index, layer in enumerate(prefix_model.layers):
                prefix_handles.append(
                    layer.register_forward_hook(
                        lambda _module, _inputs, output, index=layer_index: prefix_captured.__setitem__(
                            index, self._hidden_from_hook(output).detach()
                        )
                    )
                )
        try:
            _, past_key_values = self.model.paligemma_with_expert.forward(
                attention_mask=prefix_attention_mask,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )
        finally:
            for handle in prefix_handles:
                handle.remove()

        result = {
            "state": state,
            "prefix_pad_masks": prefix_pad_masks,
            "past_key_values": past_key_values,
            "visual_input_mean": visual_input_mean,
            "language_tokens": language_tokens,
            "language_masks": language_masks,
            "num_image_tokens": num_image_tokens,
        }
        if prefix_captured is not None:
            if any(hidden is None for hidden in prefix_captured):
                missing = [
                    index for index, hidden in enumerate(prefix_captured) if hidden is None
                ]
                raise RuntimeError(f"No residual captured for PaliGemma layers {missing}.")
            prompt_end_indices = num_image_tokens + language_masks.sum(dim=1) - 1
            batch_indices = torch.arange(batch_size, device=prefix_embs.device)
            result["prompt_end"] = torch.stack(
                [
                    hidden[batch_indices, prompt_end_indices]
                    for hidden in prefix_captured
                    if hidden is not None
                ],
                dim=1,
            )
            result["prompt_end_indices"] = prompt_end_indices
        return result

    @torch.no_grad()
    def extract(
        self,
        observation: model_types.Observation,
        *,
        noise: torch.Tensor,
        timestep: float,
        capture_prefix: bool = False,
    ) -> dict[str, torch.Tensor | int]:
        context = self._prepare_denoise_context(
            observation, noise, capture_prefix=capture_prefix
        )
        batch_size = context["language_tokens"].shape[0]

        captured: list[torch.Tensor | None] = [None] * len(self.layers)
        handles = []
        for layer_index, layer in enumerate(self.layers):
            handles.append(
                layer.register_forward_hook(
                    lambda _module, _inputs, output, index=layer_index: captured.__setitem__(
                        index, self._hidden_from_hook(output).detach()
                    )
                )
            )

        timestep_tensor = torch.full(
            (batch_size,), timestep, dtype=torch.float32, device=noise.device
        )
        try:
            velocity = self.model.denoise_step(
                context["state"],
                context["prefix_pad_masks"],
                context["past_key_values"],
                noise,
                timestep_tensor,
            )
        finally:
            for handle in handles:
                handle.remove()

        if any(hidden is None for hidden in captured):
            missing = [index for index, hidden in enumerate(captured) if hidden is None]
            raise RuntimeError(f"No residual captured for action-expert layers {missing}.")
        hidden_states = [hidden for hidden in captured if hidden is not None]
        expected_tokens = self.model.config.action_horizon
        if any(hidden.shape[1] != expected_tokens for hidden in hidden_states):
            shapes = [tuple(hidden.shape) for hidden in hidden_states]
            raise RuntimeError(f"Unexpected action-expert layer shapes: {shapes}.")

        action_mean = torch.stack([hidden.mean(dim=1) for hidden in hidden_states], dim=1)
        action_first = torch.stack([hidden[:, 0] for hidden in hidden_states], dim=1)
        result = {
            "action_mean": action_mean.cpu(),
            "action_first": action_first.cpu(),
            "velocity": velocity.cpu(),
            "visual_input_mean": context["visual_input_mean"].cpu(),
            "language_tokens": context["language_tokens"].cpu(),
            "language_masks": context["language_masks"].cpu(),
            "num_image_tokens": context["num_image_tokens"],
            "num_action_tokens": expected_tokens,
        }
        if capture_prefix:
            result["prompt_end"] = context["prompt_end"].cpu()
            result["prompt_end_indices"] = context["prompt_end_indices"].cpu()
        return result

    @torch.no_grad()
    def extract_paired_patch_grid(
        self,
        observation: model_types.Observation,
        *,
        noise: torch.Tensor,
        timestep: float,
        layer_indices: list[int],
        alphas: list[float],
        base_prompt_index: int = 0,
        source_prompt_index: int = 1,
    ) -> dict[str, torch.Tensor]:
        """Patch one prompt with the same-state residual difference from the other."""
        context = self._prepare_denoise_context(observation, noise)
        batch_size = context["language_tokens"].shape[0]
        if batch_size != 2:
            raise ValueError(f"Paired patching requires batch size 2, got {batch_size}.")
        if {base_prompt_index, source_prompt_index} != {0, 1}:
            raise ValueError("Base and source prompt indices must be distinct members of {0, 1}.")
        if not layer_indices:
            raise ValueError("At least one patch layer is required.")
        if len(set(layer_indices)) != len(layer_indices):
            raise ValueError("Patch layer indices must be unique.")
        if any(index < 0 or index >= len(self.layers) for index in layer_indices):
            raise ValueError(f"Patch layers must be in [0, {len(self.layers) - 1}].")
        if not alphas or not all(np.isfinite(alpha) for alpha in alphas):
            raise ValueError("Patch alphas must be a nonempty list of finite values.")

        timestep_tensor = torch.full(
            (batch_size,), timestep, dtype=torch.float32, device=noise.device
        )
        final_layer = self.layers[-1]

        natural_final: list[torch.Tensor] = []
        natural_handle = final_layer.register_forward_hook(
            lambda _module, _inputs, output: natural_final.append(
                self._hidden_from_hook(output).detach()
            )
        )
        try:
            natural_velocity = self.model.denoise_step(
                context["state"],
                context["prefix_pad_masks"],
                context["past_key_values"],
                noise,
                timestep_tensor,
            )
        finally:
            natural_handle.remove()
        if len(natural_final) != 1:
            raise RuntimeError(f"Expected one natural final residual, got {len(natural_final)}.")

        num_layers = len(layer_indices)
        num_alphas = len(alphas)
        action_horizon = self.model.config.action_horizon
        action_dim = self.model.config.action_dim
        hidden_size = natural_final[0].shape[-1]
        patched_velocity = torch.empty(
            (num_layers, num_alphas, action_horizon, action_dim), dtype=torch.float32
        )
        patched_target_mean = torch.empty(
            (num_layers, num_alphas, hidden_size), dtype=torch.float32
        )
        patched_final_mean = torch.empty_like(patched_target_mean)
        untouched_velocity_max_abs_difference = torch.empty(
            (num_layers, num_alphas), dtype=torch.float32
        )

        for layer_offset, layer_index in enumerate(layer_indices):
            for alpha_offset, alpha in enumerate(alphas):
                captured_target: list[torch.Tensor] = []
                captured_final: list[torch.Tensor] = []

                def patch_hook(_module, _inputs, output, scale=float(alpha)):
                    hidden = self._hidden_from_hook(output)
                    delta = hidden[source_prompt_index] - hidden[base_prompt_index]
                    patched = hidden.clone()
                    patched[base_prompt_index] = hidden[base_prompt_index] + scale * delta
                    captured_target.append(patched.detach())
                    return self._replace_hidden_in_hook_output(output, patched)

                patch_handle = self.layers[layer_index].register_forward_hook(patch_hook)
                final_handle = final_layer.register_forward_hook(
                    lambda _module, _inputs, output: captured_final.append(
                        self._hidden_from_hook(output).detach()
                    )
                )
                try:
                    velocity = self.model.denoise_step(
                        context["state"],
                        context["prefix_pad_masks"],
                        context["past_key_values"],
                        noise,
                        timestep_tensor,
                    )
                finally:
                    final_handle.remove()
                    patch_handle.remove()

                if len(captured_target) != 1 or len(captured_final) != 1:
                    raise RuntimeError(
                        "Patch capture failed: "
                        f"target={len(captured_target)}, final={len(captured_final)}."
                    )
                patched_velocity[layer_offset, alpha_offset] = velocity[
                    base_prompt_index
                ].float().cpu()
                patched_target_mean[layer_offset, alpha_offset] = captured_target[0][
                    base_prompt_index
                ].mean(dim=0).float().cpu()
                patched_final_mean[layer_offset, alpha_offset] = captured_final[0][
                    base_prompt_index
                ].mean(dim=0).float().cpu()
                untouched_velocity_max_abs_difference[layer_offset, alpha_offset] = (
                    velocity[source_prompt_index]
                    .float()
                    .sub(natural_velocity[source_prompt_index].float())
                    .abs()
                    .max()
                    .cpu()
                )

        return {
            "natural_velocity": natural_velocity.float().cpu(),
            "natural_final_mean": natural_final[0].mean(dim=1).float().cpu(),
            "patched_velocity": patched_velocity,
            "patched_target_mean": patched_target_mean,
            "patched_final_mean": patched_final_mean,
            "untouched_velocity_max_abs_difference": untouched_velocity_max_abs_difference,
        }

    @torch.no_grad()
    def extract_layer_tokens(
        self,
        observation: model_types.Observation,
        *,
        noise: torch.Tensor,
        timestep: float,
        layer_index: int,
    ) -> dict[str, torch.Tensor]:
        """Return all action-token residuals at one expert block."""
        if layer_index < 0 or layer_index >= len(self.layers):
            raise ValueError(f"Layer must be in [0, {len(self.layers) - 1}].")
        context = self._prepare_denoise_context(observation, noise)
        batch_size = context["language_tokens"].shape[0]
        captured: list[torch.Tensor] = []
        handle = self.layers[layer_index].register_forward_hook(
            lambda _module, _inputs, output: captured.append(
                self._hidden_from_hook(output).detach()
            )
        )
        timestep_tensor = torch.full(
            (batch_size,), timestep, dtype=torch.float32, device=noise.device
        )
        try:
            velocity = self.model.denoise_step(
                context["state"],
                context["prefix_pad_masks"],
                context["past_key_values"],
                noise,
                timestep_tensor,
            )
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError(f"Expected one layer-token capture, got {len(captured)}.")
        return {
            "action_tokens": captured[0].float().cpu(),
            "velocity": velocity.float().cpu(),
        }

    @torch.no_grad()
    def extract_paired_patch_controls(
        self,
        observation: model_types.Observation,
        *,
        noise: torch.Tensor,
        timestep: float,
        layer_index: int,
        projection_basis: torch.Tensor,
        shuffled_delta: torch.Tensor,
        random_seed: int,
        base_prompt_index: int = 0,
        source_prompt_index: int = 1,
    ) -> dict[str, torch.Tensor | list[str]]:
        """Compare fixed semantic and matched controls at one selected block."""
        if layer_index < 0 or layer_index >= len(self.layers):
            raise ValueError(f"Layer must be in [0, {len(self.layers) - 1}].")
        if {base_prompt_index, source_prompt_index} != {0, 1}:
            raise ValueError("Base and source prompt indices must be distinct members of {0, 1}.")
        context = self._prepare_denoise_context(observation, noise)
        batch_size = context["language_tokens"].shape[0]
        if batch_size != 2:
            raise ValueError(f"Paired controls require batch size 2, got {batch_size}.")
        timestep_tensor = torch.full(
            (batch_size,), timestep, dtype=torch.float32, device=noise.device
        )

        natural_target: list[torch.Tensor] = []
        natural_handle = self.layers[layer_index].register_forward_hook(
            lambda _module, _inputs, output: natural_target.append(
                self._hidden_from_hook(output).detach()
            )
        )
        try:
            natural_velocity = self.model.denoise_step(
                context["state"],
                context["prefix_pad_masks"],
                context["past_key_values"],
                noise,
                timestep_tensor,
            )
        finally:
            natural_handle.remove()
        if len(natural_target) != 1:
            raise RuntimeError(f"Expected one natural target capture, got {len(natural_target)}.")

        target = natural_target[0]
        full_delta = target[source_prompt_index] - target[base_prompt_index]
        basis = projection_basis.to(device=full_delta.device, dtype=torch.float32)
        if basis.ndim != 2 or basis.shape[0] != full_delta.shape[-1]:
            raise ValueError(
                f"Projection basis shape {tuple(basis.shape)} is incompatible with "
                f"hidden size {full_delta.shape[-1]}."
            )
        goal_delta = (full_delta.float() @ basis) @ basis.T
        goal_delta = goal_delta.to(dtype=full_delta.dtype)
        null_delta = full_delta - goal_delta

        shuffled = shuffled_delta.to(device=full_delta.device, dtype=full_delta.dtype)
        if shuffled.shape != full_delta.shape:
            raise ValueError(
                f"Shuffled delta shape {tuple(shuffled.shape)} != {tuple(full_delta.shape)}."
            )
        generator = torch.Generator(device=full_delta.device).manual_seed(random_seed)
        random_delta = torch.randn(
            full_delta.shape,
            generator=generator,
            device=full_delta.device,
            dtype=full_delta.dtype,
        )
        full_token_norm = torch.linalg.vector_norm(full_delta.float(), dim=-1, keepdim=True)
        random_token_norm = torch.linalg.vector_norm(
            random_delta.float(), dim=-1, keepdim=True
        ).clamp_min(1e-12)
        random_delta = random_delta * (full_token_norm / random_token_norm).to(
            dtype=random_delta.dtype
        )

        mode_names = ["paired", "goal", "null", "shuffled", "random"]
        mode_deltas = [full_delta, goal_delta, null_delta, shuffled, random_delta]
        patched_velocity = torch.empty(
            (len(mode_names), self.model.config.action_horizon, self.model.config.action_dim),
            dtype=torch.float32,
        )
        patch_delta_norm_ratio = torch.empty(len(mode_names), dtype=torch.float32)
        untouched_velocity_max_abs_difference = torch.empty(
            len(mode_names), dtype=torch.float32
        )
        full_norm = torch.linalg.vector_norm(full_delta.float()).clamp_min(1e-12)

        for mode_index, patch_delta in enumerate(mode_deltas):
            def patch_hook(_module, _inputs, output, delta=patch_delta):
                hidden = self._hidden_from_hook(output)
                patched = hidden.clone()
                patched[base_prompt_index] = hidden[base_prompt_index] + delta
                return self._replace_hidden_in_hook_output(output, patched)

            handle = self.layers[layer_index].register_forward_hook(patch_hook)
            try:
                velocity = self.model.denoise_step(
                    context["state"],
                    context["prefix_pad_masks"],
                    context["past_key_values"],
                    noise,
                    timestep_tensor,
                )
            finally:
                handle.remove()
            patched_velocity[mode_index] = velocity[base_prompt_index].float().cpu()
            patch_delta_norm_ratio[mode_index] = (
                torch.linalg.vector_norm(patch_delta.float()) / full_norm
            ).cpu()
            untouched_velocity_max_abs_difference[mode_index] = (
                velocity[source_prompt_index]
                .float()
                .sub(natural_velocity[source_prompt_index].float())
                .abs()
                .max()
                .cpu()
            )

        return {
            "mode_names": mode_names,
            "natural_velocity": natural_velocity.float().cpu(),
            "patched_velocity": patched_velocity,
            "patch_delta_norm_ratio": patch_delta_norm_ratio,
            "untouched_velocity_max_abs_difference": untouched_velocity_max_abs_difference,
        }

    @torch.no_grad()
    def sample_actions_paired_patch(
        self,
        observation: model_types.Observation,
        *,
        noise: torch.Tensor,
        layer_index: int,
        base_prompt_index: int,
        source_prompt_index: int,
        mode: str,
        alpha: float = 1.0,
        num_steps: int = 10,
        random_seed: int = 0,
    ) -> torch.Tensor:
        """Sample paired action chunks while patching every flow-matching step."""
        if mode not in {"none", "paired", "random"}:
            raise ValueError(f"Unsupported closed-loop patch mode: {mode!r}.")
        if layer_index < 0 or layer_index >= len(self.layers):
            raise ValueError(f"Layer must be in [0, {len(self.layers) - 1}].")
        if {base_prompt_index, source_prompt_index} != {0, 1}:
            raise ValueError("Base and source prompt indices must be distinct members of {0, 1}.")
        if num_steps <= 0:
            raise ValueError("num_steps must be positive.")
        context = self._prepare_denoise_context(observation, noise)
        batch_size = context["language_tokens"].shape[0]
        if batch_size != 2:
            raise ValueError(f"Paired action sampling requires batch size 2, got {batch_size}.")

        x_t = noise.clone()
        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=noise.device)
        time = torch.tensor(1.0, dtype=torch.float32, device=noise.device)
        for step_index in range(num_steps):
            timestep = time.expand(batch_size)
            handle = None
            if mode != "none":
                def patch_hook(_module, _inputs, output, denoise_step=step_index):
                    hidden = self._hidden_from_hook(output)
                    full_delta = hidden[source_prompt_index] - hidden[base_prompt_index]
                    if mode == "paired":
                        patch_delta = full_delta
                    else:
                        generator = torch.Generator(device=hidden.device).manual_seed(
                            random_seed + denoise_step
                        )
                        patch_delta = torch.randn(
                            full_delta.shape,
                            generator=generator,
                            device=hidden.device,
                            dtype=hidden.dtype,
                        )
                        full_token_norm = torch.linalg.vector_norm(
                            full_delta.float(), dim=-1, keepdim=True
                        )
                        random_token_norm = torch.linalg.vector_norm(
                            patch_delta.float(), dim=-1, keepdim=True
                        ).clamp_min(1e-12)
                        patch_delta = patch_delta * (
                            full_token_norm / random_token_norm
                        ).to(dtype=patch_delta.dtype)
                    patched = hidden.clone()
                    patched[base_prompt_index] = (
                        hidden[base_prompt_index] + alpha * patch_delta
                    )
                    return self._replace_hidden_in_hook_output(output, patched)

                handle = self.layers[layer_index].register_forward_hook(patch_hook)
            try:
                velocity = self.model.denoise_step(
                    context["state"],
                    context["prefix_pad_masks"],
                    context["past_key_values"],
                    x_t,
                    timestep,
                )
            finally:
                if handle is not None:
                    handle.remove()
            x_t = x_t + dt * velocity
            time += dt
        return x_t.float().cpu()

    @torch.no_grad()
    def sample_actions_full_expert_replay(
        self,
        observation: model_types.Observation,
        *,
        noise: torch.Tensor,
        base_prompt_index: int,
        source_prompt_index: int,
        num_steps: int = 10,
    ) -> dict[str, torch.Tensor]:
        """Replay every expert-block output from one paired prompt into the other."""
        if {base_prompt_index, source_prompt_index} != {0, 1}:
            raise ValueError("Base and source prompt indices must be distinct members of {0, 1}.")
        if num_steps <= 0:
            raise ValueError("num_steps must be positive.")
        context = self._prepare_denoise_context(observation, noise)
        batch_size = context["language_tokens"].shape[0]
        if batch_size != 2:
            raise ValueError(f"Full expert replay requires batch size 2, got {batch_size}.")

        x_t = noise.clone()
        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=noise.device)
        time = torch.tensor(1.0, dtype=torch.float32, device=noise.device)
        layer_hook_counts = torch.zeros(len(self.layers), dtype=torch.int64)
        velocity_max_abs_difference = torch.empty(num_steps, dtype=torch.float32)

        for step_index in range(num_steps):
            handles = []
            for layer_index, layer in enumerate(self.layers):
                def replay_hook(_module, _inputs, output, index=layer_index):
                    hidden = self._hidden_from_hook(output)
                    replayed = hidden.clone()
                    replayed[base_prompt_index] = hidden[source_prompt_index]
                    layer_hook_counts[index] += 1
                    return self._replace_hidden_in_hook_output(output, replayed)

                handles.append(layer.register_forward_hook(replay_hook))

            timestep = time.expand(batch_size)
            try:
                velocity = self.model.denoise_step(
                    context["state"],
                    context["prefix_pad_masks"],
                    context["past_key_values"],
                    x_t,
                    timestep,
                )
            finally:
                for handle in handles:
                    handle.remove()

            velocity_max_abs_difference[step_index] = (
                velocity[base_prompt_index]
                .float()
                .sub(velocity[source_prompt_index].float())
                .abs()
                .max()
                .cpu()
            )
            x_t = x_t + dt * velocity
            time += dt

        return {
            "actions": x_t.float().cpu(),
            "layer_hook_counts": layer_hook_counts,
            "velocity_max_abs_difference": velocity_max_abs_difference,
            "action_max_abs_difference": x_t[base_prompt_index]
            .float()
            .sub(x_t[source_prompt_index].float())
            .abs()
            .max()
            .cpu(),
        }

    @torch.no_grad()
    def sample_actions_capture_residuals(
        self,
        observation: model_types.Observation,
        *,
        noise: torch.Tensor,
        layer_index: int,
        num_steps: int = 10,
    ) -> dict[str, torch.Tensor]:
        """Sample actions and capture token-mean post-block residuals per denoising step."""
        if layer_index < 0 or layer_index >= len(self.layers):
            raise ValueError(f"Layer must be in [0, {len(self.layers) - 1}].")
        if num_steps <= 0:
            raise ValueError("num_steps must be positive.")
        context = self._prepare_denoise_context(observation, noise)
        batch_size = context["language_tokens"].shape[0]
        x_t = noise.clone()
        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=noise.device)
        time = torch.tensor(1.0, dtype=torch.float32, device=noise.device)
        captured_means = []

        for _step_index in range(num_steps):
            captured = []
            handle = self.layers[layer_index].register_forward_hook(
                lambda _module, _inputs, output: captured.append(
                    self._hidden_from_hook(output).detach().float().mean(dim=1).cpu()
                )
            )
            try:
                velocity = self.model.denoise_step(
                    context["state"],
                    context["prefix_pad_masks"],
                    context["past_key_values"],
                    x_t,
                    time.expand(batch_size),
                )
            finally:
                handle.remove()
            if len(captured) != 1:
                raise RuntimeError(f"Expected one residual capture, got {len(captured)}.")
            captured_means.append(captured[0])
            x_t = x_t + dt * velocity
            time += dt

        return {
            "actions": x_t.float().cpu(),
            "residual_mean": torch.stack(captured_means, dim=1),
            "layer_hook_count": torch.tensor(len(captured_means), dtype=torch.int64),
        }

    @torch.no_grad()
    def sample_actions_conceptor(
        self,
        observation: model_types.Observation,
        *,
        noise: torch.Tensor,
        layer_index: int,
        conceptor: torch.Tensor,
        beta: float,
        num_steps: int = 10,
    ) -> dict[str, torch.Tensor]:
        """Apply a COAST-style multiplicative gate at every denoising step."""
        if layer_index < 0 or layer_index >= len(self.layers):
            raise ValueError(f"Layer must be in [0, {len(self.layers) - 1}].")
        if not np.isfinite(beta) or beta < 0 or beta > 1:
            raise ValueError("Conceptor strength beta must be in [0, 1].")
        if num_steps <= 0:
            raise ValueError("num_steps must be positive.")
        hidden_size = int(conceptor.shape[-1])
        if tuple(conceptor.shape) not in {
            (hidden_size, hidden_size),
            (num_steps, hidden_size, hidden_size),
        }:
            raise ValueError(
                "Conceptor must have shape [hidden, hidden] or "
                f"[steps, hidden, hidden], got {tuple(conceptor.shape)}."
            )

        context = self._prepare_denoise_context(observation, noise)
        batch_size = context["language_tokens"].shape[0]
        matrices = conceptor.to(device=noise.device, dtype=torch.float32)
        identity = torch.eye(hidden_size, device=noise.device, dtype=torch.float32)
        x_t = noise.clone()
        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=noise.device)
        time = torch.tensor(1.0, dtype=torch.float32, device=noise.device)
        hook_count = 0
        relative_change = torch.zeros(num_steps, dtype=torch.float32)

        for step_index in range(num_steps):
            handle = None
            if beta > 0:
                current = matrices if matrices.ndim == 2 else matrices[step_index]
                gate = (1.0 - beta) * identity + beta * current

                def gate_hook(_module, _inputs, output, matrix=gate, index=step_index):
                    nonlocal hook_count
                    hidden = self._hidden_from_hook(output)
                    working_matrix = matrix.to(dtype=hidden.dtype)
                    steered = hidden @ working_matrix.T
                    difference = steered.float() - hidden.float()
                    relative_change[index] = (
                        torch.linalg.vector_norm(difference)
                        / torch.linalg.vector_norm(hidden.float()).clamp_min(1e-12)
                    ).detach().cpu()
                    hook_count += 1
                    return self._replace_hidden_in_hook_output(output, steered)

                handle = self.layers[layer_index].register_forward_hook(gate_hook)
            try:
                velocity = self.model.denoise_step(
                    context["state"],
                    context["prefix_pad_masks"],
                    context["past_key_values"],
                    x_t,
                    time.expand(batch_size),
                )
            finally:
                if handle is not None:
                    handle.remove()
            x_t = x_t + dt * velocity
            time += dt

        expected_hooks = num_steps if beta > 0 else 0
        if hook_count != expected_hooks:
            raise RuntimeError(
                f"Conceptor hook fired {hook_count} times; expected {expected_hooks}."
            )
        return {
            "actions": x_t.float().cpu(),
            "hook_count": torch.tensor(hook_count, dtype=torch.int64),
            "relative_residual_change": relative_change,
        }
