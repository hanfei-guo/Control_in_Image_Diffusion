from __future__ import annotations

from diffusers.callbacks import PipelineCallback
from diffusers.configuration_utils import register_to_config

from .schedules import semantic_weight


class IPAdapterScaleEnableCallback(PipelineCallback):
    """Enable the IP-Adapter after a configurable fraction of denoising steps."""

    tensor_inputs = []

    @register_to_config
    def __init__(self, cutoff_step_ratio: float = 0.5, cutoff_step_index: int | None = None, scale: float = 0.8):
        self.scale = scale
        super().__init__(cutoff_step_ratio=cutoff_step_ratio, cutoff_step_index=cutoff_step_index)

    def callback_fn(self, pipeline, step_index, timestep, callback_kwargs) -> dict:
        cutoff_step_ratio = self.config.cutoff_step_ratio
        cutoff_step_index = self.config.cutoff_step_index
        cutoff_step = (
            cutoff_step_index if cutoff_step_index is not None else int(pipeline.num_timesteps * cutoff_step_ratio)
        )
        if step_index == cutoff_step:
            pipeline.set_ip_adapter_scale(self.scale)
        return callback_kwargs


class DynamicIPAdapterScaleCallback:
    """Update IP-Adapter scale at every step using a smooth schedule."""

    def __init__(self, tau: float, sharpness: float, max_scale: float):
        self.tau = tau
        self.sharpness = sharpness
        self.max_scale = max_scale

    def __call__(self, pipeline, step_index, timestep, callback_kwargs) -> dict:
        if pipeline.num_timesteps <= 1:
            progress = 1.0
        else:
            progress = step_index / (pipeline.num_timesteps - 1)
        pipeline.set_ip_adapter_scale(semantic_weight(progress, self.tau, self.sharpness, self.max_scale))
        return callback_kwargs
