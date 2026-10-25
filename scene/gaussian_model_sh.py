"""
Spherical-harmonic capacity helpers for GaussianModel.

Some methods start with a low SH degree and enable more color coefficients later.
These helpers expand the `f_rest` tensor and migrate optimizer state so training
can continue after SH capacity changes.
"""
from __future__ import annotations

import torch
from torch import nn


class GaussianModelSHMixin:
    """
        Mixin that grows SH capacity and advances the active SH degree.
    """

    def _replace_feature_rest_with_optimizer_state(self, new_features_rest: torch.Tensor) -> None:
        """
            Replace `f_rest` and copy compatible optimizer momentum into the new tensor.
        """
        new_features_rest = new_features_rest.contiguous()
        old_parameter = self._features_rest
        new_parameter = nn.Parameter(new_features_rest.requires_grad_(True))
        if getattr(self, "optimizer", None) is None:
            self._features_rest = new_parameter
            return

        for group in self.optimizer.param_groups:
            if group.get("name") != "f_rest":
                continue
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                for state_name in ("exp_avg", "exp_avg_sq"):
                    state_value = stored_state.get(state_name)
                    if not torch.is_tensor(state_value) or tuple(state_value.shape) != tuple(old_parameter.shape):
                        stored_state[state_name] = torch.zeros_like(new_features_rest)
                        continue
                    expanded_state = torch.zeros_like(new_features_rest)
                    expanded_state[:, : state_value.shape[1], :] = state_value
                    stored_state[state_name] = expanded_state.contiguous()
                del self.optimizer.state[group["params"][0]]
            group["params"][0] = new_parameter
            if stored_state is not None:
                self.optimizer.state[group["params"][0]] = stored_state
            self._features_rest = new_parameter
            return

        self._features_rest = new_parameter

    def ensure_sh_degree_capacity(self, target_max_sh_degree: int) -> bool:
        """
            Expand stored SH coefficients when the target degree needs more channels.

            Returns True only when the tensor was actually expanded.
        """
        target_max_sh_degree = int(target_max_sh_degree)
        if target_max_sh_degree < 0:
            raise ValueError("target_max_sh_degree must be non-negative.")
        required_rest_channels = (target_max_sh_degree + 1) ** 2 - 1
        current_rest_channels = int(self._features_rest.shape[1]) if self._features_rest.ndim >= 2 else 0
        self.max_sh_degree = max(int(self.max_sh_degree), target_max_sh_degree)
        if current_rest_channels >= required_rest_channels:
            return False
        if self._features_rest.ndim != 3 or self._features_rest.shape[2] != 3:
            raise ValueError("features_rest must have shape [N, K, 3].")

        expanded_features = torch.zeros(
            (self._features_rest.shape[0], required_rest_channels, 3),
            dtype=self._features_rest.dtype,
            device=self._features_rest.device,
        )
        if current_rest_channels > 0:
            expanded_features[:, :current_rest_channels, :] = self._features_rest.detach()
        self._replace_feature_rest_with_optimizer_state(expanded_features)
        return True

    def oneupSHdegree(self) -> None:
        """
            Increase the active SH degree by one, expanding capacity first if needed.
        """
        if self.active_sh_degree < self.max_sh_degree:
            self.ensure_sh_degree_capacity(self.max_sh_degree)
            self.active_sh_degree += 1
