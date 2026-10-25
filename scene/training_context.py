"""
Shared training context.

The training loop is split into optimization, logging, densification, pruning,
and scheduling modules. This file defines the context object they all receive,
plus the runtime-state cache used for method-specific temporary data.
"""
from dataclasses import dataclass, field
from typing import Any, Optional, TypedDict

from scene import Scene
from scene.methods.training_config import build_training_config


class RuntimeState(TypedDict, total=False):
    """
        Optional scratch fields shared across training stages.

        Values here are created once and updated over many iterations, such as
        GNS pruning state.
    """
    opacity_min: Optional[float]
    active_bounded_execution_stats: dict[str, Any]
    gns_lr_scaled: bool
    gns_finished: bool
    reg_prune_finished: bool
    reg_prune_target_budget: Optional[int]
    reg_prune_budget_reference_count: Optional[int]
    scene: Optional[Scene]
    pipe: Any


@dataclass(slots=True)
class TrainingContext:
    """
        Bundle long-lived objects needed by training stages.

        Passing this object keeps stage function signatures short while still
        making dataset, optimizer options, renderer options, scene, and runtime
        state explicit.
    """
    dataset: Any
    opt: Any
    pipe: Any
    runtime_args: Any
    scene: Optional[Scene] = None
    method_config: dict[str, Any] = field(default_factory=dict)
    runtime_state: RuntimeState = field(default_factory=dict)


def initialize_runtime_state() -> RuntimeState:
    """
        Create method-specific runtime state after the scene and model exist.
    """
    runtime_state: RuntimeState = {
        "opacity_min": None,
        "gns_lr_scaled": False,
        "gns_finished": False,
        "reg_prune_finished": False,
        "reg_prune_target_budget": None,
        "reg_prune_budget_reference_count": None,
        "scene": None,
        "pipe": None,
    }
    return runtime_state


def build_training_context(dataset: Any, opt: Any, pipe: Any, runtime_args: Any) -> TrainingContext:
    """
        Build the context skeleton before scene creation.

        Method settings are parsed early and written back to `opt`, so later
        modules can read both normal options and resolved method switches from
        the same object.
    """
    method_config = build_training_config(opt)
    if int(method_config.get("group_grid", 1)) > 1 or method_config.get("projection_mode") == "group_compact":
        group_grid = int(method_config["group_grid"])
        projection_mode = str(method_config.get("projection_mode", "full"))
        execution_name = {
            "full": "Simple BoundedGS shared projection",
            "simple_split": "independent Simple Split",
            "group_compact": "Full BoundedGS compact projection",
        }[projection_mode]
        print(
            "[BoundedGS] Enabled {} with a {}x{} Group grid and GNS={}; SSIM is evaluated independently inside each Leaf.".format(
                execution_name,
                group_grid,
                group_grid,
                method_config["use_gns_pruning"],
            )
        )
    # Store resolved method settings on opt so later stage calls do not repeat parsing.
    for key, value in method_config.items():
        setattr(opt, key, value)
    return TrainingContext(
        dataset=dataset,
        opt=opt,
        pipe=pipe,
        runtime_args=runtime_args,
        method_config=method_config,
        runtime_state={},
    )


def attach_scene_to_context(
    context: TrainingContext,
    scene: Scene,
) -> TrainingContext:
    """
        Attach scene-dependent objects and initialize the runtime-state cache.
    """
    context.scene = scene
    context.runtime_state = initialize_runtime_state()
    return context
