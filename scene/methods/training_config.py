"""
BoundedGS execution configuration parser.

Validate BoundedGS execution options and the optional GNS pruning stage.
"""
from typing import Any


def _read_bool(opt: Any, field_name: str, default_value: bool) -> bool:
    """
        Read a boolean option and reject non-boolean values after argparse parsing.
    """
    raw_value = getattr(opt, field_name, default_value)
    if isinstance(raw_value, bool):
        return raw_value
    raise ValueError("{} only accepts true/false.".format(field_name))


def build_training_config(opt: Any) -> dict[str, Any]:
    """
        Validate spatial execution settings and the GNS switch.
    """
    group_grid = int(getattr(opt, "group_grid", 1))
    if group_grid < 1:
        raise ValueError("group_grid must be >= 1.")
    projection_mode = str(getattr(opt, "projection_mode", "full")).strip().lower()
    if projection_mode not in {"full", "simple_split", "group_compact"}:
        raise ValueError("projection_mode only accepts full/simple_split/group_compact.")
    if projection_mode == "simple_split" and group_grid <= 1:
        raise ValueError("projection_mode={} requires group_grid > 1.".format(projection_mode))
    structure_update_mode = str(
        getattr(opt, "structure_update_mode", "single_pass")
    ).strip().lower()
    if structure_update_mode not in {"multi_pass", "single_pass"}:
        raise ValueError("structure_update_mode only accepts multi_pass/single_pass.")
    structure_rebuild_backend = str(
        getattr(opt, "structure_rebuild_backend", "auto")
    ).strip().lower()
    if structure_rebuild_backend not in {"auto", "gpu", "cpu"}:
        raise ValueError("structure_rebuild_backend only accepts auto/gpu/cpu.")
    use_memory_lifecycle_trim = _read_bool(
        opt,
        "use_memory_lifecycle_trim",
        True,
    )
    leaf_raster_budget_mib = float(
        getattr(opt, "leaf_raster_budget_mib", 0.0)
    )
    if leaf_raster_budget_mib < 0.0:
        raise ValueError("leaf_raster_budget_mib must be >= 0.")
    if leaf_raster_budget_mib > 0.0:
        if projection_mode not in {"full", "group_compact"} or (
            group_grid == 1 and projection_mode != "group_compact"
        ):
            raise ValueError(
                "Leaf budgeting requires group_compact, or projection_mode=full with group_grid > 1."
            )
    return {
        "use_gns_pruning": _read_bool(opt, "use_gns_pruning", True),
        "group_grid": group_grid,
        "projection_mode": projection_mode,
        "use_compact_parameter_gradients": _read_bool(opt, "use_compact_parameter_gradients", True),
        "structure_update_mode": structure_update_mode,
        "structure_rebuild_backend": structure_rebuild_backend,
        "use_memory_lifecycle_trim": use_memory_lifecycle_trim,
        "leaf_raster_budget_mib": leaf_raster_budget_mib,
    }
