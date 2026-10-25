"""
Command-line argument system.

Each parameter group declares defaults as class attributes. ParamGroup then
turns those attributes into argparse options and later extracts only the fields
that belong to that group. This keeps the training entry point readable while
still producing normal Python objects for the rest of the code.
"""
from __future__ import annotations
import ast
from argparse import ArgumentParser, Namespace
import os
import sys


def parse_bool_arg(raw_value: str | bool) -> bool:
    """
        Parse flexible CLI boolean text into real `True` or `False` values.

        Argparse passes strings for values such as `--flag false`; this helper
        accepts common spellings and raises a clear error for anything else.
    """
    if isinstance(raw_value, bool):
        return raw_value
    normalized = str(raw_value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError("Unsupported boolean value: {}".format(raw_value))


class GroupParams:
    """
        Simple object used to hold extracted parameters as attributes.

        The project historically reads options with dot access, so this class
        intentionally stays empty and lets ParamGroup attach fields dynamically.
    """
    pass


class ParamGroup:
    """
        Base class that registers one group of argparse options from defaults.
    """
    def __init__(self, parser: ArgumentParser, name: str, fill_none: bool = False) -> None:
        self._group_name = name
        self._fill_none = fill_none
        if not hasattr(self, "_fields"):
            self._fields: dict[str, object] = {}
        self._register_group(parser)

    def _register_group(self, parser: ArgumentParser) -> None:
        """
            Convert fields in `self._fields` into argparse options.

            A leading underscore means the option also gets a one-letter short
            form, for example `_source_path` becomes `--source_path` and `-s`.
        """
        group = parser.add_argument_group(self._group_name)
        for raw_key, raw_value in self._fields.items():
            shorthand = False
            key = raw_key
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            value = raw_value if not self._fill_none else None
            value_type = type(raw_value)
            if shorthand:
                if value_type is bool or raw_value is None:
                    group.add_argument(
                        "--" + key,
                        "-" + key[0:1],
                        default=value,
                        nargs="?",
                        const=True,
                        type=parse_bool_arg,
                    )
                else:
                    group.add_argument("--" + key, "-" + key[0:1], default=value, type=value_type)
            else:
                if value_type is bool or raw_value is None:
                    group.add_argument(
                        "--" + key,
                        default=value,
                        nargs="?",
                        const=True,
                        type=parse_bool_arg,
                    )
                else:
                    group.add_argument("--" + key, default=value, type=value_type)

    def extract(self, args: Namespace) -> GroupParams:
        """
            Copy only this group's parsed fields from the shared argparse namespace.
        """
        group = GroupParams()
        for raw_key, raw_default in self._fields.items():
            key = raw_key[1:] if raw_key.startswith("_") else raw_key
            value = getattr(args, key, raw_default)
            if self._fill_none and value is None:
                value = raw_default
            setattr(group, key, value)
        return group


class BoundedGSParams(ParamGroup):
    """Group/Leaf execution, memory layout, and bounded image supply."""
    data_defaults = {
        "use_in_memory_jpeg_cache": False,
        "jpeg_prefetch_count": 10,
        "training_gt_mode": "leaf",
    }

    def __init__(self, parser: ArgumentParser, sentinel: bool = False) -> None:
        for name, default in self.data_defaults.items():
            setattr(self, name, default)
        self.group_grid = 1  # Projection groups per image side
        self.projection_mode = "full"  # full, simple_split, or group_compact
        self.structure_update_mode = "single_pass"  # single_pass or multi_pass
        self.structure_rebuild_backend = "auto"  # auto, gpu, or cpu
        self.use_compact_parameter_gradients = True
        self.use_memory_lifecycle_trim = True
        self.leaf_raster_budget_mib = 0.0  # Leaf payload budget; 0 uses fixed rectangles
        self._fields = dict(vars(self))
        super().__init__(parser, "BoundedGS Parameters", sentinel)


class ModelParams(ParamGroup):
    """Dataset and output-path options used before scene creation."""
    def __init__(self, parser: ArgumentParser, sentinel: bool = False) -> None:
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._depths = ""
        self._resolution = -1
        self._white_background = False
        self.init_type = "sfm"
        self.max_initial_points = -1
        self.data_device = "cpu"
        self.training_image_upsample_factor = 1
        self.eval = True
        self._fields = dict(vars(self))
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args: Namespace) -> GroupParams:
        group = super().extract(args)
        group.source_path = os.path.abspath(group.source_path)
        for name, default in BoundedGSParams.data_defaults.items():
            value = getattr(args, name, None)
            setattr(group, name, default if value is None else value)
        return group


class PipelineParams(ParamGroup):
    """Renderer options shared by training and evaluation."""
    def __init__(self, parser: ArgumentParser, sentinel: bool = False) -> None:
        self.debug = False
        self.antialiasing = False
        self._fields = dict(vars(self))
        super().__init__(parser, "Pipeline Parameters", sentinel)


class OptimizationBaseParams(ParamGroup):
    """Learning rates and image/depth objectives."""
    def __init__(self, parser: ArgumentParser) -> None:
        self.iterations = 30_000
        self.position_lr_init = 0.00004
        self.position_lr_final = 0.000002
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.025
        self.initial_opacity = 0.1
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.lambda_dssim = 0.2
        self.depth_l1_weight_init = 1.0
        self.depth_l1_weight_final = 0.01
        self.random_background = False
        self.optimizer_type = "ours_adam"  # default selects torch.optim.Adam
        self._fields = dict(vars(self))
        super().__init__(parser, "Optimization Parameters")


class DensificationParams(ParamGroup):
    """Mean ABS-gradient thresholding and budgeted long-axis splitting."""
    def __init__(self, parser: ArgumentParser) -> None:
        self.densification_interval = 100
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0003
        self.min_opacity = 0.005
        self.split_distance = 0.45
        self.opacity_reduction = 0.6
        self.budget = 3_000_000
        self.budget_multiplier = 3.0
        self.budget_warmup_until_offset = 500
        self._fields = dict(vars(self))
        super().__init__(parser, "Densification Parameters")


class GNSParams(ParamGroup):
    """Optional GNS regularization, opacity LR scaling, and budget pruning."""
    def __init__(self, parser: ArgumentParser) -> None:
        self.use_gns_pruning = True
        self.gns_opacity_reg = 0.002
        self.gns_opacity_lr_scale = 5.0
        self.final_budget = 600_000
        self.final_budget_mode = "rate"
        self.final_rate = 0.2
        self.reg_prune_from_iter = 15_000
        self.reg_prune_until_iter = 20_000
        self._fields = dict(vars(self))
        super().__init__(parser, "GNS Parameters")


class OptimizationParams:
    """
        Wrapper that registers and merges all optimization-related groups.
    """
    def __init__(self, parser: ArgumentParser) -> None:
        self.groups = [
            OptimizationBaseParams(parser),
            BoundedGSParams(parser),
            DensificationParams(parser),
            GNSParams(parser),
        ]

    def extract(self, args: Namespace) -> GroupParams:
        """
            Merge all optimization parameter groups into one object for training.
        """
        merged = GroupParams()
        for group in self.groups:
            partial = group.extract(args)
            for key, value in vars(partial).items():
                setattr(merged, key, value)
        return merged


def parse_cfg_args_namespace(cfgfile_string: str) -> Namespace:
    """
        Safely parse the legacy `cfg_args` Namespace(...) text format.
    """
    cfgfile_string = str(cfgfile_string).strip()
    if cfgfile_string == "":
        return Namespace()
    expression = ast.parse(cfgfile_string, mode="eval").body
    if not isinstance(expression, ast.Call) or not isinstance(expression.func, ast.Name):
        raise ValueError("cfg_args must contain a Namespace(...) expression.")
    if expression.func.id != "Namespace":
        raise ValueError("cfg_args must use Namespace(...).")
    if expression.args:
        raise ValueError("cfg_args only supports keyword arguments.")

    parsed_values = {}
    for keyword in expression.keywords:
        if keyword.arg is None:
            raise ValueError("cfg_args does not support **kwargs.")
        parsed_values[keyword.arg] = ast.literal_eval(keyword.value)
    return Namespace(**parsed_values)


def get_combined_args(parser: ArgumentParser) -> Namespace:
    """
        Merge command-line arguments with the saved `cfg_args` file in model_path.

        Values provided on the command line take priority, while missing values
        can be recovered from the training run configuration saved with a model.
    """
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except (TypeError, FileNotFoundError, OSError):
        print("Config file not found at {}".format(getattr(args_cmdline, "model_path", "")))
    try:
        args_cfgfile = parse_cfg_args_namespace(cfgfile_string)
    except (SyntaxError, ValueError) as error:
        print("Could not parse cfg_args safely: {}".format(error))
        args_cfgfile = Namespace()

    merged_dict = vars(args_cfgfile).copy()
    for key, value in vars(args_cmdline).items():
        if value is not None:
            merged_dict[key] = value
    return Namespace(**merged_dict)
