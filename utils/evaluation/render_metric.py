"""
Post-processing rendering and metric helpers.

This module loads a saved Gaussian model, renders requested data splits, writes
optional render/ground-truth images, computes PSNR/SSIM/LPIPS, benchmarks FPS,
and stores compact JSON summaries for batch aggregation.
"""
import os
import time
from os import makedirs
from typing import Any, Optional

import torch
import torchvision
from tqdm import tqdm

from gaussian_renderer import render
from lpipsPyTorch.modules.lpips import LPIPS
from scene import Scene
from scene.gaussian_model import GaussianModel as GaussianModel3DGS
from utils.experiment_utils import TRAINING_PARAMETERS_FILENAME, load_json, save_json
from utils.evaluation.lpips_metric import compute_lpips
from utils.evaluation.bounded_evaluation import render_bounded_inference_image
from utils.image_prefetch import iter_prefetched_camera_images, resolve_jpeg_prefetch_count
from utils.in_memory_jpeg_cache import InMemoryJpegCache
from utils.loss_utils import psnr
from utils.loss_utils import ssim

PROGRESS_BAR_WIDTH = 100
OPACITY_INTERVAL_BOUNDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]


def format_metric(value: Optional[float], decimals: int) -> str:
    """
        Format a metric for JSON/CSV summaries, leaving missing values blank.
    """
    if value in (None, ""):
        return ""
    return f"{float(value):.{decimals}f}"


def render_path_for(model_path: str, split_name: str, iteration: int) -> str:
    """
        Return the output folder for one split and model iteration.
    """
    return os.path.join(model_path, split_name, f"ours_{iteration}")


def read_training_elapsed_minutes(model_path: str) -> Optional[float]:
    """
        Read pure training time saved by the training script.

    """
    training_parameters = load_json(
        os.path.join(model_path, TRAINING_PARAMETERS_FILENAME),
        default={},
    )
    elapsed_minutes = training_parameters.get("elapsed_minutes")
    return None if elapsed_minutes is None else float(elapsed_minutes)


def build_background(dataset: Any) -> torch.Tensor:
    """
        Create the CUDA background tensor that matches the dataset setting.
    """
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    return torch.tensor(bg_color, dtype=torch.float32, device="cuda")


def render_inference_image(
    view: Any,
    gaussians: GaussianModel3DGS,
    pipeline: Any,
    background: torch.Tensor,
    group_grid: int = 1,
) -> torch.Tensor:
    """
        Render one view in inference mode without keeping training gradients.
    """
    if int(group_grid) > 1:
        return render_bounded_inference_image(
            view,
            gaussians,
            pipeline,
            background,
            int(group_grid),
        )
    image = render(
        view,
        gaussians,
        pipeline,
        background,
        track_gradients=False,
        inference_only=True,
    )["render"]
    return image


def build_split_specs(
    scene: Scene,
    include_train: bool,
    skip_test: bool,
    no_save_test_images: bool,
    no_save_train_images: bool,
) -> list[tuple[str, list[Any], bool]]:
    """
        Build the train/test split plan requested by CLI flags.
    """
    split_specs = []
    if not skip_test:
        test_views = scene.getTestCameras()
        split_specs.append(("test", test_views, not no_save_test_images))
    if include_train:
        train_views = scene.getTrainCameras()
        split_specs.append(("train", train_views, not no_save_train_images))
    return split_specs


def benchmark_fps(
    views: list[Any],
    gaussians: GaussianModel3DGS,
    pipeline: Any,
    background: torch.Tensor,
    warmup_rounds: int,
    measure_rounds: int,
    group_grid: int = 1,
) -> Optional[float]:
    """
        Measure average FPS by repeatedly rendering all views.

        Warmup rounds let CUDA kernels and caches settle before the timed rounds.
    """
    if not views or measure_rounds <= 0:
        return None

    for _ in range(max(0, warmup_rounds)):
        with torch.inference_mode():
            for view in views:
                render_inference_image(
                    view,
                    gaussians,
                    pipeline,
                    background,
                    group_grid,
                )
        torch.cuda.synchronize()

    total_time = 0.0
    total_frames = 0
    for _ in range(measure_rounds):
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        with torch.inference_mode():
            for view in views:
                render_inference_image(
                    view,
                    gaussians,
                    pipeline,
                    background,
                    group_grid,
                )
        torch.cuda.synchronize()
        total_time += time.perf_counter() - start_time
        total_frames += len(views)

    if total_time <= 0:
        return None
    return total_frames / total_time


def summarize_opacity_intervals(gaussians: GaussianModel3DGS) -> dict[str, dict[str, float]]:
    """
        Count Gaussians in opacity intervals to make model sparsity easy to inspect.
    """
    opacity = gaussians.get_opacity.detach().flatten()
    total_count = int(opacity.numel())
    mean_opacity = 0.0 if total_count == 0 else float(opacity.mean().item())
    interval_summary = {}
    for idx in range(len(OPACITY_INTERVAL_BOUNDS) - 1):
        left = OPACITY_INTERVAL_BOUNDS[idx]
        right = OPACITY_INTERVAL_BOUNDS[idx + 1]
        if idx == len(OPACITY_INTERVAL_BOUNDS) - 2:
            mask = (opacity >= left) & (opacity <= 1.0)
            label = "[{:.1f}, {:.1f}]".format(left, 1.0)
        else:
            mask = (opacity >= left) & (opacity < right)
            label = "[{:.1f}, {:.1f})".format(left, right)
        count = int(mask.sum().item())
        ratio = 0.0 if total_count == 0 else count / total_count
        interval_summary[label] = {"count": count, "ratio": ratio}
    return {"mean_opacity": mean_opacity, "bins": interval_summary}


def build_opacity_report(
    model_path: str,
    iteration: int,
    opacity_stats: dict[str, Any],
    total_count: int,
) -> str:
    """
        Build the text content for the standalone opacity report.
    """
    lines = [
        "Opacity Statistics Report",
        "=" * 78,
        f"Model Path     : {model_path}",
        f"Iteration      : {iteration}",
        f"Total Gaussians: {total_count}",
        f"Mean Opacity   : {format_metric(opacity_stats['mean_opacity'], 6)}",
        "-" * 78,
        f"{'Opacity Interval':<20} {'Count':>12} {'Ratio':>12}",
        "-" * 78,
    ]
    for interval_name, interval_stats in opacity_stats["bins"].items():
        lines.append(
            "{:<20} {:>12} {:>12.2%}".format(
                interval_name,
                interval_stats["count"],
                interval_stats["ratio"],
            )
        )
    lines.append("=" * 78)
    return "\n".join(lines) + "\n"


def write_opacity_report(model_path: str, iteration: int, gaussians: GaussianModel3DGS) -> None:
    """
        Save opacity interval statistics next to the experiment output.
    """
    opacity_stats = summarize_opacity_intervals(gaussians)
    report_text = build_opacity_report(
        model_path=model_path,
        iteration=iteration,
        opacity_stats=opacity_stats,
        total_count=int(gaussians.get_opacity.shape[0]),
    )
    report_path = os.path.join(model_path, "opacity_report.txt")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(report_text)
    print("Opacity statistics saved to {}".format(report_path))


def print_split_summary(split_name: str, summary: dict[str, Any]) -> None:
    """
        Print one split's aggregate metrics in a compact readable block.
    """
    print("Split: {}".format(split_name))
    print("  SSIM : {:>12}".format(format_metric(summary["SSIM"], 5)))
    print("  PSNR : {:>12}".format(format_metric(summary["PSNR"], 3)))
    print("  LPIPS: {:>12}".format(format_metric(summary["LPIPS"], 5)))
    print("  FPS  : {:>12}".format(format_metric(summary["FPS"], 1)))
    print("  NUM  : {:>12}".format(summary["NUM"]))
    print("  Time : {:>12}".format(format_metric(summary["Training_time"], 2)))
    print("")


def evaluate_split(
    dataset: Any,
    split_name: str,
    views: list[Any],
    iteration: int,
    gaussians: GaussianModel3DGS,
    pipeline: Any,
    background: torch.Tensor,
    lpips_model: LPIPS,
    save_images: bool,
    fps_warmup_rounds: int,
    fps_measure_rounds: int,
    training_time_minutes: Optional[float],
    group_grid: int = 1,
    low_memory_lpips: bool = False,
) -> None:
    """
        Evaluate one split from start to finish.

        For each view, the function renders the image, optionally saves render
        and ground truth PNGs, records per-view metrics, then writes aggregate
        summary JSON and FPS.
    """
    if not views:
        return

    split_root = render_path_for(dataset.model_path, split_name, iteration)
    render_root = os.path.join(split_root, "renders")
    gt_root = os.path.join(split_root, "gt")
    if save_images:
        makedirs(render_root, exist_ok=True)
        makedirs(gt_root, exist_ok=True)

    per_view = {"PSNR": {}, "SSIM": {}, "LPIPS": {}}
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0

    for idx, view in enumerate(
        tqdm(
            iter_prefetched_camera_images(
                views,
                prefetch_count=resolve_jpeg_prefetch_count(dataset),
            ),
            total=len(views),
            desc=f"{split_name} render+metric",
            ncols=PROGRESS_BAR_WIDTH,
            dynamic_ncols=False,
        )
    ):
        image_name = "{:05d}.png".format(idx)
        with torch.inference_mode():
            rendering = torch.clamp(
                render_inference_image(
                    view,
                    gaussians,
                    pipeline,
                    background,
                    group_grid=int(group_grid),
                ),
                0.0,
                1.0,
            )
        gt = torch.clamp(view.original_image_cuda()[0:3, :, :], 0.0, 1.0)

        if save_images:
            torchvision.utils.save_image(rendering, os.path.join(render_root, image_name))
            torchvision.utils.save_image(gt, os.path.join(gt_root, image_name))

        rendering_batch = rendering.unsqueeze(0)
        gt_batch = gt.unsqueeze(0)
        psnr_value = float(psnr(rendering_batch, gt_batch).item())
        ssim_value = float(ssim(rendering_batch, gt_batch).item())
        lpips_value = float(
            compute_lpips(
                lpips_model,
                rendering_batch,
                gt_batch,
                low_memory_lpips,
            ).item()
        )
        del rendering, gt, rendering_batch, gt_batch

        view.release_image_cache()
        total_psnr += psnr_value
        total_ssim += ssim_value
        total_lpips += lpips_value

        per_view["PSNR"][image_name] = psnr_value
        per_view["SSIM"][image_name] = ssim_value
        per_view["LPIPS"][image_name] = lpips_value

    save_json(os.path.join(split_root, "per_view.json"), per_view)
    fps_value = benchmark_fps(
        views=views,
        gaussians=gaussians,
        pipeline=pipeline,
        background=background,
        warmup_rounds=fps_warmup_rounds,
        measure_rounds=fps_measure_rounds,
        group_grid=int(group_grid),
    )

    summary = {
        "method": "ours_{}".format(iteration),
        "iteration": int(iteration),
        "image_count": int(len(views)),
        "PSNR": total_psnr / len(views),
        "SSIM": total_ssim / len(views),
        "LPIPS": total_lpips / len(views),
        "FPS": fps_value,
        "NUM": int(gaussians.get_opacity.shape[0]),
        "Training_time": training_time_minutes,
        "Metric_protocol": {
            "render_group_grid": int(group_grid),
            "projection": "full",
            "PSNR": "global",
            "SSIM": "global",
            "LPIPS": "global_tiled_equivalent" if low_memory_lpips else "global_full",
        },
    }
    save_json(os.path.join(dataset.model_path, "result_{}.json".format(split_name)), summary)
    print_split_summary(split_name, summary)


def process_splits(
    dataset: Any,
    iteration: int,
    pipeline: Any,
    include_train: bool,
    skip_test: bool,
    no_save_test_images: bool,
    no_save_train_images: bool,
    fps_warmup_rounds: int,
    fps_measure_rounds: int,
    group_grid: int = 1,
    low_memory_lpips: bool = False,
) -> None:
    """
        Load the trained scene and evaluate every requested split.

        The function creates the Gaussian model, loads the requested iteration
        through Scene, prepares LPIPS/background helpers, writes the opacity
        report, then calls `evaluate_split` for test and/or train cameras.
    """
    group_grid = int(group_grid)
    if group_grid < 1:
        raise ValueError("group_grid must be >= 1.")
    with torch.no_grad():
        dataset.group_grid = group_grid
        dataset.lazy_image_loading = group_grid > 1
        InMemoryJpegCache(dataset).prepare()
        gaussians = GaussianModel3DGS(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        lpips_model = LPIPS("vgg").to("cuda").eval()
        background = build_background(dataset)
        training_time_minutes = read_training_elapsed_minutes(dataset.model_path)
        write_opacity_report(dataset.model_path, scene.loaded_iter, gaussians)

        split_specs = build_split_specs(
            scene,
            include_train,
            skip_test,
            no_save_test_images,
            no_save_train_images,
        )
        for split_name, views, save_images in split_specs:
            evaluate_split(
                dataset=dataset,
                split_name=split_name,
                views=views,
                iteration=scene.loaded_iter,
                gaussians=gaussians,
                pipeline=pipeline,
                background=background,
                lpips_model=lpips_model,
                save_images=save_images,
                fps_warmup_rounds=fps_warmup_rounds,
                fps_measure_rounds=fps_measure_rounds,
                training_time_minutes=training_time_minutes,
                group_grid=group_grid,
                low_memory_lpips=low_memory_lpips,
            )
