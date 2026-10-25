"""
Main training script for 3DGS experiments.

The script builds the experiment output folder, creates the Gaussian model and
scene, then runs the iteration loop. Each loop iteration renders one training
view, computes losses, updates model parameters, and optionally changes the
Gaussian structure.
"""
from collections import deque
import signal
import sys
from argparse import Namespace
import torch
from tqdm import tqdm
from arguments import GroupParams
from utils.experiment_utils import finalize_training_parameters
from utils.general_utils import safe_state
from utils.image_prefetch import CameraImagePrefetcher, resolve_jpeg_prefetch_count
from utils.in_memory_jpeg_cache import InMemoryJpegCache
from utils.resource_monitor import (
    TrainingResourceMonitor,
    resource_status_for_exception,
)
from scene import Scene
from scene.methods.initialization_3dgs import build_gaussian_model_3dgs
from scene.methods.optimization_methods import run_3dgs_optimization_method
from scene.methods.pruning_methods import run_3dgs_pruning_method
from scene.methods.densification_stage import run_3dgs_densification_method
from scene.training_loop import (
    run_checkpoint_stage,
    run_3dgs_parameter_update_method,
    run_log_and_save_stage,
    update_3dgs_training_schedule,
    update_training_progress_bar,
)
from scene.training_context import attach_scene_to_context, build_training_context
from scene.training_runtime import (
    build_training_loop_state,
    is_inplace_checkpoint_resume,
    load_training_checkpoint,
    parse_training_arguments,
    prepare_output_and_logger,
    resolve_start_checkpoint_path,
    restore_iteration_state,
    sample_training_viewpoint,
    synchronized_timestamp,
)






def training(
    dataset: GroupParams,
    opt: GroupParams,
    pipe: GroupParams,
    runtime_args: Namespace,
    resource_monitor: TrainingResourceMonitor,
) -> float | None:
    """
        Run one complete training job and return pure training time in seconds.

        The function keeps the high-level training order in one place: prepare
        files, build shared context, restore checkpoint if needed, then repeat
        optimization, logging, densification, and pruning until the final
        iteration.
    """
    # Prepare the output folder before any model state is created, so config and logs match this run.
    prepare_output_and_logger(dataset, runtime_args)
    resource_monitor.set_output_path(dataset.model_path)

    first_iter = 0
    checkpoint_payload: dict = {}
    checkpoint_path = resolve_start_checkpoint_path(dataset.model_path, runtime_args)
    if checkpoint_path is not None:
        model_params, first_iter, checkpoint_payload = load_training_checkpoint(checkpoint_path)
        if first_iter > opt.iterations:
            raise ValueError("Checkpoint iteration exceeds the planned training iterations.")
        if (first_iter == opt.iterations
                and is_inplace_checkpoint_resume(runtime_args, dataset.model_path)
                and resource_monitor.keep_completed_summary(first_iter)):
            print("[checkpoint] Training is already complete; keeping its resource summary and timing.")
            return None
        resource_monitor.restore_checkpoint_state(checkpoint_payload.get("resource_state"), first_iter)
    InMemoryJpegCache(dataset).prepare()
    resource_monitor.record_jpeg_cache(dataset)

    # Build shared objects in dependency order: config context, Gaussian model, scene, then optimizer state.
    training_context = build_training_context(dataset, opt, pipe, runtime_args)
    dataset.group_grid = int(training_context.method_config.get("group_grid", 1))
    gaussians = build_gaussian_model_3dgs(dataset, opt)
    scene = Scene(dataset, gaussians)
    training_context = attach_scene_to_context(training_context, scene)
    gaussians.training_setup(opt)

    # Restore checkpoint state before loop state is built, so later stages see the resumed model.
    if checkpoint_path is not None:
        gaussians.restore(model_params, opt)

    loop_state = build_training_loop_state(
        training_context,
        bool(runtime_args.report_lpips_test),
        bool(runtime_args.report_lpips_train),
    )
    resumed_pending_viewpoints = restore_iteration_state(
        checkpoint_payload,
        scene,
        training_context.runtime_state,
        loop_state,
        opt,
    )
    if checkpoint_path is not None:
        print("[checkpoint] Resuming from iteration {}: {}".format(first_iter, checkpoint_path))
        del model_params
    del checkpoint_payload
    resource_monitor.configure_scene(scene, training_context.method_config, first_iter)
    resource_monitor.update_gaussian_count(gaussians.get_xyz.shape[0])
    if first_iter == opt.iterations:
        return resource_monitor.pure_training_seconds
    ema_loss_for_log = 0.0
    report_time_seconds = 0.0
    progress_bar = tqdm(
        range(first_iter, opt.iterations),
        desc="Training progress",
        ncols=runtime_args.progress_bar_width,
        dynamic_ncols=False,
    )
    active_training_start = synchronized_timestamp()
    jpeg_cache_enabled = bool(getattr(dataset, "in_memory_jpeg_images", {}))
    prefetch_count = resolve_jpeg_prefetch_count(dataset) if jpeg_cache_enabled else 1
    image_prefetcher = CameraImagePrefetcher(
        jpeg_cache_enabled,
        prefetch_count,
    )

    def update_resource_statistics() -> None:
        resource_monitor.record_image_prefetch(image_prefetcher.statistics())
        resource_monitor.record_structure_rebuild_backends(gaussians)
        ply_seconds = loop_state.get("ply_save_seconds", 0.0)
        pure_seconds = (synchronized_timestamp() - active_training_start
                        - report_time_seconds - ply_seconds - resource_monitor.monitor_overhead_seconds)
        resource_monitor.set_training_times(max(pure_seconds, 0.0), report_time_seconds, ply_seconds)

    try:
        first_iter += 1
        viewpoint_queue: deque = deque()
        remaining_iterations = max(int(opt.iterations) - first_iter + 1, 0)
        target_prefetch_count = min(prefetch_count, remaining_iterations)
        if len(resumed_pending_viewpoints) > target_prefetch_count:
            raise ValueError("Checkpoint contains more pending viewpoints than remaining training iterations.")
        for camera in resumed_pending_viewpoints:
            viewpoint_queue.append(camera)
            image_prefetcher.submit(camera)
        while len(viewpoint_queue) < target_prefetch_count:
            camera = sample_training_viewpoint(scene, loop_state["viewpoint_stack"])
            viewpoint_queue.append(camera)
            image_prefetcher.submit(camera)
        remaining_submissions = remaining_iterations - len(viewpoint_queue)
        for iteration in range(first_iter, opt.iterations + 1):
            resource_monitor.record_iteration_start(
                iteration,
                gaussians.get_xyz.shape[0],
                gaussians,
            )
            # Update time-based settings, such as SH degree and method-specific training cadence.
            update_3dgs_training_schedule(training_context, gaussians, iteration)

            # Pick one training camera and build gradients from rendering plus all active loss terms.
            viewpoint_cam = viewpoint_queue.popleft()
            resource_monitor.set_camera(viewpoint_cam)
            image_prefetcher.wait(viewpoint_cam)
            if remaining_submissions > 0:
                camera = sample_training_viewpoint(scene, loop_state["viewpoint_stack"])
                viewpoint_queue.append(camera)
                image_prefetcher.submit(camera)
                remaining_submissions -= 1
            try:
                with resource_monitor.phase("optimization"):
                    optimization_outputs = run_3dgs_optimization_method(
                        training_context,
                        gaussians,
                        loop_state,
                        iteration,
                        viewpoint_cam,
                    )
            except Exception:
                resource_monitor.record_bounded_execution(
                    training_context.runtime_state.pop(
                        "active_bounded_execution_stats",
                        None,
                    )
                )
                raise
            viewpoint_cam.release_image_cache()
            loss = optimization_outputs["loss"]
            render_state = optimization_outputs["render_state"]
            resource_monitor.record_bounded_execution(
                render_state.get("bounded_execution_stats")
            )
            training_context.runtime_state.pop("active_bounded_execution_stats", None)

            with torch.no_grad():
                # After gradients are ready, run side-effect stages that should not be tracked by autograd.
                ema_loss_for_log = update_training_progress_bar(
                    progress_bar,
                    iteration,
                    ema_loss_for_log,
                    loss.item(),
                    gaussians.get_xyz.shape[0],
                )
                # Apply optimizer steps only on iterations allowed by the current method schedule.
                run_3dgs_parameter_update_method(training_context, gaussians, iteration)
                # Backward and all optimizer steps are complete. Keep the extracted render state for
                # densification, but release the loss graph before a structure rebuild replaces parameters.
                del loss, optimization_outputs
                # Save images and evaluation logs as one auxiliary stage.
                # Keep this before densification/pruning so normal report/save semantics match the original training loop.
                has_report_or_save = (
                    iteration in loop_state["testing_iterations"]
                    or iteration in loop_state["saving_iterations"]
                )
                if has_report_or_save:
                    with resource_monitor.phase("report_save"):
                        report_time_seconds += run_log_and_save_stage(
                            training_context,
                            gaussians,
                            loop_state,
                            iteration,
                        )
                else:
                    report_time_seconds += run_log_and_save_stage(
                        training_context,
                        gaussians,
                        loop_state,
                        iteration,
                    )
                # Grow, split, or rebuild Gaussians according to the selected method.
                with resource_monitor.phase("structure"):
                    structure_count_before = int(gaussians.get_xyz.shape[0])
                    run_3dgs_densification_method(training_context, gaussians, iteration, render_state)
                    use_memory_lifecycle_trim = bool(
                        training_context.method_config.get(
                            "use_memory_lifecycle_trim",
                            True,
                        )
                    )
                    if use_memory_lifecycle_trim:
                        render_state.clear()
                        del render_state
                    # Remove low-importance Gaussians after densification so the model stays within budget.
                    run_3dgs_pruning_method(training_context, gaussians, iteration)
                    if not use_memory_lifecycle_trim:
                        render_state.clear()
                        del render_state
                    structure_count_after = int(gaussians.get_xyz.shape[0])
                    resource_monitor.record_structure_transition(
                        iteration,
                        structure_count_before,
                        structure_count_after,
                    )
                    resource_monitor.update_gaussian_count(structure_count_after)
                resource_monitor.mark_iteration_completed(iteration, gaussians.get_xyz.shape[0])
                resource_state = None
                if iteration in loop_state["checkpoint_iterations"]:
                    update_resource_statistics()
                    resource_state = resource_monitor.checkpoint_state()
                run_checkpoint_stage(
                    training_context,
                    gaussians,
                    loop_state,
                    iteration,
                    list(viewpoint_queue),
                    resource_state,
                )
    except BaseException as error:
        # Persist failure/interrupt peaks before decoder-thread cleanup can delay
        # process exit long enough for the batch runner to force-kill this job.
        try:
            update_resource_statistics()
        except Exception:
            pass  # Preserve the original training/CUDA failure.
        _finalize_resource_summary(
            resource_monitor,
            dataset.model_path,
            resource_status_for_exception(error),
            error,
        )
        raise
    finally:
        image_prefetcher.close()
        resource_monitor.record_image_prefetch(image_prefetcher.statistics())
        if image_prefetcher.enabled:
            print(image_prefetcher.summary())
        progress_bar.close()

    update_resource_statistics()
    return resource_monitor.pure_training_seconds


def _raise_keyboard_interrupt_on_sigterm(signum: int, frame: object) -> None:
    del signum, frame
    raise KeyboardInterrupt


def _finalize_resource_summary(
    resource_monitor: TrainingResourceMonitor,
    model_path: str,
    status: str,
    error: BaseException | None = None,
) -> None:
    resource_monitor.set_output_path(model_path)
    try:
        resource_monitor.finalize(status, error)
    except Exception as summary_error:
        print("[Resources] Failed to write resource summary: {}".format(summary_error))


if __name__ == "__main__":
    lp, op, pp, args = parse_training_arguments(sys.argv[1:])
    if bool(args.save_final_model) and args.iterations not in args.save_iterations:
        args.save_iterations.append(args.iterations)
    if not bool(args.save_final_model):
        args.save_iterations = [
            iteration for iteration in args.save_iterations if iteration != args.iterations
        ]
    # Split the flat argparse namespace into model, optimization, and renderer parameter groups.
    model_args = lp.extract(args)
    opt_args = op.extract(args)
    pipe_args = pp.extract(args)
    resource_monitor = TrainingResourceMonitor(model_args, opt_args, args)
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt_on_sigterm)
    if sys.platform == "win32":
        signal.signal(signal.SIGBREAK, _raise_keyboard_interrupt_on_sigterm)
    try:
        safe_state(args.quiet)
        torch.autograd.set_detect_anomaly(args.detect_anomaly)
        print("Optimizing " + args.model_path)
        pure_training_seconds = training(model_args, opt_args, pipe_args, args, resource_monitor)
        if pure_training_seconds is not None:
            finalize_training_parameters(model_args.model_path, pure_training_seconds)
    except KeyboardInterrupt as error:
        _finalize_resource_summary(resource_monitor, model_args.model_path, "interrupted", error)
        print("\nTraining interrupted by user.")
        raise SystemExit(130)
    except BaseException as error:
        _finalize_resource_summary(
            resource_monitor,
            model_args.model_path,
            resource_status_for_exception(error),
            error,
        )
        raise
    _finalize_resource_summary(resource_monitor, model_args.model_path, "completed")

    print("\nTraining complete.")
