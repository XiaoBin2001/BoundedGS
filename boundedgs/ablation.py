"""Fixed-count training simulation, with one isolated process per method/resolution."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _worker(config, method, width, output, max_views):
    import math
    from utils.compact_gradients import (
        gradient_storage_bytes,
        long_term_storage_bytes,
        parameter_adam_storage_bytes,
    )
    from types import SimpleNamespace
    import numpy as np
    from PIL import Image
    import torch
    from diff_gaussian_rasterization_boundedgs import _ensure_gaussian_adam_state
    from scene.cameras import Camera
    from scene.gaussian_model import GaussianModel
    from scene.training_context import TrainingContext, initialize_runtime_state
    from scene.methods.training_config import build_training_config
    from scene.methods.optimization_methods import run_3dgs_optimization_method

    from utils import platform_utils
    if not platform_utils.WINDOWS:
        torch.set_num_threads(len(os.sched_getaffinity(0)))
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    output = Path(output)
    summary = dict(method=method, width=width, status='initializing', completed_views=0,
                   gpu_name=torch.cuda.get_device_name(), total_device_bytes=torch.cuda.get_device_properties(0).total_memory)
    dump(output / 'result.json', summary)
    rows = []
    phase = 'setup'
    current_camera = None
    begun = time.time()
    try:
        model_path = Path(config['model_path'])
        parameters = json.loads((model_path / 'training_parameters.json').read_text())
        all_entries = sorted(json.loads((model_path / 'cameras.json').read_text()), key=lambda c: c['img_name'])
        assert len({e['img_name'] for e in all_entries}) == len(all_entries)
        entries = [e for i, e in enumerate(all_entries) if i % 8 != 0]
        if config.get('expected_train_views') is not None:
            assert len(entries) == config['expected_train_views']
        camera_names = set(config.get('camera_names', []))
        if camera_names:
            entries = [entry for entry in entries if entry['img_name'] in camera_names]
            missing = camera_names - {entry['img_name'] for entry in entries}
            if missing:
                raise ValueError('Unknown camera_names: ' + ', '.join(sorted(missing)))
        centers = np.asarray([e['position'] for e in entries], dtype=np.float32)
        radius = float(np.linalg.norm(centers - centers.mean(axis=0), axis=1).max() * 1.1)
        if max_views:
            entries = entries[:max_views]
        parameters.update(group_grid=1 if method == 'Original' else config['initial_grid'],
                          projection_mode='full' if method == 'Original' else 'group_compact',
                          leaf_raster_budget_mib=0.0 if method == 'Original' else config['leaf_budget_mib'])
        if config.get('data_root'):
            parameters['source_path'] = str(Path(config['data_root']) / config['scene_name'])
        parameters.update(config.get('train_overrides', {}))
        parameters.update(config.get('spatial_methods', {}).get(method, {}))
        opt = SimpleNamespace(**parameters)
        context = TrainingContext(dataset=SimpleNamespace(white_background=parameters['white_background']),
                                  opt=opt, pipe=SimpleNamespace(debug=False, antialiasing=parameters['antialiasing']),
                                  runtime_args=SimpleNamespace(debug_from=-1),
                                  method_config=build_training_config(opt), runtime_state=initialize_runtime_state())
        # This is post-pruning reconstruction; no topology or opacity-pruning penalty.
        context.runtime_state['gns_finished'] = True
        model = GaussianModel(parameters['sh_degree'], 'ours_adam')
        model.load_ply(str(model_path / f"point_cloud/iteration_{config['model_iteration']}/point_cloud.ply"))
        n = len(model._xyz)
        if config.get('expected_gaussians') is not None:
            assert n == config['expected_gaussians']
        if n < config.get('minimum_gaussians', 1):
            raise ValueError('Source PLY does not contain the expected pre-pruning model population.')
        model.spatial_lr_scale = radius
        model.training_setup(opt)
        model.update_learning_rate(config['model_iteration'])
        raw = [g['params'][0] for g in model.optimizer.param_groups]
        gradient_bytes = sum(p.numel() * p.element_size() for p in raw)
        assert gradient_bytes == 236 * n
        for p in raw:
            _ensure_gaussian_adam_state(model.optimizer.state[p], n, p, model.optimizer.max_steps, 0)
        assert all(p.grad is None for p in raw)
        initial_samples = [p.detach().reshape(-1)[::max(1, p.numel() // 2048)].cpu().clone() for p in raw]
        cameras = []
        for i, e in enumerate(entries):
            camera_width = width or e['width']
            height = round(camera_width * e['height'] / e['width'])
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :3] = np.asarray(e['rotation'], dtype=np.float32)
            c2w[:3, 3] = np.asarray(e['position'], dtype=np.float32)
            w2c = np.linalg.inv(c2w)
            path = Path(parameters['source_path']) / parameters['images'] / e['img_name']
            assert path.is_file()
            cameras.append(Camera(resolution=(camera_width, height), colmap_id=i, R=w2c[:3, :3].T,
                                  T=w2c[:3, 3], FoVx=2*math.atan(e['width']/(2*e['fx'])),
                                  FoVy=2*math.atan(e['height']/(2*e['fy'])), depth_params=None,
                                  image=None, image_path=str(path), encoded_jpeg=None, depth_path='',
                                  depth_is_nerf_synthetic=False, image_name=e['img_name'], uid=i,
                                  data_device='cpu', lazy_image_loading=True,
                                  training_gt_mode=parameters.get('training_gt_mode', 'leaf')))
        assert len({c.image_height for c in cameras}) == 1
        passes = int(config.get('passes', 1))
        if passes < 1:
            raise ValueError('passes must be >= 1')
        total_steps = len(cameras) * passes
        loop = {'background': torch.tensor([1., 1., 1.] if parameters['white_background'] else [0., 0., 0.], device='cuda'),
                'depth_l1_weight': lambda _: 0.0}
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        summary.update(status='running', num_gaussians=n, gradient_bytes=gradient_bytes,
                       width=cameras[0].image_width, height=cameras[0].image_height,
                       requested_views=len(cameras), requested_steps=total_steps, passes=passes,
                       spatial_lr_scale=radius, parameter_lrs={g['name']:g['lr'] for g in model.optimizer.param_groups},
                       camera_order=[c.image_name for c in cameras], adam_state='fresh zero moments, step0; allocated before measurement',
                       setup_seconds=time.time()-begun)
        dump(output / 'result.json', summary)
        with (output / 'views.jsonl').open('w') as log:
            for index, camera in enumerate(cameras * passes, 1):
                current_camera = camera.image_name
                phase = 'cpu_decode_resize'
                prep_begin = time.perf_counter()
                with Image.open(camera._image_path) as source:
                    image = source.convert('RGB').resize((camera.image_width, camera.image_height), Image.Resampling.NEAREST)
                    target = torch.from_numpy(np.array(image, dtype=np.uint8, copy=True)).permute(2, 0, 1).contiguous()
                camera.install_prefetched_image_uint8(target)
                del target, image
                prep_seconds = time.perf_counter() - prep_begin
                torch.cuda.synchronize()
                assert all(p.grad is None for p in raw)
                parameter_adam_resident = parameter_adam_storage_bytes(model)
                resident = long_term_storage_bytes(model)
                entry = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                grid_before = context.method_config['group_grid']
                phase = 'render_loss_backward'
                start = time.perf_counter()
                outputs = run_3dgs_optimization_method(context, model, loop, index, camera)
                torch.cuda.synchronize()
                render_seconds = time.perf_counter() - start
                render_peak = torch.cuda.max_memory_allocated()
                render_reserved = torch.cuda.max_memory_reserved()
                loss = float(outputs['loss'])
                assert math.isfinite(loss)
                compact_gradient = any(p.grad.is_sparse for p in raw)
                actual_gradient_bytes = gradient_storage_bytes(raw)
                assert all(p.grad.shape == p.shape for p in raw)
                bounded_stats = outputs['render_state'].get('bounded_execution_stats', {})
                # Read compact validation samples outside measured regions.
                if index in (1, total_steps):
                    for p in raw:
                        values = p.grad.values() if p.grad.is_sparse else p.grad
                        assert torch.isfinite(values.reshape(-1)[::max(1, values.numel()//2048)]).all().item()
                        del values
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                phase = 'optimizer'
                start = time.perf_counter()
                model.optimizer.step(n, index)
                model.optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                optimizer_seconds = time.perf_counter()-start
                optimizer_peak = torch.cuda.max_memory_allocated()
                optimizer_reserved = torch.cuda.max_memory_reserved()
                assert len(model._xyz) == n
                row = dict(view_index=index, view_in_pass=1 + (index - 1) % len(cameras),
                           pass_index=1 + (index - 1) // len(cameras), camera=current_camera,
                           width=camera.image_width, height=camera.image_height,
                           num_gaussians=n, loss=loss, cpu_decode_resize_seconds=prep_seconds,
                           render_seconds=render_seconds, optimizer_seconds=optimizer_seconds,
                           training_seconds=render_seconds+optimizer_seconds, entry_allocated_bytes=entry,
                           render_peak_allocated_bytes=render_peak, render_peak_reserved_bytes=render_reserved,
                           render_incremental_bytes=render_peak-entry,
                           memory_metric_version=3,
                           parameter_adam_resident_bytes=parameter_adam_resident,
                           long_term_resident_bytes=resident,
                           render_working_set_peak_bytes=render_peak-resident,
                           training_working_set_peak_bytes=max(render_peak,optimizer_peak)-resident,
                           compact_parameter_gradients=compact_gradient, actual_gradient_storage_bytes=actual_gradient_bytes,
                           render_temporary_estimate_bytes=None if compact_gradient else render_peak-entry-gradient_bytes,
                           gradient_bytes=gradient_bytes, optimizer_peak_allocated_bytes=optimizer_peak,
                           training_peak_allocated_bytes=max(render_peak, optimizer_peak),
                           training_peak_reserved_bytes=max(render_reserved, optimizer_reserved),
                           base_grid=grid_before, next_base_grid=context.method_config['group_grid'],
                           raster_leaves=bounded_stats.get('raster_leaf_count', 1), splits=bounded_stats.get('split_count', 0),
                           maximum_actual_leaf_payload_bytes=bounded_stats.get('maximum_actual_payload_bytes'),
                           forced_leaf_count=bounded_stats.get('forced_leaf_count', 0))
                row['projection_group_count'] = bounded_stats.get('projection_group_count', 1)
                row['projection_candidate_count_max'] = bounded_stats.get('projection_candidate_count_max', n)
                if config.get('record_bounded_stats', False):
                    row['bounded_execution_stats'] = dict(bounded_stats)
                del outputs
                context.runtime_state.pop('active_bounded_execution_stats', None)
                camera.release_image_cache()
                log.write(json.dumps(row, allow_nan=False)+'\n'); log.flush()
                rows.append(row)
                summary['completed_views'] = len(rows)
                if index == 1 or index % 16 == 0 or index == total_steps:
                    summary.update(current_camera=current_camera, current_grid=context.method_config['group_grid'])
                    dump(output/'result.json', summary)
                    print(f'{method} {width}: {index}/{total_steps} grid={grid_before} workset={row["render_working_set_peak_bytes"]/2**30:.3f}GiB', flush=True)
        changed = any(not torch.equal(before, p.detach().reshape(-1)[::max(1, p.numel()//2048)].cpu()) for before,p in zip(initial_samples,raw))
        assert changed, 'optimizer did not change sampled model parameters'
        if config.get('save_parameter_samples', False):
            summary['final_parameter_samples'] = [p.detach().reshape(-1)[::max(1,p.numel()//2048)].cpu().tolist() for p in raw]
        summary.update(status='completed', parameter_update_verified=True,
                       final_num_gaussians=len(model._xyz))
    except torch.cuda.OutOfMemoryError as error:
        summary.update(status='oom', oom_phase=phase, oom_camera=current_camera,
                       oom_view_index=len(rows)+1, error=str(error))
    except Exception as error:
        summary.update(status='failed', failed_phase=phase, failed_camera=current_camera,
                       error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        traceback.print_exc()
    summary['wall_seconds'] = time.time()-begun
    if rows:
        peak = max(rows, key=lambda r:r['render_working_set_peak_bytes'])
        summary.update(partial_render_peak_context=peak,
                       training_seconds=sum(r['training_seconds'] for r in rows),
                       cpu_decode_resize_seconds=sum(r['cpu_decode_resize_seconds'] for r in rows),
                       measured_training_peak_allocated_bytes=max(r['training_peak_allocated_bytes'] for r in rows),
                       measured_training_peak_reserved_bytes=max(r['training_peak_reserved_bytes'] for r in rows))
        if summary['status']=='completed':
            summary['memory_metric_version'] = 3
            summary['working_set_definition'] = ('allocated peak minus Gaussian parameters, initialized Adam state, '
                                                 'and persistent densification statistics')
            summary['render_working_set_peak_bytes'] = peak['render_working_set_peak_bytes']
            summary['training_working_set_peak_bytes'] = max(r['training_working_set_peak_bytes'] for r in rows)
            summary['render_temporary_peak_estimate_bytes'] = (max(r['render_temporary_estimate_bytes'] for r in rows)
                if all(r['render_temporary_estimate_bytes'] is not None for r in rows) else None)
    if summary['status']=='completed' and config.get('evaluate_final', False):
        # All training timers and peaks above are finalized before saving/evaluation.
        from boundedgs.components import evaluate_final_ssim as evaluate_final_model
        try:
            summary['final_evaluation'] = evaluate_final_model(
                model, parameters, all_entries, output, config, completed_steps=len(rows),
            )
        except Exception as error:
            summary.update(status='failed', failed_phase='final_evaluation',
                           error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
            traceback.print_exc()
        summary['wall_with_evaluation_seconds'] = time.time()-begun
    dump(output/'result.json', summary)
    return 0 if summary['status']=='completed' else 42 if summary['status']=='oom' else 1



def worker(config, method, width, output, max_views):
    from contextlib import ExitStack
    from unittest.mock import patch
    import scene.methods.optimization_methods as optimization
    from boundedgs.components import finish_direct_grad

    original_run = optimization.run_3dgs_optimization_method
    block_update = config.get('block_update', 'standard')
    if block_update not in {'standard', 'compact_independent', 'compact_accumulated', 'original_independent'}:
        raise ValueError('Unknown block update protocol: ' + block_update)

    def blocks(context, gaussians, loop_state, iteration, camera):
        from boundedgs.blocks import run_compact_blocks, run_original_blocks
        grid = int(context.method_config['group_grid'])
        if block_update == 'original_independent':
            return run_original_blocks(context, gaussians, loop_state, iteration, camera, grid)
        return run_compact_blocks(context, gaussians, loop_state, iteration, camera, grid,
                                  independent=block_update == 'compact_independent')

    def direct(context, gaussians, loop_state, iteration, camera):
        with patch.object(optimization, "finish_staged_projection_backward",
                          lambda projection: finish_direct_grad(projection, gaussians)):
            if method == "StagedOriginal1x1DirectGrad":
                optimization._enable_debug_if_needed(context, iteration)
                background = optimization._select_training_background(context, loop_state)
                return optimization._run_independent_split_optimization(
                    context, gaussians, loop_state, iteration, camera, background, 1)
            return original_run(context, gaussians, loop_state, iteration, camera)

    with ExitStack() as stack:
        optimizer_calls = 0
        if config.get('record_optimizer_steps', False):
            from diff_gaussian_rasterization_boundedgs import GaussianAdam
            original_step = GaussianAdam.step

            def counted_step(optimizer, *args, **kwargs):
                nonlocal optimizer_calls
                result = original_step(optimizer, *args, **kwargs)
                optimizer_calls += 1
                return result

            stack.enter_context(patch.object(GaussianAdam, 'step', counted_step))
        if block_update != 'standard':
            stack.enter_context(patch.object(optimization, 'run_3dgs_optimization_method', blocks))
        elif method in {"StagedOriginal1x1DirectGrad", "AutoLeaf384DirectGrad"}:
            stack.enter_context(patch.object(optimization, "run_3dgs_optimization_method", direct))
        code = _worker(config, method, width, output, max_views)
    if config.get('record_optimizer_steps', False):
        result_path = Path(output) / 'result.json'
        result = json.loads(result_path.read_text())
        steps_per_view = int(config['initial_grid']) ** 2 if block_update.endswith('_independent') else 1
        expected = result['completed_views'] * steps_per_view
        result.update(optimizer_calls=optimizer_calls, optimizer_steps_per_view=steps_per_view)
        if code == 0:
            result['optimizer_steps_verified'] = optimizer_calls == expected
            if optimizer_calls != expected:
                result.update(status='failed', error='Unexpected number of block optimizer updates')
                code = 1
        dump(result_path, result)
    return code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--max_views", type=int, default=0)
    args = parser.parse_args()
    raise SystemExit(worker(json.loads(Path(args.config).read_text()), args.method,
                            args.width, Path(args.output), args.max_views))
