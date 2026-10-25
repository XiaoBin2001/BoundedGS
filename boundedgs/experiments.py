"""Config-driven fixed-model component, resolution and block experiments."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

from utils.batch_training.gpu import release_gpu_lock, select_scene_gpu
from utils.batch_training.processes import run_command
from utils.experiment_utils import save_json

KINDS = {"components", "resolution", "pixel_blocks", "independent_blocks"}


def select(items, names):
    requested = set(filter(None, names.split(',')))
    missing = requested - {item['name'] for item in items}
    if missing:
        raise ValueError('Unknown selections: ' + ', '.join(sorted(missing)))
    return [item for item in items if not requested or item['name'] in requested]


def build_jobs(config, only_scene='', only_variant='', widths=None):
    kind = config['experiment_type']
    if kind not in {'components', 'resolution', 'pixel_blocks', 'independent_blocks'}:
        raise ValueError('Unknown fixed-model experiment_type: ' + kind)
    resolution_sweep = kind in {'resolution', 'pixel_blocks'}
    scenes = (config.get('scenes') or [dict(name='bicycle', expected_train_views=config.get('expected_train_views'))])
    variants = select(config['variants'], only_variant)
    # Similarity is paired with the final Original state from the same run.
    if kind == 'components' and not any(v['name'] == 'Original' for v in variants):
        variants.insert(0, next(v for v in config['variants'] if v['name'] == 'Original'))
    if kind == 'components':
        variants.sort(key=lambda variant: variant['name'] != 'Original')
    for scene in select(scenes, only_scene):
        for width in ((widths or config['widths']) if resolution_sweep else [0]):
            for variant in variants:
                parameters = dict(variant['parameters'])
                if kind == 'pixel_blocks':
                    block_width = int(config['block_width'])
                    if block_width <= 0 or int(width) <= 0 or int(width) % block_width:
                        raise ValueError('Pixel-block widths must be positive multiples of block_width.')
                    if variant.get('block_update', 'standard') != 'standard':
                        parameters['group_grid'] = int(width) // block_width
                model_path = (str(Path(config['source_root']) / scene['name'])
                              if kind == 'components' else config['model_path'])
                output = Path(config['output_root']).resolve() / scene['name']
                if resolution_sweep:
                    output /= f'{int(width):05d}px'
                output /= variant['name']
                worker = dict(
                    model_path=model_path, model_iteration=config['model_iteration'],
                    seed=config['seed'], passes=config['passes'],
                    initial_grid=parameters['group_grid'],
                    leaf_budget_mib=parameters['leaf_raster_budget_mib'],
                    spatial_methods={variant['name']: parameters},
                    train_overrides=config['train_overrides'],
                    scene_name=scene['name'], variant=variant['name'],
                    expected_train_views=scene.get('expected_train_views', config.get('expected_train_views')),
                    record_bounded_stats=True, evaluate_final=kind == 'components',
                    quality_protocol='final_test_render_similarity_ssim',
                    block_update=variant.get('block_update', 'standard'),
                    record_optimizer_steps=kind in {'pixel_blocks', 'independent_blocks'},
                )
                if config.get('data_root'):
                    worker['data_root'] = config['data_root']
                yield scene['name'], variant['name'], int(width), output, worker


def summarize(jobs, output_root):
    rows = []
    for scene, variant, width, output, _ in jobs:
        row = dict(scene=scene, variant=variant, width=width, status='missing')
        path = output / 'result.json'
        if path.is_file():
            try:
                result = json.loads(path.read_text())
                status = result['status']
                row['status'] = status if status in {'completed', 'failed', 'oom'} else 'incomplete'
                if status == 'completed':
                    saved = json.loads((output / 'effective_config.json').read_text())
                    passes = saved['passes']
                    views = [json.loads(line) for line in (output / 'views.jsonl').read_text().splitlines()]
                    warm = [view for view in views if view['pass_index'] == passes]
                    if (not warm or result['passes'] != passes
                            or len(views) != result['requested_steps']
                            or len(views) != result['completed_views']
                            or len(warm) != result['requested_views']):
                        raise ValueError('Saved view records do not match the completed run.')
                    row.update(warm_seconds=sum(v['training_seconds'] for v in warm),
                               warm_working_set_gib=max(v['training_working_set_peak_bytes'] for v in warm) / 2**30,
                               SSIM_to_Original=result.get('final_evaluation', {}).get('metrics', {}).get('SSIM_to_Original'))
                elif result.get('error'):
                    row['reason'] = result['error']
            except (OSError, ValueError, KeyError, TypeError) as error:
                row.update(status='incomplete', reason=str(error))
        else:
            row['reason'] = 'result.json is missing'
        rows.append(row)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row)) or ['scene', 'variant', 'width', 'status']
    with (root / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    counts = {status: sum(row['status'] == status for row in rows)
              for status in ('completed', 'oom', 'failed', 'missing', 'incomplete')}
    # OOM is an observed outcome of resolution/memory sweeps, with no timing metric.
    report = dict(expected_tasks=len(rows), **counts,
                  complete=bool(rows) and counts['completed'] + counts['oom'] == len(rows), tasks=rows)
    save_json(root / 'summary.json', report)
    print('Ablation summary: {} ({}); OOM rows contain no timing metrics.'.format(
        'complete' if report['complete'] else 'partial', counts))
    return report


def validate_experiment(config, scene_filter=(), dry_run=False, rebuild=False):
    if config.get('experiment_type') not in KINDS:
        raise ValueError('Unknown fixed-model experiment_type.')
    for key in ('passes', 'model_iteration'):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(key + ' must be a positive integer.')
    if type(config.get('seed')) is not int:
        raise ValueError('seed must be an integer.')
    if type(config.get('max_views', 0)) is not int or config.get('max_views', 0) < 0:
        raise ValueError('max_views must be a nonnegative integer.')
    if type(config.get('repeat', 1)) is not int or config.get('repeat', 1) != 1:
        raise ValueError('Fixed-model ablations use one paired run per output_root.')
    if not isinstance(config.get('output_root'), str) or not config['output_root']:
        raise ValueError('output_root must be a nonempty path.')
    source_key = 'source_root' if config['experiment_type'] == 'components' else 'model_path'
    if not isinstance(config.get(source_key), str) or not config[source_key]:
        raise ValueError(source_key + ' must be a nonempty path.')
    if not isinstance(config.get('train_overrides'), dict):
        raise ValueError('train_overrides must be an object.')
    variants = config.get('variants')
    if not isinstance(variants, list) or not variants or any(not isinstance(v, dict) for v in variants):
        raise ValueError('variants must be a nonempty list.')
    names = []
    for variant in variants:
        name = variant.get('name')
        if not isinstance(name, str) or not name or Path(name).name != name or name in {'.', '..'}:
            raise ValueError('Invalid variant name.')
        names.append(name)
        parameters = variant.get('parameters', {})
        if not isinstance(parameters, dict):
            raise ValueError('Variant parameters must be an object.')
        if type(parameters.get('group_grid')) is not int or parameters['group_grid'] < 1:
            raise ValueError('Each variant requires a positive group_grid.')
        if 'leaf_raster_budget_mib' not in parameters:
            raise ValueError('Each variant requires leaf_raster_budget_mib.')
        if variant.get('block_update', 'standard') not in {
                'standard', 'compact_independent', 'compact_accumulated', 'original_independent'}:
            raise ValueError('Invalid block_update protocol.')
    if len(names) != len(set(names)):
        raise ValueError('Duplicate variants.')
    if config['experiment_type'] == 'components' and 'Original' not in names:
        raise ValueError('Component similarity requires the paired Original variant.')
    scenes = config.get('scenes', [dict(name='bicycle')])
    if not isinstance(scenes, list) or not scenes or any(not isinstance(scene, dict) for scene in scenes):
        raise ValueError('scenes must be a nonempty list.')
    for scene in scenes:
        name = scene.get('name')
        if not isinstance(name, str) or not name or Path(name).name != name or name in {'.', '..'}:
            raise ValueError('Invalid scene name.')
    if config['experiment_type'] in {'resolution', 'pixel_blocks'}:
        widths = config.get('widths')
        if not isinstance(widths, list) or not widths or any(type(w) is not int or w < 1 for w in widths):
            raise ValueError('widths must contain positive integers.')
    if config['experiment_type'] == 'pixel_blocks' and (type(config.get('block_width')) is not int or config['block_width'] < 1):
        raise ValueError('block_width must be a positive integer.')
    if not isinstance(config.get('only_variant', ''), str):
        raise ValueError('only_variant must be a comma-separated string.')
    jobs = list(build_jobs(config, ','.join(sorted(scene_filter)), config.get('only_variant', '')))
    outputs = [output.resolve() for _, _, _, output, _ in jobs]
    if len(outputs) != len(set(outputs)):
        raise ValueError('Ablation jobs have conflicting outputs.')
    for _, _, _, output, worker in jobs:
        source = Path(worker['model_path']).resolve()
        if source == output or source in output.parents or output in source.parents:
            raise ValueError('Ablation input and output must be separate.')
        if rebuild or dry_run:
            continue
        if output.exists():
            raise FileExistsError(output)
        for relative in ('training_parameters.json', 'cameras.json',
                         f"point_cloud/iteration_{worker['model_iteration']}/point_cloud.ply"):
            if not (source / relative).is_file():
                raise FileNotFoundError(source / relative)
    return jobs


def run_config(config, scene_filter=(), dry_run=False, rebuild=False):
    jobs = validate_experiment(config, scene_filter, dry_run, rebuild)
    if rebuild:
        if dry_run:
            print('Would rebuild ablation summary in', config['output_root'])
        else:
            return 0 if summarize(jobs, config['output_root'])['complete'] else 1
        return 0
    lock = None
    try:
        gpu, lock = select_scene_gpu(config, 'paper_ablation', config.get('gpu_id', 0), dry_run)
        for _, variant, width, output, worker in jobs:
            worker['max_views'] = config.get('max_views', 0)
            path = output / 'effective_config.json'
            if dry_run:
                print(json.dumps(dict(worker, width=width, output=str(output))))
            else:
                output.mkdir(parents=True)
                path.write_text(json.dumps(worker, indent=2) + '\n')
            command = [config.get('python_executable') or sys.executable, '-m', 'boundedgs.ablation',
                       '--config', str(path), '--output', str(output), '--method', variant,
                       '--width', str(width), '--max_views', str(worker['max_views'])]
            code = run_command(command, gpu, dry_run, resource_limits=config.get('resource_limits'),
                               resource_path=str(output / 'process_resources.json'))
            if code != 0 and not (code == 42 and config['experiment_type'] in {'resolution', 'pixel_blocks'}):
                if not dry_run:
                    summarize(jobs, config['output_root'])
                return code
        if not dry_run:
            return 0 if summarize(jobs, config['output_root'])['complete'] else 1
    finally:
        release_gpu_lock(lock)
    return 0


