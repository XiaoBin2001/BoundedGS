"""Compare final rendered test images directly with the paired Original renders."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch
from scene.cameras import Camera
from diff_gaussian_rasterization_boundedgs import _C

PARAMETERS = ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation")


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def file_sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_original_state(model, output: Path, config: dict, names: list[str], completed_steps: int):
    state = {
        "scene": config["scene_name"],
        "training_steps": completed_steps,
        "max_sh_degree": int(model.max_sh_degree),
        "active_sh_degree": int(model.active_sh_degree),
        "test_camera_names": names,
        "parameters": {name: getattr(model, name).detach().cpu().contiguous() for name in PARAMETERS},
    }
    temporary = output / "reference_state.pt.tmp"
    destination = output / "reference_state.pt"
    torch.save(state, temporary)
    temporary.replace(destination)
    loaded = torch.load(destination, map_location="cpu", weights_only=True, mmap=True)
    assert loaded["scene"] == state["scene"]
    assert loaded["training_steps"] == state["training_steps"]
    assert loaded["test_camera_names"] == state["test_camera_names"]
    assert all(torch.equal(state["parameters"][name], loaded["parameters"][name]) for name in PARAMETERS)
    meta = {
        "path": str(destination),
        "sha256": file_sha256(destination),
        "bytes": destination.stat().st_size,
        "scene": state["scene"],
        "training_steps": completed_steps,
        "num_gaussians": int(model.get_xyz.shape[0]),
        "test_camera_names": names,
        "raw_float32_parameters_verified": True,
    }
    write_json(output / "reference_state_meta.json", meta)
    return meta


def load_original_model(path: Path, config: dict, names: list[str], completed_steps: int):
    from scene.gaussian_model import GaussianModel

    meta = json.loads((path.parent / "reference_state_meta.json").read_text())
    assert meta["scene"] == config["scene_name"]
    assert meta["training_steps"] == completed_steps
    assert meta["test_camera_names"] == names
    assert Path(meta["path"]) == path and path.stat().st_size == meta["bytes"]
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    assert state["scene"] == meta["scene"] and state["test_camera_names"] == names
    assert state["training_steps"] == completed_steps
    reference = GaussianModel(int(state["max_sh_degree"]), "ours_adam")
    reference.active_sh_degree = int(state["active_sh_degree"])
    for name in PARAMETERS:
        value = state["parameters"][name]
        assert value.dtype == torch.float32
        setattr(reference, name, torch.nn.Parameter(value.to("cuda"), requires_grad=False))
    assert int(reference.get_xyz.shape[0]) == meta["num_gaussians"]
    del state
    return reference, meta


@torch.inference_mode()
def evaluate_final_ssim(model, parameters, all_entries, output, config, completed_steps):
    from utils.evaluation.render_metric import render_inference_image
    from utils.loss_utils import ssim

    if config["quality_protocol"] != "final_test_render_similarity_ssim":
        raise ValueError(f"Unsupported quality protocol: {config['quality_protocol']}")
    started = time.time()
    output = Path(output)
    entries = all_entries[::8]
    if config.get("expected_test_views") is not None:
        assert len(entries) == int(config["expected_test_views"])
    if config.get("eval_max_views", 0):
        entries = entries[:int(config["eval_max_views"])]
    names = [entry["img_name"] for entry in entries]
    assert names and len(set(names)) == len(names)
    method = config["variant"]

    # ablation.worker has finalized the warm-pass timers and allocator peaks.
    model.optimizer = None
    for name in PARAMETERS:
        getattr(model, name).grad = None
    torch.cuda.empty_cache()

    if method == "Original":
        meta = save_original_state(model, output, config, names, completed_steps)
        per_view = [{"camera": name, "SSIM_to_Original": 1.0, "L1_to_Original": 0.0} for name in names]
        reference_hash = meta["sha256"]
    else:
        path = output.parent / "Original/reference_state.pt"
        reference, meta = load_original_model(path, config, names, completed_steps)
        reference_hash = meta["sha256"]
        pipeline = SimpleNamespace(debug=False, antialiasing=parameters["antialiasing"])
        background = torch.tensor(
            [1., 1., 1.] if parameters["white_background"] else [0., 0., 0.], device="cuda"
        )
        per_view = []
        for index, entry in enumerate(entries):
            camera = camera_from_entry(entry, parameters, index)
            original_image = render_inference_image(camera, reference, pipeline, background, group_grid=1).clamp(0, 1)
            current_image = render_inference_image(camera, model, pipeline, background, group_grid=1).clamp(0, 1)
            assert current_image.shape == original_image.shape
            value = float(ssim(current_image.unsqueeze(0), original_image.unsqueeze(0)))
            l1 = float(torch.abs(current_image - original_image).mean())
            if not math.isfinite(value) or not -1.0001 <= value <= 1.0001 or not math.isfinite(l1):
                raise ValueError(f"Invalid rendered-image similarity for {entry['img_name']}: {value}, {l1}")
            per_view.append({"camera": entry["img_name"], "SSIM_to_Original": value,
                             "L1_to_Original": l1})
            camera.release_image_cache()
            del current_image, original_image, camera
            if (index + 1) % 5 == 0 or index + 1 == len(entries):
                print(f"final render similarity: {index + 1}/{len(entries)}", flush=True)

    result = {
        "status": "completed",
        "scene": config["scene_name"],
        "method": method,
        "training_steps": completed_steps,
        "image_count": len(per_view),
        "evaluation_resolution": [int(entries[0]["width"]), int(entries[0]["height"])],
        "render": "common full-frame inference-only on identical held-out camera views",
        "SSIM_protocol": "utils.loss_utils.ssim(current render, paired Original render); no GT",
        "reference_state_sha256": reference_hash,
        "metrics": {
            "SSIM_to_Original": sum(view["SSIM_to_Original"] for view in per_view) / len(per_view),
            "L1_to_Original": sum(view["L1_to_Original"] for view in per_view) / len(per_view),
        },
        "per_view": per_view,
        "elapsed_seconds": time.time() - started,
        "reference_state_saved": method == "Original",
    }
    write_json(output / "final_evaluation.json", result)
    return result


def camera_from_entry(entry, parameters, index):
    c2w=np.eye(4,dtype=np.float32)
    c2w[:3,:3]=entry['rotation'];c2w[:3,3]=entry['position']
    w2c=np.linalg.inv(c2w)
    return Camera(resolution=(entry['width'],entry['height']),colmap_id=index,
                  R=w2c[:3,:3].T,T=w2c[:3,3],FoVx=2*math.atan(entry['width']/(2*entry['fx'])),
                  FoVy=2*math.atan(entry['height']/(2*entry['fy'])),depth_params=None,
                  image=None,image_path=str(Path(parameters['source_path'])/parameters['images']/entry['img_name']),
                  encoded_jpeg=None,depth_path='',depth_is_nerf_synthetic=False,
                  image_name=entry['img_name'],uid=index,data_device='cpu',lazy_image_loading=True)


def finish_direct_grad(projection, gaussians):
    """Use identity candidates to write staged projection gradients into raw .grad."""
    if projection.indexed_raw or projection.candidate_indices is not None:
        raise ValueError("DirectGrad requires a full-N activated projection")
    if projection.projected_gradients is None:
        raise ValueError("The staged projection has no raster-to-projection gradients")
    raw = (
        gaussians._xyz, gaussians._features_dc, gaussians._features_rest,
        gaussians._opacity, gaussians._scaling, gaussians._rotation,
    )
    n = int(raw[0].shape[0])
    if int(projection.compact_radii.numel()) != n:
        raise ValueError("The staged projection must contain every Gaussian")
    for parameter in raw:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        elif parameter.grad.is_sparse:
            parameter.grad = parameter.grad.to_dense()
    identity_candidates = torch.arange(n, device=raw[0].device, dtype=torch.int32)
    _C.project_gaussians_indexed_raw_backward(
        identity_candidates,
        raw[0], raw[3], raw[4], raw[5],
        projection.scale_modifier,
        projection.viewmatrix, projection.projmatrix,
        projection.tanfovx, projection.tanfovy,
        projection.image_height, projection.image_width,
        raw[1], raw[2], projection.sh_degree, projection.campos,
        projection.compact_radii, projection.geometry,
        *projection.projected_gradients,
        *(parameter.grad for parameter in raw),
        projection.antialiasing, projection.debug,
    )
    if projection.viewspace_points.numel() == projection.projected_gradients[0].numel():
        projection.viewspace_points.grad = projection.projected_gradients[0]
