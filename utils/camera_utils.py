"""
Camera loading and serialization helpers.

Dataset readers produce lightweight CameraInfo records. This module opens the
image/depth files, applies the requested resolution scale, creates Camera
objects, and exports camera metadata for reproducibility.
"""
from __future__ import annotations

from scene.cameras import Camera
import numpy as np
from numpy.typing import NDArray
from utils.graphics_utils import fov2focal
from PIL import Image
from typing import Any

WARNED = False
LAZY_WARNED = False

def loadCam(
    args: Any,
    id: int,
    cam_info: Any,
    resolution_scale: float,
    is_nerf_synthetic: bool,
) -> Camera:
    """
        Load one image/depth pair and convert a CameraInfo record into a Camera.

        Resolution handling follows the original 3DGS behavior: fixed divisors,
        automatic 1.6K downscale for very large images, or a user-provided width.
    """
    jpeg_images = getattr(args, "in_memory_jpeg_images", {})
    encoded_jpeg = jpeg_images.get(cam_info.image_name) if isinstance(jpeg_images, dict) else None
    if bool(getattr(args, "use_in_memory_jpeg_cache", False)) and encoded_jpeg is None:
        raise FileNotFoundError("Missing in-memory JPEG for camera {}.".format(cam_info.image_name))
    lazy_image_loading = encoded_jpeg is not None or bool(getattr(args, "lazy_image_loading", False))
    if lazy_image_loading:
        global LAZY_WARNED
        if not LAZY_WARNED:
            source_label = "in-memory JPEG" if encoded_jpeg is not None else "per-view source image"
            print("[image-loader] Camera RGB/alpha tensors use {} loading.".format(source_label))
            LAZY_WARNED = True

    orig_w, orig_h = int(cam_info.width), int(cam_info.height)
    if args.resolution in [1, 2, 4, 8]:
        resolution = (
            round(orig_w / (resolution_scale * args.resolution)),
            round(orig_h / (resolution_scale * args.resolution)),
        )
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print(
                        "[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1"
                    )
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    def build_camera(image: Image.Image | None) -> Camera:
        return Camera(
            resolution,
            colmap_id=cam_info.uid,
            R=cam_info.R,
            T=cam_info.T,
            FoVx=cam_info.FovX,
            FoVy=cam_info.FovY,
            depth_params=cam_info.depth_params,
            image=image,
            image_path=cam_info.image_path,
            encoded_jpeg=encoded_jpeg,
            depth_path=cam_info.depth_path,
            depth_is_nerf_synthetic=is_nerf_synthetic,
            image_name=cam_info.image_name,
            uid=id,
            data_device=args.data_device,
            lazy_image_loading=lazy_image_loading,
            training_image_upsample_factor=int(
                getattr(args, "training_image_upsample_factor", 1)
            ),
            training_gt_mode=getattr(args, "training_gt_mode", "leaf"),
        )
    if lazy_image_loading:
        return build_camera(None)
    with Image.open(cam_info.image_path) as image:
        return build_camera(image)

def cameraList_from_camInfos(
    cam_infos: list[Any],
    resolution_scale: float,
    args: Any,
    is_nerf_synthetic: bool,
) -> list[Camera]:
    """
        Build Camera objects for every CameraInfo entry in a scene split.
    """
    camera_list: list[Camera] = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale, is_nerf_synthetic))

    return camera_list

def camera_to_JSON(id: int, camera: Camera) -> dict[str, Any]:
    """
        Convert a Camera object into the compact JSON format saved with outputs.
    """
    Rt: NDArray[np.float64] = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
