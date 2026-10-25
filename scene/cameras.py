"""
Camera objects used by rendering and training.

Camera loads image tensors, optional inverse-depth maps, alpha masks, and camera
matrices.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from numpy.typing import NDArray
from PIL import Image as PILImage
from torchvision.io import ImageReadMode, decode_jpeg

from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import PILtoTorch
import cv2


_DEPTH_NPZ_KEYS = ("invdepth", "inv_depth", "inverse_depth", "depth", "arr_0")
_IMAGE_UPLOAD_STREAM: torch.cuda.Stream | None = None


def _get_image_upload_stream() -> torch.cuda.Stream:
    global _IMAGE_UPLOAD_STREAM
    if _IMAGE_UPLOAD_STREAM is None:
        _IMAGE_UPLOAD_STREAM = torch.cuda.Stream()
    return _IMAGE_UPLOAD_STREAM


def _resolve_camera_device(data_device: str) -> torch.device:
    """
        Convert a device string to torch.device, falling back to CUDA on failure.
    """
    try:
        return torch.device(data_device)
    except Exception as e:
        print(e)
        print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
        return torch.device("cuda")


def _build_alpha_mask(
    resized_image_uint8: torch.Tensor,
    data_device: torch.device,
) -> torch.Tensor | None:
    """
        Build the alpha mask used to ignore invalid image regions.

        RGBA images provide the mask; RGB images have no explicit mask.
    """
    if resized_image_uint8.shape[0] != 4:
        return None
    return resized_image_uint8[3:4, ...].to(data_device)


def _uint8_training_crop_cuda(
    source_uint8: torch.Tensor,
    rect_y: int,
    rect_x: int,
    rect_height: int,
    rect_width: int,
    upsample_factor: int = 1,
) -> torch.Tensor:
    """Convert one source crop to a contiguous CUDA float tensor."""
    factor = int(upsample_factor)
    if factor < 1:
        raise ValueError("training_image_upsample_factor must be >= 1.")
    source_y = int(rect_y) // factor
    source_x = int(rect_x) // factor
    source_y_end = (int(rect_y) + int(rect_height) + factor - 1) // factor
    source_x_end = (int(rect_x) + int(rect_width) + factor - 1) // factor
    source_crop = source_uint8[
        :,
        source_y:source_y_end,
        source_x:source_x_end,
    ]
    if source_crop.device.type == "cpu":
        source_crop = source_crop.contiguous()
        if not source_crop.is_pinned():
            source_crop = source_crop.pin_memory()
        # Transfer compact uint8 pixels first; conversion happens on the GPU.
        source_crop = source_crop.to(device="cuda", non_blocking=True)
    training_crop = source_crop.to(
        device="cuda",
        dtype=torch.float32,
        non_blocking=True,
    ).div_(255.0)
    if factor > 1:
        training_crop = F.interpolate(
            training_crop.unsqueeze(0),
            size=(training_crop.shape[1] * factor, training_crop.shape[2] * factor),
            mode="nearest",
        ).squeeze(0)
        offset_y = int(rect_y) - source_y * factor
        offset_x = int(rect_x) - source_x * factor
        training_crop = training_crop[
            :,
            offset_y : offset_y + int(rect_height),
            offset_x : offset_x + int(rect_width),
        ]
    return training_crop.contiguous()


def _uint8_training_image_cuda(
    source_uint8: torch.Tensor,
    upsample_factor: int = 1,
) -> torch.Tensor:
    """Convert a complete uint8 image and optionally repeat each pixel spatially."""
    factor = int(upsample_factor)
    if factor < 1:
        raise ValueError("training_image_upsample_factor must be >= 1.")
    if source_uint8.device.type == "cpu":
        source_uint8 = source_uint8.contiguous().pin_memory()
    training_image = source_uint8.to(
        device="cuda",
        dtype=torch.float32,
        non_blocking=True,
    ).div_(255.0)
    if factor > 1:
        training_image = F.interpolate(
            training_image.unsqueeze(0),
            size=(training_image.shape[1] * factor, training_image.shape[2] * factor),
            mode="nearest",
        ).squeeze(0)
    return training_image.contiguous()


def _read_depth_array(depth_path: str) -> NDArray[np.generic]:
    """
        Read one inverse-depth map from disk into CPU memory.
    """
    if not os.path.exists(depth_path):
        raise FileNotFoundError("Depth file not found: {}".format(depth_path))
    suffix = os.path.splitext(depth_path)[1].lower()
    if suffix == ".npz":
        with np.load(depth_path) as payload:
            for key in _DEPTH_NPZ_KEYS:
                if key in payload:
                    return np.asarray(payload[key])
            if not payload.files:
                raise ValueError("Depth npz has no arrays: {}".format(depth_path))
            return np.asarray(payload[payload.files[0]])
    if suffix == ".npy":
        return np.asarray(np.load(depth_path))
    depth_array = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_array is None:
        raise ValueError("Unable to read depth file: {}".format(depth_path))
    return np.asarray(depth_array)


def _squeeze_depth_array(depth_array: NDArray[np.generic]) -> NDArray[np.generic]:
    """
        Normalize common HxW, HxWx1, HxWxC, and 1xHxW layouts to HxW.
    """
    depth_array = np.asarray(depth_array)
    depth_array = np.squeeze(depth_array)
    if depth_array.ndim == 3:
        if depth_array.shape[0] <= 4 and depth_array.shape[1] > 4 and depth_array.shape[2] > 4:
            depth_array = depth_array[0]
        else:
            depth_array = depth_array[..., 0]
    if depth_array.ndim != 2:
        raise ValueError("Depth array must be 2D after squeezing; got shape {}".format(depth_array.shape))
    return np.ascontiguousarray(depth_array)


def _numpy_depth_to_cuda_float(depth_array: NDArray[np.generic]) -> torch.Tensor:
    """
        Move raw depth data to CUDA first, then convert numeric dtype on GPU.
    """
    depth_array = _squeeze_depth_array(depth_array)
    unsigned_bits = 0
    if depth_array.dtype == np.uint16:
        depth_array = depth_array.view(np.int16)
        unsigned_bits = 16
    elif depth_array.dtype == np.uint32:
        depth_array = depth_array.view(np.int32)
        unsigned_bits = 32

    depth_tensor = torch.from_numpy(depth_array).to("cuda", non_blocking=True)
    depth_tensor = depth_tensor.float()
    if unsigned_bits > 0:
        depth_tensor = torch.where(depth_tensor < 0.0, depth_tensor + float(2**unsigned_bits), depth_tensor)
    return depth_tensor[None, None]


def _resize_depth_tensor(depth_tensor: torch.Tensor, resolution: tuple[int, int]) -> torch.Tensor:
    """
        Resize a CUDA depth tensor to the training image resolution.
    """
    target_width, target_height = int(resolution[0]), int(resolution[1])
    if depth_tensor.shape[-2:] != (target_height, target_width):
        depth_tensor = F.interpolate(depth_tensor, size=(target_height, target_width), mode="bilinear", align_corners=False)
    return depth_tensor[0]


def _is_legacy_png_depth(depth_path: str) -> bool:
    """
        PNG depth files follow the original 3DGS integer inverse-depth convention.
    """
    return os.path.splitext(depth_path)[1].lower() == ".png"


class Camera(nn.Module):
    """
        Full camera record with image data and projection matrices.
    """

    def __init__(
        self,
        resolution: tuple[int, int],
        colmap_id: int,
        R: NDArray[np.floating],
        T: NDArray[np.floating],
        FoVx: float,
        FoVy: float,
        depth_params: dict[str, float] | None,
        image: PILImage.Image | None,
        image_path: str,
        encoded_jpeg: torch.Tensor | None,
        depth_path: str,
        depth_is_nerf_synthetic: bool,
        image_name: str,
        uid: int,
        trans: NDArray[np.floating] = np.array([0.0, 0.0, 0.0]),
        scale: float = 1.0,
        data_device: str = "cuda",
        lazy_image_loading: bool = False,
        training_image_upsample_factor: int = 1,
        training_gt_mode: str = "leaf",
    ) -> None:
        """
            Load image/depth tensors and precompute transforms for rendering.
        """
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        self.data_device = _resolve_camera_device(data_device)
        self._image_path = str(image_path)
        self._encoded_jpeg = encoded_jpeg
        self._image_resolution = (int(resolution[0]), int(resolution[1]))
        self.training_image_upsample_factor = int(training_image_upsample_factor)
        if training_gt_mode not in {"leaf", "leaf_prefetch"}:
            raise ValueError("Unknown training_gt_mode: {}".format(training_gt_mode))
        self.training_gt_mode = training_gt_mode
        self._leaf_gt_prefetcher = None
        if self.training_image_upsample_factor < 1:
            raise ValueError("training_image_upsample_factor must be >= 1.")
        self._lazy_image_loading = bool(lazy_image_loading)
        self._image_cache_loaded = False
        self._alpha_mask_uint8: torch.Tensor | None = None
        self._original_image_uint8: torch.Tensor | None = None
        self._original_image_uint8_cuda_cache: torch.Tensor | None = None
        self._original_image_uint8_cuda_ready: torch.cuda.Event | None = None
        self._alpha_mask_uint8_cuda_cache: torch.Tensor | None = None
        self.image_width = self._image_resolution[0] * self.training_image_upsample_factor
        self.image_height = self._image_resolution[1] * self.training_image_upsample_factor
        if not self._lazy_image_loading:
            if image is None:
                raise ValueError("Non-lazy camera loading requires an image.")
            self._set_image_cache(PILtoTorch(image, self._image_resolution))

        self._depth_path = str(depth_path or "")
        self._depth_is_nerf_synthetic = bool(depth_is_nerf_synthetic)
        self._depth_params = dict(depth_params) if depth_params is not None else None
        self.depth_reliable = self._depth_path != ""
        if self.depth_reliable and self._depth_params is not None:
            depth_scale = float(self._depth_params["scale"])
            median_scale = float(self._depth_params["med_scale"])
            if depth_scale < 0.2 * median_scale or depth_scale > 5.0 * median_scale:
                self.depth_reliable = False

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def _set_image_cache(self, resized_image_uint8: torch.Tensor) -> None:
        expected_shape = (self._image_resolution[1], self._image_resolution[0])
        if resized_image_uint8.dtype != torch.uint8:
            raise ValueError("Camera image cache must use uint8.")
        if resized_image_uint8.ndim != 3 or resized_image_uint8.shape[0] not in (3, 4):
            raise ValueError(
                "Camera image cache must have CHW RGB/RGBA shape; got {}.".format(
                    tuple(resized_image_uint8.shape)
                )
            )
        if tuple(resized_image_uint8.shape[1:]) != expected_shape:
            raise ValueError(
                "Cached image resolution {} does not match requested {} for {}.".format(
                    tuple(reversed(resized_image_uint8.shape[1:])),
                    self._image_resolution,
                    self.image_name,
                )
            )
        self._alpha_mask_uint8 = _build_alpha_mask(
            resized_image_uint8,
            self.data_device,
        )
        self._original_image_uint8 = resized_image_uint8[:3, ...].to(self.data_device)
        self._original_image_uint8_cuda_cache = None
        self._original_image_uint8_cuda_ready = None
        self._alpha_mask_uint8_cuda_cache = None
        self._image_cache_loaded = True

    def _load_uint8_image(self) -> torch.Tensor:
        if self._encoded_jpeg is not None:
            return decode_jpeg(
                self._encoded_jpeg,
                mode=ImageReadMode.RGB,
                device="cpu",
            )
        with PILImage.open(self._image_path) as image:
            return PILtoTorch(image, self._image_resolution)

    def _ensure_image_cache(self) -> None:
        if self._image_cache_loaded:
            return
        self._set_image_cache(self._load_uint8_image())

    @property
    def supports_image_prefetch(self) -> bool:
        return self._lazy_image_loading

    def load_image_uint8_for_prefetch(self):
        if not self.supports_image_prefetch:
            raise ValueError("Camera {} does not use lazy image loading.".format(self.image_name))
        image_uint8 = self._load_uint8_image()
        if image_uint8.device.type == "cpu" and not image_uint8.is_pinned():
            image_uint8 = image_uint8.pin_memory()
        return image_uint8

    def install_prefetched_image_uint8(self, image_uint8: torch.Tensor) -> None:
        self._set_image_cache(image_uint8)

    def release_image_cache(self) -> None:
        if self._leaf_gt_prefetcher is not None:
            raise RuntimeError("Cannot release an image while Leaf prefetch is active.")
        self._original_image_uint8_cuda_cache = None
        self._original_image_uint8_cuda_ready = None
        self._alpha_mask_uint8_cuda_cache = None
        if not self._lazy_image_loading:
            return
        self._original_image_uint8 = None
        self._alpha_mask_uint8 = None
        self._image_cache_loaded = False

    def prefetch_original_image_uint8_cuda(self) -> None:
        self._ensure_image_cache()
        if self._original_image_uint8_cuda_cache is not None:
            return
        if self._original_image_uint8.device.type == "cuda":
            self._original_image_uint8_cuda_cache = self._original_image_uint8
            return
        upload_stream = _get_image_upload_stream()
        with torch.cuda.stream(upload_stream):
            self._original_image_uint8_cuda_cache = self._original_image_uint8.to(
                "cuda",
                non_blocking=True,
            )
            self._original_image_uint8_cuda_ready = torch.cuda.Event()
            self._original_image_uint8_cuda_ready.record(upload_stream)

    def original_image_uint8_cuda(self) -> torch.Tensor:
        self.prefetch_original_image_uint8_cuda()
        if self._original_image_uint8_cuda_ready is not None:
            current_stream = torch.cuda.current_stream()
            current_stream.wait_event(self._original_image_uint8_cuda_ready)
            self._original_image_uint8_cuda_cache.record_stream(current_stream)
        return self._original_image_uint8_cuda_cache

    def original_image_cuda(self) -> torch.Tensor:
        if self.training_gt_mode == "leaf_prefetch":
            return self.original_image_crop_cuda(0, 0, self.image_height, self.image_width)
        return _uint8_training_image_cuda(
            self.original_image_uint8_cuda(),
            self.training_image_upsample_factor,
        )

    def original_image_crop_cuda(
        self,
        rect_y: int,
        rect_x: int,
        rect_height: int,
        rect_width: int,
    ) -> torch.Tensor:
        if self._leaf_gt_prefetcher is not None:
            pixels = self._leaf_gt_prefetcher.take((rect_y, rect_x, rect_height, rect_width))
            factor = self.training_image_upsample_factor
            return _uint8_training_crop_cuda(
                pixels, rect_y % factor, rect_x % factor, rect_height, rect_width, factor,
            )
        if self.training_gt_mode == "leaf_prefetch":
            self._ensure_image_cache()
            source_uint8 = self._original_image_uint8
        else:
            source_uint8 = self.original_image_uint8_cuda()
        return _uint8_training_crop_cuda(
            source_uint8,
            rect_y,
            rect_x,
            rect_height,
            rect_width,
            self.training_image_upsample_factor,
        )

    @contextmanager
    def leaf_gt_prefetch(self):
        from utils.leaf_gt_prefetch import LeafGtPrefetcher
        self._ensure_image_cache()
        if self._leaf_gt_prefetcher is not None:
            raise RuntimeError("Nested Leaf prefetch is not supported.")
        prefetcher = LeafGtPrefetcher(self._original_image_uint8, self.training_image_upsample_factor)
        self._leaf_gt_prefetcher = prefetcher
        try:
            yield prefetcher
        finally:
            self._leaf_gt_prefetcher = None
            prefetcher.close()


    @property
    def alpha_mask(self) -> torch.Tensor | None:
        self._ensure_image_cache()
        if self._alpha_mask_uint8 is None:
            return None
        return self._alpha_mask_uint8.float() / 255.0

    def alpha_mask_cuda(self) -> torch.Tensor | None:
        self._ensure_image_cache()
        if self._alpha_mask_uint8 is None:
            return None
        if self._alpha_mask_uint8_cuda_cache is None:
            self._alpha_mask_uint8_cuda_cache = self._alpha_mask_uint8.to(
                "cuda",
                non_blocking=True,
            )
        return _uint8_training_image_cuda(
            self._alpha_mask_uint8_cuda_cache,
            self.training_image_upsample_factor,
        )

    def alpha_mask_crop_cuda(
        self,
        rect_y: int,
        rect_x: int,
        rect_height: int,
        rect_width: int,
    ) -> torch.Tensor | None:
        self._ensure_image_cache()
        if self._alpha_mask_uint8 is None:
            return None
        if self._alpha_mask_uint8_cuda_cache is None:
            self._alpha_mask_uint8_cuda_cache = self._alpha_mask_uint8.to(
                "cuda",
                non_blocking=True,
            )
        return _uint8_training_crop_cuda(
            self._alpha_mask_uint8_cuda_cache,
            rect_y,
            rect_x,
            rect_height,
            rect_width,
            self.training_image_upsample_factor,
        )

    @property
    def invdepthmap(self) -> torch.Tensor | None:
        invdepthmap = self.invdepthmap_cuda()
        return None if invdepthmap is None else invdepthmap.cpu()

    def invdepthmap_cuda(self) -> torch.Tensor | None:
        if self._depth_path == "" or not self.depth_reliable:
            return None
        invdepthmap = _resize_depth_tensor(
            _numpy_depth_to_cuda_float(_read_depth_array(self._depth_path)),
            (self.image_width, self.image_height),
        )
        if _is_legacy_png_depth(self._depth_path):
            invdepthmap = invdepthmap / (512.0 if self._depth_is_nerf_synthetic else float(2**16))
        invdepthmap = torch.clamp(invdepthmap, min=0.0)
        if self._depth_params is not None and float(self._depth_params["scale"]) > 0.0:
            invdepthmap = invdepthmap * float(self._depth_params["scale"]) + float(self._depth_params["offset"])
        return invdepthmap

    @property
    def depth_mask(self) -> torch.Tensor | None:
        depth_mask = self.depth_mask_cuda()
        return None if depth_mask is None else depth_mask.cpu()

    def depth_mask_cuda(self) -> torch.Tensor | None:
        if self._depth_path == "":
            return None
        fill_value = 1.0 if self.depth_reliable else 0.0
        return torch.full((1, self.image_height, self.image_width), fill_value, device="cuda", dtype=torch.float32)
