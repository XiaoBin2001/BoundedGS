"""
Bounded asynchronous decoding for lazily loaded camera images.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
import time
from typing import Any, Iterable, Iterator


def resolve_jpeg_prefetch_count(options: Any) -> int:
    count = int(getattr(options, "jpeg_prefetch_count", 10))
    if count < 1:
        raise ValueError("jpeg_prefetch_count must be >= 1.")
    return count


class CameraImagePrefetcher:
    def __init__(self, enabled: bool, prefetch_count: int = 1) -> None:
        self.enabled = bool(enabled)
        self.prefetch_count = int(prefetch_count)
        if self.prefetch_count < 1:
            raise ValueError("prefetch_count must be >= 1.")
        self._executor = (
            ThreadPoolExecutor(
                max_workers=self.prefetch_count,
                thread_name_prefix="camera-image-decode",
            )
            if self.enabled
            else None
        )
        self._pending: deque[tuple[Any, Future[tuple[Any, float]]]] = deque()
        self.loaded_images = 0
        self.load_seconds = 0.0
        self.wait_seconds = 0.0

    def submit(self, camera: Any) -> None:
        if not self.enabled:
            return
        if len(self._pending) >= self.prefetch_count:
            raise RuntimeError("Image prefetch queue is full.")
        if not bool(getattr(camera, "supports_image_prefetch", False)):
            raise ValueError("Image prefetch requires a camera with lazy image loading.")
        self._pending.append((camera, self._executor.submit(self._load_camera, camera)))

    @staticmethod
    def _load_camera(camera: Any) -> tuple[Any, float]:
        started = time.perf_counter()
        image_uint8 = camera.load_image_uint8_for_prefetch()
        return image_uint8, time.perf_counter() - started

    def wait(self, camera: Any) -> None:
        if not self.enabled:
            return
        if not self._pending or self._pending[0][0] is not camera:
            raise RuntimeError("The requested camera does not match the pending image prefetch.")
        _, future = self._pending.popleft()
        wait_started = time.perf_counter()
        image_uint8, load_seconds = future.result()
        self.wait_seconds += time.perf_counter() - wait_started
        self.load_seconds += float(load_seconds)
        self.loaded_images += 1
        camera.install_prefetched_image_uint8(image_uint8)

    def close(self) -> None:
        if self._executor is None:
            return
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._executor = None
        self._pending.clear()

    def summary(self) -> str:
        return (
            "[image-prefetch] decoded={} background_decode={:.3f}s main_wait={:.3f}s".format(
                self.loaded_images,
                self.load_seconds,
                self.wait_seconds,
            )
        )

    def statistics(self) -> dict[str, Any]:
        return dict(enabled=self.enabled, prefetch_count=self.prefetch_count,
                    decoded_views=self.loaded_images, background_decode_seconds=self.load_seconds,
                    main_wait_seconds=self.wait_seconds)


def iter_prefetched_camera_images(
    cameras: Iterable[Any],
    *,
    prefetch_count: int = 1,
) -> Iterator[Any]:
    iterator = iter(cameras)
    pending_cameras: deque[Any] = deque()
    first = next(iterator, None)
    if first is None:
        return
    pending_cameras.append(first)
    enabled = bool(getattr(first, "supports_image_prefetch", False))
    prefetcher = CameraImagePrefetcher(enabled, prefetch_count)
    prefetcher.submit(first)
    while len(pending_cameras) < prefetcher.prefetch_count:
        camera = next(iterator, None)
        if camera is None:
            break
        pending_cameras.append(camera)
        prefetcher.submit(camera)
    try:
        while pending_cameras:
            current = pending_cameras.popleft()
            prefetcher.wait(current)
            following = next(iterator, None)
            if following is not None:
                pending_cameras.append(following)
                prefetcher.submit(following)
            yield current
    finally:
        prefetcher.close()
