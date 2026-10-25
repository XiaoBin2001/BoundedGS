"""
Load source JPEG bitstreams into CPU memory for repeated asynchronous decoding.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
from typing import Any, Iterable

import torch
from torchvision.io import read_file
from tqdm import tqdm


JPEG_SUFFIXES = {".jpg", ".jpeg"}


def _read_jpeg(path: Path) -> tuple[str, torch.Tensor]:
    # torchvision uses a file-backed tensor here. Clone it so later JPEG
    # decoding reads anonymous resident memory rather than faulting file pages.
    encoded = read_file(str(path)).clone()
    if encoded.dtype != torch.uint8 or encoded.ndim != 1:
        raise ValueError("Invalid encoded JPEG tensor for {}.".format(path))
    return path.name, encoded


class InMemoryJpegCache:
    def __init__(self, dataset: Any) -> None:
        self.dataset = dataset
        self.enabled = bool(getattr(dataset, "use_in_memory_jpeg_cache", False))

    def prepare(self, image_names: Iterable[str] | None = None) -> dict[str, torch.Tensor]:
        if not self.enabled:
            setattr(self.dataset, "in_memory_jpeg_images", {})
            return {}

        image_dir = Path(self.dataset.source_path) / str(self.dataset.images)
        if not image_dir.is_dir():
            raise FileNotFoundError("Image directory not found: {}".format(image_dir))
        requested_names = None if image_names is None else {str(name) for name in image_names}
        image_paths = sorted(
            path
            for path in image_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in JPEG_SUFFIXES
            and (requested_names is None or path.name in requested_names)
        )
        if not image_paths:
            raise ValueError("No JPEG images found in {}.".format(image_dir))
        if requested_names is not None:
            missing_names = requested_names.difference(path.name for path in image_paths)
            if missing_names:
                raise FileNotFoundError(
                    "Missing JPEG images in {}: {}".format(
                        image_dir,
                        ", ".join(sorted(missing_names)),
                    )
                )

        started = time.perf_counter()
        worker_count = min(4, len(image_paths))
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="jpeg-memory-load") as executor:
            entries = list(
                tqdm(
                    executor.map(_read_jpeg, image_paths),
                    total=len(image_paths),
                    desc="Loading JPEGs into memory",
                    dynamic_ncols=True,
                )
            )
        images = dict(entries)
        setattr(self.dataset, "in_memory_jpeg_images", images)
        total_bytes = sum(int(encoded.numel()) for encoded in images.values())
        print(
            "[jpeg-cache] loaded {} images ({:.2f} GiB) into CPU memory in {:.2f}s".format(
                len(images),
                total_bytes / float(1024**3),
                time.perf_counter() - started,
            )
        )
        return images
