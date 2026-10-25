"""Bounded CPU packing and uint8 H2D lookahead on a non-default CUDA stream."""
from concurrent.futures import ThreadPoolExecutor
import time

import torch


_PACK_EXECUTOR = None
_UPLOAD_STREAMS = {}


def pack_group_pixels(source, rect, factor=1):
    y, x, height, width = rect
    crop = source[:, y//factor:(y+height+factor-1)//factor,
                  x//factor:(x+width+factor-1)//factor]
    if crop.is_contiguous() and crop.is_pinned():
        return crop
    packed = torch.empty(crop.shape, dtype=torch.uint8, pin_memory=True)
    packed.copy_(crop)
    return packed




class LeafGtPrefetcher:
    """At most two requested leaves (current and next); no full-view CUDA cache."""

    def __init__(self, source, factor=1):
        global _PACK_EXECUTOR
        if source.device.type != 'cpu' or source.dtype != torch.uint8:
            raise ValueError('Leaf prefetch requires CPU uint8 pixels.')
        if _PACK_EXECUTOR is None:
            _PACK_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix='leaf-gt-pack')
        self.device = torch.cuda.current_device()
        if self.device not in _UPLOAD_STREAMS:
            _UPLOAD_STREAMS[self.device] = torch.cuda.Stream(device=self.device)
        self.stream = _UPLOAD_STREAMS[self.device]
        self.source = source
        self.factor = int(factor)
        self.pending = {}
        self.stats = dict(requests=0, upload_bytes=0, max_pending=0, cpu_wait_seconds=0.0,
                          cpu_pack_requests=0)

    def _pack_and_upload(self, rect):
        packed = pack_group_pixels(self.source, rect, self.factor)
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            pixels = packed.to(device=f'cuda:{self.device}', non_blocking=True)
            ready = torch.cuda.Event()
            ready.record(self.stream)
        # Keep the pinned source until consumption; PyTorch's host allocator also
        # records the outstanding copy event when this tensor is subsequently freed.
        return pixels, ready, packed

    def submit(self, rect):
        key = tuple(rect)
        if key in self.pending:
            raise RuntimeError('Duplicate Leaf prefetch request.')
        if len(self.pending) >= 2:
            raise RuntimeError('Leaf prefetch queue exceeded current + next capacity.')
        self.pending[key] = _PACK_EXECUTOR.submit(self._pack_and_upload, key)
        y, x, height, width = key
        factor = self.factor
        self.stats['requests'] += 1
        self.stats['cpu_pack_requests'] += 1
        self.stats['upload_bytes'] += (self.source.shape[0] *
            ((y+height+factor-1)//factor-y//factor) * ((x+width+factor-1)//factor-x//factor))
        self.stats['max_pending'] = max(self.stats['max_pending'], len(self.pending))

    def take(self, rect):
        future = self.pending.pop(tuple(rect))
        start = time.perf_counter()
        pixels, ready, packed = future.result()
        self.stats['cpu_wait_seconds'] += time.perf_counter() - start
        current = torch.cuda.current_stream(self.device)
        current.wait_event(ready)
        pixels.record_stream(current)
        return pixels

    def close(self):
        error = None
        for future in self.pending.values():
            if not future.cancel():
                try:
                    future.result()
                except Exception as exc:
                    error = error or exc
        self.pending.clear()
        self.source = None
        if error is not None:
            raise error
