"""Compact per-view parameter gradients; opacity stays dense for global losses."""
import torch


class CompactGradientAccumulator:
    def __init__(self, parameters, candidates):
        self.parameters = parameters
        self.candidates = candidates
        self.index_map = None

    @torch.no_grad()
    def prepare(self):
        if self.index_map is not None:
            return
        n = self.parameters[0].shape[0]
        device = self.parameters[0].device
        selected = torch.zeros(n, dtype=torch.bool, device=device)
        selected[self.candidates.long()] = True
        indices = selected.nonzero(as_tuple=True)[0]
        del selected
        self.index_map = torch.full((n,), -1, dtype=torch.int32, device=device)
        self.index_map[indices] = torch.arange(indices.numel(), dtype=torch.int32, device=device)
        for i, parameter in enumerate(self.parameters):
            if i == 3:
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
                continue
            values = torch.zeros((indices.numel(), *parameter.shape[1:]),
                                 dtype=parameter.dtype, device=device)
            gradient = torch.sparse_coo_tensor(indices.unsqueeze(0), values,
                size=parameter.shape, is_coalesced=True)
            gradient._boundedgs_index_map = self.index_map
            parameter.grad = gradient

    def buffers(self):
        self.prepare()
        return tuple(p.grad.values() if p.grad.is_sparse else p.grad for p in self.parameters)


def can_use_compact_gradients(parameters, optimizer):
    return bool(getattr(optimizer, "supports_compact_gradients", False)) and all(
        i == 3 or p.grad is None or p.grad.is_sparse for i, p in enumerate(parameters)
    )


def parameter_adam_storage_bytes(model):
    """Unique CUDA storage of parameters and all initialized optimizer state."""
    optimizer = model.optimizer
    seen = set()
    total = 0
    for group in optimizer.param_groups:
        for parameter in group['params']:
            tensors = [parameter, *optimizer.state.get(parameter, {}).values()]
            for value in tensors:
                if not torch.is_tensor(value) or value.device.type != 'cuda':
                    continue
                storage = value.untyped_storage()
                key = (value.device, storage.data_ptr())
                if key not in seen:
                    seen.add(key)
                    total += storage.nbytes()
    return total


def long_term_storage_bytes(model):
    """Unique CUDA storage kept across views.

    Besides Gaussian parameters and initialized optimizer state, densification
    statistics live for many iterations and therefore belong to the persistent
    baseline rather than the per-view training working set.
    """
    optimizer = model.optimizer
    tensors = []
    if optimizer is not None:
        for group in optimizer.param_groups:
            for parameter in group['params']:
                tensors.append(parameter)
                tensors.extend(optimizer.state.get(parameter, {}).values())
    for field_name in ("xyz_gradient_accum", "xyz_gradient_accum_abs", "denom"):
        value = getattr(model, field_name, None)
        if torch.is_tensor(value):
            tensors.append(value)

    seen = set()
    total = 0
    for value in tensors:
        if not torch.is_tensor(value) or value.device.type != 'cuda':
            continue
        storage = value.untyped_storage()
        key = (value.device, storage.data_ptr())
        if key in seen:
            continue
        seen.add(key)
        total += storage.nbytes()
    return total


def gradient_storage_bytes(parameters):
    seen = set()
    total = 0
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        tensors = [gradient] if not gradient.is_sparse else [gradient._indices(), gradient._values()]
        mapping = getattr(gradient, '_boundedgs_index_map', None)
        if mapping is not None:
            tensors.append(mapping)
        for tensor in tensors:
            storage = tensor.untyped_storage()
            key = (tensor.device, storage.data_ptr())
            if key not in seen:
                seen.add(key)
                total += storage.nbytes()
    return total
