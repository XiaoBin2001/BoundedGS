"""Per-camera CPU layout hints; every rendered leaf is recounted on the GPU."""
from dataclasses import dataclass
from weakref import WeakKeyDictionary


@dataclass
class _Node:
    rect: tuple[int, int, int, int]
    children: tuple | None = None

    def copy(self):
        return _Node(self.rect, None if self.children is None else tuple(c.copy() for c in self.children))


def _visit_node(node, count, estimate_size, split, budget, stats, merge_ratio):
    if node.children is None:
        estimate = count(node.rect)
        stats['count_call_count'] += 1
        children = split(node.rect) if estimate.payload_bytes > budget else None
        if children is None:
            if estimate.payload_bytes > budget:
                stats['forced_leaf_count'] += 1
            yield node.rect, estimate.payload_bytes, estimate.num_rendered
            return estimate.num_rendered, estimate.payload_bytes
        stats['split_count'] += 1
        node.children = tuple(_Node(tuple(rect)) for rect in children)

    # Merge at most one tree level per visit. Newly proposed merged nodes
    # do not qualify their own parents for immediate multi-level merging.
    terminal_children = all(child.children is None for child in node.children)
    rendered = payload = 0
    for child in node.children:
        child_rendered, child_payload = yield from _visit_node(child, count, estimate_size, split, budget, stats, merge_ratio)
        rendered += child_rendered
        payload += child_payload
    if (terminal_children and all(child.children is None for child in node.children)
            and payload <= budget * merge_ratio):
        # Cheap conservative gate above avoids size queries for full leaves.
        # Child counts form a parent hint for a fixed projection and disjoint
        # Tile-aligned children; next visit still performs an exact recount.
        parent = estimate_size(rendered, node.rect)
        if parent.payload_bytes <= budget * merge_ratio:
            node.children = None
            stats['leaf_cache_merge_proposals'] = stats.get('leaf_cache_merge_proposals', 0) + 1
    return rendered, payload


class CachedLeafPlan:
    def __init__(self, rect):
        self.root = _Node(tuple(rect))
        self.visits = 0

    def leaves(self, count, estimate_size, split, budget, stats, merge_ratio=0.65):
        """Yield immediately after counting; advance only once raster consumed offsets.

        A private tree is committed on exhaustion, so a failed/abandoned traversal
        cannot leave a partially changed cached layout. Merge proposals only affect
        the next visit, whose fresh count checks them before any raster allocation.
        """
        root = self.root.copy()
        cache_field = 'leaf_cache_hits' if self.visits else 'leaf_cache_misses'
        stats[cache_field] = stats.get(cache_field, 0) + 1


        yield from _visit_node(root, count, estimate_size, split, budget, stats, merge_ratio)
        self.root = root
        self.visits += 1


def get_cached_leaf_plan(context, camera, root, budget, render_depth):
    """Invalidate layout hints when image geometry, Group layout or budget changes."""
    cache = context.runtime_state.get('_adaptive_leaf_plans')
    if cache is None:
        cache = WeakKeyDictionary()
        context.runtime_state['_adaptive_leaf_plans'] = cache
    signature = (int(camera.image_height), int(camera.image_width),
                 context.method_config['group_grid'],
                 context.method_config['projection_mode'], int(budget), bool(render_depth))
    entry = cache.get(camera)
    if entry is None or entry[0] != signature:
        entry = (signature, {})
        cache[camera] = entry
    key = tuple(root)
    if key not in entry[1]:
        entry[1][key] = CachedLeafPlan(key)
    return entry[1][key]


def capture_cached_leaf_plans(runtime_state):
    """Save CPU layout trees by camera name, without serializing camera objects."""
    def pack(node):
        return node.rect, None if node.children is None else tuple(pack(c) for c in node.children)

    cache = runtime_state.get('_adaptive_leaf_plans', {})
    return {
        str(camera.image_name): (signature, [
            (key, pack(plan.root), plan.visits) for key, plan in plans.items()
        ])
        for camera, (signature, plans) in cache.items()
    }


def restore_cached_leaf_plans(runtime_state, cameras, saved):
    """Restore layout history; the next visit still checks every leaf's budget."""
    def unpack(value):
        rect, children = value
        return _Node(tuple(rect), None if children is None else tuple(unpack(c) for c in children))

    if not saved:
        return
    cameras_by_name = {str(camera.image_name): camera for camera in cameras}
    cache = WeakKeyDictionary()
    for name, (signature, plans) in saved.items():
        if name not in cameras_by_name:
            raise ValueError("Cached Leaf checkpoint contains unknown camera: {}".format(name))
        restored = {}
        for key, root, visits in plans:
            plan = CachedLeafPlan(key)
            plan.root = unpack(root)
            plan.visits = int(visits)
            restored[tuple(key)] = plan
        cache[cameras_by_name[name]] = (tuple(signature), restored)
    runtime_state['_adaptive_leaf_plans'] = cache
