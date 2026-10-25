"""Resolve per-task resource budgets before launching training or postprocessing."""
import math
import os
from pathlib import Path


def _percent(config, name):
    value = config.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 100:
        raise ValueError(f'resource_limits.{name} must be a number in (0, 100] or null')
    return float(value)


def memory_capacity_bytes():
    """Use total capacity, never fluctuating MemAvailable, for repeatable budgets."""
    total = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                 if line.startswith('MemTotal:'))
    limit_file = Path('/sys/fs/cgroup/memory.max')
    if limit_file.is_file():
        limit = limit_file.read_text().strip()
        if limit != 'max':
            total = min(total, int(limit))
    return total


def resolve_resource_limits(config, allowed_cpus=None, memory_capacity=None):
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise ValueError('resource_limits must be an object')
    unknown = set(config) - {'cpu_max_percent', 'memory_max_percent', 'memory_mode'}
    if unknown:
        raise ValueError(f'Unknown resource_limits fields: {sorted(unknown)}')
    cpu_percent = _percent(config, 'cpu_max_percent')
    memory_percent = _percent(config, 'memory_max_percent')
    mode = config.get('memory_mode', 'strict')
    if mode not in ('strict', 'guard'):
        raise ValueError('resource_limits.memory_mode must be strict or guard')
    allowed = sorted(os.sched_getaffinity(0) if allowed_cpus is None else allowed_cpus)
    if not allowed:
        raise ValueError('No allowed CPUs')
    count = len(allowed) if cpu_percent is None else math.floor(len(allowed) * cpu_percent / 100)
    if count < 1:
        raise ValueError('CPU percentage is too small to allocate even one logical CPU')
    memory_capacity = memory_capacity_bytes() if memory_capacity is None else memory_capacity
    return {'cpu_max_percent': cpu_percent, 'allowed_cpus': allowed, 'cpu_count': count,
            'effective_cpu_percent': 100 * count / len(allowed),
            'memory_max_percent': memory_percent, 'memory_capacity_bytes': memory_capacity,
            'memory_limit_bytes': None if memory_percent is None else math.floor(memory_capacity * memory_percent / 100),
            'memory_mode': mode}


def check_resource_support(config):
    """Fail before archiving outputs if a requested hard limit is unavailable."""
    budget = resolve_resource_limits(config)
    if budget['memory_limit_bytes'] is not None and budget['memory_mode'] == 'strict':
        root = Path('/sys/fs/cgroup')
        readonly = os.statvfs(root).f_flag & os.ST_RDONLY
        controllers = (root / 'cgroup.subtree_control').read_text().split()
        if readonly or not os.access(root, os.W_OK) or 'memory' not in controllers:
            raise RuntimeError('Strict memory limit requires a writable, delegated cgroup memory controller; '
                               'this environment cannot enforce it. Use memory_mode=guard only if '
                               'stop-on-excess monitoring is acceptable.')
    return budget


def resource_description(config):
    budget = resolve_resource_limits(config)
    cpu = ('unlimited' if budget['cpu_max_percent'] is None else
           f"{budget['cpu_count']}/{len(budget['allowed_cpus'])} logical CPUs "
           f"({budget['effective_cpu_percent']:.2f}%)")
    memory = ('unlimited' if budget['memory_limit_bytes'] is None else
              f"{budget['memory_limit_bytes'] / 1024**3:.2f} GiB ({budget['memory_mode']})")
    return f'Resource limits: CPU={cpu}, memory={memory}'


def cpu_order_for_gpu(allowed, gpu):
    """Distribute whole physical cores by GPU, keeping SMT siblings together."""
    try:
        gpu_count = max(1, len(list(Path('/proc/driver/nvidia/gpus').iterdir())))
        slot = int(gpu) % gpu_count
        cores = {}
        for cpu in allowed:
            topology = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
            key = ((topology / 'physical_package_id').read_text().strip(),
                   (topology / 'core_id').read_text().strip())
            cores.setdefault(key, []).append(cpu)
        buckets = {}
        for core in sorted(cores.values(), key=lambda group: group[0]):
            buckets.setdefault(len(core), []).append(core)
        preferred, remaining = [], []
        for width in sorted(buckets, reverse=True):
            for index, core in enumerate(buckets[width]):
                (preferred if index % gpu_count == slot else remaining).extend(core)
        return preferred + remaining
    except (OSError, ValueError):
        return list(allowed)


class ResourceLimiter:
    """Apply inherited CPU affinity and optional cgroup or RSS guard to a child tree."""
    def __init__(self, config, gpu):
        import threading
        self.budget = check_resource_support(config)
        self.cgroup = None
        self.thread = None
        self.stop_event = threading.Event()
        self.exceeded = None
        self.peak_rss_bytes = None
        allowed = self.budget['allowed_cpus']
        count = self.budget['cpu_count']
        self.cpu_ids = cpu_order_for_gpu(allowed, gpu)[:count]
        if self.budget['cpu_max_percent'] is None:
            self.cpu_ids = None

    def prepare(self, cmd, env):
        import json
        import sys
        import uuid
        memory = self.budget['memory_limit_bytes']
        if memory is not None and self.budget['memory_mode'] == 'strict':
            self.cgroup = Path('/sys/fs/cgroup') / f'gs_{os.getpid()}_{uuid.uuid4().hex}'
            self.cgroup.mkdir()
            (self.cgroup / 'memory.max').write_text(str(memory))
            (self.cgroup / 'memory.oom.group').write_text('1')
        env = env.copy()
        if self.cpu_ids is not None:
            count = str(len(self.cpu_ids))
            for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
                env[key] = count
        env['GS_RESOURCE_LAUNCH'] = json.dumps({'cpu_ids': self.cpu_ids,
                                               'cgroup': str(self.cgroup) if self.cgroup else None})
        return [sys.executable, str(Path(__file__).resolve()), '--exec', *cmd], env

    def monitor(self, process):
        if self.budget['memory_limit_bytes'] is None or self.budget['memory_mode'] != 'guard':
            return
        import signal
        import threading
        import psutil

        def watch():
            while not self.stop_event.is_set() and process.poll() is None:
                try:
                    parent = psutil.Process(process.pid)
                    members = [parent, *parent.children(recursive=True)]
                    rss = 0
                    for member in members:
                        try:
                            rss += member.memory_info().rss
                        except psutil.NoSuchProcess:
                            pass
                    self.peak_rss_bytes = max(self.peak_rss_bytes or 0, rss)
                    if rss > self.budget['memory_limit_bytes']:
                        self.exceeded = f"Memory guard exceeded: RSS={rss} > limit={self.budget['memory_limit_bytes']} bytes"
                        try:
                            os.killpg(process.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            return
                        if not self.stop_event.wait(5) and process.poll() is None:
                            try:
                                os.killpg(process.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                        return
                except psutil.NoSuchProcess:
                    return
                except psutil.AccessDenied:
                    self.exceeded = 'Memory guard cannot inspect the child process; stopping instead of running unmonitored'
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    return
                self.stop_event.wait(0.2)

        self.thread = threading.Thread(target=watch, daemon=True)
        self.thread.start()

    def close(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=6)
        if self.cgroup is not None:
            if (self.cgroup / 'cgroup.kill').exists():
                (self.cgroup / 'cgroup.kill').write_text('1')
            self.cgroup.rmdir()

    def report(self):
        return {**self.budget, 'cpu_ids': self.cpu_ids, 'memory_peak_sampled_rss_bytes': self.peak_rss_bytes,
                'memory_guard_interval_seconds': 0.2, 'limit_exceeded': self.exceeded}


if __name__ == '__main__':
    import json
    import sys
    if len(sys.argv) < 3 or sys.argv[1] != '--exec':
        raise SystemExit('This helper is launched by run.py')
    policy = json.loads(os.environ.pop('GS_RESOURCE_LAUNCH'))
    if policy['cpu_ids'] is not None:
        os.sched_setaffinity(0, policy['cpu_ids'])
    if policy['cgroup'] is not None:
        (Path(policy['cgroup']) / 'cgroup.procs').write_text(str(os.getpid()))
    os.execvpe(sys.argv[2], sys.argv[2:], os.environ)
