"""A discoverable, self-healing *local* Dask cluster, importable as ``zetteldev.dask``.

Where the sibling :mod:`zetteldev.della` reaches *out* to SLURM, this module keeps
its labour *at home*: it raises a Dask cluster on the local machine (your beefy CPU
and two 4090s), and — crucially — makes it **discoverable** so that any notebook,
script, or fresh kernel can *find and join* the cluster the first one raised, rather
than each spawning its own short-lived rival::

    from zetteldev import dask as zdask

    client = zdask.client()          # discover a running cluster, or launch one — idempotent
    futures = client.map(train, configs)
    results = client.gather(futures)

The cluster outlives the kernel that launched it: a detached ``tmux`` session
(``zdev-dask-<name>``) holds the scheduler and workers, and a Dask *scheduler-file*
on disk (under the gitignored ``.zetteldev/dask/run/``) is the rendezvous point every
client reads. Call ``zdask.client()`` from a second notebook and you join the *same*
cluster, dashboard and all.

Three layers of self-healing, by deliberate design (see the discussion in the zettel):

* **Worker death** — each worker runs under a Dask *nanny*, which restarts it on crash
  or when it crosses its memory limit, plus a staggered ``--lifetime`` rollover so a
  slow leak from a long run never fells it.
* **Scheduler death** — the one layer Dask does not give you: each ``tmux`` window
  wraps its process in a ``while true`` supervisor loop, so a dead scheduler (or a
  dead worker whose nanny also died) is relaunched within seconds.
* **Stale discovery** — :func:`client` probes the recorded address before trusting it;
  an orphaned scheduler-file (process gone, file remains) is garbage-collected and the
  cluster relaunched.

GPU policy (the ``gpus`` argument), distilled from the design conversation:

* ``"all"`` (default) — every worker sees *both* 4090s, with JAX's eager
  pre-allocation disabled so a second worker cannot immolate the first. This suits the
  *butler* regime: one big job (a 100-epoch training, a large JAX computation) that
  parallelizes *itself* across the cards while Dask merely orchestrates.
* ``"pin"`` — one worker per card, each pinned via ``CUDA_VISIBLE_DEVICES`` and tagged
  ``resources={"GPU": 1}``. This suits the *work-farm* regime: many independent
  GPU jobs (a hyper-parameter sweep, sharded embedding) where the scheduler must
  guarantee one job per card so VRAM never collides. Submit with
  ``client.map(fn, items, resources={"GPU": 1})``.
* ``None`` / ``"none"`` — the cards are hidden; a CPU-only cluster for the I/O- and
  CPU-bound floods (API deluges, bootstrap resampling) where you want many workers.

Scaling. Workers join a *live* cluster at any moment, so growth is cheap:
:func:`scale` adds or removes worker processes on demand, and :func:`adapt` runs a
light controller that grows the pool under backlog and shrinks it when idle — guarded
by a conservative ``gpu_ceiling``, since adaptive scaling reads the task queue, never
the VRAM gauge.
"""

from __future__ import annotations

import json
import socket
import subprocess
import threading
import time
import zlib
from pathlib import Path
from select import select
from typing import Any

__all__ = [
    "client",
    "connect",
    "disconnect",
    "scale",
    "adapt",
    "status",
    "dashboard_url",
    "restart",
    "shutdown",
]

# --------------------------------------------------------------------------- #
# Paths and environment. The package lives at <repo>/.zetteldev/, so the repo
# root is one directory up; the local cluster's runtime state (scheduler-files,
# generated launch scripts) lives in a gitignored run/ dir beside this module.
# --------------------------------------------------------------------------- #

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RUN_DIR = _REPO_ROOT / ".zetteldev" / "dask" / "run"

# Prefer the repo's own venv binaries; fall back to `uv run` if absent.
_VENV_DASK = _REPO_ROOT / ".venv" / "bin" / "dask"
_VENV_PY = _REPO_ROOT / ".venv" / "bin" / "python"
if _VENV_DASK.exists():
    _DASK = str(_VENV_DASK)
    _PY = str(_VENV_PY)
else:  # pragma: no cover - depends on host layout
    _DASK = f"uv run --project {_REPO_ROOT} dask"
    _PY = f"uv run --project {_REPO_ROOT} python"

# Healing/rollover defaults (overridable per call).
_LIFETIME = "2h"
_LIFETIME_STAGGER = "4m"
_RESTART_DELAY = 3  # seconds the supervisor waits before relaunching a dead process

# A GPU cluster scales on the task queue, which is blind to VRAM; cap it hard.
_DEFAULT_GPU_CEILING = 2

# Live SSH tunnels opened by connect(), keyed by cluster name: {name: (ssh, [tunnels])}.
_TUNNELS: dict[str, Any] = {}


def _ports(name: str) -> tuple[int, int]:
    """Deterministic (scheduler, dashboard) ports for a named cluster.

    Derived from the name so distinct clusters never collide and — critically —
    so the address is *stable across restarts*, letting clients and workers
    reconnect to a relaunched scheduler via the same scheduler-file.
    """
    if name == "default":
        return 8786, 8787
    offset = (zlib.crc32(name.encode()) % 200) * 2
    sched = 8800 + offset
    return sched, sched + 1


def _sched_file(name: str) -> Path:
    return _RUN_DIR / f"scheduler-{name}.json"


def _worker_script(name: str) -> Path:
    return _RUN_DIR / f"worker-{name}.sh"


def _scheduler_script(name: str) -> Path:
    return _RUN_DIR / f"scheduler-{name}.sh"


def _meta_file(name: str) -> Path:
    return _RUN_DIR / f"meta-{name}.json"


def _read_meta(name: str) -> dict:
    try:
        return json.loads(_meta_file(name).read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _session(name: str) -> str:
    return f"zdev-dask-{name}"


def _detect_gpus() -> list[int]:
    """Indices of locally-visible NVIDIA GPUs (via ``nvidia-smi -L``)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10
        ).stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    return [i for i, line in enumerate(out.splitlines()) if line.startswith("GPU ")]


# --------------------------------------------------------------------------- #
# tmux helpers — the cluster lives in a detached session, one window per
# process, so you can `tmux attach -t zdev-dask-default` and watch it heal.
# --------------------------------------------------------------------------- #

def _tmux(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["tmux", *args], capture_output=True, text=True)


def _session_exists(name: str) -> bool:
    return _tmux("has-session", "-t", _session(name)).returncode == 0


def _windows(name: str) -> list[str]:
    if not _session_exists(name):
        return []
    res = _tmux("list-windows", "-t", _session(name), "-F", "#{window_name}")
    return [w for w in res.stdout.splitlines() if w]


def _worker_indices(name: str) -> list[int]:
    """Worker-window indices, sorted *numerically* (so w10 sorts after w9)."""
    return sorted(
        int(w[1:]) for w in _windows(name) if w.startswith("w") and w[1:].isdigit()
    )


# --------------------------------------------------------------------------- #
# Discovery: read the scheduler-file, probe the address, garbage-collect stale.
# --------------------------------------------------------------------------- #

def _read_address(name: str) -> str | None:
    sf = _sched_file(name)
    if not sf.exists():
        return None
    try:
        return json.loads(sf.read_text()).get("address")
    except (json.JSONDecodeError, OSError):
        return None


def _probe(address: str, timeout: float = 2.0) -> bool:
    """True iff a TCP connection to ``tcp://host:port`` succeeds (cheap liveness)."""
    try:
        host, port = address.rsplit("/", 1)[-1].rsplit(":", 1)
    except ValueError:
        return False
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except (OSError, ValueError):
        return False


def _alive(name: str) -> bool:
    addr = _read_address(name)
    return bool(addr) and _probe(addr)


def _gc(name: str) -> None:
    """Tear down any (possibly dead) session and remove its scheduler-file."""
    if _session_exists(name):
        _tmux("kill-session", "-t", _session(name))
    _sched_file(name).unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# Launch-script generation. We write small shell scripts (rather than wrestle
# nested tmux quoting) that wrap each process in a supervisor loop.
# --------------------------------------------------------------------------- #

def _write_scheduler_script(
    name: str, sched_port: int, dash_port: int, host: str
) -> Path:
    sf = _sched_file(name)
    script = f"""#!/usr/bin/env bash
# Auto-generated by zetteldev.dask — supervises the scheduler for cluster '{name}'.
while true; do
  {_DASK} scheduler --host {host} --port {sched_port} \\
    --dashboard-address :{dash_port} --scheduler-file {sf}
  echo "[zdev-dask:{name}] scheduler exited; relaunch in {_RESTART_DELAY}s" >&2
  sleep {_RESTART_DELAY}
done
"""
    path = _scheduler_script(name)
    path.write_text(script)
    path.chmod(0o755)
    return path


def _write_worker_script(
    name: str,
    *,
    gpus: str | None,
    gpu_ids: list[int],
    slots: int,
    memory_limit: str,
    nthreads: int | None,
    lifetime: str,
    lifetime_stagger: str,
) -> Path:
    """One script handling every GPU mode; ``$1`` is the worker index.

    For ``pin`` the index selects a card from ``gpu_ids``; for ``all`` every
    worker sees the whole visible set with JAX pre-allocation disabled; for the
    CPU cluster the cards are hidden so a stray ``.cuda`` call fails loudly.
    """
    sf = _sched_file(name)
    visible_all = ",".join(str(g) for g in gpu_ids)
    gpu_bash_list = " ".join(str(g) for g in gpu_ids)
    nth = f"--nthreads {nthreads}" if nthreads else ""
    mode = gpus or "none"

    script = f"""#!/usr/bin/env bash
# Auto-generated by zetteldev.dask — supervises worker $1 for cluster '{name}'.
idx="$1"
MODE="{mode}"
GPUS=({gpu_bash_list})
if [ "$MODE" = "pin" ]; then
  export CUDA_VISIBLE_DEVICES="${{GPUS[$idx]}}"
  RES="--resources GPU={slots}"
elif [ "$MODE" = "all" ]; then
  export CUDA_VISIBLE_DEVICES="{visible_all}"
  RES=""
else
  export CUDA_VISIBLE_DEVICES=""
  RES=""
fi
# Keep JAX from swallowing the whole card on first use (lets workers coexist).
export XLA_PYTHON_CLIENT_PREALLOCATE=false
while true; do
  {_DASK} worker --scheduler-file {sf} --name "w$idx" \\
    --memory-limit {memory_limit} \\
    --lifetime {lifetime} --lifetime-restart --lifetime-stagger {lifetime_stagger} \\
    $RES {nth}
  echo "[zdev-dask:{name}] worker $idx exited; relaunch in {_RESTART_DELAY}s" >&2
  sleep {_RESTART_DELAY}
done
"""
    path = _worker_script(name)
    path.write_text(script)
    path.chmod(0o755)
    return path


def _start_worker_window(name: str, idx: int) -> None:
    _tmux(
        "new-window", "-t", _session(name), "-n", f"w{idx}",
        f"bash {_worker_script(name)} {idx}",
    )


# --------------------------------------------------------------------------- #
# SSH tunnelling — forward a local port to a remote cluster over an
# authenticated paramiko transport (ported from reason_reckon's beast.py, so
# zetteldev stays self-contained). Used by connect() to reach a privately-bound
# remote cluster without exposing Dask's unauthenticated protocol to any network.
# --------------------------------------------------------------------------- #

class _Tunnel:
    """Forward ``127.0.0.1:local_port`` to ``remote_host:remote_port`` over an
    open paramiko transport, in a daemon thread (one direct-tcpip channel per
    accepted connection)."""

    # Per-read chunk (not a cap — this is a streaming pump). Matched to paramiko's
    # DEFAULT_MAX_PACKET_SIZE, so each recv pulls ~one channel packet; reading more
    # buys little. Throughput is bounded by paramiko's Python crypto and the channel
    # window, not this number — for bulk transfers prefer subprocess `ssh -L`.
    _BUF = 32 * 1024

    def __init__(self, local_port: int, remote_host: str, remote_port: int, transport):
        self._sock = socket.socket()
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", local_port))
        self._sock.listen(5)
        self._sock.settimeout(1.0)
        self._shutdown = False
        self._transport = transport
        self._remote = (remote_host, remote_port)
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()

    def _handle(self, client_sock: socket.socket) -> None:
        try:
            chan = self._transport.open_channel(
                "direct-tcpip", self._remote, client_sock.getpeername()
            )
        except Exception:
            client_sock.close()
            return
        try:
            while True:
                r, _, _ = select([client_sock, chan], [], [], 5)
                if client_sock in r:
                    data = client_sock.recv(self._BUF)
                    if not data:
                        break
                    chan.sendall(data)
                if chan in r:
                    data = chan.recv(self._BUF)
                    if not data:
                        break
                    client_sock.sendall(data)
        finally:
            chan.close()
            client_sock.close()

    def _accept_loop(self) -> None:
        while not self._shutdown:
            try:
                cs, _ = self._sock.accept()
                threading.Thread(target=self._handle, args=(cs,), daemon=True).start()
            except socket.timeout:
                continue
            except OSError:
                break

    def close(self) -> None:
        self._shutdown = True
        self._sock.close()


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def client(
    name: str = "default",
    *,
    n_workers: int = 2,
    gpus: str | None = "all",
    memory_limit: str = "auto",
    nthreads: int | None = None,
    slots: int = 1,
    host: str = "127.0.0.1",
    timeout: float = 60.0,
) -> "Any":
    """Connect to a running local Dask cluster, launching one if none is found.

    Idempotent: the first call raises a detached, self-healing cluster; every
    later call (from any kernel) discovers it via its scheduler-file and returns
    a connected client to the *same* cluster.

    Args:
        name: Cluster name; distinct names are independent clusters (distinct
            ports, tmux session, scheduler-file).
        n_workers: Worker floor at launch. Ignored for ``gpus="pin"``, where the
            count is fixed to the number of detected GPUs (one per card).
        gpus: ``"all"`` (every worker sees all cards; default), ``"pin"`` (one
            worker per card, GPU-resource tagged), or ``None``/``"none"`` (CPU).
        memory_limit: Per-worker memory cap (Dask syntax, e.g. ``"16GB"``);
            ``"auto"`` lets Dask divide host RAM.
        nthreads: Threads per worker; ``None`` lets Dask choose.
        slots: For ``gpus="pin"``, the GPU-resource units a worker advertises
            (raise above 1 to let several light tasks share one card).
        host: Interface the scheduler binds and advertises. Default
            ``"127.0.0.1"`` (local-only). Pass a reachable address (e.g. this
            box's Tailscale IP) to let remote clients — SolveIt, another machine
            — connect. Note: Dask's protocol is unauthenticated, so only expose
            on a trusted network. Takes effect on a *fresh* launch.
        timeout: Seconds to wait for a freshly-launched scheduler to come up.

    Returns:
        A ``distributed.Client`` connected to the cluster (renders inline in a
        notebook, with a dashboard link).

    Raises:
        TimeoutError: the scheduler did not come up within ``timeout``.
    """
    from distributed import Client

    _RUN_DIR.mkdir(parents=True, exist_ok=True)
    gpu_ids = _detect_gpus() if gpus in ("all", "pin") else []
    if gpus == "pin":
        if not gpu_ids:
            raise RuntimeError("gpus='pin' requested but no GPUs detected.")
        n_workers = len(gpu_ids)

    # Fast path: a healthy cluster already exists — join it, top up to the floor.
    if _alive(name):
        cl = Client(scheduler_file=str(_sched_file(name)))
        if len(_worker_indices(name)) < n_workers:
            scale(n_workers, name=name)
        return cl

    # Otherwise (re)launch from clean state.
    _gc(name)
    sched_port, dash_port = _ports(name)
    _write_scheduler_script(name, sched_port, dash_port, host)
    _write_worker_script(
        name,
        gpus=gpus,
        gpu_ids=gpu_ids,
        slots=slots,
        memory_limit=memory_limit,
        nthreads=nthreads,
        lifetime=_LIFETIME,
        lifetime_stagger=_LIFETIME_STAGGER,
    )

    # Record cluster metadata (read by the adaptive controller and status()).
    _meta_file(name).write_text(
        json.dumps(
            {"gpus": gpus or "none", "slots": slots, "gpu_ids": gpu_ids, "host": host}
        )
    )

    # Detached session; first window is the supervised scheduler. `new-session`
    # fails if the session already exists, which is how we lose a launch race
    # against a concurrent kernel — in that case we drop to discovery rather than
    # double-launch (the winner brings up the scheduler and floor workers).
    launched = _tmux(
        "new-session", "-d", "-s", _session(name), "-n", "scheduler",
        f"bash {_scheduler_script(name)}",
    ).returncode == 0

    deadline = time.time() + timeout
    while time.time() < deadline:
        if _alive(name):
            break
        time.sleep(1.0)
    else:
        raise TimeoutError(
            f"Scheduler for cluster '{name}' did not come up within {timeout}s; "
            f"inspect it with `tmux attach -t {_session(name)}`."
        )

    if launched:
        for idx in range(n_workers):
            _start_worker_window(name, idx)

    return Client(scheduler_file=str(_sched_file(name)))


def scale(n: int, name: str = "default") -> int:
    """Set the cluster to ``n`` worker processes, adding or removing as needed.

    Scaling up launches fresh worker windows (indexed above the highest existing
    one, to avoid name collisions) that join the live cluster immediately.
    Scaling down kills the highest-numbered worker windows; Dask reschedules any
    tasks they were running, so the shrink is safe though abrupt.

    For a ``gpus="pin"`` cluster ``n`` is clamped to the GPU count — there is one
    card per worker, and a surplus worker would claim a GPU it cannot see.

    Returns:
        The resulting worker count.
    """
    if not _session_exists(name):
        raise RuntimeError(f"No cluster '{name}' running; call client() first.")
    meta = _read_meta(name)
    if meta.get("gpus") == "pin":
        n = min(n, len(meta.get("gpu_ids", [])))
    idxs = _worker_indices(name)
    have = len(idxs)
    if n > have:
        nxt = (idxs[-1] + 1) if idxs else 0
        for k in range(n - have):
            _start_worker_window(name, nxt + k)
    elif n < have:
        for i in idxs[n:]:  # numerically highest windows
            _tmux("kill-window", "-t", f"{_session(name)}:w{i}")
    return n


def adapt(
    minimum: int = 1,
    maximum: int = 12,
    *,
    name: str = "default",
    interval: float = 10.0,
    gpu_ceiling: int = _DEFAULT_GPU_CEILING,
) -> None:
    """Launch a background controller that scales the cluster to its load.

    The controller polls the scheduler's own ``adaptive_target()`` (Dask's
    estimate of the workers a backlog warrants), clamps it to
    ``[minimum, maximum]`` — and, for a GPU cluster, to ``gpu_ceiling``, since
    the queue length it reads is blind to VRAM — then calls :func:`scale`.

    It runs in a ``adapt`` tmux window; stop it by killing that window or via
    :func:`shutdown`.
    """
    if not _session_exists(name):
        raise RuntimeError(f"No cluster '{name}' running; call client() first.")
    cmd = (
        f"{_PY} -m zetteldev.dask _adapt --name {name} "
        f"--min {minimum} --max {maximum} --interval {interval} "
        f"--gpu-ceiling {gpu_ceiling}"
    )
    _tmux("new-window", "-t", _session(name), "-n", "adapt", cmd)


def status(name: str = "default") -> dict:
    """Live summary of a cluster: liveness, worker count, dashboard, resources."""
    info: dict[str, Any] = {
        "name": name,
        "alive": _alive(name),
        "gpus": _read_meta(name).get("gpus", "none"),
        "address": _read_address(name),
        "worker_windows": len(_worker_indices(name)),
        "dashboard": dashboard_url(name),
    }
    if info["alive"]:
        from distributed import Client

        with Client(scheduler_file=str(_sched_file(name)), timeout="5s") as cl:
            sched = cl.scheduler_info()
            info["workers"] = len(sched.get("workers", {}))
            info["threads"] = sum(
                w.get("nthreads", 0) for w in sched.get("workers", {}).values()
            )
    return info


def dashboard_url(name: str = "default") -> str:
    _, dash_port = _ports(name)
    host = _read_meta(name).get("host", "127.0.0.1")
    return f"http://{host}:{dash_port}/status"


def restart(name: str = "default") -> None:
    """Restart all workers (clears their memory) without dropping the scheduler."""
    from distributed import Client

    if not _alive(name):
        raise RuntimeError(f"No live cluster '{name}'.")
    with Client(scheduler_file=str(_sched_file(name)), timeout="10s") as cl:
        cl.restart()


def shutdown(name: str = "default") -> None:
    """Tear the cluster down entirely: kill the session, remove its run files."""
    _gc(name)
    _worker_script(name).unlink(missing_ok=True)
    _scheduler_script(name).unlink(missing_ok=True)
    _meta_file(name).unlink(missing_ok=True)


def connect(
    host: str,
    *,
    name: str = "default",
    local_port: int | None = None,
    dashboard: bool = True,
    ssh_config: str = "~/.ssh/config",
    username: str | None = None,
    key_filename: str | None = None,
    ignore_version_mismatch: bool = True,
) -> "Any":
    """Connect to a *remote* cluster over an authenticated SSH tunnel (paramiko).

    The counterpart to :func:`client` for when you are *not* on the box hosting
    the cluster — a laptop, or SolveIt, reaching an Athomia cluster. SSH keys do
    the authenticating; the unauthenticated Dask protocol never leaves the remote
    box's loopback. The port-forward is carried in-process over paramiko, so no
    separate ``ssh -L`` is needed.

    ``host`` is resolved through ``~/.ssh/config`` — ``HostName``, ``User``,
    ``IdentityFile`` and ``ProxyCommand`` are all honoured — so connection
    details (including a tailnet ProxyCommand) live in standard ssh config you
    own, not baked into this module.

    Pairs with a **privately-bound** cluster (``client()``'s ``host="127.0.0.1"``
    default), *not* the tailscale-exposed ``dask-athomia`` launch — else two auth
    stories fight. Note: keep ``gather`` routing through the scheduler
    (``direct=False``, the default); remote workers advertise loopback addresses
    the client cannot reach through the single tunnel.

    Args:
        host: SSH host (an alias in ``~/.ssh/config``, or a hostname/IP).
        name: Which remote cluster — selects its deterministic ports.
        local_port: Local port for the tunnel's near end (default: the remote
            scheduler port, so the address reads identically on both ends).
        dashboard: Also forward the dashboard port.
        ssh_config: Path to the ssh config to resolve ``host`` against.
        username: Override the ssh user (else from config / current user).
        key_filename: Override the identity file (else from config / agent).
        ignore_version_mismatch: Suppress Dask's client/cluster version-mismatch
            warning (default ``True``). A deliberate-but-benign version gap is the
            normal case across the tunnel — numpy 1.26 bridges 2.x array pickles,
            python/tornado diffs are patch-level, and the serialization-critical
            libs (cloudpickle, msgpack) must still match. Set ``False`` to see the
            warning. *Caveat:* pandas pickling across a major (2.x↔3.x) is fragile;
            ship DataFrames as parquet/arrow/dict, not as bare objects.

    Returns:
        A ``distributed.Client`` connected through the tunnel. Tear down with
        :func:`disconnect`.
    """
    import paramiko

    sched_port, dash_port = _ports(name)
    local_port = local_port or sched_port

    disconnect(name)  # free any prior tunnel on this name (and its local port)

    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    conf: dict = {}
    cfg_path = Path(ssh_config).expanduser()
    if cfg_path.exists():
        sc = paramiko.SSHConfig()
        with cfg_path.open() as fh:
            sc.parse(fh)
        conf = sc.lookup(host)

    sock = paramiko.ProxyCommand(conf["proxycommand"]) if "proxycommand" in conf else None
    ssh.connect(
        conf.get("hostname", host),
        port=int(conf.get("port", 22)),
        username=username or conf.get("user"),
        key_filename=key_filename or (conf.get("identityfile") or [None])[0],
        sock=sock,
        allow_agent=True,
        look_for_keys=True,
    )
    transport = ssh.get_transport()
    assert transport is not None  # set by a successful connect()
    transport.set_keepalive(30)  # keep an idle tunnel alive between submissions

    tunnels = [_Tunnel(local_port, "127.0.0.1", sched_port, transport)]
    if dashboard:
        tunnels.append(_Tunnel(dash_port, "127.0.0.1", dash_port, transport))
    _TUNNELS[name] = (ssh, tunnels)

    from distributed import Client

    if not ignore_version_mismatch:
        return Client(f"tcp://127.0.0.1:{local_port}")

    import warnings

    from distributed.versions import VersionMismatchWarning

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=VersionMismatchWarning)
        return Client(f"tcp://127.0.0.1:{local_port}")


def disconnect(name: str = "default") -> None:
    """Close the SSH tunnel(s) and connection opened by :func:`connect`."""
    entry = _TUNNELS.pop(name, None)
    if not entry:
        return
    ssh, tunnels = entry
    for t in tunnels:
        t.close()
    ssh.close()


# --------------------------------------------------------------------------- #
# Adaptive controller entry point (run as a subprocess by adapt()).
# --------------------------------------------------------------------------- #

def _run_adapt_controller(
    name: str, minimum: int, maximum: int, interval: float, gpu_ceiling: int
) -> None:  # pragma: no cover - long-running loop
    from distributed import Client

    # The GPU ceiling guards VRAM, so it applies only to GPU clusters — not to a
    # CPU cluster that merely happens to run on a GPU-equipped machine. A pinned
    # cluster is further capped at one worker per card.
    meta = _read_meta(name)
    mode = meta.get("gpus", "none")
    if mode == "pin":
        ceiling = min(maximum, len(meta.get("gpu_ids", [])))
    elif mode == "all":
        ceiling = min(maximum, gpu_ceiling)
    else:
        ceiling = maximum
    with Client(scheduler_file=str(_sched_file(name))) as cl:
        while True:
            try:
                # run_on_scheduler injects the scheduler as kwarg `dask_scheduler`.
                target = cl.run_on_scheduler(
                    lambda dask_scheduler: dask_scheduler.adaptive_target()
                )
            except Exception as exc:  # scheduler bouncing; wait and retry
                print(f"[zdev-dask:{name}] adapt poll failed: {exc}", flush=True)
                time.sleep(interval)
                continue
            target = max(minimum, min(ceiling, int(target)))
            if target != len(_worker_indices(name)):
                print(f"[zdev-dask:{name}] adapt -> {target} workers", flush=True)
                scale(target, name=name)
            time.sleep(interval)


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="zetteldev.dask internal CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("_adapt", help="run the adaptive controller (internal)")
    p.add_argument("--name", default="default")
    p.add_argument("--min", type=int, default=1)
    p.add_argument("--max", type=int, default=12)
    p.add_argument("--interval", type=float, default=10.0)
    p.add_argument("--gpu-ceiling", type=int, default=_DEFAULT_GPU_CEILING)

    pu = sub.add_parser("up", help="launch/discover a cluster and print its address")
    pu.add_argument("--name", default="default")
    pu.add_argument("--host", default="127.0.0.1")
    pu.add_argument("--gpus", default="all", help="all | pin | none")
    pu.add_argument("--workers", type=int, default=2)

    a = parser.parse_args()
    if a.cmd == "_adapt":
        _run_adapt_controller(a.name, a.min, a.max, a.interval, a.gpu_ceiling)
    elif a.cmd == "up":
        cl = client(name=a.name, host=a.host, gpus=a.gpus, n_workers=a.workers)
        addr = cl.scheduler_info().get("address")
        print(f"cluster:   {a.name}")
        print(f"scheduler: {addr}")
        print(f"dashboard: {dashboard_url(a.name)}")
        print(f'connect:   Client("{addr}")')
