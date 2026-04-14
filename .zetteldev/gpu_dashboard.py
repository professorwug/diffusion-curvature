#!/usr/bin/env python3
"""
GPU Dashboard for Della cluster
Shows real-time GPU utilization for running jobs via nvidia-smi.

Usage:
    ./gpu_dashboard.py                    # Run locally, SSH to della (default, real-time)
    ./gpu_dashboard.py --once             # One-shot mode
    ./gpu_dashboard.py -i 10              # Refresh every 10 seconds
    ./gpu_dashboard.py --host della-gpu   # Use different SSH host
    ./gpu_dashboard.py --local            # Run directly on cluster (no SSH)
    ./gpu_dashboard.py --jobstats         # Use jobstats (long-running averages) instead of nvidia-smi
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Optional

# Default SSH host for della
DEFAULT_HOST = "della"


@dataclass
class GPUStats:
    gpu_id: str
    utilization: float  # percent
    mem_used: int       # bytes
    mem_total: int      # bytes

    @property
    def mem_used_gb(self) -> float:
        return self.mem_used / (1024**3)

    @property
    def mem_total_gb(self) -> float:
        return self.mem_total / (1024**3)

    @property
    def mem_percent(self) -> float:
        return (self.mem_used / self.mem_total * 100) if self.mem_total > 0 else 0


@dataclass
class NodeStats:
    name: str
    cpus: int
    cpu_time: float
    mem_used: int
    mem_total: int
    gpus: list  # list of GPUStats

    @property
    def mem_used_gb(self) -> float:
        return self.mem_used / (1024**3)

    @property
    def mem_total_gb(self) -> float:
        return self.mem_total / (1024**3)

    @property
    def mem_percent(self) -> float:
        return (self.mem_used / self.mem_total * 100) if self.mem_total > 0 else 0


@dataclass
class JobStats:
    job_id: str
    job_name: str
    partition: str
    state: str
    runtime: str
    nodes: list  # list of NodeStats
    error: Optional[str] = None


def run_cmd(cmd: list[str], timeout: int = 30, ssh_host: Optional[str] = None) -> tuple[str, str, int]:
    """Run a command and return stdout, stderr, returncode.

    If ssh_host is provided, run the command on the remote host via SSH.
    """
    try:
        if ssh_host:
            # Wrap command for SSH execution with proper quoting
            cmd_str = " ".join(shlex.quote(c) for c in cmd)
            full_cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", ssh_host, cmd_str]
        else:
            full_cmd = cmd
        result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout)
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "Command timed out", 1
    except Exception as e:
        return "", str(e), 1


# Global SSH host setting (set by main)
_ssh_host: Optional[str] = None
_cluster_user: Optional[str] = None


def get_running_jobs() -> list[dict]:
    """Get list of running jobs for current user (excludes pending jobs)."""
    # Use cluster user if set, otherwise fall back to local USER env
    user = _cluster_user or os.environ.get("USER", "")
    stdout, _, rc = run_cmd([
        "squeue", "-u", user, "-h",
        "-o", "%i|%j|%P|%T|%M|%N"  # Added %N for nodelist
    ], ssh_host=_ssh_host)
    if rc != 0:
        return []

    jobs = []
    for line in stdout.strip().split("\n"):
        if not line:
            continue
        parts = line.split("|")
        if len(parts) >= 6:
            state = parts[3]
            runtime = parts[4]
            nodelist = parts[5]
            # Skip pending jobs and jobs with 0:00 runtime
            if state in ("PENDING", "PD") or runtime == "0:00":
                continue
            jobs.append({
                "job_id": parts[0],
                "job_name": parts[1],
                "partition": parts[2],
                "state": state,
                "runtime": runtime,
                "nodelist": nodelist,
            })
    return jobs


def get_jobstats(job_id: str) -> Optional[dict]:
    """Get jobstats JSON for a job."""
    stdout, _, rc = run_cmd(["jobstats", "--json", job_id], ssh_host=_ssh_host)
    if rc != 0 or not stdout.strip():
        return None
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        return None


def get_nvidia_smi_stats(node: str) -> list[GPUStats]:
    """Get real-time GPU stats via nvidia-smi on a node."""
    # SSH to della first, then to the node
    if _ssh_host:
        # Two-hop: local -> della -> node
        cmd_str = f"ssh {node} 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits'"
        full_cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", _ssh_host, cmd_str]
    else:
        # Single hop: della -> node (when running on cluster)
        full_cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", node,
                    "nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits"]

    try:
        result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=15)
        if result.returncode != 0:
            return []

        gpus = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                gpus.append(GPUStats(
                    gpu_id=parts[0],
                    utilization=float(parts[1]),
                    mem_used=int(parts[2]) * 1024 * 1024,  # MiB to bytes
                    mem_total=int(parts[3]) * 1024 * 1024,
                ))
        return gpus
    except Exception:
        return []


def parse_job_stats(job_info: dict, stats_json: Optional[dict]) -> JobStats:
    """Parse jobstats JSON into JobStats dataclass."""
    if stats_json is None:
        return JobStats(
            job_id=job_info["job_id"],
            job_name=job_info["job_name"],
            partition=job_info["partition"],
            state=job_info["state"],
            runtime=job_info["runtime"],
            nodes=[],
            error="No stats available (job may be starting)"
        )

    nodes = []
    for node_name, node_data in stats_json.get("nodes", {}).items():
        gpus = []
        gpu_utils = node_data.get("gpu_utilization", {})
        gpu_used = node_data.get("gpu_used_memory", {})
        gpu_total = node_data.get("gpu_total_memory", {})

        for gpu_id in gpu_utils.keys():
            gpus.append(GPUStats(
                gpu_id=gpu_id,
                utilization=gpu_utils.get(gpu_id, 0),
                mem_used=gpu_used.get(gpu_id, 0),
                mem_total=gpu_total.get(gpu_id, 0),
            ))

        # Sort GPUs by ID
        gpus.sort(key=lambda g: int(g.gpu_id))

        nodes.append(NodeStats(
            name=node_name,
            cpus=node_data.get("cpus", 0),
            cpu_time=node_data.get("total_time", 0),
            mem_used=node_data.get("used_memory", 0),
            mem_total=node_data.get("total_memory", 0),
            gpus=gpus,
        ))

    return JobStats(
        job_id=job_info["job_id"],
        job_name=job_info["job_name"],
        partition=job_info["partition"],
        state=job_info["state"],
        runtime=job_info["runtime"],
        nodes=nodes,
    )


def make_bar(percent: float, width: int = 20, fill: str = "█", empty: str = "░") -> str:
    """Create an ASCII progress bar."""
    percent = max(0, min(100, percent))
    filled = int(width * percent / 100)
    return fill * filled + empty * (width - filled)


def format_bytes(b: float) -> str:
    """Format bytes as human-readable string."""
    if b >= 1024**3:
        return f"{b / 1024**3:.1f}G"
    elif b >= 1024**2:
        return f"{b / 1024**2:.1f}M"
    else:
        return f"{b / 1024:.1f}K"


def clear_screen():
    """Clear terminal screen."""
    print("\033[2J\033[H", end="")


def render_dashboard(jobs: list[JobStats]):
    """Render the dashboard display."""
    now = time.strftime("%H:%M:%S")
    W = 50  # Total width

    print(f"┌{'─' * W}┐")
    print(f"│ {'GPU DASHBOARD':^{W-2}} │")
    print(f"├{'─' * W}┤")

    if not jobs:
        print(f"│ {'No running jobs':^{W-2}} │")
        print(f"└{'─' * W}┘")
        return

    for job in jobs:
        # Job header - compact with node name
        node_name = job.nodes[0].name if job.nodes else "?"
        header = f"{job.job_id} {job.partition} {node_name} {job.runtime}"
        print(f"│ {header:<{W-2}} │")

        if job.error:
            print(f"│  {job.error[:W-4]:<{W-4}} │")
        elif not job.nodes:
            print(f"│  {'Waiting...':^{W-4}} │")
        else:
            for node in job.nodes:
                # GPUs only - one line each
                for gpu in node.gpus:
                    util_bar = make_bar(gpu.utilization, width=10)
                    gpu_line = f"GPU{gpu.gpu_id} {util_bar} {gpu.utilization:4.0f}% {gpu.mem_used_gb:3.0f}G"
                    print(f"│  {gpu_line:<{W-4}} │")

        print(f"├{'─' * W}┤")

    # Summary
    total_gpus = sum(len(n.gpus) for j in jobs for n in j.nodes)
    all_utils = [g.utilization for j in jobs for n in j.nodes for g in n.gpus]
    avg_util = sum(all_utils) / len(all_utils) if all_utils else 0

    summary = f"{len(jobs)} jobs | {total_gpus} GPUs | {avg_util:.0f}% avg | {now}"
    print(f"│ {summary:<{W-2}} │")
    print(f"└{'─' * W}┘")


def main():
    global _ssh_host, _cluster_user

    parser = argparse.ArgumentParser(description="GPU Dashboard for Della cluster")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("-i", "--interval", type=int, default=5, help="Refresh interval in seconds (default: 5)")
    parser.add_argument("--host", type=str, default=DEFAULT_HOST, help=f"SSH host to connect to (default: {DEFAULT_HOST})")
    parser.add_argument("--local", action="store_true", help="Run locally without SSH (use when already on cluster)")
    parser.add_argument("-u", "--user", type=str, default=None, help="Cluster username (auto-detected if not specified)")
    parser.add_argument("--realtime", action="store_true", default=True, help="Use nvidia-smi for real-time stats (default: True)")
    parser.add_argument("--jobstats", action="store_true", help="Use jobstats instead of nvidia-smi (shows long-running averages)")
    args = parser.parse_args()

    # --jobstats overrides --realtime
    use_realtime = not args.jobstats

    # Set SSH host (None if running locally on cluster)
    _ssh_host = None if args.local else args.host

    if _ssh_host:
        print(f"Connecting to {_ssh_host}...", end="", flush=True)
        # Test SSH connection and get remote username
        stdout, stderr, rc = run_cmd(["whoami"], ssh_host=_ssh_host)
        if rc != 0:
            print(f" FAILED\nCould not connect to {_ssh_host}: {stderr}")
            sys.exit(1)
        _cluster_user = args.user or stdout.strip()
        print(f" OK (user: {_cluster_user})")
    else:
        _cluster_user = args.user or os.environ.get("USER", "")

    try:
        while True:
            # Get running jobs
            job_infos = get_running_jobs()

            # Get stats for each job
            jobs = []
            for job_info in job_infos:
                if job_info["state"] == "RUNNING":
                    if use_realtime and job_info.get("nodelist"):
                        # Use nvidia-smi for real-time stats
                        gpus = get_nvidia_smi_stats(job_info["nodelist"])
                        if gpus:
                            node = NodeStats(
                                name=job_info["nodelist"],
                                cpus=0,
                                cpu_time=0,
                                mem_used=0,
                                mem_total=0,
                                gpus=gpus,
                            )
                            jobs.append(JobStats(
                                job_id=job_info["job_id"],
                                job_name=job_info["job_name"],
                                partition=job_info["partition"],
                                state=job_info["state"],
                                runtime=job_info["runtime"],
                                nodes=[node],
                            ))
                        else:
                            # Fallback to jobstats if nvidia-smi fails
                            stats = get_jobstats(job_info["job_id"])
                            jobs.append(parse_job_stats(job_info, stats))
                    else:
                        # Use jobstats
                        stats = get_jobstats(job_info["job_id"])
                        jobs.append(parse_job_stats(job_info, stats))
                else:
                    jobs.append(parse_job_stats(job_info, None))

            # Render
            clear_screen()
            render_dashboard(jobs)

            if args.once:
                break

            print(f"\nRefreshing in {args.interval}s... (Ctrl+C to exit)")
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)


if __name__ == "__main__":
    main()
