"""`vllm serve` lifecycle: start with /health gating, /metrics scrapes, stop with a GPU-memory wait.

tpprof polls GET /health itself before any client starts, because `vllm bench serve` does
not wait for the server by default (spec 4.4, D4-22, D4-23). While the engine starts, the
API server's port is bound but not listening: Linux refuses the connection, macOS may let
it time out (D5-16), so any connection error or timeout counts as "not ready yet".
"""
from __future__ import annotations

import csv
import http.client
import os
import socket
import threading
import time
import urllib.parse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from tpprof import procs, promparse
from tpprof.engine import EngineConfig

HOST = "127.0.0.1"                 # serve_argv always binds here (C1)
HEALTH_POLL_S = 1.0
HEALTH_REQUEST_TIMEOUT_S = 2.0
PORT_PROBE_TIMEOUT_S = 1.0
LOG_TAIL_LINES = 40
POLLER_FIELDS = ("t_wall", "name", "value")


class ServerStartError(RuntimeError):
    """The server exited, or never answered /health with 200, before the startup timeout."""


@dataclass
class ServerHandle:
    cfg: EngineConfig
    port: int
    proc: procs.Proc
    log_path: str
    base_url: str
    ready_s: float                 # spawn -> first 200 from /health, monotonic seconds


def _get(base_url: str, path: str, timeout_s: float) -> tuple[int, bytes]:
    """GET base_url + path over a direct connection (no proxy environment applies)."""
    u = urllib.parse.urlsplit(base_url)
    conn = http.client.HTTPConnection(u.hostname or HOST, u.port or 80, timeout=timeout_s)
    try:
        conn.request("GET", u.path.rstrip("/") + path)
        resp = conn.getresponse()
        return resp.status, resp.read()
    finally:
        conn.close()


def _healthy(base_url: str) -> bool:
    try:
        status, _ = _get(base_url, "/health", HEALTH_REQUEST_TIMEOUT_S)
    except (OSError, http.client.HTTPException):
        return False
    return status == 200


def _port_accepts_connections(port: int) -> bool:
    try:
        with socket.create_connection((HOST, port), timeout=PORT_PROBE_TIMEOUT_S):
            return True
    except OSError:
        return False


def start_server(cfg: EngineConfig, model_dir: str, port: int, run_dir: str, base_env: Mapping[str, str],
                 run_id: str, startup_timeout_s: float, vllm_bin: str = "vllm") -> ServerHandle:
    """Spawn `vllm serve` for cfg and return once /health answers 200.

    The output goes to run_dir/server-<port>.log. If the process exits or the timeout
    passes first, its process group is stopped and ServerStartError carries the log tail.
    A port that already accepts connections (a leftover server) is refused before
    spawning, since its /health would answer for the wrong server.
    The stop after a failed start sweeps run_id's tag (AM32), so a live server started
    under the same run_id (the other DP2-rand server) is killed with it; the session has
    failed at that point anyway.
    """
    base_url = f"http://{HOST}:{port}"
    log_path = os.path.join(run_dir, f"server-{port}.log")
    what = f"vllm serve {cfg.name}/{cfg.arm} on port {port}"
    if _port_accepts_connections(port):
        raise ServerStartError(f"{what}: port {port} already accepts connections (a leftover server?)")
    proc = procs.spawn(cfg.serve_argv(model_dir, port, vllm_bin), cfg.environment(base_env), log_path, run_id)
    deadline = proc.t_mono_start + startup_timeout_s
    try:
        while True:
            if _healthy(base_url):
                return ServerHandle(cfg=cfg, port=port, proc=proc, log_path=log_path, base_url=base_url,
                                    ready_s=time.monotonic() - proc.t_mono_start)
            code = proc.popen.poll()
            if code is not None:
                reason = f"exited with code {code} before /health returned 200"
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                reason = f"not healthy after {startup_timeout_s:g} s"
                break
            time.sleep(min(HEALTH_POLL_S, remaining))
    except BaseException:          # e.g. KeyboardInterrupt: the server is in its own session
        procs.stop(proc)
        raise
    procs.stop(proc)
    raise ServerStartError(f"{what} {reason}; last {LOG_TAIL_LINES} lines of {log_path}:\n"
                           f"{procs.tail(log_path, LOG_TAIL_LINES)}")


def scrape_metrics(base_url: str, timeout_s: float = 5.0) -> str:
    """The Prometheus text of GET /metrics; raises on a connection error or a non-200 status."""
    status, body = _get(base_url, "/metrics", timeout_s)
    if status != 200:
        raise RuntimeError(f"GET {base_url}/metrics returned HTTP {status}")
    return body.decode("utf-8", "replace")


def stop_servers(handles: Sequence[ServerHandle], nvidia_smi: str = "nvidia-smi",
                 env: Mapping[str, str] | None = None) -> bool:
    """Stop servers that may share a run tag, then sweep the tags and wait for all their GPUs.

    Each process group gets SIGINT, SIGTERM, SIGKILL (spec 7.4). The TPPROF_RUN_ID sweep
    (AM32) runs only after every group is down: sweeping after the first server would
    SIGKILL a sibling that carries the same tag, as the two DP2-rand servers do.
    True once every GPU of every handle uses < GPU_FREE_MIB; False if that does not
    happen within procs.wait_gpu_memory_free's timeout. env is the environment for nvidia-smi.
    If a group survives SIGKILL, the other servers are still stopped, swept and waited
    for before the first such error is raised.
    """
    errors: list[RuntimeError] = []
    for h in handles:
        try:
            procs.stop(h.proc, sweep=False)
        except RuntimeError as e:
            errors.append(e)
    for run_id in dict.fromkeys(h.proc.run_id for h in handles):
        procs.sweep_tagged(run_id)
    gpus = sorted({g for h in handles for g in h.cfg.gpus})
    freed = procs.wait_gpu_memory_free(gpus, nvidia_smi=nvidia_smi, env=env)
    if errors:
        raise errors[0]
    return freed


def stop_server(h: ServerHandle, nvidia_smi: str = "nvidia-smi", env: Mapping[str, str] | None = None) -> bool:
    """stop_servers([h]). With two live servers under one run tag, stop them together instead."""
    return stop_servers([h], nvidia_smi=nvidia_smi, env=env)


class MetricsPoller:
    """Context manager: a thread that scrapes the GAUGES every interval_s into a csv.

    Each round writes one row (t_wall, name, value) per gauge, the value summed over the
    servers in base_urls, as promparse sums a DP server's engines (spec 4.4). A round in
    which a server cannot be reached is skipped and counted in scrape_errors (a crashed
    server is reported by its client's result). A reachable server whose scrape lacks a
    gauge stops the poller, and leaving the context raises with the metric's name.
    """

    def __init__(self, base_urls: Sequence[str], csv_path: str, interval_s: float = 1.0):
        self.base_urls = list(base_urls)
        self.csv_path = csv_path
        self.interval_s = interval_s
        self.rows = 0
        self.scrape_errors = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def __enter__(self) -> MetricsPoller:
        d = os.path.dirname(self.csv_path)
        if d:
            os.makedirs(d, exist_ok=True)
        self._fh = open(self.csv_path, "a", newline="")
        self._writer = csv.writer(self._fh)
        if self._fh.tell() == 0:
            self._writer.writerow(POLLER_FIELDS)
            self._fh.flush()
        self._thread = threading.Thread(target=self._loop, name="tpprof-metrics-poller", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._fh.close()
        if self._error is not None and exc[0] is None:
            raise RuntimeError(f"MetricsPoller for {self.csv_path} failed: {self._error}") from self._error

    def _loop(self) -> None:
        try:
            while True:
                self._round()
                if self._stop.wait(self.interval_s):
                    return
        except BaseException as e:  # surfaced by __exit__
            self._error = e

    def _round(self) -> None:
        t_wall = time.time()
        totals = dict.fromkeys(promparse.GAUGES, 0.0)
        for url in self.base_urls:
            try:
                text = scrape_metrics(url)
            except (OSError, http.client.HTTPException, RuntimeError):
                self.scrape_errors += 1
                return
            samples = promparse.parse_prometheus(text)
            for name in promparse.GAUGES:
                totals[name] += promparse.metric_sum(samples, name)
        for name, value in totals.items():
            self._writer.writerow((f"{t_wall:.3f}", name, repr(value)))
            self.rows += 1
        self._fh.flush()
