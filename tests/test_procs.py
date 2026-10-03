from __future__ import annotations

import os
import pathlib
import resource
import sys
import textwrap
import time

import psutil
import pytest

from tpprof import procs

PY = sys.executable


def _gone(pid: int) -> bool:
    try:
        return psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


def _wait_for_file(path: pathlib.Path, timeout_s: float = 10.0) -> str:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists() and path.read_text().strip():
            return path.read_text().strip()
        time.sleep(0.02)
    raise AssertionError(f"{path} was not written within {timeout_s} s")


def _fake_nvidia_smi(tmp_path: pathlib.Path, memory_lines: str, apps_lines: str = "") -> str:
    """A tiny nvidia-smi that answers the two C8 memory queries and rejects anything else."""
    script = tmp_path / "nvidia-smi"
    script.write_text(textwrap.dedent(f"""\
        #!{PY}
        import sys
        args = sys.argv[1:]
        if args == ["--query-gpu=index,memory.used", "--format=csv,noheader,nounits"]:
            sys.stdout.write({memory_lines!r})
        elif args == ["--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]:
            sys.stdout.write({apps_lines!r})
        else:
            sys.stderr.write("unexpected args: %r\\n" % (args,))
            sys.exit(2)
        """))
    script.chmod(0o755)
    return str(script)


def test_spawn_tags_env_and_appends_output(tmp_path):
    log = tmp_path / "child.log"
    log.write_text("previous line\n")
    code = "import os, sys; print(os.environ['TPPROF_RUN_ID'], flush=True); print('to-stderr', file=sys.stderr)"
    p = procs.spawn([PY, "-c", code], dict(os.environ), str(log), "run-abc")
    assert procs.wait(p, 10) == 0
    assert log.read_text().splitlines() == ["previous line", "run-abc", "to-stderr"]
    assert p.run_id == "run-abc" and p.log_path == str(log)
    assert p.t_wall_start > 0 and p.t_mono_start > 0


def test_spawn_starts_a_new_session(tmp_path):
    out = tmp_path / "sid"
    code = f"import os; open({str(out)!r}, 'w').write('%d %d' % (os.getpid(), os.getsid(0)))"
    p = procs.spawn([PY, "-c", code], dict(os.environ), str(tmp_path / "log"), "r")
    assert procs.wait(p, 10) == 0
    pid, sid = out.read_text().split()
    assert pid == sid


def test_wait_returns_none_on_timeout(tmp_path):
    p = procs.spawn([PY, "-c", "import time; time.sleep(60)"], dict(os.environ), str(tmp_path / "log"), "r")
    try:
        t0 = time.monotonic()
        assert procs.wait(p, 0.2) is None
        assert time.monotonic() - t0 < 2
    finally:
        procs.stop(p, 1, 1)


def test_run_timeout_kills_group(tmp_path):
    pidfile = tmp_path / "pid"
    code = f"import os, time; open({str(pidfile)!r}, 'w').write(str(os.getpid())); time.sleep(60)"
    out = procs.run([PY, "-c", code], dict(os.environ), str(tmp_path / "log"), "run-timeout", timeout_s=0.5)
    assert out.timed_out is True
    assert out.exit_code is None
    assert 0.5 <= out.duration_s < 10
    assert out.t_wall_end > 0 and out.t_mono_end > 0
    assert _gone(int(pidfile.read_text()))


def test_run_normal_exit(tmp_path):
    out = procs.run([PY, "-c", "import sys; sys.exit(3)"], dict(os.environ), str(tmp_path / "log"), "r",
                    timeout_s=10)
    assert out.timed_out is False
    assert out.exit_code == 3
    assert out.duration_s < 10


def test_stop_uses_sigint_first(tmp_path):
    # A Python child dies of KeyboardInterrupt on SIGINT; no escalation is needed.
    p = procs.spawn([PY, "-c", "import time; print('up', flush=True); time.sleep(60)"], dict(os.environ),
                    str(tmp_path / "log"), "r")
    _wait_for_file(tmp_path / "log")
    t0 = time.monotonic()
    code = procs.stop(p, 5, 5)
    assert time.monotonic() - t0 < 3
    assert code != -9
    assert "KeyboardInterrupt" in (tmp_path / "log").read_text()


def test_stop_escalates_to_sigkill_for_sigint_ignoring_child(tmp_path):
    code = ("import signal, time; signal.signal(signal.SIGINT, signal.SIG_IGN); "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); print('up', flush=True); time.sleep(60)")
    p = procs.spawn([PY, "-c", code], dict(os.environ), str(tmp_path / "log"), "run-ignore")
    _wait_for_file(tmp_path / "log")
    t0 = time.monotonic()
    rc = procs.stop(p, grace_int_s=0.3, grace_term_s=0.3)
    assert time.monotonic() - t0 < 2
    assert rc == -9
    assert _gone(p.popen.pid)
    # idempotent
    assert procs.stop(p, 0.3, 0.3) == -9


def test_stop_kills_other_group_members(tmp_path):
    # The leader exits on SIGINT but a group member ignoring SIGINT/SIGTERM is still killed.
    gpid = tmp_path / "gpid"
    child = ("import signal, time; signal.signal(signal.SIGINT, signal.SIG_IGN); "
             "signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)")
    leader = textwrap.dedent(f"""\
        import subprocess, sys, time
        c = subprocess.Popen([sys.executable, "-c", {child!r}])
        open({str(gpid)!r}, "w").write(str(c.pid))
        time.sleep(60)
        """)
    p = procs.spawn([PY, "-c", leader], dict(os.environ), str(tmp_path / "log"), "run-group")
    member = int(_wait_for_file(gpid))
    time.sleep(0.2)
    procs.stop(p, 0.3, 0.3)
    deadline = time.monotonic() + 2
    while not _gone(member) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert _gone(member)


def test_sweep_tagged_kills_escaped_grandchild(tmp_path):
    gpid = tmp_path / "gpid"
    grandchild = f"import os, time; open({str(gpid)!r}, 'w').write(str(os.getpid())); time.sleep(60)"
    child = textwrap.dedent(f"""\
        import os, subprocess, sys, time
        subprocess.Popen([sys.executable, "-c", {grandchild!r}], preexec_fn=os.setsid)
        time.sleep(60)
        """)
    p = procs.spawn([PY, "-c", child], dict(os.environ), str(tmp_path / "log"), "run-escape")
    g = int(_wait_for_file(gpid))
    assert os.getsid(g) == g  # escaped the child's session and group
    procs.stop(p, 0.3, 0.3)
    deadline = time.monotonic() + 2
    while not _gone(g) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert _gone(g)


def test_sweep_tagged_matches_exact_run_id_only(tmp_path):
    env = dict(os.environ)
    p = procs.spawn([PY, "-c", "import time; time.sleep(60)"], env, str(tmp_path / "log"), "run-x")
    try:
        time.sleep(0.3)
        assert procs.sweep_tagged("run-x-other") == []
        assert procs.sweep_tagged("run") == []
        assert p.popen.poll() is None
        assert procs.sweep_tagged("run-x") == [p.popen.pid]
        assert p.popen.wait(5) == -9
    finally:
        procs.stop(p, 0.3, 0.3)


def test_raise_nofile(tmp_path):
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    code = "import resource; print(resource.getrlimit(resource.RLIMIT_NOFILE)[0])"
    resource.setrlimit(resource.RLIMIT_NOFILE, (256, hard))
    try:
        p = procs.spawn([PY, "-c", code], dict(os.environ), str(tmp_path / "raised.log"), "r", raise_nofile=True)
        assert procs.wait(p, 10) == 0
        q = procs.spawn([PY, "-c", code], dict(os.environ), str(tmp_path / "plain.log"), "r")
        assert procs.wait(q, 10) == 0
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
    assert int((tmp_path / "raised.log").read_text()) >= min(hard, 4096)
    assert int((tmp_path / "plain.log").read_text()) == 256


def test_raise_nofile_never_lowers(tmp_path):
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    # a soft limit above the target, but not above the hard limit (GitHub runners: hard = 65536)
    above = 70000 if hard == resource.RLIM_INFINITY else min(70000, hard)
    if above <= procs.NOFILE_TARGET:
        pytest.skip(f"hard limit {hard} leaves no soft limit above {procs.NOFILE_TARGET} to test")
    code = "import resource; print(resource.getrlimit(resource.RLIMIT_NOFILE)[0])"
    resource.setrlimit(resource.RLIMIT_NOFILE, (above, hard))
    try:
        p = procs.spawn([PY, "-c", code], dict(os.environ), str(tmp_path / "log"), "r", raise_nofile=True)
        assert procs.wait(p, 10) == 0
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
    assert int((tmp_path / "log").read_text()) == above


def test_gpu_memory_used_and_processes(tmp_path):
    smi = _fake_nvidia_smi(tmp_path, "0, 30000\n1, 12\n", "4242, 29990\n4343, [N/A]\n")
    assert procs.gpu_memory_used(smi) == {0: 30000, 1: 12}
    assert procs.gpu_processes(smi) == [(4242, 29990), (4343, -1)]


def test_gpu_processes_empty(tmp_path):
    smi = _fake_nvidia_smi(tmp_path, "0, 0\n", "")
    assert procs.gpu_processes(smi) == []


def test_gpu_query_failure_raises_with_stderr(tmp_path):
    script = tmp_path / "nvidia-smi"
    script.write_text(f"#!{PY}\nimport sys\nsys.stderr.write('NVIDIA-SMI has failed\\n')\nsys.exit(9)\n")
    script.chmod(0o755)
    with pytest.raises(RuntimeError, match="NVIDIA-SMI has failed"):
        procs.gpu_memory_used(str(script))


def test_gpu_memory_used_uses_path_from_env(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _fake_nvidia_smi(bindir, "0, 7\n")
    env = dict(os.environ, PATH=os.pathsep.join([str(bindir), os.environ.get("PATH", "")]))
    assert procs.gpu_memory_used(env=env) == {0: 7}


def test_wait_gpu_memory_free_times_out(tmp_path):
    smi = _fake_nvidia_smi(tmp_path, "0, 30000\n1, 30000\n")
    t0 = time.monotonic()
    assert procs.wait_gpu_memory_free([0, 1], timeout_s=1.0, nvidia_smi=smi) is False
    assert 1.0 <= time.monotonic() - t0 < 3.0


def test_wait_gpu_memory_free_only_checks_listed_gpus(tmp_path):
    smi = _fake_nvidia_smi(tmp_path, "0, 30000\n1, 5\n")
    t0 = time.monotonic()
    assert procs.wait_gpu_memory_free([1], timeout_s=5, nvidia_smi=smi) is True
    assert time.monotonic() - t0 < 2
    # a GPU missing from the output never counts as free
    assert procs.wait_gpu_memory_free([2], timeout_s=0.3, nvidia_smi=smi) is False


def test_tail(tmp_path):
    f = tmp_path / "x.log"
    f.write_text("".join(f"line {i}\n" for i in range(1000)))
    assert procs.tail(str(f), 3) == "line 997\nline 998\nline 999"
    assert procs.tail(str(f)).splitlines() == [f"line {i}" for i in range(960, 1000)]
    short = tmp_path / "s.log"
    short.write_text("a\nb")
    assert procs.tail(str(short), 40) == "a\nb"
    assert procs.tail(str(tmp_path / "missing.log")) == ""
