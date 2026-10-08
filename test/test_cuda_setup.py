"""Setup boundaries and failures must not modify the wrong environment."""

from pathlib import Path
import subprocess
import sys

import pytest

from ddtw import _cuda_config, cli


@pytest.fixture
def conda_env(tmp_path, monkeypatch):
    prefix = tmp_path / "environment with spaces"
    (prefix / "conda-meta").mkdir(parents=True)
    (prefix / "bin").mkdir()
    monkeypatch.setattr(sys, "prefix", str(prefix))
    monkeypatch.setattr(cli.platform, "system", lambda: "Linux")
    monkeypatch.setattr(cli.platform, "machine", lambda: "x86_64")
    monkeypatch.setenv("CONDA_PREFIX", str(prefix))
    monkeypatch.setenv("CONDA_EXE", "/opt/conda/bin/conda")
    monkeypatch.setattr(cli.shutil, "which", lambda name: name)
    monkeypatch.setattr(cli.subprocess, "check_output", lambda *a, **kw: "/opt/conda\n")
    return prefix


def test_help_does_not_import_torch():
    result = subprocess.run(
        [sys.executable, "-c", "import sys; from ddtw.cli import main; "
         "assert 'torch' not in sys.modules; main(['--help'])"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "setup-cuda" in result.stdout


@pytest.mark.parametrize("case", ["base", "venv", "nested", "windows", "arm"])
def test_reject_wrong_environment_before_install(conda_env, monkeypatch, case):
    if case == "base":
        monkeypatch.setattr(cli.subprocess, "check_output", lambda *a, **kw: str(conda_env))
    elif case == "venv":
        monkeypatch.delenv("CONDA_PREFIX")
    elif case == "nested":
        monkeypatch.setattr(sys, "prefix", str(conda_env / "venv"))
    elif case == "windows":
        monkeypatch.setattr(cli.platform, "system", lambda: "Windows")
    else:
        monkeypatch.setattr(cli.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(cli, "_run", lambda *a, **kw: pytest.fail("must not install"))
    assert cli.main(["setup-cuda"]) == 1
    assert not (conda_env / "etc").exists()


def test_dry_run_is_read_only(conda_env, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_run", lambda *a, **kw: pytest.fail("must not install"))
    monkeypatch.setattr(cli, "_check_driver", lambda: pytest.fail("dry run does not probe GPU"))
    assert cli.main(["setup-cuda", "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "--prefix" in output and str(conda_env) in output
    assert "cuda=12.8" in output and "gcc_linux-64=13" in output
    assert "torch==2.11.0+cu128" in output
    assert not (conda_env / "etc").exists()


@pytest.mark.parametrize("failed_step", [0, 1, 2, 3])
def test_failure_stops_setup(conda_env, monkeypatch, capsys, failed_step):
    monkeypatch.setattr(cli, "_check_driver", lambda: None)
    for name in ("nvcc", "x86_64-conda-linux-gnu-cc", "x86_64-conda-linux-gnu-c++"):
        executable = conda_env / "bin" / name
        executable.touch()
        executable.chmod(0o755)
    commands = []

    def fail(command):
        commands.append(command)
        if len(commands) == failed_step + 1:
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(cli, "_run", fail)
    assert cli.main(["setup-cuda"]) == 1
    assert len(commands) == failed_step + 1
    assert "Setup complete" not in capsys.readouterr().out
    assert _cuda_config.config_path().exists() == (failed_step == 3)


def test_setup_targets_same_environment_and_checks_in_fresh_process(conda_env, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_check_driver", lambda: None)
    for name in ("nvcc", "x86_64-conda-linux-gnu-cc", "x86_64-conda-linux-gnu-c++"):
        executable = conda_env / "bin" / name
        executable.touch()
        executable.chmod(0o755)
    commands = []
    monkeypatch.setattr(cli, "_run", commands.append)
    assert cli.main(["setup-cuda"]) == 0
    assert commands[0][commands[0].index("--prefix") + 1] == str(conda_env)
    assert all(command[0] == sys.executable for command in commands[1:])
    for command in commands[1:3]:
        assert command[command.index("--prefix") + 1] == str(conda_env)
        assert "--no-user" in command
    assert commands[-1][-1] == "check-cuda" and "-I" in commands[-1]
    assert _cuda_config.config_path().read_text().strip() == "12.8"
    assert "Setup complete" in capsys.readouterr().out


@pytest.mark.parametrize("driver_cuda", ["11.8", "12.7", "N/A"])
def test_driver_rejected_before_install(monkeypatch, driver_cuda):
    monkeypatch.setattr(cli.shutil, "which", lambda name: name)
    monkeypatch.setattr(cli.subprocess, "check_output", lambda *a, **kw: f"CUDA Version: {driver_cuda}")
    with pytest.raises(RuntimeError, match="driver"):
        cli._check_driver()


@pytest.mark.parametrize("driver_cuda", ["12.8", "13.2"])
def test_driver_can_be_newer_than_toolkit(monkeypatch, driver_cuda):
    monkeypatch.setattr(cli.shutil, "which", lambda name: name)
    monkeypatch.setattr(cli.subprocess, "check_output", lambda cmd, **kw:
                        f"CUDA Version: {driver_cuda}" if len(cmd) == 1 else "GPU, 595.91.07\n")
    cli._check_driver()


def test_build_flags_restored_even_on_failure(conda_env, monkeypatch):
    import os

    config = _cuda_config.config_path()
    config.parent.mkdir(parents=True)
    config.write_text("12.8\n")
    for name in ("x86_64-conda-linux-gnu-cc", "x86_64-conda-linux-gnu-c++"):
        executable = conda_env / "bin" / name
        executable.touch()
        executable.chmod(0o755)
    monkeypatch.setenv("CUDA_HOME", "/unrelated/cuda")
    monkeypatch.setenv("CXX", "/unrelated/compiler")
    monkeypatch.setenv("NVCC_PREPEND_FLAGS", "-ccbin=/wrong/compiler")
    before = dict(os.environ)
    with pytest.raises(RuntimeError, match="build failed"):
        with _cuda_config.build_environment():
            assert os.environ["CUDA_HOME"] == str(conda_env)
            assert os.environ["CXX"].startswith(str(conda_env))
            assert "NVCC_PREPEND_FLAGS" not in os.environ
            raise RuntimeError("build failed")
    assert dict(os.environ) == before
