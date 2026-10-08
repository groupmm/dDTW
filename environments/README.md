# GPU setup on Linux

Use Linux x86_64, Conda, and an NVIDIA GPU with a working driver that reports
CUDA support of 12.8 or newer in `nvidia-smi`. The driver is the only system
prerequisite; the CUDA toolkit and compilers are installed inside the environment.

Create and activate a dedicated environment once:

```bash
conda create -n ddtw python=3.12 pip
conda activate ddtw
```

Install and configure the package:

```bash
python -m pip install ddtw
ddtw setup-cuda
```

There is no need to clone the repository. Setup installs the dependencies,
compiles the extension and tests a GPU loss and its gradient. Wait for
**GPU check passed** and **Setup complete**. The first run can take several
minutes and needs internet access and several GB of free disk space.
In future sessions, activate the same environment with `conda activate ddtw`.

## What gets installed

The command selects one tested configuration: CUDA 12.8, GCC/G++ 13,
PyTorch 2.11.0+cu128, Ninja and NumPy. It retains your Python minor version and replaces
other versions of those dependencies in the active environment. Use a dedicated
environment to avoid conflicts with other applications, including torchvision
or torchaudio releases that require a different PyTorch version.

The installer accepts standard CPython 3.10–3.13. Python 3.12 is the validated
default; accepting another version does not mean it has received the same GPU
testing. Free-threaded Python builds are not supported by this setup command.

Conda installs the CUDA toolkit and compiler packages using strict channel
priority and explicit versions. Pip installs PyTorch from its CUDA 12.8 index.
The command does not change your global Conda configuration, system packages or
NVIDIA driver. It requires a dedicated Conda environment and refuses to modify
Conda base. For CPU use, install dDTW in the Conda environment and skip GPU setup.

The driver version is a compatibility check, not the version to install.
For example, a driver reporting CUDA 13.x can run this CUDA 12.8 setup.
`torch.version.cuda` and the environment's `nvcc --version` must both report 12.8.

## Inspect or check the setup

```bash
ddtw setup-cuda --dry-run
ddtw check-cuda
```

The dry run prints the commands without installing anything; it does not resolve
dependencies or test GPU access. Use it to inspect the exact dependency versions
and installation commands; separate setup scripts or environment YAML files
are not needed. Run setup without `--dry-run` to install the dependencies, save
the build configuration and perform the GPU check.

The check compiles/loads the extension and runs
forward/backward on the first visible GPU. It does not install dependencies.
Use `CUDA_VISIBLE_DEVICES` to select a different GPU if needed.
`python -m ddtw setup-cuda` and `python -m ddtw check-cuda` are equivalent commands
that explicitly use the current Python interpreter.

Build settings live in `$CONDA_PREFIX/etc/ddtw/`. Compiled extensions live in
`$CONDA_PREFIX/var/cache/ddtw/`, separated by Python and PyTorch version.
Package managers may also use their normal download caches. No manual activation
hooks are needed: dDTW selects its configured toolkit and compilers during the
build and restores the process's original compiler settings afterward.

The setup command can be rerun after an interrupted installation or a PyTorch
upgrade. It reapplies the pinned dependencies and repeats the GPU check. It does
not roll back dependency installations if a later step fails; errors return a
nonzero exit status and never print a success message.

## Troubleshooting

- **`ddtw` command not found:** activate the environment and try
  `python -m ddtw --help`. Older dDTW releases do not include this command;
  install a release containing the CLI or install this checkout as described below.
- **Conda environment required:** use the Conda commands above and run setup
  with that environment's Python interpreter. Python and compiler installations
  must use the same environment.
- **GPU or driver unavailable:** check `nvidia-smi`. Driver installation is
  managed outside this tool, usually by the system administrator.
- **Conda reports a dependency conflict:** start with a fresh environment using
  Python 3.12. CUDA and GCC versions must stay pinned together.
- **Build or GPU check fails:** the compiler/runtime error is printed. Run
  `ddtw check-cuda` to repeat it. To limit compiler memory use, run
  `MAX_JOBS=1 ddtw check-cuda` (managed builds default to two workers).
- **PyTorch was changed after setup:** rerun `ddtw setup-cuda` to restore the
  matching CUDA 12.8 build.

## Development

To install this checkout, use an activated Conda environment and run:

```bash
python -m pip install -e ".[test,benchmark]"
ddtw setup-cuda
python -m pytest test
```

For validation of the published package, run tests from a directory containing
`test/` but no local `ddtw/` package, so the checkout cannot shadow the installation.

The `ddtw[cuda]` extra adds Ninja only; pip installation itself never runs GPU
setup. Use the packaged `ddtw setup-cuda` command for both published installations
and source checkouts.
