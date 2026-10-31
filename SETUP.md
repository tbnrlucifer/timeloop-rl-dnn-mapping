# Setup

This repository combines PPO-based DNN mapping, GAMMA, and Timeloop. GAMMA and PyTimeloop are included as Git submodules, with their integration commits pinned by this repository.

## 1. Clone the repository

Clone the repository with all its submodules:

```bash
git clone --recurse-submodules https://github.com/tbnrlucifer/timeloop-rl-dnn-mapping.git
cd timeloop-rl-dnn-mapping
```

If you have already cloned the repository without its submodules, initialize them with:

```bash
git submodule update --init --recursive
```

## 2. Install system dependencies

On Ubuntu, install the required build dependencies:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  scons \
  cmake \
  libconfig++-dev \
  libboost-dev \
  libboost-iostreams-dev \
  libboost-serialization-dev \
  libyaml-cpp-dev \
  libncurses-dev \
  libtinfo-dev \
  libgpm-dev
```

The project uses Python 3.10. Create and activate a Python environment. For example, with Conda:

```bash
conda create -n timeloop python=3.10
conda activate timeloop
```

## 3. Build Timeloop

Timeloop is included as a nested submodule inside PyTimeloop.

First, create the PAT source symlink:

```bash
cd pytimeloop/lib/timeloop
ln -sfn ../pat-public/src/pat src/pat
```

Build Timeloop with Accelergy support:

```bash
scons -j4 --accelergy
```

Return to the repository root:

```bash
cd ../../..
```

## 4. Install PyTimeloop

From the repository root:

```bash
cd pytimeloop

export CMAKE_GENERATOR="Unix Makefiles"
export CMAKE_POLICY_VERSION_MINIMUM=3.5

pip install -e .
```

Return to the repository root:

```bash
cd ..
```

The CMake environment variables above were required for the tested build environment. They may not be necessary with every CMake version.

## 5. Verify the installation

Check that the Timeloop command-line application is available:

```bash
timeloop-model.py --help
```

## Notes

- Run `git submodule update --init --recursive` after cloning or switching to a revision that changes submodule references.
- The PAT symlink should resolve to `pytimeloop/lib/timeloop/pat-public/src/pat` within the repository checkout.
- Building Timeloop and PyTimeloop can take several minutes and requires sufficient memory and disk space.