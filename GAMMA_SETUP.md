# GAMMA + Timeloop Setup Guide

This document provides a **fully reproducible setup guide** for installing and configuring:

• GAMMA (RL-based DNN mapping framework)
• pytimeloop (Python bindings for Timeloop)
• Timeloop simulator
• PAT (Power-Area-Timing) model
• Full integration required for GAMMA-based mapping generation and evaluation

This guide incorporates all required fixes and workarounds discovered during installation.

---

# Tested Environment

This setup has been tested on:

| Component | Version                                    |
| --------- | ------------------------------------------ |
| OS        | Ubuntu 22.04 (WSL also works)              |
| Python    | 3.8                                        |
| GCC       | 11.4                                       |
| CMake     | 3.24.x                                     |
| Ninja     | 1.13.x                                     |
| Conda     | Miniconda                                  |
| Timeloop  | NVLabs Timeloop (submodule via pytimeloop) |
| GAMMA     | gamma-timeloop repo                        |

---

# High-Level Architecture

Pipeline:

GAMMA (RL agent)
↓
pytimeloop (Python bindings)
↓
Timeloop (C++ simulator)
↓
PAT model (power/area estimation)
↓
Stats output (cycles, energy, EDP)
↓
Reward signal returned to GAMMA

---

# Step 1: Create clean conda environment

```
conda create -n gamma python=3.8 -y
conda activate gamma
```

Install dependencies:

```
pip install numpy pyyaml cmake ninja
```

Verify:

```
cmake --version
ninja --version
```

---

# Step 2: Clone GAMMA-Timeloop

Choose a clean directory:

```
mkdir ~/gamma_clean
cd ~/gamma_clean
```

Clone:

```
git clone https://github.com/<gamma-repo-url>/gamma-timeloop.git
cd gamma-timeloop
```

---

# Step 3: Build pytimeloop

Run:

```
python build_pytimeloop.py
```

This will clone:

```
gamma_clean/
├── gamma-timeloop/
├── pytimeloop/
```

---

# Step 4: Build Timeloop (CRITICAL STEP)

Navigate:

```
cd ~/gamma_clean/pytimeloop/lib/timeloop
```

---

## Step 4.1: Link PAT model

Timeloop requires the PAT power model.

Create symbolic link:

```
ln -s pat-public/src/pat src/pat
```

Verify:

```
ls src/pat
```

Should show:

```
pat.cpp
pat.hpp
```

---

## Step 4.2: Fix missing include error

Open file:

```
nano src/loop-analysis/nest-analysis-tile-info.cpp
```

Add at top:

```
#include <numeric>
```

This fixes:

```
error: ‘accumulate’ is not a member of ‘std’
```

---

## Step 4.3: Fix libconfig range loop error

Open file:

```
nano src/compound-config/compound-config.cpp
```

Find line:

```
for (const std::string& m: LNode->lookup(name))
```

Change to:

```
for (const std::string m: LNode->lookup(name))
```

This fixes:

```
error: loop variable binds to temporary
```

---

## Step 4.4: Build Timeloop

Run:

```
scons -j4
```

Expected output:

```
scons: done building targets.
```

This produces:

```
build/libtimeloop-model.so
```

Verify:

```
ls build/
```

---

# Step 5: Install pytimeloop

Navigate:

```
cd ~/gamma_clean/pytimeloop
```

Set required environment variables:

```
export CMAKE_POLICY_VERSION_MINIMUM=3.5
export CMAKE_GENERATOR="Unix Makefiles"
```

Install:

```
pip install . --no-build-isolation
```

Expected output:

```
Successfully installed pytimeloop
```

---

# Step 6: Verify Installation

Run Python:

```
python
```

Test import:

```
import pytimeloop
print("SUCCESS")
```

Expected output:

```
SUCCESS
```

---

# Step 7: Verify Timeloop Library Detection

Run:

```
ls ~/gamma_clean/pytimeloop/lib/timeloop/build/
```

Should contain:

```
libtimeloop-model.so
```

---

# Common Errors and Fixes

---

## Error: accumulate not a member of std

Fix:

Add:

```
#include <numeric>
```

to:

```
src/loop-analysis/nest-analysis-tile-info.cpp
```

---

## Error: Could not find LIBTIMELOOP

Cause:

Timeloop not built.

Fix:

```
cd pytimeloop/lib/timeloop
scons -j4
```

---

## Error: ninja not found during pip install

Fix:

```
pip install ninja
pip install . --no-build-isolation
```

---

## Error: PAT model missing

Fix:

```
ln -s pat-public/src/pat src/pat
```

---

## Error: undefined symbol ParseSpecs

Cause:

Wrong Timeloop version or incomplete build

Fix:

Rebuild Timeloop fully:

```
scons -c
scons -j4
```

---

# Final Directory Structure

```
gamma_clean/
│
├── gamma-timeloop/
│
├── pytimeloop/
│   ├── pytimeloop/
│   ├── lib/
│   │   ├── timeloop/
│   │   │   ├── build/
│   │   │   ├── src/
│   │   │   ├── include/
│
└── (conda environment)
```

---

# Final Verification Checklist

All must pass:

```
✓ conda activate gamma
✓ import pytimeloop works
✓ libtimeloop-model.so exists
✓ scons build successful
✓ no missing symbols
✓ no cmake errors
```

---

# Result

You now have a fully working GAMMA + Timeloop environment.

This enables:

• GAMMA-based RL mapping generation
• Timeloop-based mapping evaluation
• Energy, latency, and EDP reward computation

---

# Next Steps

You can now run:

• GAMMA training
• Mapping search
• Reward evaluation
• Integration with RL pipelines
