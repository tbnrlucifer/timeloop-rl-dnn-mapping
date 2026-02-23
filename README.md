# Timeloop GAMMA Integration

This branch contains the complete setup and integration of the GAMMA mapping generator with Timeloop and pytimeloop.

GAMMA is used as the mapping generator, and Timeloop is used as the evaluation engine.

---

# Overview

The mapping evaluation pipeline:

```
GAMMA → pytimeloop → Timeloop → stats → reward metrics
```

GAMMA generates mappings and Timeloop evaluates them to produce:

• latency (cycles)
• energy
• energy-delay product (EDP)
• memory accesses
• utilization

These metrics can be used for optimization and analysis.

---

# What This Branch Contains

This branch includes:

• Fully working GAMMA setup
• pytimeloop build and integration
• Timeloop compilation and fixes
• Environment configuration
• Complete setup guide

See:

```
GAMMA_SETUP.md
```

for full installation instructions.

---

# Environment Requirements

Recommended configuration:

```
OS: Ubuntu 20.04 (native or WSL)
Python: 3.8
Conda: Miniconda or Anaconda
CMake: 3.10 – 3.24
GCC: 9 – 11
```

---

# Setup

Follow the complete guide:

```
GAMMA_SETUP.md
```

This includes:

• Conda environment creation
• pytimeloop installation
• Timeloop build
• Required fixes
• Verification steps

---

# Verification

Test pytimeloop import:

```
python -c "import pytimeloop"
```

If no error appears, setup is successful.

---

# Relationship with main branch

Repository structure:

• main branch → RL-based mapping generator
• gamma branch → GAMMA-based mapping generator

Both use Timeloop as the evaluation engine.

---
