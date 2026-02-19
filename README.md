# Timeloop RL-Based DNN Mapping Evaluation

This repository contains the setup, modifications, and documentation for reproducing the RL-based DNN mapping evaluation workflow described in:

**Computationally Efficient DNN Mapping using Reinforcement Learning**

This includes:

* Timeloop setup
* RL candidate mapping generation
* Timeloop-based reevaluation
* Debugged Python reevaluation pipeline
* Fixes for modern Timeloop versions

---

# Overview

The workflow consists of two stages:

### Stage 1: RL Mapping Generation

The RL script generates candidate mappings and stores them in a pickle file.

```
modified_script.py → output_mappings.pkl
```

---

### Stage 2: Timeloop Reevaluation

Each candidate mapping is evaluated using Timeloop to obtain:

* Energy
* Cycles
* EDP

```
model_for_RL_reevaluation.py → best_map.yaml
```

---

# Key Contributions / Fixes

This repo fixes several compatibility issues:

• Python bindings removed in newer Timeloop versions
• Replaced Python API calls with CLI subprocess execution
• Fixed LD_LIBRARY_PATH linking issues
• Fixed mapping YAML formatting issues
• Implemented full reevaluation pipeline
• Verified consistency between CLI and Python execution

---

# Current Status

Working:

* Timeloop setup
* Mapping reevaluation
* Best mapping extraction
* Full stats generation

Partial / In Progress:

* PPO training implementation
* RL convergence validation

See RL_STATUS.md for details.

---

# Repository Purpose

This repository documents:

* Complete reproducible setup
* Debugging steps
* Fixed reevaluation implementation
* RL pipeline status

---

# Full setup instructions

See:

SETUP.md

---

# Reevaluation details

See:

REEVALUATION.md

---

# RL implementation status

See:

RL_STATUS.md

---

## Reference Output

The `reference_output/` folder contains Timeloop evaluation results for the best mapping found.

These files were generated using:

timeloop-model arch.yaml prob.yaml best_map.yaml

They include:

• timeloop-model.stats.txt – energy, cycles, EDP
• timeloop-model.map.txt – loop mapping
• timeloop-model.map+stats.html – visualization
• best_map.yaml – mapping evaluated

These serve as validation reference.
