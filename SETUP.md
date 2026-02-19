# Full Setup Guide

This guide explains how to set up Timeloop and run the RL mapping evaluation pipeline.

---

# Step 1: Clone Timeloop

```
git clone https://github.com/NVlabs/timeloop.git
cd timeloop
```

---

# Step 2: Clone pat-public

```
git clone https://github.com/NVlabs/pat-public.git
```

Link it:

```
ln -s pat-public/src/pat src/pat
```

---

# Step 3: Build Timeloop

Install dependencies:

```
sudo apt install build-essential scons libconfig++-dev libboost-all-dev
```

Build:

```
scons -j2
```

Binary will appear:

```
build/timeloop-model
```

---

# Step 4: Clone Timeloop Exercises Repo

```
git clone https://github.com/NVlabs/timeloop-accelergy-exercises.git
```

---

# Step 5: Place Computationally-Efficient-DNN-Mapping

Move the folder into:

```
timeloop-accelergy-exercises/workspace/
```

Result:

```
timeloop-accelergy-exercises/workspace/Computationally-Efficient-DNN-Mapping
```

---

# Step 6: Python Environment

Recommended:

Python 3.10 or 3.11

Install dependencies:

```
pip install torch numpy pyyaml
```

---

# Step 7: Verify Timeloop works

```
build/timeloop-model arch.yaml prob.yaml map.yaml
```

Should produce:

```
timeloop-model.stats.txt
```

---

# Step 8: Verify reevaluation script

```
python model_for_RL_reevaluation.py 12
```

---

# Expected output

```
Mappings to evaluate: 1304
Best EDP: XXXXX
```

---
