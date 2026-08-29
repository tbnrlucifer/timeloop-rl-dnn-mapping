import random
import pickle
import yaml
import time

# Import your existing functions
from modified_script import (
    prime_factors_with_index,
    getmapping,
    format_mapping_dict,
    reward_func
)

# ----------------------------------------
# Base problem dimensions (Layer 12 example)
# Change if needed
# ----------------------------------------

BASE_PROBLEM = [3, 3, 3, 120, 120, 120, 16, 32]


# ----------------------------------------
# Generate valid shuffled state
# ----------------------------------------

def generate_random_state():

    state = prime_factors_with_index(BASE_PROBLEM)

    # Shuffle order only — factors remain valid
    random.shuffle(state)

    return state


# ----------------------------------------
# Generate valid mappings list
# ----------------------------------------

def generate_mapping_list(num_mappings=500):

    mapping_list = []

    print(f"Generating {num_mappings} valid mappings...")

    for i in range(num_mappings):

        state = generate_random_state()

        memmap = getmapping(state)

        mapping_yaml_dict = format_mapping_dict(memmap)

        reward = reward_func(memmap)

        entry = [
            reward,
            mapping_yaml_dict,
            state,
            state,
            0,
            time.time(),
            {},
            0,
            0
        ]

        mapping_list.append(entry)

        if i % 50 == 0:
            print(f"Generated {i}/{num_mappings}")

    return mapping_list


# ----------------------------------------
# Save mappings to pickle file
# ----------------------------------------

def save_mapping_file(filename, num_mappings):

    mapping_list = generate_mapping_list(num_mappings)

    with open(filename, "wb") as f:
        pickle.dump(mapping_list, f)

    print(f"\nSaved to: {filename}")


# ----------------------------------------
# Main
# ----------------------------------------

if __name__ == "__main__":

    save_mapping_file(
        "random_valid_mappings.pkl",
        num_mappings=1
    )
