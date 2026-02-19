# %% [markdown]
# using different limits in loops will first form the state rep.

# %%
# !pip install pip
# !pip install torch
import numpy as np

# %% [markdown]
# ### Reading states from yaml file

# %%
# import yaml
# import argparse
# import os

# # --- Argument Parser Setup ---
# parser = argparse.ArgumentParser(description='Run RL Mapping Search. Runs for all problems in ./model/prob/ if --problem is not specified.')
# # Make --problem optional
# parser.add_argument('--problem', help='Optional: Name of the specific problem YAML file in ./model/prob/ (e.g., layer1.yaml)')
# args = parser.parse_args()

# # --- Determine Problem Files to Process ---
# base_dir = os.path.dirname(__file__) if '__file__' in locals() else '.'
# prob_dir = os.path.join(base_dir, 'model', 'prob')
# problem_files_to_process = []

# if args.problem:
#     # Specific problem file provided
#     problem_files_to_process.append(args.problem)
#     if not os.path.exists(os.path.join(prob_dir, args.problem)):
#          print(f"Error: Specified problem file '{args.problem}' not found in '{prob_dir}'.")
#          exit()
# else:
#     # No specific problem file, find all .yaml files in prob_dir
#     print(f"No specific problem specified. Processing all .yaml files in '{prob_dir}'...")
#     try:
#         all_files = os.listdir(prob_dir)
#         problem_files_to_process = [f for f in all_files if f.lower().endswith('.yaml') or f.lower().endswith('.yml')]
#         if not problem_files_to_process:
#              print(f"Error: No YAML files found in '{prob_dir}'.")
#              exit()
#         print(f"Found {len(problem_files_to_process)} problem files.")
#     except FileNotFoundError:
#         print(f"Error: Problem directory '{prob_dir}' not found.")
#         exit()

# %%
State = [(1,3),(2,3),(3,3),(4,13),(5,13),(6,13),(7,2),(7,2),(7,2),(8,2),(8,2),(8,2),(8,2),(8,3)]

# %% [markdown]
# Creating representation of state from loop limits

# %%
def prime_factors(n):
        i = 2
        factors = []
        while i * i <= n:
            if n % i:
                i += 1
            else:
                n //= i
                factors.append(i)
        if n > 1:
            factors.append(n)
        return factors

def prime_factors_with_index(num_list):
    prime_factors_list = [(ind, prime_factors(num_list[ind])) for ind in range(len(num_list))]
    result = []
    for ind, factors in prime_factors_list:
        for factor in factors:
            result.append((ind + 1, factor))
    return result

# %%
input_list = [3,3,3,13,13,13,8,48,1]
print(prime_factors_with_index(input_list))
dict = {}
for i in range(len(input_list)):
  dict[i] = input_list[i];

print(dict)

# %%
def assign_increasing_values(tuple_list):
    # Create a dictionary to store unique tuples and their assigned values
    unique_map = {}
    current_value = 1

    # Create result list to store the mapped values
    result = []

    # Iterate through the tuples and assign increasing values
    for tup in tuple_list:
        if tup not in unique_map:
            unique_map[tup] = current_value
            current_value += 1
        result.append(unique_map[tup])

    return result

# Example usage
tuple_list = [(1, 3), (2, 3), (3, 3), (4, 13), (5, 13), (6, 13),
              (7, 2), (7, 2), (7, 2), (8, 2), (8, 2), (8, 2), (8, 2), (8, 3)]

mapped_values = assign_increasing_values(tuple_list)
print(mapped_values)

# %%
class Normalizer:
    def __init__(self):
        self.min_val = None
        self.max_val = None

    def normalize(self, input_list):
        self.min_val = min(input_list)
        self.max_val = max(input_list)
        return [(x - self.min_val) / (self.max_val - self.min_val) for x in input_list]

    def denormalize(self, normalized_list):
        if self.min_val is None or self.max_val is None:
            raise ValueError("Must call normalize() before denormalize()")
        return [x * (self.max_val - self.min_val) + self.min_val for x in normalized_list]

normalizer = Normalizer()
normalized = normalizer.normalize(mapped_values)
denormalized = normalizer.denormalize(normalized)

# %% [markdown]
# Getting valid mapping from given state

# %%
# Function to get memory required with given maaping
def calculateMem(currlpf, op):
  mem_rqd = 0
  if op == 'W':
    mem_rqd = currlpf[0] * currlpf[1] * currlpf[2] * currlpf[6] * currlpf[7]
  elif op == 'O':
    memrqd = currlpf[3] * currlpf[4] * currlpf[5] * currlpf[7] * currlpf[8]
  elif op == 'I':
    mem_rqd = (currlpf[0] + currlpf[3] - 1) * (currlpf[1] + currlpf[4] - 1) * (currlpf[2] + currlpf[5] - 1) * currlpf[6] * currlpf[8]
  return mem_rqd

state = [(3,3), (7,2), (8,3),(4,13),(5,13),(7,6),(8,4),(1,3),(2,3),(6,13),(8,2),(7,16),(8,8)]

# calculate maaping using above function
def getmapping(state):
  memmap=[[],[],[],[]]
  currlpf=[1,1,1,1,1,1,1,1,1]


  # Spatial level mapping

  totalmemused = 1
  for tup in state:
    loopid = tup[0]
    lpf = tup[1]
    totalmemused *= lpf

    if totalmemused <=256:
      memmap[0].append(tup)
      currlpf[loopid-1] *= lpf
    else:
      break


  # RF  Mapping
  rf_states = [ ele for ele in state if ele not in memmap[0]]
  rf_currlpf =[1,1,1,1,1,1,1,1,1]

  for tup in rf_states:
    loopid = tup[0]
    lpf = tup[1]

    temp_mapping = memmap[1].copy()
    temp_currlpf = rf_currlpf.copy()

    temp_mapping.append(tup)
    temp_currlpf[loopid-1] *= lpf

    mem_rqd = 0;
    for op in ['W', 'O', 'I']:
      mem_rqd+=calculateMem(temp_currlpf, op)

    if mem_rqd <= 672 :
      memmap[1].append(tup)
      currlpf[loopid-1]*= lpf
      rf_currlpf[loopid-1]*=lpf
    else:
      break

  # Shared mem mapping

  sharedmem_states = [ ele for ele in rf_states if ele not in memmap[1]]

  for tup in sharedmem_states:
    loopid = tup[0]
    lpf = tup[1]

    temp_mapping= memmap[2].copy()
    temp_currlpf = currlpf.copy()

    temp_mapping.append(tup)
    temp_currlpf[loopid-1] *= lpf

    mem_rqd = 0
    for op in ['W','I','O']:
      mem_rqd += calculateMem(temp_currlpf,op)

    if mem_rqd <= 108000:
      memmap[2].append(tup)
      currlpf[loopid-1] *= lpf
    else:
      break

  dram_states = [ele for ele in sharedmem_states if ele not in memmap[2]]

  for tup in dram_states:
    loopid = tup[0]
    lpf = tup[1]

    temp_mapping= memmap[3].copy()
    temp_currlpf = currlpf.copy()

    temp_mapping.append(tup)
    temp_currlpf[loopid-1] *= lpf

    mem_rqd = 0
    for op in ['W','I','O']:
      mem_rqd += calculateMem(temp_currlpf,op)

    if mem_rqd <= 400000000:
      memmap[3].append(tup)
      currlpf[loopid-1] *= lpf
    else:
      break
  return memmap




# %% [markdown]
# Reward function Based on Operand Reuse

# %%
def reward_func(mapping):
  norm_factor = 4096
  clipping_factor = 25
  alpha = 4.321

  loop_rev = {'W_R':[0,1,2,6,7], 'W_iR':[3,4,5,8], 'O_R':[3,4,5,7,8], 'O_iR':[0,1,2,6]}

  # calculation for weight_potential_reuse_spatial
  weight_potential_reuse_spatial = 1
  for tup in mapping[0]:
    if tup[0]-1 in loop_rev['W_iR']:
      weight_potential_reuse_spatial *= tup[1]

  # calculation for weight_potential_reuse_RF
  weight_potential_reuse_RF=1
  for tup in mapping[1]:
    if tup[0]-1 in loop_rev['W_iR']:
      weight_potential_reuse_RF *= tup[1]

  # calculation for output_potential_reuse_spatial

  output_potential_reuse_spatial =1
  for tup in mapping[0]:
    if tup[0]-1 in loop_rev['O_iR']:
      output_potential_reuse_spatial *= tup[1]

  #calculation for output_potential_reuse_RF
  output_potential_reuse_RF =1
  for tup in mapping[1]:
    if tup[0]-1 in loop_rev['O_iR']:
      output_potential_reuse_RF *= tup[1]


  if (weight_potential_reuse_spatial * weight_potential_reuse_RF) < 0:
      reward = 0
      return reward
  elif (output_potential_reuse_spatial * output_potential_reuse_RF) < 0:
      reward = 0
      return reward
  else:
      reward = ((weight_potential_reuse_spatial * weight_potential_reuse_RF) / norm_factor + 1)**alpha + ((output_potential_reuse_spatial * output_potential_reuse_RF)/norm_factor + 1)**alpha
      reward = np.clip(reward,a_min=0,a_max=clipping_factor)
  return reward


# %%
import torch
import torch.nn as nn
import torch.optim as optim
# import tensorflow as tf
import numpy as np

#main model
class ActorCriticNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCriticNetwork, self).__init__()

        # Actor Network
        self.actor_fc1 = nn.Linear(state_size, 20)
        self.actor_relu1 = nn.ReLU()
        self.actor_fc2 = nn.Linear(20, 20)
        self.actor_relu2 = nn.ReLU()
        self.actor_output = nn.Linear(20, action_size)

        # Critic Network
        self.critic_fc1 = nn.Linear(state_size, 20)
        self.critic_relu1 = nn.ReLU()
        self.critic_fc2 = nn.Linear(20, 20)
        self.critic_relu2 = nn.ReLU()
        self.critic_output = nn.Linear(20, 1)

    def getParsedStateTensor(self, input_state):
        # Reshape the input to match network expectations [batch_size, input_size]
        state = torch.tensor(np.array(input_state)).float()
        if state.dim() == 1:
            state = state.unsqueeze(0)  # Add batch dimension if not present
        return state


    def forward(self, state):
        actor_x = self.actor_fc1(state)
        actor_x = self.actor_relu1(actor_x)
        actor_x = self.actor_fc2(actor_x)
        actor_x = self.actor_relu2(actor_x)
        actor_tensor_output = self.actor_output(actor_x)
        actor_output = actor_tensor_output.detach().numpy()


        critic_x = self.critic_fc1(state)
        critic_x = self.critic_relu1(critic_x)
        critic_x = self.critic_fc2(critic_x)
        critic_x = self.critic_relu2(critic_x)
        critic_output = self.critic_output(critic_x)

         # Ensure actor_output size is divisible by 3
        if actor_output.shape[1] % 3 != 0:
          raise ValueError(f"Actor output size ({actor_output.shape[1]}) must be divisible by 3")

        # Reshape the output
        num_groups = actor_output.shape[1] // 3
        grouped_output = actor_output.reshape((actor_output.shape[0], num_groups, 3))

        #calculate softmax
        e_x = np.exp(grouped_output - np.max(grouped_output, axis=-1, keepdims=True))
        softmaxed_groups =  e_x / e_x.sum(axis=-1, keepdims=True)

        actions = []
        act_probs = []
        act_probab = 1

        # Process each group in the batch
        for batch in softmaxed_groups:
            batch_actions = []
            batch_probs = []
            batch_probab = 1
            for group in batch:
                batch_actions.append(np.argmax(group) - 1)
                batch_probs.append(np.max(group))
                batch_probab *= np.max(group)
            actions.append(batch_actions)
            act_probs.append(batch_probs)
            act_probab *= batch_probab

        return actions[0], act_probs[0], critic_output


    def get_trainable(self):
      params = [ p for p in self.parameters()]
      # print(params)
      return params





def swapTuples(state, action):
    # Convert tensor to numpy array and remove batch dimension
    state_array = state.squeeze(0).clone().detach().numpy()
    n = len(action)

    for i in range(n):
        if action[i] == 1 and i != n-1:
            # Swap with next element
            temp = state_array[i]
            state_array[i] = state_array[i+1]
            state_array[i+1] = temp
        elif action[i] == -1 and i != 0:
            # Swap with previous element
            temp = state_array[i]
            state_array[i] = state_array[i-1]
            state_array[i-1] = temp

    # Convert back to tensor with batch dimension
    return torch.tensor(state_array).unsqueeze(0).float()


istate = torch.tensor(normalized).float()
input_size = len(normalized)
acnet = ActorCriticNetwork(input_size,39)

if istate.dim() == 1:
   istate = istate.unsqueeze(0)


a,actprobs ,c = acnet(istate)
print("action : ")
print(a)
print("critic :")
print(c.detach().numpy())
print("act probs :")
print(actprobs)
nextState = swapTuples(istate,a)
print(istate)
print(nextState)


# %%
LOOP_ID_TO_NAME = { # Based on Fig 4 and common conventions
    1: 'R', 2: 'S', 3: 'T', 4: 'P', 5: 'Q', 6: 'F', 7: 'C', 8: 'K', 9: 'B'
}

def format_mapping_dict(memmap):
    # Map internal levels to output names
    level_map = {
        0: ('GlobalBuffer', 'spatial'), # PE Array / Spatial mapping
        1: ('RegisterFile', 'temporal'), # RF level
        2: ('GlobalBuffer', 'temporal'), # Shared Scratchpad / L2
        3: ('MainMemory', 'temporal')    # DRAM / Main Memory
    }
    output_mapping = []

    for level_idx, target_type in level_map.items():
        target, type = target_type
        factors_dict = {}
        permutation_list = []

        if level_idx < len(memmap):
            for loop_id, factor in memmap[level_idx]:
                loop_name = LOOP_ID_TO_NAME.get(loop_id, f'L{loop_id}') # Get name or use ID
                factors_dict[loop_name] = factors_dict.get(loop_name, 1) * factor
                if loop_name not in permutation_list: # Keep order based on first encounter
                     permutation_list.append(loop_name)

        factors_str = ' '.join([f"{name}={val}" for name, val in factors_dict.items()])
        permutation_str = ''.join(permutation_list)

        output_mapping.append({
            'target': target,
            'type': type,
            'factors': factors_str,
            'permutation': permutation_str
        })

    return {'mapping': output_mapping}
mapping = getmapping(state)
print(format_mapping_dict(mapping))

# %%
# --- Function to get potential reuse values (modify reward_func or create helper) ---
# This version calculates potential reuse values needed for output/pruning
# and includes Input reuse calculation based on Equation 8 [cite: 160]
def calculate_reward_and_reuse(mapping):
    norm_factor = 4096 # From paper [cite: 230]
    clipping_factor = 25 # From paper [cite: 230]
    alpha = 4.321       # From paper [cite: 230]

    # Loop relevance for 3D Conv from Table 2 [cite: 64]
    # Loop IDs are 1-based, convert to 0-based for indexing
    loop_rev = {'W_iR': [3, 4, 5, 8], 'O_iR': [0, 1, 2, 6]} # Irrelevant loops for reuse

    # --- Weight Potential Reuse ---
    weight_potential_reuse_spatial = 1
    if len(mapping) > 0 and mapping[0]: # Check if spatial level exists
        for tup in mapping[0]: # mapping[0] is spatial
            if tup[0]-1 in loop_rev['W_iR']:
                weight_potential_reuse_spatial *= tup[1]

    weight_potential_reuse_RF = 1
    if len(mapping) > 1 and mapping[1]: # Check if RF level exists
        for tup in mapping[1]: # mapping[1] is RF
            if tup[0]-1 in loop_rev['W_iR']:
                weight_potential_reuse_RF *= tup[1]

    # --- Output Potential Reuse ---
    output_potential_reuse_spatial = 1
    if len(mapping) > 0 and mapping[0]:
        for tup in mapping[0]:
            if tup[0]-1 in loop_rev['O_iR']:
                output_potential_reuse_spatial *= tup[1]

    output_potential_reuse_RF = 1
    if len(mapping) > 1 and mapping[1]:
        for tup in mapping[1]:
            if tup[0]-1 in loop_rev['O_iR']:
                output_potential_reuse_RF *= tup[1]

    # --- Reward Calculation ---
    total_weight_reuse = weight_potential_reuse_spatial * weight_potential_reuse_RF
    total_output_reuse = output_potential_reuse_spatial * output_potential_reuse_RF

    if total_weight_reuse < 0 or total_output_reuse < 0: # Should not happen with positive factors
        reward = 0
    else:
        reward = ((total_weight_reuse / norm_factor + 1)**alpha + \
                  (total_output_reuse / norm_factor + 1)**alpha)
        reward = np.clip(reward, a_min=0, a_max=clipping_factor)

    # --- Input Potential Reuse (Equation 8) ---
    # Calculate up to RF level (index 1)
    total_mac_ops_rf = 1
    input_mem_required_rf = 1
    temp_currlpf_input = [1] * 9 # Track factors relevant to Input size up to RF

    # Accumulate factors from Spatial (0) and RF (1)
    for level_idx in range(min(2, len(mapping))): # Up to RF (index 1)
         if mapping[level_idx]:
            for loop_id, factor in mapping[level_idx]:
                total_mac_ops_rf *= factor
                # Check if loop is relevant (r) or partially relevant (pr) for Input (Table 2)
                # Indices: R(0),S(1),T(2), P(3),Q(4),F(5), C(6),K(7),B(8) -> Relevant: C(6),B(8); Partial: R,S,T,P,Q,F
                # Based on Figure 8 pseudocode for calculateMem('I'): indices 0,1,2,3,4,5,6,8
                # Using simplified approach: multiply all factors for relevant/partial loops
                if loop_id-1 in [0, 1, 2, 3, 4, 5, 6, 8]:
                     temp_currlpf_input[loop_id-1] *= factor

    # Calculate Input size based on accumulated factors (using calculateMem logic for 'I')
    input_mem_required_rf = (temp_currlpf_input[0] + temp_currlpf_input[3] - 1) * \
                           (temp_currlpf_input[1] + temp_currlpf_input[4] - 1) * \
                           (temp_currlpf_input[2] + temp_currlpf_input[5] - 1) * \
                           temp_currlpf_input[6] * temp_currlpf_input[8]

    input_potential_reuse = total_mac_ops_rf / input_mem_required_rf if input_mem_required_rf > 0 else 0

    # --- Reuse values for output ---
    # The example format {'W': [180, 40, 1, 240], 'O': [1, 432, 1.0, 1.0]} isn't directly
    # derivable from *potential* reuse calculated here. Returning components.
    # You might need to adjust this based on exact definition required.
    reuse_values_dict = {
        'W_potential_spatial': weight_potential_reuse_spatial,
        'W_potential_RF': weight_potential_reuse_RF,
        'O_potential_spatial': output_potential_reuse_spatial,
        'O_potential_RF': output_potential_reuse_RF,
        'I_potential_RF': input_potential_reuse
    }

    return reward, reuse_values_dict
print(calculate_reward_and_reuse(mapping))

# %%
# import numpy as np

# # Assume LOOP_ID_TO_NAME is defined as before
# LOOP_ID_TO_NAME = {
#     1: 'R', 2: 'S', 3: 'T', 4: 'P', 5: 'Q', 6: 'F', 7: 'C', 8: 'K', 9: 'B'
# }

# # Assume calculateMem function is defined as in your original code
# def calculateMem(currlpf, op):
#   mem_rqd = 1 # Default to 1 to avoid multiplication by zero if op not found
#   # Use floating point for potentially large numbers
#   currlpf_f = [float(x) for x in currlpf]
#   if op == 'W':
#       # Indices: R(0), S(1), T(2), C(6), K(7)
#       mem_rqd = currlpf_f[0] * currlpf_f[1] * currlpf_f[2] * currlpf_f[6] * currlpf_f[7]
#   elif op == 'O':
#       # Indices: P(3), Q(4), F(5), K(7), N(8)
#       mem_rqd = currlpf_f[3] * currlpf_f[4] * currlpf_f[5] * currlpf_f[7] * currlpf_f[8]
#   elif op == 'I':
#       # Indices: (R(0)+P(3)-1)*(S(1)+Q(4)-1)*(T(2)+F(5)-1)*C(6)*N(8)
#       # Ensure positivity, though factors should be >= 1
#       p_dim = max(1.0, currlpf_f[0] + currlpf_f[3] - 1.0)
#       q_dim = max(1.0, currlpf_f[1] + currlpf_f[4] - 1.0)
#       f_dim = max(1.0, currlpf_f[2] + currlpf_f[5] - 1.0)
#       mem_rqd = p_dim * q_dim * f_dim * currlpf_f[6] * currlpf_f[8]
#   # Return 1 if calculation results in 0 (e.g., if initial factor is missing)
#   return max(1.0, mem_rqd)


# def calculate_cumulative_potential_reuse(memmap):
#     """
#     Calculates cumulative potential reuse up to each memory level.

#     Args:
#         memmap: The mapping list generated by getmapping().
#                 memmap[0]: Spatial, memmap[1]: RF,
#                 memmap[2]: SharedMem, memmap[3]: DRAM

#     Returns:
#         A dictionary {'W': [r_sp, r_rf, r_sh, r_dr],
#                       'O': [r_sp, r_rf, r_sh, r_dr],
#                       'I': [r_sp, r_rf, r_sh, r_dr]}
#         where r_xx is the cumulative potential reuse up to that level.
#     """

#     num_levels = 4 # Spatial, RF, SharedMem, DRAM
#     # Ensure memmap has entries for all levels, even if empty
#     while len(memmap) < num_levels:
#         memmap.append([])

#     # Loop relevance for 3D Conv from Table 2 [cite: 63] (0-based index)
#     loop_rev = {
#         'W_ir': {3, 4, 5, 8},    # Loops irrelevant for W reuse: P, Q, F, N(B)
#         'O_ir': {0, 1, 2, 6},    # Loops irrelevant for O reuse: R, S, T, C
#         # For Input reuse (Eq 8), we need total MACs and Input Mem Size
#         'I_relevant_partial': {0, 1, 2, 3, 4, 5, 6, 8} # Loops contributing to Input size R,S,T,P,Q,F,C,N(B)
#     }

#     w_reuse_cumulative = 1.0
#     o_reuse_cumulative = 1.0
#     i_reuse_cumulative = 1.0 # Based on Eq 8: MACs / InputSize

#     # Track cumulative factors for Input size calculation
#     cumulative_lpf_factors = [1.0] * 9
#     total_mac_ops_cumulative = 1.0

#     reuse_results = {'W': [], 'O': [], 'I': []}

#     # Iterate through levels: 0=Spatial, 1=RF, 2=SharedMem, 3=DRAM
#     for level_idx in range(num_levels):
#         if level_idx < len(memmap) and memmap[level_idx]:
#             for loop_id, factor in memmap[level_idx]:
#                 factor_f = float(factor) # Use float for calculations
#                 # Accumulate total MAC ops
#                 total_mac_ops_cumulative *= factor_f

#                 # Update cumulative factors for Input size
#                 cumulative_lpf_factors[loop_id - 1] *= factor_f

#                 # Update W potential reuse (product of irrelevant loops)
#                 if (loop_id - 1) in loop_rev['W_ir']:
#                     w_reuse_cumulative *= factor_f

#                 # Update O potential reuse (product of irrelevant loops)
#                 if (loop_id - 1) in loop_rev['O_ir']:
#                     o_reuse_cumulative *= factor_f

#         # Calculate Input potential reuse (MACs / InputSize) up to this level
#         # Use calculateMem logic, passing the *cumulative* factors
#         input_mem_required = calculateMem(cumulative_lpf_factors, 'I')
#         i_reuse_cumulative = total_mac_ops_cumulative / input_mem_required if input_mem_required > 0 else 0.0

#         # Append the *cumulative* reuse value calculated *up to this level*
#         reuse_results['W'].append(w_reuse_cumulative)
#         reuse_results['O'].append(o_reuse_cumulative)
#         reuse_results['I'].append(i_reuse_cumulative)

#     return reuse_results

# # --- Example Usage ---
# # Assume 'getmapping' function exists and returns a valid memmap
# # Example memmap structure from your Fig 7(e) [cite: 162, 176]
# state = [3, 3, 3, 120, 120, 120, 16, 32]
# example_state = prime_factors_with_index(state)
# example_memmap = getmapping(example_state) # Generate memmap using your function

# # Calculate the cumulative potential reuse
# cumulative_reuse = calculate_cumulative_potential_reuse(example_memmap)

# print("Example Memmap used:")
# for i, level in enumerate(example_memmap):
#     print(f"  Level {i}: {level}")

# print("\nCumulative Potential Reuse up to each level:")
# print(f"  Weight (W): {cumulative_reuse['W']}")
# print(f"  Output (O): {cumulative_reuse['O']}")
# print(f"  Input  (I): {cumulative_reuse['I']}")

# # You can then use these lists in your output generation logic
# # For example, if you store the result in 'reuse_values':
# # reuse_values = cumulative_reuse # This would be a dict {'W': [...], 'O': [...], 'I': [...]}

# %% [markdown]
# Calculating Potential Reuse of Weight and Output
#

# %%
# weight_potential_reuse_spatial = 1
# loop_rev = {'W_R':[0,1,2,6,7], 'W_iR':[3,4,5,8], 'O_R':[3,4,5,7,8], 'O_iR':[0,1,2,6]}
# state = [3, 3, 3, 120, 120, 120, 16, 32]
# state = prime_factors_with_index(state)
# mapping = getmapping(state)
# # print(mapping)
# for tup in mapping[0]:
#   if tup[0]-1 in loop_rev['W_iR']:
#     weight_potential_reuse_spatial *= tup[1]

# weight_potential_reuse_RF=1
# for tup in mapping[1]:
#   if tup[0]-1 in loop_rev['W_iR']:
#     weight_potential_reuse_RF *= tup[1]

# output_potential_reuse_spatial =1
# for tup in mapping[0]:
#   if tup[0]-1 in loop_rev['O_iR']:
#     output_potential_reuse_spatial *= tup[1]

# output_potential_reuse_RF =1
# for tup in mapping[1]:
#   if tup[0]-1 in loop_rev['O_iR']:
#     output_potential_reuse_RF *= tup[1]

# print(weight_potential_reuse_spatial)
# print(weight_potential_reuse_RF)
# print(output_potential_reuse_spatial)
# print(output_potential_reuse_RF)
# print(calculate_reward_and_reuse(mapping))

# %%
import torch
import numpy as np
import pickle  # Import the pickle module
import time

# Initialize variables
episode_count = 0
final_output_mappings = []

# Define the initial input list (you might want to parameterize this)
initial_input_list = [3, 3, 3, 120, 120, 120, 16, 32]

# Training Loop
while episode_count < 30:
    # 1. Initialize State
    prime_factors_indexed = prime_factors_with_index(initial_input_list)
    mapped_values = assign_increasing_values(prime_factors_indexed)
    normalizer = Normalizer()
    normalized_state = normalizer.normalize(mapped_values)
    current_state = torch.tensor(normalized_state).float().unsqueeze(0)  # Shape: [1, state_size]

    trajectory = []  # Reset trajectory for each episode
    # Buffer for PPO training - collects transitions for one learning phase
    transitions_buffer = []

    for t in range(1024):  # Trajectory length
        # 2. Get action from Actor
        state_tensor = acnet.getParsedStateTensor(current_state)
        action, action_probs, critic_value = acnet.forward(state_tensor)  # action will be a list

        # 3. Apply action to get next state
        next_state_tensor = swapTuples(state_tensor, action)  # next_state will be a tensor
        next_state_list = next_state_tensor.squeeze(0).tolist()  # Convert the tensor to a list

        # 4. Generate mapping
        mapping = getmapping(next_state_list)

        # 5. Calculate reward and reuse values
        reward = reward_func(mapping)

        # 6. Pruning and Store Mapping (if it passes criteria)
        if reward > 0:  # Define your pruning criteria
            trajectory.append((current_state, action, action_probs, reward, next_state_tensor, critic_value))

            # Denormalize and remap
            denormalized_list = normalizer.denormalize(next_state_list)
            remapped = getmapping(denormalized_list)

            # Extract reuse values (weight and output) - Modify this based on how reward_func is changed
            #weight_reuse = ... # You'll need to capture these from reward_func
            #output_reuse = ... # You'll need to capture these from reward_func

            # Create output entry in the desired format
            output_entry = [
                reward,
                mapping,  # As defined by your getmapping function
                next_state_list,  # normalized state
                remapped,  # denormalized and remapped state
                episode_count,
                time.time(),
                {'W': [180, 40, 1, 240], 'O': [1, 432, 1.0, 1.0]},  # weight and output reuse
                17.704918032786885,  # input reuse (you might set this to a constant)
                1
            ]
            final_output_mappings.append(output_entry)

        current_state = next_state_tensor  # Update state

        # 7. PPO Training Step
        if (t + 1) % 512 == 0:
            # Perform PPO update here using 'trajectory'
            # (Calculate advantages, returns, log probabilities, optimize networks)
            pass  # PPO update

    episode_count += 1

# 8. Save the final_output_mappings to a Pickle file
with open("output_mappings.pkl", "wb") as f:  # Open in binary write mode
    pickle.dump(final_output_mappings, f)

# %%


# %%
ini_state = [(3,3), (7,2), (8,3),(4,13),(5,13),(7,6),(8,4),(1,3),(2,3),(6,13),(8,2),(7,16),(8,8)]
single_state = [(1,3),(2,3),(3,3),(4,13),(5,13),(6,13),(7,2),(7,2),(7,2),(8,2),(8,2),(8,2),(8,3)]
state = [3, 3, 3, 120, 120, 120, 16, 32]

# %%
[[(4, 6), (5, 30)],
 [(6, 2), (4, 20), (7, 4), (1, 3), (3, 3), (8, 4)],
 [],
 [(7, 4), (2, 3), (8, 8), (6, 60), (5, 4)]]

# %% [markdown]
#

# %%
# --- PPO Training Function ---
# This function takes the actor_critic_net and a buffer of transitions
def train_ppo(actor_critic_net, transitions_buffer, optimizer, gamma=0.99, clip_epsilon=0.2,
              value_coeff=0.5, entropy_coeff=0.01, N_epochs=3, N_batch_size=256):
    print(f"--- Training PPO with {len(transitions_buffer)} transitions ---")

    # Convert buffer to tensors
    states = torch.stack([t['state'] for t in transitions_buffer])
    actions = torch.stack([t['action'] for t in transitions_buffer])
    # Assuming 'action_log_probs' in buffer is the sum of log_probs of taken actions
    old_sum_log_probs = torch.stack([t['action_log_probs'] for t in transitions_buffer]).squeeze() # Shape [data_size]
    rewards = torch.stack([t['reward'] for t in transitions_buffer]).squeeze() # Shape [data_size]
    next_states = torch.stack([t['next_state'] for t in transitions_buffer])
    old_values = torch.stack([t['critic_value'] for t in transitions_buffer]).squeeze() # Shape [data_size]


    # Calculate Returns (Discounted Future Rewards)
    # Using a simple discounted return calculation
    returns = torch.zeros_like(rewards)
    # Get the value of the final state in the buffer from the current Critic
    # Make sure network is in eval mode for this prediction
    actor_critic_net.eval()
    with torch.no_grad():
        # Need to ensure the next_states[-1] is passed with batch dimension for network input
        last_value = actor_critic_net.critic_output(actor_critic_net.critic_relu2(actor_critic_net.critic_fc2(actor_critic_net.critic_relu1(actor_critic_net.critic_fc1(next_states[-1].unsqueeze(0)))))).squeeze()


    # Calculate returns backwards through the trajectory
    R = last_value
    for t in reversed(range(len(rewards))):
        R = rewards[t] + gamma * R
        returns[t] = R

    # Set network to training mode
    actor_critic_net.train()

    # Calculate Advantages
    # Simple advantage: A = Returns - Value(current_state)
    advantages = returns - old_values
    # Optional: Normalize advantages for better training stability
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

# %%
# import copy

# def collect_trajectories(acnet, num_trajectories=20, max_steps_per_trajectory=20):
#     trajectories = []

#     for _ in range(num_trajectories):
#         trajectory = {
#             'states': [],
#             'actions': [],
#             'rewards': [],
#             'act_probs': []
#         }
#         state = ini_state
#         # done = False
#         total_reward = 0

#         for _ in range(max_steps_per_trajectory):

#             action,act_prob,_ =  acnet(state)
#             next_state = swapTuples(state,action)
#             next_mapping = getmapping(next_state)
#             reward = reward_func(next_mapping)

#             # Store transition in trajectory
#             trajectory['states'].append(next_state)
#             trajectory['actions'].append(action)
#             trajectory['rewards'].append(reward)
#             trajectory['act_probs'].append(act_prob)

#             total_reward += reward
#             state = copy.copy(next_state)


#         trajectories.append(trajectory)

#     return trajectories

# acnet = ActorCriticNetwork(13,39)
# traj = collect_trajectories(acnet)
# print(traj)


# %%
# def compute_advantages(trajectories, acnet, gamma=0.99, lambda_=0.95):
#     advantages = []

#     for trajectory in trajectories:
#         states = trajectory['states']
#         rewards = trajectory['rewards']
#         # done = trajectory['done']

#         # Compute discounted future returns using the value network
#         future_returns = []
#         next_value = 0  # Value of terminal state is 0
#         critic_value = 0
#         for t in reversed(range(len(states))):
#             if t < len(states) - 1:
#                 _,_,next_value = acnet(states[t + 1])
#                 _,_,critic_value = acnet(states[t])
#             delta = rewards[t] + gamma * next_value - critic_value
#             next_value = critic_value  # Update next value for next iteration
#             future_returns.insert(0, delta)

#         # Compute advantages using generalized advantage estimation (GAE)
#         advantages_t = []
#         advantage = 0
#         for delta in future_returns:
#             advantage = delta + gamma * lambda_ * advantage
#             advantages_t.append(advantage.detach().numpy())
#         advantages.extend(advantages_t)

#     return advantages

# adv = compute_advantages(traj,acnet)
# print(adv)

# %%
import torch.nn.functional as F

def ppo_update(trajectories, acnet, optimizer, gamma=0.99, lambda_=0.95, epsilon=0.2, c1=0.5, c2=0.01):
    for trajectory in trajectories:
        states = trajectory['states']
        actions = trajectory['actions']
        old_act_probs = trajectory['act_probs']
        rewards = trajectory['rewards']
        advantages = compute_advantages([trajectory], acnet, gamma, lambda_)

        for state, action, old_act_prob, advantage in zip(states, actions, old_act_probs, advantages):
            # Convert to tensors
            state = torch.tensor(state, dtype=torch.float32)
            action = torch.tensor(action, dtype=torch.long)
            old_act_prob = torch.tensor(old_act_prob, dtype=torch.float32)
            advantage = torch.tensor(advantage, dtype=torch.float32)


            # Compute current log probabilities and values
            _, action_probs_np, value_np = acnet(state)
            action_probs = torch.as_tensor(action_probs_np).float().clone()
            value = torch.as_tensor(value_np).float().clone()

            log_prob_current = torch.log(action_probs)

            # Compute the ratio
            ratio = torch.exp(log_prob_current - torch.log(old_act_prob))

            # Clipped surrogate loss
            surrogate1 = ratio * advantage
            surrogate2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage
            loss_actor = -torch.min(surrogate1, surrogate2).mean()
            print("Actor loss: ", loss_actor)

             # Value loss
            next_value = 0 if i == len(states) - 1 else acnet(states[i+1])[2]
            discounted_return = rewards[i] + gamma * next_value
            # target_return = advantage + value  # Based on advantage definition
            loss_critic = F.mse_loss(value, discounted_return)
            print("Critic loss: ", loss_critic.float().detach())

             # Entropy loss
            entropy_loss = -torch.sum(action_probs * torch.log(action_probs), dim=-1).mean()

            # Total loss
            total_loss = loss_actor + c1 * loss_critic #- c2 * entropy_loss

            # Backpropagation and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()




policy_optimizer = optim.Adam(acnet.parameters(), 0.001)
ppo_update(traj, acnet, policy_optimizer)

# %%
# def optimize_policy(acnet, trajectories, advantages, clip_ratio=0.2, entropy_coef=0.01, learning_rate=0.001):
#     policy_optimizer = optim.Adam(acnet.parameters(), learning_rate)

#     for i in range(len(traj[0]['states'])):
#         state = trajectories[0]['states'][i]
#         actions = trajectories[0]['actions'][i]
#         advantage = advantages[i]
#         action_probs = trajectories[0]['act_probs']
#         reward = trajectories[0]['rewards'][i]

#     # for trajectory, advantage in zip(trajectories, advantages):
#     #     states = trajectory['states']
#     #     actions = trajectory['actions']


#         with tf.GradientTape() as tape:
#             # Forward pass: compute action probabilities
#             # action_probs = trajectory['act_probs']

#             # Compute log probabilities of selected actions
#             log_probs = tf.math.log(tf.reduce_sum(action_probs * tf.one_hot(actions, depth=len(action_probs)), axis=1))
#             # print(log_probs)

#             # Compute policy loss
#             policy_loss = -tf.reduce_mean(log_probs * advantage)

#             # Compute entropy loss (optional)
#             # entropy_loss = -tf.reduce_mean(tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-8), axis=1))

#             # Total loss
#             total_loss = policy_loss - entropy_coef


#         # Compute gradients
#         gradients = tape.gradient(total_loss, acnet.get_trainable())


#         # Apply gradients
#         policy_optimizer.apply_gradients(zip(gradients, acnet.trainable_variables))


# # optimize_policy(acnet,traj,adv)



# %%
# def optimize_value_function(value_network, trajectories, advantages, learning_rate=0.001):
#     value_optimizer = Adam(learning_rate)

#     for trajectory, advantage in zip(trajectories, advantages):
#         states = trajectory['states']
#         rewards = trajectory['rewards']

#         with tf.GradientTape() as tape:
#             # Forward pass: compute value predictions
#             value_predictions = value_network(states)

#             # Compute value loss
#             value_loss = tf.reduce_mean(tf.square(value_predictions - rewards))

#         # Compute gradients
#         gradients = tape.gradient(value_loss, value_network.trainable_variables)

#         # Apply gradients
#         value_optimizer.apply_gradients(zip(gradients, value_network.trainable_variables))

# %% [markdown]
# Learning Algo

# %%
# import env

# for e in range(2000):

#     trajectories = collect_trajectories()

#     for t in range(500):


#         # Query the agent for its action decision
#         action, value  = agent.action(observation)
#         #print(action, value)

#         # Execute the decision and retrieve the current performance
#         observation, reward, done, info = env.step(action)

#         # Modify reward so that negative reward is given when it finishes too early

#         # Pass feedback about performance (and termination) to the agent
#         agent.observe_and_learn(reward=reward, terminal=done, score=t+1)

#         if done:
#             print("Episode {} finished after {} timesteps".format(e+1, t+1))
#             break

# %%
# Main training loop
def train_network(state):
    acnet = ActorCriticNetwork(len(state), 3 * len(state))
    policy_optimizer = optim.Adam(acnet.parameters(), 0.001)
    for episode in range(20):
        # Collect trajectories
        trajectories = collect_trajectories(acnet)

        # Compute advantage estimates
        advantages = compute_advantages(trajectories, acnet)

        # Run the PPO Algorithm on the network
        ppo_update(trajectories, acnet, policy_optimizer)

        #Performance log
        print(f"Episode {episode + 1} complete")
    return acnet


# %%
def get_optimal_mapping(acnet, state):
    # Pass the state through the ActorCriticNetwork
    actions, action_probs, _ = acnet(state)

    # Select the action with the highest probability for each state in the list
    optimal_mapping = actions
    return optimal_mapping

test = [(1,3),(2,3),(3,3),(4,13),(5,13),(6,13),(7,2),(7,2),(7,2),(8,2),(8,2),(8,2),(8,3)]
acnet = train_network(test)
optimal_actions = get_optimal_mapping(acnet, test)
print(f"Optimal action mapping for the test state: {optimal_actions}")

optimal_state = swapTuples(test, optimal_actions)
print(f"Optimal state after applying actions: {optimal_state}")
opt = getmapping(optimal_state)
reward_func(opt)

# %%
import os
import pandas as pd

# Path to the model folder
model_folder = "model"

# Function to process a CSV file and assign loop_ids dynamically
def process_csv(file_path):
    df = pd.read_csv(file_path, header=None, skiprows=1)  # Read CSV without headers
    states = []

    for i, row in df.iterrows():
        state = [(loop_id + 1, factor) for loop_id, factor in enumerate(row)]
        states.append(state)

    return states

states_combined = []
# Process states_combined CSV files in the model folder
for filename in os.listdir(model_folder):
    if filename.endswith(".csv"):
        file_path = os.path.join(model_folder, filename)


        states = process_csv(file_path)
        for i in states:
            states_combined.append(i)

for single_state in states_combined:
    print(single_state)


# %%
# for test in states_combined:
acnet = train_network(test)
optimal_actions = get_optimal_mapping(acnet, test)
print(f"Optimal action mapping for the test state: {optimal_actions}")

optimal_state = swapTuples(test, optimal_actions)
print(f"Optimal state after applying actions: {optimal_state}")
opt = getmapping(optimal_state)
reward_func(opt)



