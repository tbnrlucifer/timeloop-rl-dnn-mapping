# %% [markdown]
# using different limits in loops will first form the state rep.

# %%
# !pip install pip
# !pip install torch
import numpy as np

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


# %% [markdown]
# Calculating Potential Reuse of Weight and Output
# 

# %%
# weight_potential_reuse_spatial = 1
# loop_rev = {'W_R':[0,1,2,6,7], 'W_iR':[3,4,5,8], 'O_R':[3,4,5,7,8], 'O_iR':[0,1,2,6]}
# mapping = [[(3, 3), (7, 2), (8, 3), (4, 13)], [(5, 13), (7, 6), (8, 4), (1, 3), (2, 3)], [(6, 13), (8, 2)], [(7, 16), (8, 8)]]
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

# %%
ini_state = [(3,3), (7,2), (8,3),(4,13),(5,13),(7,6),(8,4),(1,3),(2,3),(6,13),(8,2),(7,16),(8,8)]
single_state = [(1,3),(2,3),(3,3),(4,13),(5,13),(6,13),(7,2),(7,2),(7,2),(8,2),(8,2),(8,2),(8,3)]

# %% [markdown]
# 

# %%
import copy

def collect_trajectories(acnet, num_trajectories=20, max_steps_per_trajectory=20):
    trajectories = []

    for _ in range(num_trajectories):
        trajectory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'act_probs': []
        }
        state = ini_state
        # done = False
        total_reward = 0

        for _ in range(max_steps_per_trajectory):

            action,act_prob,_ =  acnet(state)
            next_state = swapTuples(state,action)
            next_mapping = getmapping(next_state)
            reward = reward_func(next_mapping)

            # Store transition in trajectory
            trajectory['states'].append(next_state)
            trajectory['actions'].append(action)
            trajectory['rewards'].append(reward)
            trajectory['act_probs'].append(act_prob)

            total_reward += reward
            state = copy.copy(next_state)


        trajectories.append(trajectory)

    return trajectories

acnet = ActorCriticNetwork(13,39)
traj = collect_trajectories(acnet)
print(traj)


# %%
def compute_advantages(trajectories, acnet, gamma=0.99, lambda_=0.95):
    advantages = []

    for trajectory in trajectories:
        states = trajectory['states']
        rewards = trajectory['rewards']
        # done = trajectory['done']

        # Compute discounted future returns using the value network
        future_returns = []
        next_value = 0  # Value of terminal state is 0
        critic_value = 0
        for t in reversed(range(len(states))):
            if t < len(states) - 1:
                _,_,next_value = acnet(states[t + 1])
                _,_,critic_value = acnet(states[t])
            delta = rewards[t] + gamma * next_value - critic_value
            next_value = critic_value  # Update next value for next iteration
            future_returns.insert(0, delta)

        # Compute advantages using generalized advantage estimation (GAE)
        advantages_t = []
        advantage = 0
        for delta in future_returns:
            advantage = delta + gamma * lambda_ * advantage
            advantages_t.append(advantage.detach().numpy())
        advantages.extend(advantages_t)

    return advantages

adv = compute_advantages(traj,acnet)
print(adv)

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
    


