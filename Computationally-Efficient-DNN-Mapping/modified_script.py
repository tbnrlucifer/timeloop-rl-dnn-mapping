    import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import time
import pickle


# ── Core functions (imported by generate_random_mappings.py) ─────────────────

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
    result = []
    for ind in range(len(num_list)):
        for factor in prime_factors(num_list[ind]):
            result.append((ind + 1, factor))
    return result


def assign_increasing_values(tuple_list):
    unique_map = {}
    current_value = 1
    result = []
    for tup in tuple_list:
        if tup not in unique_map:
            unique_map[tup] = current_value
            current_value += 1
        result.append(unique_map[tup])
    return result


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


def calculateMem(currlpf, op):
    mem_rqd = 0
    if op == 'W':
        mem_rqd = currlpf[0] * currlpf[1] * currlpf[2] * currlpf[6] * currlpf[7]
    elif op == 'O':
        mem_rqd = currlpf[3] * currlpf[4] * currlpf[5] * currlpf[7] * currlpf[8]
    elif op == 'I':
        mem_rqd = (currlpf[0] + currlpf[3] - 1) * (currlpf[1] + currlpf[4] - 1) * \
                  (currlpf[2] + currlpf[5] - 1) * currlpf[6] * currlpf[8]
    return mem_rqd


def getmapping(state):
    memmap = [[], [], [], []]
    currlpf = [1, 1, 1, 1, 1, 1, 1, 1, 1]

    # Spatial level
    totalmemused = 1
    for tup in state:
        loopid, lpf = tup[0], tup[1]
        totalmemused *= lpf
        if totalmemused <= 256:
            memmap[0].append(tup)
            currlpf[loopid - 1] *= lpf
        else:
            totalmemused //= lpf  # undo
            break  # spatial must be contiguous — keep break here

    # RF level
    rf_states = [e for e in state if e not in memmap[0]]
    rf_currlpf = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    for tup in rf_states:
        loopid, lpf = tup[0], tup[1]
        temp_currlpf = rf_currlpf.copy()
        temp_currlpf[loopid - 1] *= lpf
        mem_rqd = sum(calculateMem(temp_currlpf, op) for op in ['W', 'O', 'I'])
        if mem_rqd <= 672:
            memmap[1].append(tup)
            currlpf[loopid - 1] *= lpf
            rf_currlpf[loopid - 1] *= lpf
        # no break — continue trying remaining factors

    # GlobalBuffer level
    sharedmem_states = [e for e in rf_states if e not in memmap[1]]
    for tup in sharedmem_states:
        loopid, lpf = tup[0], tup[1]
        temp_currlpf = currlpf.copy()
        temp_currlpf[loopid - 1] *= lpf
        mem_rqd = sum(calculateMem(temp_currlpf, op) for op in ['W', 'I', 'O'])
        if mem_rqd <= 108000:
            memmap[2].append(tup)
            currlpf[loopid - 1] *= lpf
        # no break

    # DRAM — catch ALL remaining factors unconditionally
    placed = set(id(x) for level in memmap[:3] for x in level)
    for tup in state:
        if id(tup) not in placed:
            memmap[3].append(tup)

    return memmap


def reward_func(mapping):
    norm_factor = 4096
    clipping_factor = 25
    alpha = 4.321
    loop_rev = {'W_iR': [3, 4, 5, 8], 'O_iR': [0, 1, 2, 6]}

    w_sp, w_rf, o_sp, o_rf = 1, 1, 1, 1
    for tup in mapping[0]:
        if tup[0] - 1 in loop_rev['W_iR']: w_sp *= tup[1]
        if tup[0] - 1 in loop_rev['O_iR']: o_sp *= tup[1]
    for tup in mapping[1]:
        if tup[0] - 1 in loop_rev['W_iR']: w_rf *= tup[1]
        if tup[0] - 1 in loop_rev['O_iR']: o_rf *= tup[1]

    if w_sp * w_rf < 0 or o_sp * o_rf < 0:
        return 0
    reward = ((w_sp * w_rf / norm_factor + 1) ** alpha +
              (o_sp * o_rf / norm_factor + 1) ** alpha)
    return float(np.clip(reward, 0, clipping_factor))


LOOP_ID_TO_NAME = {1: 'R', 2: 'S', 3: 'T', 4: 'P', 5: 'Q', 6: 'F', 7: 'C', 8: 'K', 9: 'N'}


def format_mapping_dict(memmap):
    level_map = {
        0: ('GlobalBuffer', 'spatial'),
        1: ('RegisterFile', 'temporal'),
        2: ('GlobalBuffer', 'temporal'),
        3: ('MainMemory',   'temporal'),
    }
    output_mapping = []
    for level_idx, (target, mtype) in level_map.items():
        factors_dict = {}
        permutation_list = []
        if level_idx < len(memmap):
            for loop_id, factor in memmap[level_idx]:
                name = LOOP_ID_TO_NAME.get(loop_id, f'L{loop_id}')
                factors_dict[name] = factors_dict.get(name, 1) * factor
                if name not in permutation_list:
                    permutation_list.append(name)
        factors_str = ' '.join(f"{n}={v}" for n, v in factors_dict.items())
        permutation_str = ''.join(permutation_list)
        output_mapping.append({
            'target': target,
            'type': mtype,
            'factors': factors_str,
            'permutation': permutation_str,
        })
    return {'mapping': output_mapping}


# ── RL Model ──────────────────────────────────────────────────────────────────

class ActorCriticNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCriticNetwork, self).__init__()
        self.actor_fc1    = nn.Linear(state_size, 20)
        self.actor_relu1  = nn.ReLU()
        self.actor_fc2    = nn.Linear(20, 20)
        self.actor_relu2  = nn.ReLU()
        self.actor_output = nn.Linear(20, action_size)
        self.critic_fc1    = nn.Linear(state_size, 20)
        self.critic_relu1  = nn.ReLU()
        self.critic_fc2    = nn.Linear(20, 20)
        self.critic_relu2  = nn.ReLU()
        self.critic_output = nn.Linear(20, 1)

    def getParsedStateTensor(self, input_state):
        state = torch.tensor(np.array(input_state)).float()
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return state

    def forward(self, state):
        actor_x = self.actor_relu2(self.actor_fc2(self.actor_relu1(self.actor_fc1(state))))
        actor_output = self.actor_output(actor_x).detach().numpy()

        critic_x = self.critic_relu2(self.critic_fc2(self.critic_relu1(self.critic_fc1(state))))
        critic_output = self.critic_output(critic_x)

        if actor_output.shape[1] % 3 != 0:
            raise ValueError(f"Actor output size ({actor_output.shape[1]}) must be divisible by 3")

        num_groups = actor_output.shape[1] // 3
        grouped_output = actor_output.reshape((actor_output.shape[0], num_groups, 3))
        e_x = np.exp(grouped_output - np.max(grouped_output, axis=-1, keepdims=True))
        softmaxed_groups = e_x / e_x.sum(axis=-1, keepdims=True)

        actions, act_probs = [], []
        for batch in softmaxed_groups:
            batch_actions, batch_probs = [], []
            for group in batch:
                batch_actions.append(np.argmax(group) - 1)
                batch_probs.append(np.max(group))
            actions.append(batch_actions)
            act_probs.append(batch_probs)

        return actions[0], act_probs[0], critic_output

    def get_trainable(self):
        return list(self.parameters())


def swapTuples(state, action):
    state_array = state.squeeze(0).clone().detach().numpy()
    n = len(action)
    for i in range(n):
        if action[i] == 1 and i != n - 1:
            state_array[i], state_array[i + 1] = state_array[i + 1], state_array[i]
        elif action[i] == -1 and i != 0:
            state_array[i], state_array[i - 1] = state_array[i - 1], state_array[i]
    return torch.tensor(state_array).unsqueeze(0).float()


def ppo_update(trajectories, acnet, optimizer, gamma=0.99, lambda_=0.95,
               epsilon=0.2, c1=0.5, c2=0.01):
    for trajectory in trajectories:
        states       = trajectory['states']
        actions      = trajectory['actions']
        old_act_probs = trajectory['act_probs']
        rewards      = trajectory['rewards']

        for i, (state, action, old_act_prob) in enumerate(
                zip(states, actions, old_act_probs)):
            state        = torch.tensor(state, dtype=torch.float32)
            action       = torch.tensor(action, dtype=torch.long)
            old_act_prob = torch.tensor(old_act_prob, dtype=torch.float32)

            _, action_probs_np, value_np = acnet(state)
            action_probs = torch.as_tensor(action_probs_np).float().clone()
            value        = torch.as_tensor(value_np).float().clone()

            log_prob_current = torch.log(action_probs + 1e-8)
            ratio = torch.exp(log_prob_current - torch.log(old_act_prob + 1e-8))

            advantage = torch.tensor(rewards[i], dtype=torch.float32) - value.detach()
            surrogate1 = ratio * advantage
            surrogate2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage
            loss_actor = -torch.min(surrogate1, surrogate2).mean()

            next_value = 0 if i == len(states) - 1 else \
                acnet(torch.tensor(states[i + 1], dtype=torch.float32))[2].detach()
            discounted_return = rewards[i] + gamma * next_value
            loss_critic  = F.mse_loss(value, torch.tensor([[discounted_return]], dtype=torch.float32))
            entropy_loss = -torch.sum(action_probs * torch.log(action_probs + 1e-8), dim=-1).mean()
            total_loss   = loss_actor + c1 * loss_critic - c2 * entropy_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()


def train_network(initial_input_list, episodes=20):
    prime_factors_indexed = prime_factors_with_index(initial_input_list)
    mapped_values = assign_increasing_values(prime_factors_indexed)
    state_size = len(mapped_values)
    action_size = 3 * state_size

    acnet = ActorCriticNetwork(state_size, action_size)
    policy_optimizer = optim.Adam(acnet.parameters(), 0.001)

    for episode in range(episodes):
        normalizer = Normalizer()
        normalized_state = normalizer.normalize(mapped_values)
        current_state = torch.tensor(normalized_state).float().unsqueeze(0)

        trajectory = {'states': [], 'actions': [], 'act_probs': [], 'rewards': []}

        for t in range(1024):
            state_tensor = acnet.getParsedStateTensor(current_state)
            action, action_probs, _ = acnet.forward(state_tensor)
            next_state_tensor = swapTuples(state_tensor, action)
            next_state_list = next_state_tensor.squeeze(0).tolist()
            mapping = getmapping(next_state_list)
            reward = reward_func(mapping)

            if reward > 0:
                trajectory['states'].append(next_state_list)
                trajectory['actions'].append(action)
                trajectory['act_probs'].append(action_probs)
                trajectory['rewards'].append(reward)

            current_state = next_state_tensor

        if trajectory['states']:
            ppo_update([trajectory], acnet, policy_optimizer)

        print(f"Episode {episode + 1} complete")

    return acnet


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yaml

    initial_input_list = [3, 3, 3, 120, 120, 120, 16, 32]
    print("Training network...")
    acnet = train_network(initial_input_list, episodes=5)

    # Test a mapping
    state = prime_factors_with_index(initial_input_list)
    mapping = getmapping(state)
    print("Sample mapping:", format_mapping_dict(mapping))
    print("Reward:", reward_func(mapping))