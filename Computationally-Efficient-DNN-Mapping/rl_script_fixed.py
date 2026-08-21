"""
rl_script_fixed.py
==================
PyTorch implementation of the RL mapping search heuristic from:
  "Computationally Efficient DNN Mapping Search Heuristic using Deep RL"
  Bakshi & Johnsson, ACM TECS 2023.

Key paper details implemented:
  - Section 6.1: State = LPF 2-tuples list, normalized via assign_increasing_values
  - Section 6.3: Actor samples stochastically from softmax (NOT argmax)
  - Section 6.4.2: getmapping parses tuple-list left-to-right per Fig 8 pseudocode
  - Section 6.4.3: Reward = potential reuse of W and O at spatial+RF levels
  - Section 7.4.1: 10 fixed initial tuple-list orderings, one independent run each
  - Table 6: gamma=0.95, GAE lambda=0.99, lr=1e-3, clip=0.2,
             NtotalTraj=1024, NlearningTraj=512, Nbatch=256, Nepochs=3
  - Fig 10 Layer 12: W threshold=375, O threshold=216
  - Table 4: spatial<=256 PEs, RF<=672B, SharedMem<=108kB
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import pickle
from collections import defaultdict


# ── Utility ───────────────────────────────────────────────────────────────────

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
        if self.max_val == self.min_val:
            return [0.0 for _ in input_list]
        return [(x - self.min_val) / (self.max_val - self.min_val) for x in input_list]


# ── Memory / Mapping ──────────────────────────────────────────────────────────

def calculateMem(currlpf, op):
    if op == 'W':
        return currlpf[0]*currlpf[1]*currlpf[2]*currlpf[6]*currlpf[7]
    elif op == 'O':
        return currlpf[3]*currlpf[4]*currlpf[5]*currlpf[7]*currlpf[8]
    elif op == 'I':
        return ((currlpf[0]+currlpf[3]-1) *
                (currlpf[1]+currlpf[4]-1) *
                (currlpf[2]+currlpf[5]-1) *
                currlpf[6] * currlpf[8])
    return 0


def getmapping(state):
    """Fig 8: parse tuple-list left-to-right into valid 4-level mapping."""
    memmap  = [[], [], [], []]
    currlpf = [1]*9

    # Spatial: contiguous prefix, product <= 256 PEs
    sp_ids   = set()
    total_pe = 1
    for tup in state:
        loopid, lpf = tup
        if total_pe * lpf <= 256:
            total_pe         *= lpf
            currlpf[loopid-1] *= lpf
            memmap[0].append(tup)
            sp_ids.add(id(tup))
        else:
            break

    # RF: try each remaining; skip (don't break) if doesn't fit
    remaining = [t for t in state if id(t) not in sp_ids]
    rf_currlpf  = [1]*9
    rf_ids = set()
    for tup in remaining:
        loopid, lpf = tup
        test = rf_currlpf.copy()
        test[loopid-1] *= lpf
        if sum(calculateMem(test, op) for op in ['W','O','I']) <= 672:
            rf_currlpf[loopid-1]  *= lpf
            currlpf[loopid-1]     *= lpf
            memmap[1].append(tup)
            rf_ids.add(id(tup))

    # SharedMem: try each remaining; skip if doesn't fit
    remaining2 = [t for t in remaining if id(t) not in rf_ids]
    sm_ids = set()
    for tup in remaining2:
        loopid, lpf = tup
        test = currlpf.copy()
        test[loopid-1] *= lpf
        if sum(calculateMem(test, op) for op in ['W','O','I']) <= 108000:
            currlpf[loopid-1] *= lpf
            memmap[2].append(tup)
            sm_ids.add(id(tup))

    # DRAM: everything else
    placed = sp_ids | rf_ids | sm_ids
    for tup in state:
        if id(tup) not in placed:
            memmap[3].append(tup)

    return memmap


# ── Reward (Section 6.4.3, Fig 11) ───────────────────────────────────────────

LOOP_REV = {'W_iR': [3,4,5,8], 'O_iR': [0,1,2,6]}

def compute_reuse(memmap):
    w_sp, w_rf, o_sp, o_rf = 1, 1, 1, 1
    for tup in memmap[0]:
        if tup[0]-1 in LOOP_REV['W_iR']: w_sp *= tup[1]
        if tup[0]-1 in LOOP_REV['O_iR']: o_sp *= tup[1]
    for tup in memmap[1]:
        if tup[0]-1 in LOOP_REV['W_iR']: w_rf *= tup[1]
        if tup[0]-1 in LOOP_REV['O_iR']: o_rf *= tup[1]
    return w_sp, w_rf, o_sp, o_rf


def reward_func(memmap, wt_threshold=0, ot_threshold=0):
    norm_factor = 4096
    alpha       = 4.321
    clip        = 25
    w_sp, w_rf, o_sp, o_rf = compute_reuse(memmap)
    w, o = w_sp*w_rf, o_sp*o_rf
    if w < wt_threshold or o < ot_threshold:
        return 0.0
    return float(np.clip((w/norm_factor+1)**alpha + (o/norm_factor+1)**alpha, 0, clip))


# ── Actor-Critic Network (Section 6.3, Fig 6) ────────────────────────────────

class ActorCriticNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.actor_fc1    = nn.Linear(state_size, 20)
        self.actor_fc2    = nn.Linear(20, 20)
        self.actor_output = nn.Linear(20, action_size)
        self.critic_fc1   = nn.Linear(state_size, 20)
        self.critic_fc2   = nn.Linear(20, 20)
        self.critic_output= nn.Linear(20, 1)

    def forward(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        ax    = F.relu(self.actor_fc2(F.relu(self.actor_fc1(state))))
        logits= self.actor_output(ax)
        cx    = F.relu(self.critic_fc2(F.relu(self.critic_fc1(state))))
        value = self.critic_output(cx)
        nlpf  = logits.shape[1] // 3
        probs = F.softmax(logits.view(logits.shape[0], nlpf, 3), dim=-1)
        return probs, value

    def sample_action(self, state):
        """Section 6.3: STOCHASTIC sampling from softmax — essential for exploration."""
        probs, value = self.forward(state)
        probs_np = probs.squeeze(0).detach().numpy()
        actions, act_probs = [], []
        for group in probs_np:
            idx = np.random.choice(3, p=group)   # sample, not argmax
            actions.append(idx - 1)
            act_probs.append(float(group[idx]))
        return actions, act_probs, value


def swapTuples(state_tensor, action):
    arr = state_tensor.squeeze(0).clone().detach().numpy().copy()
    n   = len(action)
    for i in range(n):
        if action[i] == 1 and i < n-1:
            arr[i], arr[i+1] = arr[i+1], arr[i]
        elif action[i] == -1 and i > 0:
            arr[i], arr[i-1] = arr[i-1], arr[i]
    return torch.tensor(arr).unsqueeze(0).float()


def swapTupleList(tuples, action):
    t = list(tuples)
    n = len(action)
    for i in range(n):
        if action[i] == 1 and i < n-1:
            t[i], t[i+1] = t[i+1], t[i]
        elif action[i] == -1 and i > 0:
            t[i], t[i-1] = t[i-1], t[i]
    return t


# ── Trajectory collection ─────────────────────────────────────────────────────

def collect_trajectories(acnet, init_tensor, init_tuples,
                         wt_threshold, ot_threshold, max_steps=1024):
    traj   = {'states': [], 'actions': [], 'act_probs': [], 'rewards': []}
    pruned = {'tuples': [], 'memmaps': [], 'rewards': []}

    cur_state  = init_tensor.clone()
    cur_tuples = list(init_tuples)

    acnet.eval()
    with torch.no_grad():
        for _ in range(max_steps):
            action, act_probs, _ = acnet.sample_action(cur_state)
            nxt_state  = swapTuples(cur_state, action)
            nxt_tuples = swapTupleList(cur_tuples, action)
            memmap     = getmapping(nxt_tuples)
            reward     = reward_func(memmap)   # no threshold for training

            traj['states'].append(cur_state.squeeze(0).tolist())
            traj['actions'].append(list(action))
            traj['act_probs'].append(act_probs)
            traj['rewards'].append(reward)

            w_sp, w_rf, o_sp, o_rf = compute_reuse(memmap)
            if w_sp*w_rf >= wt_threshold and o_sp*o_rf >= ot_threshold:
                pruned['tuples'].append(list(nxt_tuples))
                pruned['memmaps'].append(memmap)
                pruned['rewards'].append(reward)

            cur_state  = nxt_state
            cur_tuples = nxt_tuples

    acnet.train()
    return traj, pruned


# ── PPO update (Table 6) ──────────────────────────────────────────────────────

def compute_advantages(states, rewards, acnet, gamma=0.95, lambda_=0.99):
    acnet.eval()
    advs = []
    adv  = 0.0
    with torch.no_grad():
        for t in reversed(range(len(states))):
            s   = torch.tensor(states[t], dtype=torch.float32)
            _, vt = acnet(s)
            vt  = vt.item()
            vn  = 0.0 if t == len(states)-1 else \
                  acnet(torch.tensor(states[t+1], dtype=torch.float32))[1].item()
            delta = rewards[t] + gamma*vn - vt
            adv   = delta + gamma*lambda_*adv
            advs.insert(0, adv)
    acnet.train()
    return advs


def ppo_update(traj, acnet, optimizer,
               gamma=0.95, lambda_=0.99, epsilon=0.2,
               c1=0.5, c2=0.01, learning_traj=512, batch_size=256, n_epochs=3):
    states    = traj['states']
    actions   = traj['actions']
    act_probs = traj['act_probs']
    rewards   = traj['rewards']
    n         = len(states)

    for cs in range(0, n, learning_traj):
        ce = min(cs+learning_traj, n)
        s_states = states[cs:ce]
        s_actions = actions[cs:ce]
        s_probs  = act_probs[cs:ce]
        s_rews   = rewards[cs:ce]

        advs     = compute_advantages(s_states, s_rews, acnet, gamma, lambda_)
        adv_t    = torch.tensor(advs, dtype=torch.float32)
        adv_norm = ((adv_t - adv_t.mean()) / (adv_t.std()+1e-8)).tolist()

        for bs in range(0, len(s_states), batch_size):
            be = min(bs+batch_size, len(s_states))
            for _ in range(n_epochs):
                for i in range(bs, be):
                    state     = torch.tensor(s_states[i], dtype=torch.float32)
                    action    = torch.tensor(s_actions[i], dtype=torch.long)
                    old_prob  = torch.tensor(s_probs[i],  dtype=torch.float32)
                    advantage = torch.tensor(adv_norm[i], dtype=torch.float32)

                    probs, value = acnet(state)
                    probs  = probs.squeeze(0)        # [nlpf, 3]
                    value  = value.squeeze()

                    action_idx = action + 1
                    curr_prob = probs[torch.arange(probs.shape[0]), action_idx]
                    ratio     = torch.exp(torch.sum(torch.log(curr_prob+1e-8)) -
                                          torch.sum(torch.log(old_prob+1e-8)))
                    s1 = ratio * advantage
                    s2 = torch.clamp(ratio, 1-epsilon, 1+epsilon) * advantage
                    loss_actor = -torch.min(s1, s2).mean()

                    vn = torch.tensor(0.0)
                    if i < len(s_states)-1:
                        with torch.no_grad():
                            _, vn = acnet(torch.tensor(s_states[i+1], dtype=torch.float32))
                        vn = vn.squeeze().detach()
                    target     = torch.tensor(float(s_rews[i]) + gamma*float(vn),
                                              dtype=torch.float32)
                    loss_critic = F.mse_loss(value, target)
                    entropy     = -(probs * torch.log(probs+1e-8)).sum(dim=-1).mean()

                    loss = loss_actor + c1*loss_critic - c2*entropy
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()


def train_network(initial_input_list, acnet=None, optimizer=None, episodes=30):

    loop_orderings = [
        ([8,7,6,5,4,3,2,1], False),
        ([1,2,3,4,5,6,7,8], False),
        ([8,7,6,5,4,3,2,1], True),
        ([1,2,3,4,5,6,7,8], True),
        ([1,2,3,4,5,6,8,7], False),
        ([1,2,3,4,5,8,6,7], False),
        ([1,2,3,4,8,5,6,7], False),
        ([1,2,3,4,5,7,6,8], False),
        ([1,2,3,4,7,5,6,8], False),
        ([1,2,3,7,4,5,6,8], False),
    ]

    wt_threshold = 375
    ot_threshold = 216

    # 👉 initialize ONLY if not provided
    if acnet is None:
        sample_tuples = build_init_tuples(initial_input_list, loop_orderings[0][0], False)
        mapped = assign_increasing_values(sample_tuples)
        acnet = ActorCriticNetwork(len(mapped), 3*len(mapped))
        optimizer = optim.Adam(acnet.parameters(), lr=1e-3)

    all_mappings = []

    for run_id, (loop_order, sort_asc) in enumerate(loop_orderings):

        init_tuples = build_init_tuples(initial_input_list, loop_order, sort_asc)

        run_mappings = train_one_run(
            initial_input_list, init_tuples,
            wt_threshold, ot_threshold,
            acnet, optimizer,
            episodes=episodes, run_id=run_id
        )

        all_mappings.extend(run_mappings)

    return acnet, optimizer, all_mappings


# ── Formatting ────────────────────────────────────────────────────────────────

LOOP_ID_TO_NAME = {1:'R',2:'S',3:'T',4:'P',5:'Q',6:'F',7:'C',8:'K',9:'N'}
ALL_DIMS        = ['R','S','T','P','Q','F','C','K','N']

def format_mapping_dict(memmap):
    level_map = {
        0: ('GlobalBuffer','spatial'),
        1: ('RegisterFile','temporal'),
        2: ('GlobalBuffer','temporal'),
        3: ('MainMemory',  'temporal'),
    }
    out = []
    for li, (target, mtype) in level_map.items():
        fd, perm = {}, []
        if li < len(memmap):
            for lid, fac in memmap[li]:
                nm = LOOP_ID_TO_NAME.get(lid, f'L{lid}')
                fd[nm] = fd.get(nm,1)*fac
                if nm not in perm: perm.append(nm)
        for d in ALL_DIMS:
            fd.setdefault(d,1)
            if d not in perm: perm.append(d)
        out.append({'target':target,'type':mtype,
                    'factors':' '.join(f"{d}={fd[d]}" for d in ALL_DIMS),
                    'permutation':''.join(perm)})
    return {'mapping': out}


# ── One RL run ────────────────────────────────────────────────────────────────

def build_init_tuples(initial_input_list, loop_order, sort_asc=False):
    pf = prime_factors_with_index(initial_input_list)
    lid_lpfs = defaultdict(list)
    for lid, lpf in pf: lid_lpfs[lid].append(lpf)
    tuples = []
    for lid in loop_order:
        lpfs = sorted(lid_lpfs[lid]) if sort_asc else lid_lpfs[lid]
        for lpf in lpfs: tuples.append((lid, lpf))
    return tuples


def train_one_run(initial_input_list, init_tuples, wt_threshold, ot_threshold,
                  acnet, optimizer,
                  episodes=30, run_id=0):
    mapped = assign_increasing_values(init_tuples)
    norm   = Normalizer().normalize(mapped)
    init_t = torch.tensor(norm, dtype=torch.float32).unsqueeze(0)


    all_mappings = []
    for ep in range(episodes):
        traj, pruned = collect_trajectories(
            acnet, init_t, init_tuples,
            wt_threshold=wt_threshold, ot_threshold=ot_threshold)

        ppo_update(traj, acnet, optimizer)

        for tuples, memmap, reward in zip(
                pruned['tuples'], pruned['memmaps'], pruned['rewards']):
            w_sp,w_rf,o_sp,o_rf = compute_reuse(memmap)
            all_mappings.append([
                reward, format_mapping_dict(memmap), [], tuples, memmap,
                ep, time.time(),
                {'W':[w_sp,w_rf,1,w_sp*w_rf],'O':[o_sp,o_rf,1,o_sp*o_rf]},
                1.0, 1
            ])

        print(f"  Run {run_id+1} Ep {ep+1}/{episodes}"
              f" | pruned={len(pruned['tuples'])}"
              f" | total={len(all_mappings)}")

    return all_mappings


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    INITIAL_INPUT_LIST = [3, 3, 3, 120, 120, 120, 16, 32]

    LOOP_ORDERINGS = [
        ([8,7,6,5,4,3,2,1], False),
        ([1,2,3,4,5,6,7,8], False),
        ([8,7,6,5,4,3,2,1], True),
        ([1,2,3,4,5,6,7,8], True),
        ([1,2,3,4,5,6,8,7], False),
        ([1,2,3,4,5,8,6,7], False),
        ([1,2,3,4,8,5,6,7], False),
        ([1,2,3,4,5,7,6,8], False),
        ([1,2,3,4,7,5,6,8], False),
        ([1,2,3,7,4,5,6,8], False),
    ]
    WT_THRESHOLD = 375
    OT_THRESHOLD = 216

    print("="*60)
    print("RL Mapping Search — 10 runs x 30 episodes (paper exact)")
    print("="*60)

    all_mappings = []
    for run_id, (loop_order, sort_asc) in enumerate(LOOP_ORDERINGS):
        init_tuples = build_init_tuples(INITIAL_INPUT_LIST, loop_order, sort_asc)
        print(f"\nRun {run_id+1}/10 | order={loop_order} sort_asc={sort_asc}")
        run_mappings = train_one_run(
            INITIAL_INPUT_LIST, init_tuples,
            WT_THRESHOLD, OT_THRESHOLD,
            episodes=30, run_id=run_id)
        all_mappings.extend(run_mappings)
        print(f"  Run {run_id+1} done | total: {len(all_mappings)}")

    with open("output_mappings.pkl", "wb") as f:
        pickle.dump(all_mappings, f)
    print(f"\nTotal mappings saved: {len(all_mappings)}")