import pickle
from rl_script_fixed import train_network, ppo_update
from rl_script_fixed import assign_increasing_values, Normalizer
from ga_from_rl import run_ga

# -------------------------------
# Utils
# -------------------------------

def tuples_to_state(tuples):
    mapped = assign_increasing_values(tuples)
    norm = Normalizer().normalize(mapped)
    return norm


def create_feedback_traj(elite_set):
    traj = {
        'states': [],
        'actions': [],
        'act_probs': [],
        'rewards': []
    }

    for edp, tuples in elite_set:
        state = tuples_to_state(tuples)

        action = [0]*len(state)
        prob = [1.0]*len(state)

        reward = 1e7 / edp

        traj['states'].append(state)
        traj['actions'].append(action)
        traj['act_probs'].append(prob)
        traj['rewards'].append(reward)

    return traj


# -------------------------------
# MAIN LOOP
# -------------------------------

if __name__ == "__main__":

    arch_yaml = open("arch/arch.yaml").read()
    prob_yaml = open("prob/prob_3d_12.yaml").read()

    INITIAL_INPUT_LIST = [3,3,3,120,120,120,16,32]

    # Step 0: initial RL run
    print("Running initial RL...")
    acnet, optimizer, mappings = train_network(INITIAL_INPUT_LIST)

    best_ga = []
    for iteration in range(3):

        print(f"\n===== ITERATION {iteration+1} =====")

        # 1. Select top RL mappings
        top = sorted(mappings, key=lambda x: x[0], reverse=True)[:20]
        rl_population = [item[3] for item in top]

        # merge with previous GA elites if exist
        if iteration == 0:
            population = rl_population
        else:
            population = rl_population + best_ga

        # 2. Run GA
        elite = run_ga(population, arch_yaml, prob_yaml)

        best_ga = [t for _, t in elite]

        # 3. Feedback to RL
        feedback_traj = create_feedback_traj(elite)

        # For now just print
        print("Top GA EDPs:", [e[0] for e in elite])

        print("Applying PPO feedback...")
        for _ in range(4):   # stronger signal
            ppo_update(feedback_traj, acnet, optimizer)


        # 4. Re-run RL (fresh)
        _, _, mappings = train_network(INITIAL_INPUT_LIST, acnet, optimizer, episodes=4)  # MUST be small

    print("\nDONE")