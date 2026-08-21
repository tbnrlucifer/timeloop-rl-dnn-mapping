import pickle
import random
import subprocess
import tempfile
import os
import yaml

TIMELOOP_BIN = "/home/ishaans/timeloop/build/timeloop-model"
TIMELOOP_LIB = "/home/ishaans/timeloop/build"

ALL_DIMS = ['R','S','T','P','Q','F','C','K','N']



# ─────────────────────────────────────────────
# Mapping utils (reuse your logic)
# ─────────────────────────────────────────────

def format_mapping(memmap):
    from rl_script_fixed import format_mapping_dict
    return format_mapping_dict(memmap)


def run_timeloop(mapping_dict, arch_yaml, prob_yaml):
    def fix_mapping(map_dict):
        fixed = []
        for entry in map_dict['mapping']:
            fd = {}
            for tok in entry['factors'].split():
                if '=' in tok:
                    d,v = tok.split('=')
                    fd[d]=int(v)
            for d in ALL_DIMS:
                fd.setdefault(d,1)

            perm = list(entry['permutation'])
            for d in ALL_DIMS:
                if d not in perm:
                    perm.append(d)

            fixed.append({
                'target':entry['target'],
                'type':entry['type'],
                'factors':' '.join(f"{d}={fd[d]}" for d in ALL_DIMS),
                'permutation':''.join(perm)
            })
        return {'mapping': fixed}

    map_yaml = yaml.dump(fix_mapping(mapping_dict))

    with tempfile.TemporaryDirectory() as d:
        open(f"{d}/arch.yaml","w").write(arch_yaml)
        open(f"{d}/prob.yaml","w").write(prob_yaml)
        open(f"{d}/map.yaml","w").write(map_yaml)

        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = TIMELOOP_LIB + ":" + env.get("LD_LIBRARY_PATH","")

        r = subprocess.run([
            TIMELOOP_BIN,
            f"{d}/arch.yaml",
            f"{d}/prob.yaml",
            f"{d}/map.yaml"
        ], cwd=d, capture_output=True, text=True, env=env)

        if r.returncode != 0:
            return None

        energy, cycles = None, None
        for line in open(f"{d}/timeloop-model.stats.txt"):
            if line.startswith("Energy:") and "uJ" in line:
                energy=float(line.split(":")[1].strip().split()[0])*1e-6
            if line.startswith("Cycles:"):
                cycles=float(line.split(":")[1].strip())

        if energy and cycles:
            return energy * cycles

    return None


# ─────────────────────────────────────────────
# GA operators
# ─────────────────────────────────────────────

def mutate(tuples):
    t = list(tuples)
    i, j = random.sample(range(len(t)), 2)
    t[i], t[j] = t[j], t[i]
    return t


def crossover(p1, p2):
    n = len(p1)
    cut = random.randint(1, n-2)

    child = p1[:cut]

    for x in p2:
        if x not in child:
            child.append(x)

    return child


# ─────────────────────────────────────────────
# GA main
# ─────────────────────────────────────────────

def run_ga(population, arch_yaml, prob_yaml, generations=5, pop_size=20):

    from rl_script_fixed import getmapping

    for gen in range(generations):
        print(f"\n=== Generation {gen+1} ===")

        scored = []

        for tuples in population:
            memmap = getmapping(tuples)
            mapping_dict = format_mapping(memmap)

            edp = run_timeloop(mapping_dict, arch_yaml, prob_yaml)

            if edp is not None:
                scored.append((edp, tuples))

        scored.sort(key=lambda x: x[0])

        if not scored:
            print("All failed")
            return None

        best_edp = scored[0][0]
        print(f"Best EDP: {best_edp:.4e}")

        # Keep best solutions (elitism)
        elite = [t for _, t in scored[:5]]   # keep top 5

        # Selection
        top = [t for _, t in scored[:pop_size//2]]

        # Reproduce
        new_pop = elite.copy()

        while len(new_pop) < pop_size:
            if random.random() < 0.5:
                p1, p2 = random.sample(top, 2)
                child = crossover(p1, p2)
            else:
                p = random.choice(top)
                child = mutate(p)

            new_pop.append(child)

        population = new_pop

    return scored[:5]  # top 5 instead of best


# ─────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # Load RL results
    data = pickle.load(open("output_mappings.pkl","rb"))

    # Top-K from RL
    top = sorted(data, key=lambda x: x[0], reverse=True)[:20]

    initial_population = [item[3] for item in top]  # tuples

    arch_yaml = open("arch/arch.yaml").read()
    prob_yaml = open("prob/prob_3d_12.yaml").read()

    best = run_ga(initial_population, arch_yaml, prob_yaml)

    print("\nFINAL BEST:", best)