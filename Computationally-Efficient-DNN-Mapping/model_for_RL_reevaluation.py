# """
# Example of using the class `pytimeloop.app.Model`. This is the default way for
# implementing functionality of the `timeloop-model` application.
# """
# # import pytimeloop.app.model as tlmodel
# from pytimeloop import Config

# # import tensorflow.math as tf

# import glob
# import logging
# import sys
# import os

# import pickle
# import time
# import yaml
# import numpy as np
# from multiprocessing import Pool, Lock
# from copy import deepcopy

# from bindings.mapping import ArchProperties
# from pytimeloop.config import Config
# from pytimeloop.engine import Accelerator
# from pytimeloop.model import ArchSpecs, SparseOptimizationInfo
# from pytimeloop.mapping import ArchConstraints, Mapping
# from pytimeloop.problem import Workload

# class myModelApp:
#     def __init__(self, cfg: Config, out_dir: str, auto_bypass_on_failure=False,
#                  out_prefix='', log_level=logging.INFO):
#         # Setup logger
#         self.log_level = log_level
#         self.model_logger = logging.getLogger('pytimeloop.app.Model')
#         self.model_logger.setLevel(10000)

#         # timeloop-model configurations
#         self.auto_bypass_on_failure = auto_bypass_on_failure
#         self.out_prefix = out_prefix
#         semi_qualified_prefix = 'timeloop-model'
#         self.out_prefix = out_dir + '/' + semi_qualified_prefix

#         # Architecture configuration
#         self.arch_specs = ArchSpecs(cfg['architecture'])
#         self.arch_specs.generate_tables(
#             cfg, semi_qualified_prefix, out_dir, self.out_prefix, log_level)

#         # Problem configuration
#         self.workload = Workload(cfg['problem'])
#         self.model_logger.info('Problem configuration complete.')

#         self.arch_props = ArchProperties(self.arch_specs)

#         # Architecture constraints
#         self.constraints = ArchConstraints(
#             self.arch_props, self.workload, cfg['architecture_constraints'])
#         self.model_logger.info('Architecture configuration complete.')

#         # Mapping configuration
#         self.mapping = None
#         self.model_logger.info('Mapping construction complete.')

#         # Sparse optimizations
#         if 'sparse_optimizations' in cfg:
#             sparse_opt_cfg = cfg['sparse_optimizations']
#         else:
#             sparse_opt_cfg = Config()
#         self.sparse_optimizations = SparseOptimizationInfo(
#             sparse_opt_cfg, self.arch_specs)

#     def run(self, cfg):
#         self.mapping = Mapping(cfg['mapping'], self.arch_specs, self.workload)
#         if not self.constraints.satisfied_by(self.mapping):
#             self.model_logger.error(
#                 'Mapping violates architecture constraints.')
#             raise ValueError('Mapping violates architecture constraints.')

#         engine = Accelerator(self.arch_specs)

#         eval_stat = engine.evaluate(self.mapping,
#                                     self.workload,
#                                     self.sparse_optimizations)
#         return eval_stat


# def evaluate(mapping_list_chunk, input_yaml_str, pid):
#     # print("Proc ID:",pid)

#     mappings_evaluated = 0
#     results = []

#     # best_mapping_reward = 0
#     # best_mapping_dict = 0
#     # best_mapping = 0
#     # best_mapping_lpfs = 0
#     # best_mapping_merged = 0
#     # best_mapping_episode = 0
#     # best_time_seconds =  0
#     # best_mapping_reuse = None
#     # best_mapping_cost = 0
#     # best_mapping_utilization = 0
#     # best_mapping_latency = 0
#     # best_mapping_algorithmic_computes = 0
#     # best_mapping_edp = -1


#     config = Config.load_yaml(input_yaml_str)
#     lock.acquire()
#     app = myModelApp(config, 'out_'+str(pid), out_prefix=(str(pid)+"_"))
#     lock.release()

#     map_yaml_str = yaml.dump({
#       "mapping": mapping_list_chunk[-1][1]
#       })

#     eval_mapping_str = input_yaml_str + map_yaml_str
#     config = Config.load_yaml(eval_mapping_str)
#     eval_stats = app.run(config)

#     best_mapping_reward = mapping_list_chunk[-1][0]
#     best_mapping_dict = mapping_list_chunk[-1][1]
#     best_mapping = mapping_list_chunk[-1][2]
#     best_mapping_lpfs = mapping_list_chunk[-1][3]
#     best_mapping_merged = mapping_list_chunk[-1][4]
#     best_mapping_episode = mapping_list_chunk[-1][5]
#     best_time_seconds =  mapping_list_chunk[-1][6]
#     best_mapping_reuse = mapping_list_chunk[-1][7]
#     best_mapping_cost =  eval_stats.energy
#     best_mapping_utilization = eval_stats.utilization
#     best_mapping_latency = eval_stats.cycles
#     best_mapping_algorithmic_computes = eval_stats.algorithmic_computes
#     best_mapping_edp = best_mapping_cost * best_mapping_latency

#     for mapping_list_item in mapping_list_chunk:
    
#         mappings_evaluated += 1
#         # if pid==0:
#         #     print("mappings_evaluated:",mappings_evaluated)
#         map_yaml_str = yaml.dump(mapping_list_item[1])
#         eval_mapping_str = input_yaml_str + map_yaml_str

#         config = Config.load_yaml(eval_mapping_str)

#         eval_stats = app.run(config)

#         mapping_cost = eval_stats.energy    # Energy
#         latency = eval_stats.cycles    # Cycles
#         utilization = eval_stats.utilization      #PE utilization
#         edp = mapping_cost*latency

#         reward = mapping_list_item[0]

#         if edp < best_mapping_edp:
#         # if (mapping_cost) < (best_mapping_cost):
#             # print("new best mapping found. EDP:",edp)
#             best_mapping_reward = reward
#             best_mapping_dict = mapping_list_item[1]
#             best_mapping = mapping_list_item[2]
#             best_mapping_lpfs = mapping_list_item[3]
#             best_mapping_merged = mapping_list_item[4]
#             best_mapping_episode = mapping_list_item[5]
#             best_time_seconds =  mapping_list_item[6]
#             best_mapping_reuse = mapping_list_item[7]
#             best_mapping_cost = mapping_cost
#             best_mapping_utilization = utilization
#             best_mapping_latency = latency
#             best_mapping_algorithmic_computes = eval_stats.algorithmic_computes
#             best_mapping_edp = edp

#         # print(eval_stats.energy,"|",eval_stats.cycles,"|",eval_stats.utilization,"|",eval_stats.actual_computes,"|",eval_stats.algorithmic_computes)
#     # print("pid:",str(pid),"\tbest EDP:",best_mapping_edp)
#     # print("Proc ID:",pid,"\tMappings evaluated:",mappings_evaluated)
#     return list([mappings_evaluated, best_mapping_reward, best_mapping_dict, best_mapping, best_mapping_lpfs, best_mapping_merged, 
#                 best_mapping_episode, best_time_seconds, best_mapping_reuse, best_mapping_cost, best_mapping_utilization, 
#                 best_mapping_latency, best_mapping_algorithmic_computes, best_mapping_edp])

# def init_pool_processes(lock):
#     global mlock
#     mlock = lock

# if __name__ == '__main__':

#     if len(sys.argv) < 2:
#         print("Possible usages:\n")
#         print("1)\t", sys.argv[0], " <nn_layer_number>")
#         print("2)\t", sys.argv[0], " <nn_layer_number> <weight_reuse_threshold> <output_reuse_threshold> <init_guess_order> <use_sorted_pf>")
#         sys.exit(0)

#     nn_layer_number = int(sys.argv[1])
#     prob_file = "config/prob/prob_3d_" + str(nn_layer_number) + ".yaml"

#     arch_file = "config/arch/arch.yaml"
    
#     mapping_dir = "config/map/"
#     mapping_file_list = []

#     #if we want to evaluate mappings for a specific reuse/initial_guess/sorted case
#     if len(sys.argv) == 6:
#         weight_reuse_threshold = int(sys.argv[2])
#         output_reuse_threshold = int(sys.argv[3])
#         init_guess_order = int(sys.argv[4])
#         use_sorted_pf = int(sys.argv[5])

#         mapping_file_list.append(os.path.join(mapping_dir, 
#                             "layer_" +              str(nn_layer_number) +
#                             "_wtThres_" +           str(weight_reuse_threshold) +
#                             "_otThres_" +           str(output_reuse_threshold) +
#                             "_initGuessOrder_" +    str(init_guess_order) + 
#                             "_sorted_" +            str(use_sorted_pf) +
#                             "_map.pickle"))
#     else:
#         for file in os.listdir(mapping_dir):
#             if ("layer_" + str(nn_layer_number) + "_") in file:
#                 mapping_file_list.append(os.path.join(mapping_dir, file))

#     # Read the architecture
#     with open(arch_file,'r') as archfile:
#         arch_yaml_str = yaml.dump(yaml.safe_load(archfile))
#     # print("arch_yaml_str:",arch_yaml_str)

#     # Read the problem
#     with open(prob_file,'r') as probfile:
#         prob_yaml_str = yaml.dump(yaml.safe_load(probfile))
#     # print("prob_yaml_str:",prob_yaml_str)

#     # Preconcat the arch and prob str as they will be reused
#     input_yaml_str = arch_yaml_str + "\n" + prob_yaml_str


#     for mapping_file in mapping_file_list:

#         print("Mapping File:", mapping_file)

#         with open(mapping_file, 'rb') as mapfile:
#             mapping_list = (pickle.load(mapfile))
#         print("Mappings to be evaluated: ",len(mapping_list))

#         num_parallel_procs = 24
#         mdiv = int(len(mapping_list)/num_parallel_procs)
#         mapping_idx_chunk = [[i*mdiv,i*mdiv+mdiv-1] for i in range(num_parallel_procs)]
#         if len(mapping_list) % num_parallel_procs != 0:
#             mapping_idx_chunk[-1][1] = mapping_idx_chunk[-1][1]+len(mapping_list)%num_parallel_procs

#         lock = Lock()

#         start_time =  time.perf_counter()
#         pool = Pool(processes=num_parallel_procs, initializer=init_pool_processes, initargs=(lock,))
#         results = pool.starmap(evaluate, [[mapping_list[mapping_idx_chunk[i][0]:mapping_idx_chunk[i][1]+1]] + [input_yaml_str, i] for i in range(num_parallel_procs)])

#         time_seconds =  time.perf_counter() - start_time
#         print("\nTime taken to evaluate mappings:",time_seconds,"s")

#         best_mapping_edp = results[-1][-1]
#         best_result = results[-1]

#         total_mappings_evaluated = 0

#         for item in results:
#             # print("Item:",item)
#             total_mappings_evaluated += item[0]
#             edp = item[-1]
#             # print("EDP:",edp)
#             if edp < best_mapping_edp:
#                 # print("\tBest EDP:",edp)
#             #     print("\treuse:",item[8])
#                 best_result = item
#                 best_mapping_edp = edp

#         # print("\n\nBest result:", best_result)
#         print("Total Mappings Evaluated:", total_mappings_evaluated,"\n")

#         best_mapping_reward = best_result[1]
#         best_mapping_episode = best_result[6]
#         best_mapping_weight_reuse = best_result[8]['W'][0]*best_result[8]['W'][1]
#         best_mapping_output_reuse = best_result[8]['O'][0]*best_result[8]['O'][1]
#         best_mapping_cost = best_result[9]
#         best_mapping_utilization = best_result[10]
#         best_mapping_latency = best_result[11]
#         best_time_seconds = best_result[7]
#         best_mapping_dict = best_result[2]
#         best_mapping_merged = best_result[5]
#         best_mapping_algorithmic_computes = best_result[12]

#         print("\n\n{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
#                     best_mapping_reward, \
#                     best_mapping_episode,   \
#                     best_mapping_weight_reuse, \
#                     best_mapping_output_reuse, \
#                     best_mapping_cost, \
#                     best_mapping_utilization, \
#                     best_mapping_latency, \
#                     best_time_seconds, \
#                     best_mapping_algorithmic_computes, \
#                     best_mapping_dict, \
#                     best_mapping_merged))

#         print("\n*************************\n")


import subprocess
import tempfile
import os
import yaml
import pickle
import sys
import time
import re


TIMELLOOP_BIN = "/home/ishaans/timeloop/build/timeloop-model"
TIMELLOOP_LIB = "/home/ishaans/timeloop/build"


def run_timeloop(arch_yaml, prob_yaml, map_yaml):
    with tempfile.TemporaryDirectory() as tmpdir:

        arch_file = os.path.join(tmpdir, "arch.yaml")
        prob_file = os.path.join(tmpdir, "prob.yaml")
        map_file  = os.path.join(tmpdir, "map.yaml")

        with open(arch_file, "w") as f:
            f.write(arch_yaml)

        with open(prob_file, "w") as f:
            f.write(prob_yaml)

        with open(map_file, "w") as f:
            f.write(map_yaml)

        cmd = [TIMELLOOP_BIN, arch_file, prob_file, map_file]

        # Inject LD_LIBRARY_PATH explicitly
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = TIMELLOOP_LIB + ":" + env.get("LD_LIBRARY_PATH", "")

        result = subprocess.run(
            cmd,
            cwd=tmpdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env              # <-- critical fix
        )

        if result.returncode != 0:
            print("Timeloop failed:")
            print(result.stderr)
            return None

        stats_file = os.path.join(tmpdir, "timeloop-model.stats.txt")

        if not os.path.exists(stats_file):
            print("Stats file not found")
            return None

        with open(stats_file) as f:
            return f.read()


def parse_stats(stats_text):

    energy_j = None
    cycles = None

    for line in stats_text.splitlines():

        line = line.strip()

        # Parse summary energy (uJ)
        if line.startswith("Energy:") and "uJ" in line:
            energy_uj = float(line.split(":")[1].strip().split()[0])
            energy_j = energy_uj * 1e-6   # convert uJ → J

        # Parse summary cycles
        if line.startswith("Cycles:"):
            cycles = float(line.split(":")[1].strip())

    if energy_j is None or cycles is None:
        return None, None

    return energy_j, cycles


def evaluate(mapping_list):

    mappings_evaluated = 0
    best_edp = float("inf")
    best_mapping_yaml = None

    for mapping_list_item in mapping_list:

        if mappings_evaluated % 50 == 0:
            print(f"Processed {mappings_evaluated}/{len(mapping_list)} mappings...")

        mappings_evaluated += 1

        map_dict = mapping_list_item[1]

        # Ensure mapping is a list
        if isinstance(map_dict.get("mapping", None), dict):
            map_dict["mapping"] = [map_dict["mapping"]]

        map_yaml_str = yaml.dump(map_dict)

        stats_text = run_timeloop(
            arch_yaml_str,
            prob_yaml_str,
            map_yaml_str
        )


        if stats_text is None:
            continue

        energy, cycles = parse_stats(stats_text)

        if energy is None or cycles is None:
            continue

        edp = energy * cycles

        if edp < best_edp:
            best_edp = edp
            best_mapping_yaml = map_yaml_str

    return mappings_evaluated, best_edp, best_mapping_yaml


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python model_for_RL_reevaluation.py <layer_number>")
        sys.exit(1)

    nn_layer_number = int(sys.argv[1])

    arch_file = "arch/arch.yaml"
    prob_file = f"prob/prob_3d_{nn_layer_number}.yaml"
    # mapping_file = f"map/layer_{nn_layer_number}_wtThres_375_otThres_216_initGuessOrder_0_sorted_0_map.pickle"
    mapping_file = "output_mappings.pkl"

    # Load architecture
    with open(arch_file) as f:
        arch_yaml_str = f.read()

    # Load problem
    with open(prob_file) as f:
        prob_yaml_str = f.read()

    # Load mappings
    with open(mapping_file, "rb") as f:
        mapping_list = pickle.load(f)

    print("Mappings to evaluate:", len(mapping_list))

    start = time.time()

    total_evaluated, best_edp, best_map_yaml = evaluate(mapping_list)

    end = time.time()

    print("Total mappings evaluated:", total_evaluated)
    print("Time:", end - start, "seconds")
    print("Best EDP:", best_edp)

    if best_map_yaml is not None:
        with open("best_map.yaml", "w") as f:
            f.write(best_map_yaml)
    else:
        print("No valid mapping found.")

