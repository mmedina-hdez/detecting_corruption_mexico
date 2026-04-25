import sys
import os
from pyprojroot.here import here
sys.path.append(str(here() / 'methods')) 
import subprocess
from additional_utils.functions import generate_param_combinations


#Fix labels
features2test = [
    'network',
    'domain_knowledge',
]
subsets2test = [0,1,2,3]

num_cores = 8

all_processes = []

for param_num, label in enumerate(features2test):
    for this_subset in subsets2test:
        process_name = f"python ablation_exec.py {label} {this_subset}&"
        all_processes.append(process_name)

print(all_processes)


for i in range(0, len(all_processes), num_cores):

    print(i)
    commands = all_processes[i:i+num_cores]
    commands = " ".join(commands) + " wait"
    print(commands)
    result = subprocess.call(commands, shell=True)

    print(f"Batch {i} done.")