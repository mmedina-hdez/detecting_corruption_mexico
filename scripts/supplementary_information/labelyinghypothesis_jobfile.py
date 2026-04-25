import sys
import os
from pyprojroot.here import here
sys.path.append(str(here() / 'methods')) 
import subprocess
from additional_utils.functions import generate_param_combinations


#Fix labels
alllabtest = [
    'sanctionedA_C_all',
    'sanctionedA_C_max1',
    'sanctionedA_C_max2',
    'sanctionedA_C_max3',
    'sanctionedA_I_all',
    'sanctionedA_I_max1',
    'sanctionedA_I_max2',
    'sanctionedA_I_max3',
    'sanctionedB_C_all',
    'sanctionedB_C_max1',
    'sanctionedB_C_max2',
    'sanctionedB_C_max3',
    'sanctionedB_I_all',
    'sanctionedB_I_max1',
    'sanctionedB_I_max2',
    'sanctionedB_I_max3',
    
]

num_cores = 12

all_processes = []

for param_num, label in enumerate(alllabtest):
    process_name = f"python labelyinghypothesis_exec.py {label}&"
    all_processes.append(process_name)

print(all_processes)


for i in range(0, len(all_processes), num_cores):
    print(i)
    commands = all_processes[i:i+num_cores]
    commands = " ".join(commands) + " wait"
    print(commands)
    result = subprocess.call(commands, shell=True)

    print(f"Batch {i} done.")