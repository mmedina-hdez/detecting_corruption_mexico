import sys
import os
from pyprojroot.here import here
sys.path.append(str(here() / 'methods')) 
import subprocess
from additional_utils.functions import generate_param_combinations

trial_test = False
algorithm = 'hdsrf'
r = 42


######################## parameter combinations ########################
#remember that if you change the parameters, you also need to change cv_exec.py

if algorithm == 'hdsrf':

    if trial_test == True:

        param_grid = { #FOR TEST PURPOSES
            'max_samples':[1],
            'n_estimators':[10,15, 25],
            'max_depth':[8],
            'min_samples_split':[100, 150],
            'max_features':['sqrt'],
            'random_state':[r], #for all
            'alpha':[0.1],
            }
    else:    
        param_grid = { 
            #'max_samples':[1],
            'n_estimators':[10],
            'max_depth':[8],
            'min_samples_split':[2],
            'max_features':['sqrt'],
            'random_state':[r], #for all
            'alpha':[0.05, 0.075, 0.1, 0.125, 0.15],
            }

    
param_combinations = generate_param_combinations(param_grid)  # Generate param combinations
print('Number of combinations: ', len(param_combinations))


num_cores = 5

all_processes = []

for param_num, params in enumerate(param_combinations):
    process_name = f"python classpriortest_exec.py {trial_test} {algorithm} {param_num} &"
    all_processes.append(process_name)

print(all_processes)



for i in range(0, len(all_processes), num_cores):

    print(i)
    commands = all_processes[i:i+num_cores]
    commands = " ".join(commands) + " wait"
    print(commands)
    result = subprocess.call(commands, shell=True)

    print(f"Batch {i} done.")