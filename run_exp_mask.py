"""
RUN ME IN {project_folder}/
python run_exp_mask.py
"""
import os
import subprocess
import numpy as np

method = 'sad'
datasets = ['wikipedia', 'txn_filter']

gpu = 7
processes = []

masks = [0.1, 0.3, 0.5, 0.7, 0.9]
for dataset in datasets:
  for mask in masks:
    gpu = gpu + 1
    if gpu > 9:
      gpu = 7
    command = f'python -u train.py  ' + \
              f'--data_set {dataset} ' + \
              f'--mask_label --mask_ratio {mask} ' + \
              f'--gpu {gpu}'
    process = subprocess.Popen(command, shell=True)
    processes.append(process)
# cfg_msg = f"thre-exp: method={method}, dataset={dataset}, thershold={thresholds}, mask={mask}, gamma_con={gamma_con}, gamma_dev={gamma_dev}"
        
for process in processes:
  process.wait()

print("All processes completed.")


