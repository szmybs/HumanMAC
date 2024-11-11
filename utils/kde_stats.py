import numpy as np
import sys, os

kde_results_path = os.path.join(os.getcwd(), "kde_results")

kde_results_files = os.listdir(kde_results_path)
kde_results_list = []
for krf in kde_results_files:
    full_path = os.path.join(kde_results_path, krf)
    kde_result = np.load(full_path)
    kde_results_list.append(kde_result)
    
kde_results = np.array(kde_results_list).mean(axis=0)
print(kde_results)