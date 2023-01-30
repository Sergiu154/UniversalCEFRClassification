import pickle
import os


results = {}


for filename in os.listdir('./experiment_results'):

    with open(f'./experiment_results/{filename}', 'rb') as fin:
        results[filename] = pickle.load(fin)


import pdb; pdb.set_trace()
