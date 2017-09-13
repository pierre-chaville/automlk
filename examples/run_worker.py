import sys
from automlk.search import worker_search

# Get the arguments list
uid = sys.argv[1]
print('launching worker on dataset id:', uid)

# launch search on models for the dataset
worker_search(uid, 'auto')
