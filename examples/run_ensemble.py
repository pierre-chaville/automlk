import sys
from automlk.search import worker_search

# launch search on models for the dataset
uid = sys.argv[1]
worker_search(uid, 'ensemble')
