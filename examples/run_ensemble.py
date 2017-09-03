from automlk.search import worker_search

# launch search on models for the dataset
uid = input('dataset id ?')
worker_search(uid, 'ensemble')
