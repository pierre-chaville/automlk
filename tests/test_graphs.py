from automlk.dataset import get_dataset
from automlk.graphs import graph_correl_features

for i in range(1, 7):
    dt = get_dataset(i)
    graph_correl_features(dt)
