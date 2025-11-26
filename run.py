import torch
import sys
from graph_builders.base_grape import get_base_grape_graph, add_edge2dense_idx
from graph_builders.grape_site_connectivity import get_grape_site_connectivity_graph

from graph_builders.mask_splitter import apply_random_split
from models.grape_model import GRAPEModel
from training.training_loop import train_grape

#Sammenlign til min config in case noget ik dur
print(f"Python version: {sys.version.split()[0]}")#3.12.3
print(f"PyTorch version: {torch.__version__}")#2.9.0+cu128
print(f"CUDA available: {torch.cuda.is_available()}")#True
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")#12.8
    print(f"Device: {torch.cuda.get_device_name(0)}")#NVIDIA GeForce RTX 5070


seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# HER SÆTTES OM VI KØRER DEN 'NYE' IMPLEMENTATION (DEN RIGTIGE) ELLER DEN GAMLE.
new = True

# data = get_base_grape_graph(features_path="data/features.csv", targets_path="data/targets.csv")

#test på en lille subset lavet af: 30 features, 400 sample_sites og 200 targets.
# data = get_base_grape_graph(features_path="data/features_small.csv", targets_path="data/targets_small.csv")
data = get_grape_site_connectivity_graph(features_path="data/features_small.csv", targets_path="data/targets_small.csv", distance_threshold=1)
if new: data = add_edge2dense_idx(data)
data = apply_random_split(data, val_ratio=0.0, test_ratio=0.3)

model = GRAPEModel(data=data,
                   node_dims=[None, 16, 16],
                   edge_dims=[None, 16, 16],
                   node_head_dims=[None, 64],
                   edge_head_dims=[None, 64],
                   edge_dropout_rate=0.30,
                   new = new)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_grape(
    model=model,
    data=data,
    optimizer=optimizer,
    epochs=20000,
    checkpoint_dir="checkpoints",
    checkpoint_every=500,
    save_best=True,
    run_name=None
)

