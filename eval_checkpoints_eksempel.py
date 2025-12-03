import torch
import os
import torch.nn.functional as F

from models.grape_model import GRAPEModel
from graph_builders.base_grape import get_base_grape_graph
from graph_builders.mask_splitter import apply_random_split

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

checkpoint_dir = "checkpoints"
best_models = []
for root, _, files in os.walk(checkpoint_dir):
    for f in files:
        if f.endswith("best.pt"):
            best_models.append(os.path.join(root, f))

model_path = best_models[0]
checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
state_dict = checkpoint["model_state_dict"]

data = get_base_grape_graph(features_path="data/features_small.csv", targets_path="data/targets_small.csv")
# data = get_base_grape_graph(features_path="data/features.csv", targets_path="data/targets.csv")

#hvis man vil eksperimentere med forskellige splits, num_layers, edge_dropout_rate så skal de gemmes i payloaden for hver checkpoint, så man ikke skal hardcode dem her.
data = apply_random_split(data, val_ratio=0.1, test_ratio=0.2)
model = GRAPEModel(data=data,
                   num_layers=3, 
                   edge_dropout_rate=0.30) 
model.load_state_dict(state_dict, strict=False)


split = "validation"
edge_pred, edge_true, node_pred, node_true, dropped_is_cont = model.evaluate(data, split=split)
cont_mask = dropped_is_cont
cat_mask = ~dropped_is_cont

pred_cont = edge_pred[cont_mask]
true_cont = edge_true[cont_mask]
pred_cat = edge_pred[cat_mask]
true_cat = edge_true[cat_mask]

edge_loss_cont = torch.sqrt(F.mse_loss(pred_cont, true_cont))
edge_loss_cat = F.binary_cross_entropy_with_logits(pred_cat, true_cat)
edge_loss = edge_loss_cont + edge_loss_cat
node_loss = F.binary_cross_entropy_with_logits(node_pred, node_true)
total_loss = edge_loss + node_loss

print(f"Best training loss: {checkpoint.get('total_loss'):.4f}")
print(f"Evaluation loss on {split} set: Total {total_loss.item():.4f}")

split = "test"
edge_pred, edge_true, node_pred, node_true = model.evaluate(data, split=split)
edge_loss = F.mse_loss(edge_pred, edge_true)
node_loss = F.binary_cross_entropy_with_logits(node_pred, node_true)
total_loss = edge_loss + node_loss
print(f"Evaluation loss on {split} set: Total {total_loss.item():.4f}")