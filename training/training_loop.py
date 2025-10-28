import os
import time
import torch
import torch.nn.functional as F
from datetime import datetime

def train_grape(model, 
                data, 
                optimizer, 
                epochs, 
                edge_loss_weight=1.0, 
                node_loss_weight=1.0,
                checkpoint_dir="checkpoints",
                checkpoint_every = 1000,
                save_best = True,
                run_name=None):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    data = data.to(device)

    if run_name is None:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"grape_run_{ts}"
    out_dir = os.path.join(checkpoint_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    total_params = sum(p.numel() for p in model.parameters())
    print("=== Model Info ===")
    print(f"Device: {device}")
    print(f"Total parameters: {total_params:,}")
    print("Parameter breakdown:")
    for name, p in model.named_parameters(): 
        if p.requires_grad: 
            print(f"{name:<40} {p.numel():>10,}")

    best_loss = float('inf')
    best_path = os.path.join(out_dir, f"{run_name}_best.pt")

    model.train()
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        
        optimizer.zero_grad()
        edge_pred, edge_true, node_pred, node_true = model(data=data)
        edge_loss = F.mse_loss(edge_pred, edge_true)
        node_loss = F.binary_cross_entropy_with_logits(node_pred, node_true)
        total_loss = edge_loss_weight * edge_loss + node_loss_weight * node_loss
        total_loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        duration = time.perf_counter() - start
        print(f"Epoch {epoch:03d} | Total {total_loss.item():.4f} | Edge {edge_loss.item():.4f} | "f"Node {node_loss.item():.4f} | Time: {duration:.2f}s")

        if checkpoint_every and (epoch % checkpoint_every == 0):
            ckpt_name = f"{run_name}_epoch{epoch:06d}_loss{total_loss.item():.4f}.pt"
            ckpt_path = os.path.join(out_dir, ckpt_name)
            _save_checkpoint(ckpt_path, model, optimizer, epoch, total_loss.item(),
                       edge_loss.item(), node_loss.item(), run_name)

        if save_best and total_loss.item() < best_loss:
            best_loss = total_loss.item()
            _save_checkpoint(best_path, model, optimizer, epoch, best_loss,
                       edge_loss.item(), node_loss.item(), run_name)

def _save_checkpoint(path, model, optimizer, epoch, total_loss, edge_loss, node_loss, run_name):
    payload = {
        "epoch": epoch,
        "run_name": run_name,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "total_loss": float(total_loss),
        "edge_loss": float(edge_loss),
        "node_loss": float(node_loss),
        "torch_version": torch.__version__,
        "cuda": torch.cuda.is_available(),
    }
    torch.save(payload, path)