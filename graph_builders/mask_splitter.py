import torch

def apply_random_split(data, val_ratio=0.1, test_ratio=0.2):
    num_nodes = data.num_nodes
    n_obs = data.x.shape[0] - data.x.shape[1]
    perm = torch.randperm(n_obs)
    n_val = int(val_ratio * n_obs)
    n_test = int(test_ratio * n_obs)

    train_idx = perm[n_val+n_test:]
    val_idx = perm[:n_val]
    test_idx = perm[n_val:n_val+n_test]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    #samme for edges
    n_edges = data.edge_index.size(1)
    perm = torch.randperm(n_edges)
    n_val_edges = int(val_ratio * n_edges)
    n_test_edges = int(test_ratio * n_edges)

    val_edge_idx = perm[:n_val_edges]
    test_edge_idx = perm[n_val_edges:n_val_edges+n_test_edges]
    train_edge_idx = perm[n_val_edges+n_test_edges:]

    train_edge_mask = torch.zeros(n_edges, dtype=torch.bool)
    val_edge_mask = torch.zeros(n_edges, dtype=torch.bool)
    test_edge_mask = torch.zeros(n_edges, dtype=torch.bool)

    train_edge_mask[train_edge_idx] = True
    val_edge_mask[val_edge_idx] = True
    test_edge_mask[test_edge_idx] = True

    data.train_edge_mask = train_edge_mask
    data.val_edge_mask = val_edge_mask
    data.test_edge_mask = test_edge_mask

    return data