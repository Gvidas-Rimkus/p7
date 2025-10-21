import torch

def apply_random_split(data, val_ratio=0.1, test_ratio=0.2, seed=1):
    torch.manual_seed(seed)

    num_nodes = data.num_nodes
    n_obs = data.y.size(0) #vi splitter kun noderne med tilhørende y, da de andre ikke skal prædikteres på

    perm = torch.randperm(n_obs)
    n_val = int(val_ratio * n_obs)
    n_test = int(test_ratio * n_obs)

    val_idx = perm[:n_val]
    test_idx = perm[n_val:n_val+n_test]
    train_idx = perm[n_val+n_test:]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data