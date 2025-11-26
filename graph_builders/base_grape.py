from torch_geometric.data import Data
import pandas as pd
import torch

def get_base_grape_graph(features_path, targets_path):
    features_df = pd.read_csv(features_path, index_col=0)
    targets_df = pd.read_csv(targets_path, index_col=0)
    #sample_sites noderne får indices: 0 -> antal sample_sites, features noder får indices: antal sample_sites -> antal sample_sites + antal features.
    n_obs = features_df.shape[0]
    n_feats = features_df.shape[1]

    #her laves edges i grafen (observationer <-> features) med feature vals som edge attributter.
    edge_index = [[], []]
    edge_attr = []
    for i, (_, row) in enumerate(features_df.iterrows()):
        for j, val in enumerate(row):
            obs_node = i
            feat_node = n_obs + j
            if pd.notna(val):
                edge_index[0].append(obs_node)
                edge_index[1].append(feat_node)
                edge_attr.append(val)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    #PyG modellerer edges som directed per default, så her tilfølger vi den modsatte retning: directed -> undirected.
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_attr = torch.cat([edge_attr, edge_attr], dim=0).unsqueeze(-1)
    #her indsætter vi værdier på noderne husk at: observations noderne instantieres med constant vektorer fyldt med 1'er, feature noder instantieres med one-hot vektorer.
    x_obs = torch.ones((n_obs, n_feats), dtype=torch.float)
    x_feat = torch.eye(n_feats, dtype=torch.float)
    x = torch.cat([x_obs, x_feat], dim=0)

    #targets kommer på observations noderne, og jeg padder med 0'er så dimensionerne passer (for maskingens skyld)
    y = torch.tensor(targets_df.values, dtype=torch.float)
    n_pad = x.size(0) - y.size(0)
    pad = torch.zeros((n_pad, y.size(1)), dtype=y.dtype)
    y = torch.cat([y, pad], dim=0)

    graph = Data(edge_index=edge_index, edge_attr=edge_attr, x=x, y=y)
    return graph

def add_edge2dense_idx(data):
    source, target = data.edge_index
    n_obs = data.x.shape[0] - data.x.shape[1]
    n_feat = data.x.shape[1]

    mask = source < n_obs
    source = source[mask]
    target = target[mask]

    obs_idx = source
    feat_idx = target - n_obs
    edge2dense_idx = obs_idx * n_feat + feat_idx

    data.edge2dense_idx = edge2dense_idx
    data.edge_mask_obs_to_feat = mask
    return data