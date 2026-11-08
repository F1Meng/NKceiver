from __future__ import annotations

import numpy as np
import torch
from torch import nn
from scipy import sparse


def expression(counts):
    x = sparse.csr_matrix(counts, dtype=np.float32)
    x = sparse.diags(10000 / np.maximum(np.asarray(x.sum(1)).ravel(), 1)).dot(x).tocsr()
    x.data = np.log1p(x.data)
    return x


def mapper_network(genes, dimensions):
    return nn.Sequential(
        nn.Linear(genes, 256),
        nn.LayerNorm(256),
        nn.GELU(),
        nn.Linear(256, 128),
        nn.GELU(),
        nn.Linear(128, dimensions),
    )


@torch.inference_mode()
def transfer(counts, checkpoint, batch_size=512, device=None, return_coordinates=False):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    x = expression(counts)

    def tensor(k):
        return torch.as_tensor(checkpoint[k], dtype=torch.float32, device=device)

    mean, std = (tensor("expression_mean"), tensor("expression_std"))
    z = tensor("anchor_latent")
    labels = torch.as_tensor(checkpoint["anchor_labels"], device=device)
    zmean, zstd = (tensor("latent_mean"), tensor("latent_std"))
    zs = (z - zmean) / zstd
    normz = zs.square().sum(1)
    network = mapper_network(len(mean), z.shape[1]).to(device)
    try:
        network.load_state_dict(checkpoint["mapper"], strict=True)
    except (KeyError, RuntimeError) as error:
        raise ValueError("annotator.mapper parameters do not match the mapping network") from error
    network.eval()
    results = []
    confidence = []
    coordinates = []
    k = int(checkpoint.get("neighbors", 15))
    for start in range(0, x.shape[0], batch_size):
        xx = torch.from_numpy(x[start : start + batch_size].toarray()).to(device)
        xx = ((xx - mean) / std).clamp(-10, 10)
        queryz = network(xx)
        similarity = 2 * queryz @ zs.T - normz[None, :]
        values, indices = similarity.topk(k, dim=1)
        weights = torch.softmax((values - values[:, 0, None]) / max(z.shape[1] * 0.1, 1), dim=1)
        coordinate = queryz * zstd + zmean
        probabilities = torch.zeros((len(xx), len(checkpoint["state_names"])), device=device)
        probabilities.scatter_add_(1, labels[indices], weights)
        conf, pred = probabilities.max(1)
        results.append(pred.cpu().numpy())
        confidence.append(conf.cpu().numpy())
        if return_coordinates:
            coordinates.append(coordinate.cpu().numpy())
    result = (np.concatenate(results), np.concatenate(confidence))
    return (*result, np.concatenate(coordinates)) if return_coordinates else result
