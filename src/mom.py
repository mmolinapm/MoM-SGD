"""Coordinate-wise median aggregation of precomputed block gradients."""

import torch


def coordinatewise_median(block_gradients: torch.Tensor) -> torch.Tensor:
    """Return hat_g[c] = median_{j=1,...,K} g_j[c] for block gradients g_j.

    The input has shape (K, ...), with K >= 1. The median is taken across
    dim=0, so the output has shape block_gradients.shape[1:] and preserves
    dtype and device. The input tensor is not modified.

    For even K, torch.median selects the lower of the two central values.
    """
    if not isinstance(block_gradients, torch.Tensor):
        raise TypeError("block_gradients must be a torch.Tensor.")
    if block_gradients.ndim == 0:
        raise ValueError("block_gradients must have at least one dimension (K, ...).")
    if block_gradients.shape[0] < 1:
        raise ValueError("block_gradients must contain at least one block (K >= 1).")
    return torch.median(block_gradients, dim=0).values


def mom_gradient(
    theta: torch.Tensor,
    X_batch: torch.Tensor,
    y_batch: torch.Tensor,
    loss_fn,
    k: int,
    m_n: int,
) -> torch.Tensor:
    """Return the Median-of-Means gradient for predictions X @ theta.

    Split the supplied k * m_n observations into k consecutive blocks B_j.
    For each block, compute directly with autograd
    g_j = grad_theta [(1 / m_n) sum_{i in B_j} loss_i(theta)].
    Return g_MoM = coordinate-wise median(g_1, ..., g_k), shaped like theta.

    loss_fn(predictions, targets) must return one loss per observation.
    Neither theta nor theta.grad is modified.
    """
    for name, tensor in (("theta", theta), ("X_batch", X_batch), ("y_batch", y_batch)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")
    if not theta.requires_grad:
        raise ValueError("theta must have requires_grad=True.")
    for name, value in (("k", k), ("m_n", m_n)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be a positive integer.")
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if X_batch.ndim == 0 or y_batch.ndim == 0:
        raise ValueError("X_batch and y_batch must have an observation dimension.")
    if X_batch.shape[0] != k * m_n or y_batch.shape[0] != k * m_n:
        raise ValueError("X_batch and y_batch must each contain exactly k * m_n observations.")
    if not callable(loss_fn):
        raise TypeError("loss_fn must be callable.")

    block_gradients = []
    for j in range(k):
        start = j * m_n
        X_block = X_batch[start : start + m_n]
        y_block = y_batch[start : start + m_n]
        predictions = X_block @ theta
        losses = loss_fn(predictions, y_block)
        if not isinstance(losses, torch.Tensor):
            raise TypeError("loss_fn must return a torch.Tensor.")
        if losses.ndim == 0 or losses.shape[0] != m_n or losses.numel() != m_n:
            raise ValueError("loss_fn must return one loss value per observation.")
        block_loss = losses.mean()
        (block_gradient,) = torch.autograd.grad(block_loss, theta)
        block_gradients.append(block_gradient)

    return coordinatewise_median(torch.stack(block_gradients, dim=0))
