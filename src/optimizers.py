"""Gradient transformations and in-place parameter updates using PyTorch."""

import torch


def gradient_norm(gradient: torch.Tensor) -> torch.Tensor:
    """Euclidean norm with float64 accumulation; leave the gradient unchanged."""
    dtype = torch.complex128 if gradient.is_complex() else torch.float64
    return torch.linalg.vector_norm(gradient.to(dtype=dtype))


def clip_gradient(gradient: torch.Tensor, gamma: float) -> torch.Tensor:
    """Return g * min(1, gamma / ||g||_2), with a global norm and gamma > 0.

    The input is unchanged; a zero gradient is returned as a clone.
    """
    if not isinstance(gradient, torch.Tensor):
        raise TypeError("gradient must be a torch.Tensor.")
    if not (gradient.is_floating_point() or gradient.is_complex()):
        raise TypeError("gradient must have a floating-point or complex dtype.")
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)):
        raise TypeError("gamma must be a real number.")
    if not gamma > 0:
        raise ValueError("gamma must be strictly positive.")
    norm = gradient_norm(gradient)
    if norm == 0:
        return gradient.clone()
    return gradient * torch.clamp(gamma / norm, max=1)


def normalize_gradient(gradient: torch.Tensor, beta: float) -> torch.Tensor:
    """Return g / (||g||_2 + beta), with a global norm and beta > 0.

    The input gradient is unchanged.
    """
    if not isinstance(gradient, torch.Tensor):
        raise TypeError("gradient must be a torch.Tensor.")
    if not (gradient.is_floating_point() or gradient.is_complex()):
        raise TypeError("gradient must have a floating-point or complex dtype.")
    if isinstance(beta, bool) or not isinstance(beta, (int, float)):
        raise TypeError("beta must be a real number.")
    if not beta > 0:
        raise ValueError("beta must be strictly positive.")
    return gradient / (gradient_norm(gradient) + beta)


def sgd_step_(
    parameter: torch.Tensor,
    gradient: torch.Tensor,
    learning_rate: float,
) -> None:
    """Apply parameter <- parameter - learning_rate * gradient in-place.

    Tensors must have equal shapes and share a device; learning_rate > 0.
    The update is not tracked by autograd and returns None.
    """
    if not isinstance(parameter, torch.Tensor):
        raise TypeError("parameter must be a torch.Tensor.")
    if not isinstance(gradient, torch.Tensor):
        raise TypeError("gradient must be a torch.Tensor.")
    if not (parameter.is_floating_point() or parameter.is_complex()):
        raise TypeError("parameter must have a floating-point or complex dtype.")
    if parameter.shape != gradient.shape:
        raise ValueError("parameter and gradient must have the same shape.")
    if parameter.device != gradient.device:
        raise ValueError("parameter and gradient must be on the same device.")
    if gradient.is_complex() and not parameter.is_complex():
        raise TypeError("a complex gradient requires a complex parameter.")
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)):
        raise TypeError("learning_rate must be a real number.")
    if not learning_rate > 0:
        raise ValueError("learning_rate must be strictly positive.")
    with torch.no_grad():
        parameter.add_(gradient, alpha=-learning_rate)


def clipped_sgd_step_(
    parameter: torch.Tensor,
    gradient: torch.Tensor,
    learning_rate: float,
    gamma: float,
) -> None:
    """Apply parameter <- parameter - learning_rate * clip_gradient(g, gamma).

    Uses global L2 clipping, requires learning_rate, gamma > 0, and returns None.
    The parameter is updated in-place without autograd tracking.
    """
    with torch.no_grad():
        sgd_step_(parameter, clip_gradient(gradient, gamma), learning_rate)


def normalized_sgd_step_(
    parameter: torch.Tensor,
    gradient: torch.Tensor,
    learning_rate: float,
    beta: float,
) -> None:
    """Apply parameter <- parameter - learning_rate * g / (||g||_2 + beta).

    Uses the global L2 norm, requires learning_rate, beta > 0, and returns None.
    The parameter is updated in-place without autograd tracking.
    """
    with torch.no_grad():
        sgd_step_(parameter, normalize_gradient(gradient, beta), learning_rate)
