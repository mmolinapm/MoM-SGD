"""Unit tests for the Median-of-Means gradient estimator."""

import pytest
import torch

from src.mom import coordinatewise_median, mom_gradient


def test_scalar_gradients():
    # Consecutive block means are 2, 6, and 10.
    block_gradients = torch.tensor([2., 6., 10.])
    result = coordinatewise_median(block_gradients)
    torch.testing.assert_close(result, torch.tensor(6.))
    assert result.shape == torch.Size([])


def test_multidimensional_gradients():
    # Block means are [2, 10], [6, 2], [10, 6]; medians are coordinate-wise.
    block_gradients = torch.tensor([[2., 10.], [6., 2.], [10., 6.]])
    result = coordinatewise_median(block_gradients)
    torch.testing.assert_close(result, torch.tensor([6., 6.]))


def test_extreme_contaminated_block():
    block_gradients = torch.tensor([[2., 4.], [3., 5.], [1e8, -1e8]])
    result = coordinatewise_median(block_gradients)
    torch.testing.assert_close(result, torch.tensor([3., 4.]))


@pytest.mark.parametrize("x_count,y_count,m_n,k", [
    (5, 6, 2, 3), (6, 5, 2, 3), (6, 6, 2, 2),
    (6, 6, 4, 2), (0, 0, 2, 3),
])
def test_invalid_batch_sizes(x_count, y_count, m_n, k):
    with pytest.raises(ValueError, match=r"exactly k \* m_n observations"):
        mom_gradient(torch.zeros(2, requires_grad=True), torch.ones(x_count, 2),
                     torch.ones(y_count), squared_loss, k=k, m_n=m_n)


@pytest.mark.parametrize("name", ["k", "m_n"])
@pytest.mark.parametrize("value,error", [
    (0, ValueError), (-2, ValueError), (2.0, TypeError), (True, TypeError),
])
def test_invalid_block_sizes(name, value, error):
    sizes = {"k": 3, "m_n": 2}
    sizes[name] = value
    with pytest.raises(error, match=f"{name} must be a positive integer"):
        mom_gradient(torch.zeros(2, requires_grad=True), torch.ones(6, 2),
                     torch.ones(6), squared_loss, **sizes)


def test_missing_batch_dimension():
    with pytest.raises(ValueError, match="at least one dimension"):
        coordinatewise_median(torch.tensor(1.))


@pytest.mark.parametrize("missing", ["X_batch", "y_batch"])
def test_estimator_missing_observation_dimension(missing):
    batches = {"X_batch": torch.ones(1, 1), "y_batch": torch.ones(1)}
    batches[missing] = torch.tensor(1.)
    with pytest.raises(ValueError, match="observation dimension"):
        mom_gradient(torch.zeros(1, requires_grad=True), **batches,
                     loss_fn=squared_loss, k=1, m_n=1)


def test_empty_blocks():
    with pytest.raises(ValueError, match="at least one block"):
        coordinatewise_median(torch.empty(0, 2))


@pytest.mark.parametrize("shape", [(), (4,), (2, 3), (2, 1, 3)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_shape_dtype_and_device(shape, dtype, device):
    gradients = torch.ones((3, *shape), dtype=dtype, device=device)
    result = coordinatewise_median(gradients)
    assert result.shape == shape
    assert result.dtype == gradients.dtype
    assert result.device == gradients.device
    torch.testing.assert_close(result, torch.ones(shape, dtype=dtype, device=device))


def test_even_block_count_uses_lower_median():
    # Means of the original blocks [1, 3] and [7, 9].
    gradients = torch.tensor([2., 8.])
    torch.testing.assert_close(
        coordinatewise_median(gradients), torch.tensor(2.)
    )


def test_noncontiguous_gradients():
    # Precomputed means of the original six per-observation gradient tensors.
    gradients = torch.tensor([
        [[2., 3.], [4., 5.]], [[10., 11.], [12., 13.]],
        [[18., 19.], [20., 21.]],
    ], dtype=torch.float64).transpose(1, 2)
    assert not gradients.is_contiguous()
    result = coordinatewise_median(gradients)
    torch.testing.assert_close(result, torch.tensor([[10., 12.], [11., 13.]], dtype=torch.float64))


def test_integer_block_gradients_preserve_dtype():
    # The current aggregator accepts integer tensors (torch.median semantics).
    gradients = torch.tensor([[2, 10], [6, 2], [10, 6]], dtype=torch.int64)
    result = coordinatewise_median(gradients)
    torch.testing.assert_close(result, torch.tensor([6, 6], dtype=torch.int64))
    assert result.dtype == gradients.dtype


def squared_loss(predictions, targets):
    return 0.5 * (predictions - targets).square()


@pytest.mark.parametrize("features,expected_blocks,expected", [
    ([[1.], [3.], [5.], [7.], [9.], [11.]],
     [[2.], [6.], [10.]], [6.]),
    ([[1., 9.], [3., 11.], [5., 1.], [7., 3.], [9., 5.], [11., 7.]],
     [[2., 10.], [6., 2.], [10., 6.]], [6., 6.]),
    ([[1., 3.], [3., 5.], [2., 4.], [4., 6.], [1e8, -1e8], [1e8, -1e8]],
     [[2., 4.], [3., 5.], [1e8, -1e8]], [3., 4.]),
    ([[1.], [3.], [7.], [9.]], [[2.], [8.]], [2.]),
])
def test_complete_estimator_block_means_and_coordinatewise_median(
        monkeypatch, features, expected_blocks, expected):
    # At theta=0 and y=-1, each squared-loss gradient is exactly its X row.
    # Retain the original per-observation examples and expected answers.
    X = torch.tensor(features, dtype=torch.float64)
    y = -torch.ones(len(features), dtype=X.dtype)
    theta = torch.zeros(X.shape[1], dtype=X.dtype, requires_grad=True)
    theta.grad = torch.full_like(theta, 17.)
    original_theta, original_grad = theta.detach().clone(), theta.grad.clone()
    original_X, original_y = X.clone(), y.clone()
    k, m_n = len(expected_blocks), 2
    assert X.shape[0] == y.shape[0] == k * m_n
    observed_blocks, observed_sizes = [], []

    def observe_aggregation(block_gradients):
        observed_blocks.append(block_gradients.detach().clone())
        return coordinatewise_median(block_gradients)

    def observe_loss(predictions, targets):
        observed_sizes.append(predictions.shape[0])
        return squared_loss(predictions, targets)

    # Observe actual block gradients and delegate unchanged to the canonical
    # aggregator. This spy does not implement a replacement estimator.
    monkeypatch.setattr("src.mom.coordinatewise_median", observe_aggregation)
    result = mom_gradient(theta, X, y, observe_loss, k=k, m_n=m_n)
    assert observed_sizes == [m_n] * k
    assert len(observed_blocks) == 1
    assert observed_blocks[0].shape == (k, *theta.shape)
    torch.testing.assert_close(observed_blocks[0], torch.tensor(expected_blocks, dtype=X.dtype))
    torch.testing.assert_close(result, torch.tensor(expected, dtype=X.dtype))
    assert result.shape == theta.shape
    assert result.dtype == theta.dtype
    assert result.device == theta.device
    torch.testing.assert_close(theta, original_theta)
    torch.testing.assert_close(theta.grad, original_grad)
    torch.testing.assert_close(X, original_X)
    torch.testing.assert_close(y, original_y)
