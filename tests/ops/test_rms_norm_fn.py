import pytest
import torch

from cs336_basics.nn.basic import RMSNormFn


def _reference_rmsnorm(x: torch.Tensor, weights: torch.Tensor, eps: float) -> torch.Tensor:
    rms = torch.sqrt((x * x).mean(dim=-1, keepdim=True) + eps)
    return (x / rms) * weights


@pytest.mark.skipif(not torch.cuda.is_available(), reason="RMSNormFn forward uses CUDA kernel")
def test_rms_norm_fn_backward_matches_reference():
    torch.manual_seed(0)
    d_model = 16
    x = torch.randn(8, d_model, device="cuda",
                    dtype=torch.float32, requires_grad=True)
    weights = torch.randn(d_model, device="cuda",
                          dtype=torch.float32, requires_grad=True)
    eps = 1e-5

    # Custom autograd.Function
    x_custom = x.detach().clone().requires_grad_(True)
    w_custom = weights.detach().clone().requires_grad_(True)
    out_custom = RMSNormFn.apply(x_custom, w_custom, eps)
    out_custom.sum().backward()

    # Reference using PyTorch ops
    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = weights.detach().clone().requires_grad_(True)
    out_ref = _reference_rmsnorm(x_ref, w_ref, eps)
    out_ref.sum().backward()

    torch.testing.assert_close(out_custom, out_ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(x_custom.grad, x_ref.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(w_custom.grad, w_ref.grad, rtol=1e-4, atol=1e-4)
