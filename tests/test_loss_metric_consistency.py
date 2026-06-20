"""Loss/metric consistency checks for assembled-field diagnostics."""

from __future__ import annotations

import numpy as np
import torch

from lpcanet.losses import (
    CompositeLoss,
    interface_flux_loss,
    interface_value_loss,
    pde_residual_loss,
    reconstruction_loss,
    spectral_loss,
)
from lpcanet.metrics import interface_flux_jump, interface_jump, mre, mse, relative_spectrum_error
from lpcanet.metrics.residuals import darcy_residual, poisson_residual
from lpcanet.metrics.torch_ops import interface_flux_traces_torch, interface_value_traces_torch


def test_loss_terms_match_corresponding_metrics() -> None:
    pred_np, true_np, forcing_np, coeff_np = _fixed_fields()
    pred = torch.tensor(pred_np, dtype=torch.float64, requires_grad=True)
    true = torch.tensor(true_np, dtype=torch.float64)
    forcing = torch.tensor(forcing_np, dtype=torch.float64)
    coeff = torch.tensor(coeff_np, dtype=torch.float64)
    dx = 1.0 / float(pred_np.shape[-1] - 1)

    assert torch.allclose(
        reconstruction_loss(pred, true, mode="relative_l2"),
        torch.tensor(mre(pred_np, true_np), dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )
    assert torch.allclose(
        reconstruction_loss(pred, true, mode="mse"),
        torch.tensor(mse(pred_np, true_np), dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )
    assert torch.allclose(
        interface_value_loss(pred, patch_size=4, stride=2),
        torch.tensor(interface_jump(pred_np, patch_size=4, stride=2), dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )
    assert torch.allclose(
        interface_flux_loss(pred, patch_size=4, stride=2, dx=dx),
        torch.tensor(interface_flux_jump(pred_np, patch_size=4, stride=2, dx=dx), dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )

    poisson_metric = float(np.sqrt(np.mean(poisson_residual(pred_np, forcing_np, dx) ** 2)))
    assert torch.allclose(
        pde_residual_loss(pred, dx=dx, forcing=forcing, equation="poisson", reduction="rms"),
        torch.tensor(poisson_metric, dtype=torch.float64),
        atol=1e-10,
        rtol=1e-12,
    )

    darcy_metric = float(np.sqrt(np.mean(darcy_residual(pred_np, coeff_np, forcing_np, dx) ** 2)))
    assert torch.allclose(
        pde_residual_loss(
            pred,
            dx=dx,
            forcing=forcing,
            coefficient=coeff,
            equation="darcy",
            reduction="rms",
        ),
        torch.tensor(darcy_metric, dtype=torch.float64),
        atol=1e-10,
        rtol=1e-12,
    )

    assert torch.allclose(
        spectral_loss(pred, true, high_k_weight_power=0.0),
        torch.tensor(relative_spectrum_error(pred_np, true_np), dtype=torch.float64),
        atol=1e-10,
        rtol=1e-10,
    )


def test_each_loss_term_is_differentiable() -> None:
    pred_np, true_np, forcing_np, coeff_np = _fixed_fields()
    true = torch.tensor(true_np, dtype=torch.float64)
    forcing = torch.tensor(forcing_np, dtype=torch.float64)
    coeff = torch.tensor(coeff_np, dtype=torch.float64)
    dx = 1.0 / float(pred_np.shape[-1] - 1)

    terms = [
        lambda p: reconstruction_loss(p, true, mode="relative_l2"),
        lambda p: interface_value_loss(p, patch_size=4, stride=2),
        lambda p: interface_flux_loss(p, patch_size=4, stride=2, dx=dx),
        lambda p: pde_residual_loss(p, dx=dx, forcing=forcing, equation="poisson"),
        lambda p: pde_residual_loss(
            p,
            dx=dx,
            forcing=forcing,
            coefficient=coeff,
            equation="darcy",
        ),
        lambda p: spectral_loss(p, true, high_k_weight_power=1.0),
    ]

    for term in terms:
        pred = torch.tensor(pred_np, dtype=torch.float64, requires_grad=True)
        value = term(pred)
        (grad,) = torch.autograd.grad(value, pred)
        assert torch.isfinite(value)
        assert torch.isfinite(grad).all()
        assert torch.linalg.vector_norm(grad) > 0.0


def test_truth_referenced_interface_losses_match_trace_error() -> None:
    pred_np, true_np, _, _ = _fixed_fields()
    pred = torch.tensor(pred_np, dtype=torch.float64, requires_grad=True)
    true = torch.tensor(true_np, dtype=torch.float64)
    dx = 1.0 / float(pred_np.shape[-1] - 1)

    expected_value = torch.mean(
        torch.abs(
            interface_value_traces_torch(pred, 4, 2)
            - interface_value_traces_torch(true, 4, 2)
        )
    )
    expected_flux = torch.mean(
        torch.abs(
            interface_flux_traces_torch(pred, 4, 2, dx=dx)
            - interface_flux_traces_torch(true, 4, 2, dx=dx)
        )
    )
    value = interface_value_loss(
        pred,
        true,
        patch_size=4,
        stride=2,
        target="truth",
    )
    flux = interface_flux_loss(
        pred,
        true,
        patch_size=4,
        stride=2,
        dx=dx,
        target="truth",
    )
    assert torch.allclose(value, expected_value)
    assert torch.allclose(flux, expected_flux)
    (value + flux).backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_composite_loss_weighted_sum_and_components() -> None:
    pred_np, true_np, forcing_np, _ = _fixed_fields()
    pred = torch.tensor(pred_np, dtype=torch.float64, requires_grad=True)
    true = torch.tensor(true_np, dtype=torch.float64)
    forcing = torch.tensor(forcing_np, dtype=torch.float64)
    dx = 1.0 / float(pred_np.shape[-1] - 1)
    loss = CompositeLoss(
        weights={
            "recon": 1.0,
            "interface_value": 0.25,
            "interface_flux": 0.1,
            "pde_residual": 0.05,
            "spectral": 0.2,
        },
        patch_size=4,
        stride=2,
        dx=dx,
        spectral_high_k_weight_power=1.0,
    )

    total, components = loss(pred, true, forcing=forcing, return_components=True)
    expected = (
        components["recon"]
        + 0.25 * components["interface_value"]
        + 0.1 * components["interface_flux"]
        + 0.05 * components["pde_residual"]
        + 0.2 * components["spectral"]
    )

    assert torch.allclose(total, expected)
    total.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def _fixed_fields() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    pred = rng.normal(size=(2, 8, 8)).astype(np.float64)
    true = rng.normal(size=(2, 8, 8)).astype(np.float64)
    forcing = rng.normal(size=(2, 8, 8)).astype(np.float64)
    coeff = (1.0 + 0.1 * rng.random(size=(2, 8, 8))).astype(np.float64)
    return pred, true, forcing, coeff
