from __future__ import annotations

import numpy as np
import pytest
import torch

from lpcanet.metrics import mre
from lpcanet.models.coupling import CouplingOperator, GATCouplingBackend
from lpcanet.models.mlp import MLP
from lpcanet.pca.pca import LocalToLocalPCAEncoder
from lpcanet.train.loop import predict_latent, train_latent_model
from lpcanet.train.seeding import set_seed


def test_coupling_operator_handles_ragged_codes() -> None:
    torch.manual_seed(0)
    input_counts = [2, 3, 1, 4]
    output_counts = [1, 2, 3, 1]
    x = torch.randn(5, sum(input_counts), requires_grad=True)

    gnn = CouplingOperator(
        input_component_counts=input_counts,
        output_component_counts=output_counts,
        grid_shape=(2, 2),
        embed_dim=8,
        backend="gnn",
        num_layers=2,
        global_output_dim=2,
    )
    y = gnn(x)
    assert tuple(y.shape) == (5, 2 + sum(output_counts))
    y.square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    attention = CouplingOperator(
        input_component_counts=input_counts,
        output_component_counts=output_counts,
        grid_shape=(2, 2),
        embed_dim=8,
        backend="attention",
        num_layers=1,
        attention_heads=2,
        attention_n_max=4,
        local_residual=True,
        local_hidden_size=8,
        local_num_layers=2,
        zero_init_correction=True,
    )
    assert tuple(attention(x.detach()).shape) == (5, sum(output_counts))
    assert len(attention.local_predictors) == len(input_counts)
    assert all(torch.count_nonzero(decoder.weight) == 0 for decoder in attention.output_decoders)


def test_attention_guardrail_asserts_above_n_max() -> None:
    counts = [1] * 272
    with pytest.raises(AssertionError, match="Attention guardrail violated"):
        CouplingOperator(
            input_component_counts=counts,
            output_component_counts=counts,
            grid_shape=(17, 16),
            embed_dim=8,
            backend="attention",
            attention_heads=2,
            attention_n_max=256,
        )


def test_gat_backend_shape_and_gradient() -> None:
    torch.manual_seed(1)
    input_counts = [2, 3, 1, 4]
    output_counts = [1, 2, 3, 1]
    x = torch.randn(5, sum(input_counts), requires_grad=True)

    # Single-head GAT, with global_output_dim to exercise that path too.
    gat_single = CouplingOperator(
        input_component_counts=input_counts,
        output_component_counts=output_counts,
        grid_shape=(2, 2),
        embed_dim=8,
        backend="gat",
        num_layers=2,
        attention_heads=1,
        neighborhood=4,
        global_output_dim=3,
    )
    y = gat_single(x)
    assert tuple(y.shape) == (5, 3 + sum(output_counts)), f"unexpected shape {tuple(y.shape)}"
    y.square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all(), "non-finite gradients in single-head GAT"

    # Multi-head GAT (4 heads), 8-neighbor graph.
    x2 = torch.randn(3, sum(input_counts), requires_grad=True)
    gat_multi = CouplingOperator(
        input_component_counts=input_counts,
        output_component_counts=output_counts,
        grid_shape=(2, 2),
        embed_dim=8,
        backend="gat",
        num_layers=3,
        attention_heads=4,
        neighborhood=8,
        dropout=0.0,
        gat_negative_slope=0.1,
    )
    y2 = gat_multi(x2)
    assert tuple(y2.shape) == (3, sum(output_counts)), f"unexpected shape {tuple(y2.shape)}"
    y2.square().mean().backward()
    assert x2.grad is not None
    assert torch.isfinite(x2.grad).all(), "non-finite gradients in multi-head GAT"


def test_gat_embed_dim_not_divisible_by_heads_raises() -> None:
    with pytest.raises(ValueError, match="divisible by attention_heads"):
        CouplingOperator(
            input_component_counts=[2, 3, 1, 4],
            output_component_counts=[1, 2, 3, 1],
            grid_shape=(2, 2),
            embed_dim=9,
            backend="gat",
            attention_heads=4,
        )


def test_gat_backend_module_is_gat_coupling_backend() -> None:
    op = CouplingOperator(
        input_component_counts=[2, 2],
        output_component_counts=[2, 2],
        grid_shape=(1, 2),
        embed_dim=8,
        backend="gat",
        attention_heads=2,
    )
    assert isinstance(op.backend_module, GATCouplingBackend)


def test_coupling_poisson128_run_improves_mre_over_plain_l2l() -> None:
    set_seed(4)
    x, y = _make_neighbor_coupled_poisson128(n_samples=96)
    encoder = LocalToLocalPCAEncoder(
        patch_size=32,
        stride=32,
        input_variance=1,
        output_variance=1,
        solver="full",
        standardize=True,
        random_state=0,
    ).fit(x[:72], y[:72])
    x_latent = encoder.transform_inputs(x)
    y_latent = encoder.transform_outputs(y)

    train_slice = slice(0, 64)
    val_slice = slice(64, 80)
    test_slice = slice(80, 96)
    plain_pred = _train_and_predict(
        MLP(
            input_dim=x_latent.shape[1],
            output_dim=y_latent.shape[1],
            hidden_size=8,
            num_layers=2,
        ),
        x_latent,
        y_latent,
        train_slice,
        val_slice,
        test_slice,
    )
    counts = encoder.component_counts
    _shared_coupling_kwargs = dict(
        input_component_counts=[int(value) for value in counts["input_patches"]],
        output_component_counts=[int(value) for value in counts["output_patches"]],
        grid_shape=(4, 4),
        embed_dim=32,
        num_layers=2,
        attention_heads=4,
        attention_n_max=256,
        neighborhood=8,
    )
    gnn_pred = _train_and_predict(
        CouplingOperator(backend="gnn", **_shared_coupling_kwargs),
        x_latent,
        y_latent,
        train_slice,
        val_slice,
        test_slice,
    )
    gat_pred = _train_and_predict(
        CouplingOperator(backend="gat", **_shared_coupling_kwargs),
        x_latent,
        y_latent,
        train_slice,
        val_slice,
        test_slice,
    )

    true_fields = y[test_slice]
    plain_mre = mre(encoder.inverse_transform_outputs(plain_pred), true_fields)
    gnn_mre = mre(encoder.inverse_transform_outputs(gnn_pred), true_fields)
    gat_mre = mre(encoder.inverse_transform_outputs(gat_pred), true_fields)
    assert gnn_mre < plain_mre, f"GNN ({gnn_mre:.4f}) did not beat MLP ({plain_mre:.4f})"
    assert gat_mre < plain_mre, f"GAT ({gat_mre:.4f}) did not beat MLP ({plain_mre:.4f})"


def _train_and_predict(
    model: torch.nn.Module,
    x_latent: np.ndarray,
    y_latent: np.ndarray,
    train_slice: slice,
    val_slice: slice,
    test_slice: slice,
) -> np.ndarray:
    train_latent_model(
        model,
        x_latent[train_slice],
        y_latent[train_slice],
        x_latent[val_slice],
        y_latent[val_slice],
        device="cpu",
        batch_size=16,
        epochs=80,
        lr=3.0e-3,
        weight_decay=0.0,
        scheduler=None,
    )
    return predict_latent(model, x_latent[test_slice], device="cpu", batch_size=16)


def _make_neighbor_coupled_poisson128(n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(9)
    axis = np.linspace(0.0, 1.0, 32, dtype=np.float64)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    patch_mode = np.sin(np.pi * xx) * np.sin(np.pi * yy)
    amplitudes = rng.normal(size=(n_samples, 4, 4))
    x = np.zeros((n_samples, 128, 128), dtype=np.float64)
    y = np.zeros_like(x)
    for row in range(4):
        for col in range(4):
            row_slice = slice(32 * row, 32 * (row + 1))
            col_slice = slice(32 * col, 32 * (col + 1))
            local = amplitudes[:, row, col]
            neighbor_values = [local]
            for row_offset in (-1, 0, 1):
                for col_offset in (-1, 0, 1):
                    if row_offset == 0 and col_offset == 0:
                        continue
                    n_row = row + row_offset
                    n_col = col + col_offset
                    if 0 <= n_row < 4 and 0 <= n_col < 4:
                        neighbor_values.append(amplitudes[:, n_row, n_col])
            coupled = np.mean(np.stack(neighbor_values, axis=0), axis=0)
            x[:, row_slice, col_slice] = local[:, None, None] * patch_mode
            y[:, row_slice, col_slice] = coupled[:, None, None] * patch_mode
    x += 0.005 * rng.normal(size=x.shape)
    y += 0.005 * rng.normal(size=y.shape)
    return x.astype(np.float32), y.astype(np.float32)
