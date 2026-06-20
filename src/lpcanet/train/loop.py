"""Training loop utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from lpcanet.losses import CompositeLoss


@dataclass
class TrainingResult:
    """Container for training results."""

    best_state_dict: dict[str, torch.Tensor]
    train_loss: np.ndarray
    val_loss: np.ndarray
    best_val_loss: float
    epochs_ran: int
    stopped_early: bool


def train_latent_model(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    device: str | torch.device = "cpu",
    batch_size: int = 32,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    scheduler: str | None = None,
    patience: int = 30,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
) -> TrainingResult:
    """Train a latent-space model and keep the best validation checkpoint."""
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if early_stopping_patience is not None and early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience must be positive when enabled.")
    if early_stopping_min_delta < 0.0:
        raise ValueError("early_stopping_min_delta must be non-negative.")

    device = torch.device(device)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = None
    if scheduler == "reduce_on_plateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=patience,
            factor=0.5,
        )
    elif scheduler in (None, "none"):
        lr_scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler {scheduler!r}.")

    train_loader = _make_loader(x_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = _make_loader(x_val, y_val, batch_size=batch_size, shuffle=False)

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    stopped_early = False

    for _ in range(epochs):
        model.train()
        train_loss = _run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        model.eval()
        with torch.no_grad():
            val_loss = _run_epoch(model, val_loader, criterion, device, optimizer=None)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if lr_scheduler is not None:
            lr_scheduler.step(val_loss)

        if val_loss < best_val_loss - early_stopping_min_delta:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        else:
            epochs_without_improvement += 1
            if (
                early_stopping_patience is not None
                and epochs_without_improvement >= early_stopping_patience
            ):
                stopped_early = True
                break

    if best_state is None:
        best_state = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
    model.load_state_dict(best_state)
    return TrainingResult(
        best_state_dict=best_state,
        train_loss=np.asarray(train_losses, dtype=np.float32),
        val_loss=np.asarray(val_losses, dtype=np.float32),
        best_val_loss=float(best_val_loss),
        epochs_ran=len(train_losses),
        stopped_early=stopped_early,
    )


def train_in_loop_physical_model(
    model: nn.Module,
    decoder: nn.Module,
    x_train_latent: np.ndarray,
    y_train: np.ndarray,
    x_val_latent: np.ndarray,
    y_val: np.ndarray,
    *,
    forcing_train: np.ndarray | None = None,
    forcing_val: np.ndarray | None = None,
    coefficient_train: np.ndarray | None = None,
    coefficient_val: np.ndarray | None = None,
    loss_config: dict[str, Any],
    device: str | torch.device = "cpu",
    batch_size: int = 32,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    scheduler: str | None = None,
    patience: int = 30,
) -> TrainingResult:
    """Train with differentiable PCA decode, assembly, and physical-field loss."""
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    device = torch.device(device)
    model.to(device)
    decoder.to(device)
    decoder.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = _make_scheduler(optimizer, scheduler, patience)

    train_loader = _make_physical_loader(
        x_train_latent,
        y_train,
        forcing_train,
        coefficient_train,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = _make_physical_loader(
        x_val_latent,
        y_val,
        forcing_val,
        coefficient_val,
        batch_size=batch_size,
        shuffle=False,
    )

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    schedule = LossWarmupSchedule(loss_config)
    validation_criterion = _make_composite_loss(
        loss_config,
        schedule.weights_for_epoch(
            schedule.reconstruction_epochs + schedule.ramp_epochs
        ),
    )

    for epoch_index in range(epochs):
        weights = schedule.weights_for_epoch(epoch_index)
        criterion = _make_composite_loss(loss_config, weights)
        model.train()
        train_loss = _run_physical_epoch(
            model,
            decoder,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
        )
        model.eval()
        with torch.no_grad():
            val_loss = _run_physical_epoch(
                model,
                decoder,
                val_loader,
                validation_criterion,
                device,
                optimizer=None,
            )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if lr_scheduler is not None:
            lr_scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    if best_state is None:
        best_state = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
    model.load_state_dict(best_state)
    return TrainingResult(
        best_state_dict=best_state,
        train_loss=np.asarray(train_losses, dtype=np.float32),
        val_loss=np.asarray(val_losses, dtype=np.float32),
        best_val_loss=float(best_val_loss),
        epochs_ran=len(train_losses),
        stopped_early=False,
    )


def predict_latent(
    model: nn.Module,
    x: np.ndarray,
    *,
    device: str | torch.device = "cpu",
    batch_size: int = 256,
) -> np.ndarray:
    """Run latent model inference and return a NumPy array."""
    device = torch.device(device)
    model.to(device)
    model.eval()
    dataset = TensorDataset(torch.from_numpy(np.asarray(x, dtype=np.float32)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for (x_batch,) in loader:
            pred = model(x_batch.to(device)).detach().cpu().numpy()
            outputs.append(pred)
    return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)


def train_image_model(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    device: str | torch.device = "cpu",
    batch_size: int = 32,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    scheduler: str | None = None,
    patience: int = 30,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
) -> TrainingResult:
    """Train an image-to-image model and keep the best validation checkpoint."""
    return train_latent_model(
        model,
        _as_channel_array(x_train),
        _as_channel_array(y_train),
        _as_channel_array(x_val),
        _as_channel_array(y_val),
        device=device,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        scheduler=scheduler,
        patience=patience,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
    )


def predict_images(
    model: nn.Module,
    x: np.ndarray,
    *,
    device: str | torch.device = "cpu",
    batch_size: int = 256,
) -> np.ndarray:
    """Run image-to-image model inference and return ``(N, H, W)`` arrays."""
    pred = predict_latent(
        model,
        _as_channel_array(x),
        device=device,
        batch_size=batch_size,
    )
    if pred.ndim != 4 or pred.shape[1] != 1:
        raise ValueError(f"Expected image model output shape (N, 1, H, W), got {pred.shape}.")
    return pred[:, 0].astype(np.float32, copy=False)


def _make_loader(
    x: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(x, dtype=np.float32)),
        torch.from_numpy(np.asarray(y, dtype=np.float32)),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _make_physical_loader(
    x_latent: np.ndarray,
    y_field: np.ndarray,
    forcing: np.ndarray | None,
    coefficient: np.ndarray | None,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    tensors = [
        torch.from_numpy(np.asarray(x_latent, dtype=np.float32)),
        torch.from_numpy(np.asarray(y_field, dtype=np.float32)),
    ]
    if forcing is None:
        tensors.append(torch.empty((len(y_field), 0), dtype=torch.float32))
    else:
        tensors.append(torch.from_numpy(np.asarray(forcing, dtype=np.float32)))
    if coefficient is None:
        tensors.append(torch.empty((len(y_field), 0), dtype=torch.float32))
    else:
        tensors.append(torch.from_numpy(np.asarray(coefficient, dtype=np.float32)))
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle)


class LossWarmupSchedule:
    """Scale auxiliary loss weights after an optional reconstruction-only warmup."""

    def __init__(self, loss_config: dict[str, Any]) -> None:
        warmup = loss_config.get("warmup", {})
        self.reconstruction_epochs = int(warmup.get("reconstruction_epochs", 0))
        self.ramp_epochs = int(warmup.get("ramp_epochs", 0))
        self.base_weights = {
            name: float(value)
            for name, value in loss_config.get("weights", {}).items()
        }
        active_terms = loss_config.get("active_terms")
        self.active_terms = set(active_terms) if active_terms is not None else set(self.base_weights)

    def weights_for_epoch(self, epoch_index: int) -> dict[str, float]:
        weights: dict[str, float] = {}
        ramp = self._ramp_factor(epoch_index)
        for name, value in self.base_weights.items():
            if name not in self.active_terms:
                weights[name] = 0.0
            elif name == "recon":
                weights[name] = value
            else:
                weights[name] = value * ramp
        return weights

    def _ramp_factor(self, epoch_index: int) -> float:
        if epoch_index < self.reconstruction_epochs:
            return 0.0
        if self.ramp_epochs <= 0:
            return 1.0
        ramp_step = epoch_index - self.reconstruction_epochs + 1
        return min(1.0, max(0.0, float(ramp_step) / float(self.ramp_epochs)))


def _make_composite_loss(loss_config: dict[str, Any], weights: dict[str, float]) -> CompositeLoss:
    return CompositeLoss(
        weights=weights,
        reconstruction=str(loss_config.get("reconstruction", "relative_l2")),
        patch_size=int(loss_config["patch_size"]),
        stride=int(loss_config["stride"]),
        dx=float(loss_config["dx"]),
        include_edges=bool(loss_config.get("include_edges", False)),
        pde=str(loss_config.get("pde", "poisson")),
        pde_reduction=str(loss_config.get("pde_reduction", "rms")),
        spectral_high_k_weight_power=float(loss_config.get("spectral_high_k_weight_power", 1.0)),
        interface_target=str(loss_config.get("interface_target", "zero")),
    )


def _as_channel_array(fields: np.ndarray) -> np.ndarray:
    fields = np.asarray(fields, dtype=np.float32)
    if fields.ndim == 3:
        return fields[:, None, :, :]
    if fields.ndim == 4:
        return fields
    raise ValueError(f"Expected fields with shape (N, H, W) or (N, C, H, W), got {fields.shape}.")


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    total_loss = 0.0
    total_count = 0
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        if optimizer is not None:
            optimizer.zero_grad()
        pred = model(x_batch)
        loss = criterion(pred, y_batch)
        if optimizer is not None:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item()) * x_batch.shape[0]
        total_count += int(x_batch.shape[0])
    if total_count == 0:
        raise ValueError("Cannot train/evaluate on an empty dataset.")
    return total_loss / total_count


def _run_physical_epoch(
    model: nn.Module,
    decoder: nn.Module,
    loader: DataLoader,
    criterion: CompositeLoss,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    total_loss = 0.0
    total_count = 0
    for x_batch, y_batch, forcing_batch, coefficient_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        forcing_arg = _optional_batch(forcing_batch, device)
        coefficient_arg = _optional_batch(coefficient_batch, device)
        if optimizer is not None:
            optimizer.zero_grad()
        pred_latent = model(x_batch)
        pred_field = decoder(pred_latent)
        loss = criterion(
            pred_field,
            y_batch,
            forcing=forcing_arg,
            coefficient=coefficient_arg,
        )
        if optimizer is not None:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item()) * x_batch.shape[0]
        total_count += int(x_batch.shape[0])
    if total_count == 0:
        raise ValueError("Cannot train/evaluate on an empty dataset.")
    return total_loss / total_count


def _optional_batch(batch: torch.Tensor, device: torch.device) -> torch.Tensor | None:
    if batch.ndim == 2 and batch.shape[1] == 0:
        return None
    return batch.to(device)


def _make_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler: str | None,
    patience: int,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    if scheduler == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=patience,
            factor=0.5,
        )
    if scheduler in (None, "none"):
        return None
    raise ValueError(f"Unsupported scheduler {scheduler!r}.")
