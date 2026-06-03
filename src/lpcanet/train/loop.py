"""Training loop utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainingResult:
    """Container for training results."""

    best_state_dict: dict[str, torch.Tensor]
    train_loss: np.ndarray
    val_loss: np.ndarray
    best_val_loss: float


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
) -> TrainingResult:
    """Train a latent-space model and keep the best validation checkpoint."""
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

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
