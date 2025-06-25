import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional
import matplotlib.pyplot as plt

from checkpoint import CheckpointManager
from visualization import (
    plot_training_loss,
    compute_embeddings,
    plot_latent_overlay,
)


class Encoder(nn.Module):
    """Simple convolutional encoder."""

    def __init__(self, input_channels: int = 1, embedding_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(256, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.norm(x)
        return x


def contrastive_loss(y_true: torch.Tensor, distances: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """Compute contrastive loss."""
    y_true = y_true.float()
    positive = y_true * distances.pow(2)
    negative = (1 - y_true) * torch.clamp(margin - distances, min=0).pow(2)
    return torch.mean(positive + negative)


class SiameseNetwork(nn.Module):
    def __init__(self, encoder: Encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        e1 = self.encoder(x1)
        e2 = self.encoder(x2)
        return torch.sqrt(((e1 - e2) ** 2).sum(dim=1))


def _prepare_tensor(x: np.ndarray) -> torch.Tensor:
    if x.ndim == 3:  # N,H,W
        x = np.expand_dims(x, 1)
    elif x.ndim == 4 and x.shape[-1] in {1, 3}:  # N,H,W,C -> N,C,H,W
        x = np.transpose(x, (0, 3, 1, 2))
    return torch.from_numpy(x).float()


def train_siamese_model(
    X1: np.ndarray,
    X2: np.ndarray,
    labels: np.ndarray,
    embedding_dim: int = 128,
    epochs: int = 20,
    batch_size: int = 32,
    margin: float = 1.0,
    save_dir: str = "models",
    checkpoint_dir: str | None = None,
    checkpoint_interval: int = 1,
    plot_interval: int = 1,
    latent_interval: int | None = None,
    latent_sample_size: int = 1000,
    latent_quick_interval: int | None = None,
    latent_quick_size: int = 200,
):
    """Train Siamese network using PyTorch.

    Args:
        X1, X2: Patch pairs for contrastive training.
        labels: Pair labels (1=same, 0=different).
        embedding_dim: Dimension of the encoder output.
        epochs: Number of training epochs.
        batch_size: Batch size for DataLoader.
        margin: Contrastive loss margin.
        save_dir: Directory to save models and plots.
        checkpoint_dir: Directory for checkpoint files.
        checkpoint_interval: Save checkpoint every N epochs.
        plot_interval: Save loss plot every N epochs.
        latent_interval: If set, generate latent UMAP/t-SNE/PCA plots every N epochs.
        latent_sample_size: Number of patches to sample for latent plots.
        latent_quick_interval: If set, generate quick latent plots every N epochs.
        latent_quick_size: Sample size for quick latent plots.
    """
    X1_t = _prepare_tensor(X1)
    X2_t = _prepare_tensor(X2)
    y_t = torch.from_numpy(labels).float()

    dataset = TensorDataset(X1_t, X2_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_channels = X1_t.shape[1]
    encoder = Encoder(input_channels=input_channels, embedding_dim=embedding_dim)
    model = SiameseNetwork(encoder)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1 and device.type == "cuda":
        print(f"Using {torch.cuda.device_count()} GPUs for training")
        model = nn.DataParallel(model)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(save_dir, "checkpoints")
    ckpt = CheckpointManager(checkpoint_dir)

    start_epoch, history = ckpt.load(
        model.module if isinstance(model, nn.DataParallel) else model,
        optimizer,
    )
    model.train()
    try:
        for epoch in range(start_epoch, epochs):
            epoch_loss = 0.0
            for a, b, lbl in loader:
                a, b, lbl = a.to(device), b.to(device), lbl.to(device)
                optimizer.zero_grad()
                dist = model(a, b)
                loss = contrastive_loss(lbl, dist, margin)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * a.size(0)
            epoch_loss /= len(loader.dataset)
            history.append(epoch_loss)
            print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f}")

            if (epoch + 1) % checkpoint_interval == 0:
                ckpt.save(
                    epoch,
                    model.module if isinstance(model, nn.DataParallel) else model,
                    optimizer,
                    history,
                )
            if (epoch + 1) % plot_interval == 0:
                plot_training_loss(history, os.path.join(save_dir, "training_history.png"))
            if latent_interval and (epoch + 1) % latent_interval == 0:
                sample_idx = np.random.choice(len(X1), min(latent_sample_size, len(X1)), replace=False)
                sample_patches = X1[sample_idx]
                emb = compute_embeddings(encoder, sample_patches)
                plot_latent_overlay(
                    emb,
                    None,
                    output_dir=save_dir,
                    prefix=f"latent_epoch_{epoch+1}",
                )
            if latent_quick_interval and (epoch + 1) % latent_quick_interval == 0:
                sample_idx = np.random.choice(len(X1), min(latent_quick_size, len(X1)), replace=False)
                sample_patches = X1[sample_idx]
                emb = compute_embeddings(encoder, sample_patches)
                plot_latent_overlay(
                    emb,
                    None,
                    output_dir=save_dir,
                    prefix=f"latent_quick_epoch_{epoch+1}",
                )
    except KeyboardInterrupt:
        print("Training interrupted. Saving checkpoint...")
        ckpt.save(
            epoch,
            model.module if isinstance(model, nn.DataParallel) else model,
            optimizer,
            history,
        )
        raise

    os.makedirs(save_dir, exist_ok=True)
    torch.save(encoder.state_dict(), os.path.join(save_dir, "encoder.pt"))
    torch.save(
        model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
        os.path.join(save_dir, "siamese.pt"),
    )
    ckpt.save(
        epochs - 1,
        model.module if isinstance(model, nn.DataParallel) else model,
        optimizer,
        history,
    )
    plot_training_loss(history, os.path.join(save_dir, "training_history.png"))
    if latent_interval:
        sample_idx = np.random.choice(len(X1), min(latent_sample_size, len(X1)), replace=False)
        sample_patches = X1[sample_idx]
        emb = compute_embeddings(encoder, sample_patches)
        plot_latent_overlay(emb, None, output_dir=save_dir, prefix="latent_final")
    if latent_quick_interval:
        sample_idx = np.random.choice(len(X1), min(latent_quick_size, len(X1)), replace=False)
        sample_patches = X1[sample_idx]
        emb = compute_embeddings(encoder, sample_patches)
        plot_latent_overlay(emb, None, output_dir=save_dir, prefix="latent_quick_final")

    return encoder, model, history



def load_models(save_dir: str = "models", device: Optional[torch.device] = None):
    """Load saved PyTorch models."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder()
    model = SiameseNetwork(encoder)
    if torch.cuda.device_count() > 1 and device.type == "cuda":
        model = nn.DataParallel(model)
    encoder_path = os.path.join(save_dir, "encoder.pt")
    siamese_path = os.path.join(save_dir, "siamese.pt")
    if os.path.exists(encoder_path):
        encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))
    if os.path.exists(siamese_path):
        state = torch.load(siamese_path, map_location="cpu")
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(state)
        else:
            model.load_state_dict(state)

    encoder.to(device)
    model.to(device)
    if isinstance(model, nn.DataParallel):
        model.module.encoder = encoder
    else:
        model.encoder = encoder
    encoder.eval()
    model.eval()
    
    return encoder, model


if __name__ == "__main__":
    X1 = np.random.rand(10, 64, 64)
    X2 = np.random.rand(10, 64, 64)
    y = np.random.randint(0, 2, size=10)
    train_siamese_model(
        X1,
        X2,
        y,
        epochs=1,
        latent_interval=None,
        latent_quick_interval=None,
    )
