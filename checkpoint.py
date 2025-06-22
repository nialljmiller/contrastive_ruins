import os
import json
import torch

class CheckpointManager:
    """Simple checkpoint manager for training jobs."""

    def __init__(self, directory: str):
        self.directory = directory
        os.makedirs(directory, exist_ok=True)
        self.checkpoint_file = os.path.join(directory, "checkpoint.pt")
        self.meta_file = os.path.join(directory, "checkpoint.json")

    def load(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
        """Load checkpoint if it exists."""
        if not os.path.exists(self.checkpoint_file):
            return 0, []
        data = torch.load(self.checkpoint_file, map_location="cpu")
        model.load_state_dict(data["model_state"])
        optimizer.load_state_dict(data["optimizer_state"])
        history = data.get("history", [])
        start_epoch = int(data.get("epoch", -1)) + 1
        print(f"Loaded checkpoint from epoch {start_epoch}")
        return start_epoch, history

    def save(self, epoch: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer, history: list):
        """Save checkpoint state."""
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "history": history,
        }, self.checkpoint_file)
        with open(self.meta_file, "w") as f:
            json.dump({"epoch": epoch}, f)

    def clear(self):
        for path in [self.checkpoint_file, self.meta_file]:
            if os.path.exists(path):
                os.remove(path)
