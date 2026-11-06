import json
import torch
from pathlib import Path
from safetensors import safe_open


# Base loader that loads latent tensors in their native (N, C, H, W) format for CNNs
class BaseDataLoader():
    """Loads 3D SD3 latent tensors (C, H, W) into a 4D batch tensor (N, C, H, W) for CNNs."""
    

    def __init__(self, directory, percent, results_json_path, **kwargs):
        self.directory = Path(directory)
        self.results_json_path = Path(results_json_path)
        self.percent = percent
        self.tensor_key = f"latent_{percent:03d}"
        self.data = None
        self.labels = None

    def load_data(self):
        """
        Loads the latent tensors and binary labels for the specified percent across all safetensor files.
        Returns a data tensor of shape (n_samples, 16, 128, 128) and a labels tensor of shape (n_samples,).
        """
        # Load the review JSON and map each sample id to 1 (correct) or 0 (incorrect/uncertain)
        with open(self.results_json_path, "r", encoding="utf-8") as f:
            reviews = json.load(f)

        labels_by_id = {}
        for entry in reviews:
            sample_id = entry["id"]
            if entry["status"] == "correct":
                labels_by_id[sample_id] = 1
            else:
                labels_by_id[sample_id] = 0

        # get all safetensor files in the directory and any subdirectories in deterministic sorted order
        files = [f for f in self.directory.rglob("*.safetensors") if f.is_file()]
        files.sort()

        tensors = []
        labels = []
        for file in files:
            file_id = file.stem
            labels.append(labels_by_id[file_id])

            with safe_open(file, framework="pt", device="cpu") as f:
                # Native format: shape (16, 128, 128), dtype torch.float16
                tensor = f.get_tensor(self.tensor_key)
                tensors.append(tensor)

        # Stack into (n_samples, 16, 128, 128) Channels-First (N, C, H, W) format
        self.data = torch.stack(tensors)
        self.labels = torch.tensor(labels, dtype=torch.long)
        return self.data, self.labels


# Loader that flattens the latent tensors into (n_samples, n_features) for linear models or DNNs
class FlatDataLoader(BaseDataLoader):
    """Loads SD3 latent tensors and flattens each sample into a 1D feature vector for linear models/DNNs."""

    def __init__(self, directory, percent, results_json_path, **kwargs):
        super().__init__(directory, percent, results_json_path, **kwargs)

    def load_data(self):
        """
        Loads the latent tensors and labels using BaseDataLoader and flattens each sample.
        Returns a data tensor of shape (n_samples, 262144) and a labels tensor of shape (n_samples,).
        """
        # Load the 4D tensor (n_samples, 16, 128, 128) and labels via BaseDataLoader
        super().load_data()

        # Flatten (C, H, W) dimensions into (n_samples, 16 * 128 * 128) = (n_samples, 262144)
        self.data = self.data.flatten(start_dim=1)
        return self.data, self.labels