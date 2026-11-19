import json
import torch
from pathlib import Path
from safetensors import safe_open


# Base loader that loads cached PCA (or latent) tensors in (N, C, H, W) format along with metadata
class BaseDataLoader:
    """Loads 3D SD3 feature tensors (C, H, W) and metadata (target_count, one-hot class) for CNNs."""

    def __init__(
        self,
        directory,
        percent,
        results_json_path,
        tensor_prefix="pca",
        class_names=None,
        **kwargs,
    ):
        self.directory = Path(directory)
        self.results_json_path = Path(results_json_path)
        self.percent = percent
        self.tensor_prefix = tensor_prefix
        self.tensor_key = f"{tensor_prefix}_{percent:03d}"

        # Pass training_data.class_names to testing_data so one-hot columns align identically
        self.class_names = list(class_names) if class_names is not None else None

        self.spatial_data = None
        self.metadata = None
        self.data = None
        self.labels = None
        self.sample_ids = None

    def _load_sample_metadata(self, file_path, safe_handle):
        """Read (object_class, target_count) from safetensors metadata or sibling JSON."""
        header_meta = safe_handle.metadata() or {}
        if "object_class" in header_meta and "target_count" in header_meta:
            return str(header_meta["object_class"]), int(header_meta["target_count"])

        json_path = file_path.with_suffix(".json")
        if json_path.is_file():
            with open(json_path, "r", encoding="utf-8") as f:
                json_meta = json.load(f)
            if "object_class" in json_meta and "target_count" in json_meta:
                return str(json_meta["object_class"]), int(json_meta["target_count"])

        raise ValueError(
            f"Could not find 'object_class' and 'target_count' in {file_path} metadata or {json_path}."
        )

    def _encode_metadata(self, target_counts, object_classes):
        """Encode [target_count, one_hot(object_class)] as a (n_samples, 1 + n_classes) float32 tensor."""
        if self.class_names is None:
            self.class_names = sorted(set(object_classes))

        class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        num_classes = len(self.class_names)

        meta_rows = []
        for target_count, obj_class in zip(target_counts, object_classes):
            if obj_class not in class_to_idx:
                raise ValueError(
                    f"Unseen object class '{obj_class}'. Expected one of {self.class_names}."
                )
            one_hot = [0.0] * num_classes
            one_hot[class_to_idx[obj_class]] = 1.0
            meta_rows.append([float(target_count)] + one_hot)

        return torch.tensor(meta_rows, dtype=torch.float32)

    def load_data(self):
        """
        Loads feature tensors, encoded metadata, and binary labels across all safetensor files.
        Returns ((spatial_data, metadata), labels):
          - spatial_data: shape (n_samples, C, H, W) (e.g., (n_samples, 48, 48, 48) for PCA48)
          - metadata:     shape (n_samples, 1 + n_classes) [target_count, one-hot object_class]
          - labels:       shape (n_samples,) binary 0/1 accuracy labels
        """
        # Load the review JSON and map each sample id to 1 (correct) or 0 (incorrect/uncertain)
        with open(self.results_json_path, "r", encoding="utf-8") as f:
            reviews = json.load(f)

        labels_by_id = {}
        for entry in reviews:
            sample_id = entry["id"]
            labels_by_id[sample_id] = 1 if entry["status"] == "correct" else 0

        # Get all sample safetensor files in deterministic sorted order (skip basis/partial files)
        files = [
            f for f in self.directory.rglob("*.safetensors")
            if f.is_file() and not f.stem.endswith("_basis") and ".partial" not in f.name
        ]
        files.sort()

        if not files:
            raise FileNotFoundError(f"No sample .safetensors files found in {self.directory}.")

        tensors = []
        labels = []
        target_counts = []
        object_classes = []
        sample_ids = []

        for file in files:
            file_id = file.stem
            if file_id not in labels_by_id:
                raise KeyError(f"Sample ID '{file_id}' not found in {self.results_json_path}.")

            with safe_open(file, framework="pt", device="cpu") as f:
                tensor = f.get_tensor(self.tensor_key)
                obj_class, target_count = self._load_sample_metadata(file, f)

            tensors.append(tensor)
            labels.append(labels_by_id[file_id])
            target_counts.append(target_count)
            object_classes.append(obj_class)
            sample_ids.append(file_id)

        # Stack into Channels-First (N, C, H, W) format and pair with encoded metadata
        self.spatial_data = torch.stack(tensors)
        self.metadata = self._encode_metadata(target_counts, object_classes)
        self.data = (self.spatial_data, self.metadata)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.sample_ids = sample_ids
        return self.data, self.labels


# Loader that flattens feature tensors and appends metadata into (n_samples, n_features)
class FlatDataLoader(BaseDataLoader):
    """Loads SD3 PCA/latent tensors, flattens each sample, and appends numeracy + class features."""

    def __init__(
        self,
        directory,
        percent,
        results_json_path,
        tensor_prefix="pca",
        class_names=None,
        **kwargs,
    ):
        super().__init__(
            directory,
            percent,
            results_json_path,
            tensor_prefix=tensor_prefix,
            class_names=class_names,
            **kwargs,
        )

    def load_data(self):
        """
        Loads spatial tensors, metadata, and labels via BaseDataLoader, flattens spatial features,
        and concatenates metadata into a 2D tensor of shape (n_samples, C*H*W + 1 + n_classes).
        """
        super().load_data()

        # Cast to float32 before concatenating with float32 metadata
        flat_spatial = self.spatial_data.flatten(start_dim=1).float()
        self.data = torch.cat([flat_spatial, self.metadata], dim=1)
        return self.data, self.labels