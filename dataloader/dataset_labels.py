import json
import os
from typing import Dict, List


class DatasetLabels:
    _config = None

    @classmethod
    def _load_config(cls):
        """Load dataset labels from JSON file."""
        if cls._config is None:
            config_path = os.path.join(os.path.dirname(__file__), "..", "config", "dataset_labels.json")
            with open(config_path, "r") as f:
                cls._config = json.load(f)
        return cls._config

    @classmethod
    def get_dataset_config(cls, dataset_name: str) -> Dict[str, any]:
        """Get label configuration for a dataset.

        Args:    dataset_name: Dataset name (case-insensitive)
        Returns: Dictionary with 'coarse', 'fine', and 'hierarchy' keys
        """
        config = cls._load_config()
        dataset_name = dataset_name.upper()
        if dataset_name in config:
            return config[dataset_name]
        else:
            raise NotImplementedError(f"Dataset {dataset_name} not implemented.")

    @classmethod
    def get_coarse_labels(cls, dataset_name: str) -> List[str]:
        """Get coarse labels for a dataset."""
        config = cls.get_dataset_config(dataset_name)
        return config["coarse"]

    @classmethod
    def get_fine_labels(cls, dataset_name: str) -> List[str]:
        """Get fine labels for a dataset."""
        config = cls.get_dataset_config(dataset_name)
        return config["fine"]

    @classmethod
    def get_hierarchy_map(cls, dataset_name: str) -> Dict[str, str]:
        """Get hierarchy mapping for a dataset."""
        config = cls.get_dataset_config(dataset_name)
        return config["hierarchy"]
