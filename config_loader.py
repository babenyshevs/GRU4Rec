import json
import os
from typing import Any, Dict

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def load_config(path: str) -> Dict[str, Any]:
    """Load a JSON or YAML configuration file.

    Args:
        path: Location of the configuration file.

    Returns:
        A dictionary with the loaded configuration.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is unsupported or YAML support is
            requested but unavailable.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    with open(path, "r") as fh:
        if ext == ".json":
            return json.load(fh)
        if ext in {".yaml", ".yml"}:
            if yaml is None:
                raise ValueError("PyYAML is required to load YAML files")
            return yaml.safe_load(fh)
    raise ValueError(f"Unsupported config extension: {ext}")
