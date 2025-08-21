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


def load_paropt_config(path: str) -> Dict[str, Any]:
    """Load configuration for the paropt utility.

    The configuration is read from ``path`` and supplemented with default
    values for optional fields.

    Args:
        path: Location of the configuration file.

    Returns:
        A dictionary containing the paropt configuration.
    """
    config = load_config(path)

    defaults = {
        "gru4rec_model": "gru4rec_pytorch",
        "fixed_parameters": "",
        "measure": 20,
        "ntrials": 50,
        "final_measure": [20],
        "primary_metric": "recall",
        "eval_type": "standard",
        "device": "cuda:0",
        "item_key": "ItemId",
        "session_key": "SessionId",
        "time_key": "Time",
    }

    for key, val in defaults.items():
        config.setdefault(key, val)

    if not isinstance(config.get("final_measure"), list):
        config["final_measure"] = [config["final_measure"]]

    return config
