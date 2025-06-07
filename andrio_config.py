import os
import json
from pathlib import Path

DEFAULT_CONFIG = {
    "UE_SOURCE_DIR": str(Path.home() / "UE_Source"),
    "UE_INSTALL_DIR": str(Path.home() / "UnrealEngine"),
    "ANDRIO_OUTPUT_DIR": str(Path.home() / "AndrioOutput")
}


def _load_config_file() -> dict:
    """Load configuration from JSON file specified by ANDRIO_CONFIG_FILE or default."""
    config_file = os.environ.get("ANDRIO_CONFIG_FILE", "andrio_config.json")
    if os.path.exists(config_file):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
    return {}


def get_config() -> dict:
    """Return configuration dict merging defaults, file config and environment."""
    config = DEFAULT_CONFIG.copy()
    file_cfg = _load_config_file()
    config.update(file_cfg)

    for key in list(DEFAULT_CONFIG.keys()):
        env_val = os.getenv(key)
        if env_val:
            config[key] = env_val
    return config
