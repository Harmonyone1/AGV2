from __future__ import annotations

import os
from typing import Dict
from pathlib import Path

import yaml

from dotenv import load_dotenv, dotenv_values

MANDATORY_KEYS = [
    "DB_NAME",
    "DB_USER",
    "DB_PASSWORD",
    "DB_HOST",
    "DB_PORT",
    "TL_ENVIRONMENT",
    "TL_EMAIL",
    "TL_PASSWORD",
    "TL_SERVER",
    "TL_ACC_NUM",
]


STRATEGY_CONFIG_PATH = Path("config/strategies.yaml")


def _load_strategy_definitions(config_path: Path) -> list[dict[str, object]]:
    if not config_path.exists():
        return []
    with config_path.open("r", encoding="utf-8") as fh:
        parsed = yaml.safe_load(fh) or {}
    strategies = parsed.get("strategies", [])
    if not isinstance(strategies, list):
        raise ValueError("config/strategies.yaml must define a top-level 'strategies' list")
    return strategies


def load_config(env_path: str = ".env") -> Dict[str, str]:
    """Load configuration values from an .env file.

    Parameters
    ----------
    env_path: str, optional
        Path to the .env file. Defaults to ``".env"``.

    Returns
    -------
    Dict[str, str]
        Mapping of configuration keys to their loaded values.

    Raises
    ------
    EnvironmentError
        If any of the :data:`MANDATORY_KEYS` are not present in the loaded
        environment.
    """
    # Load values into the process environment and read key/value pairs
    load_dotenv(env_path)
    values = dotenv_values(env_path)

    # Check required keys
    missing = [k for k in MANDATORY_KEYS if not os.getenv(k)]
    if missing:
        raise EnvironmentError(
            "Missing mandatory configuration values: " + ", ".join(missing)
        )

    # Return a dictionary of all keys from the file merged with environment
    config: Dict[str, str] = {k: os.getenv(k, v) for k, v in values.items()}
    for key in MANDATORY_KEYS:
        config[key] = os.getenv(key, config.get(key))  # ensure mandatory keys present

    strategy_defs = _load_strategy_definitions(STRATEGY_CONFIG_PATH)
    config["strategies"] = strategy_defs
    return config
