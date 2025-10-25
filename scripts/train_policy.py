"""Stage-2 PPO training script."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml

from rl import ObservationStatsWrapper, RewardConfig, TradingEnv

try:  # Optional dependency for actual training
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
except Exception as exc:  # pragma: no cover
    PPO = None  # type: ignore
    DummyVecEnv = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO trading policy")
    parser.add_argument("--config", default="config/rl_policy.yaml", help="Path to RL policy YAML")
    parser.add_argument("--data", default=None, help="Override data path")
    parser.add_argument("--timesteps", type=int, default=None, help="Override total timesteps")
    parser.add_argument("--output", default=None, help="Override checkpoint directory")
    return parser.parse_args()


def main() -> None:
    if PPO is None or DummyVecEnv is None:
        raise ImportError(
            "stable-baselines3 (and torch) are required to run train_policy.py. "
            "Install them via 'pip install torch stable-baselines3'."
        ) from _IMPORT_ERROR

    args = parse_args()
    cfg = _load_yaml(args.config)
    env_cfg = cfg.get("environment", {})
    data_path = args.data or env_cfg.get("data_path")
    if data_path is None:
        raise ValueError("Data path must be provided via --data or config file")

    reward_cfg = RewardConfig(**env_cfg.get("reward", {}))
    env = TradingEnv(
        data=data_path,
        episode_length=int(env_cfg.get("episode_length", 256)),
        trading_cost_bps=float(env_cfg.get("trading_cost_bps", reward_cfg.trading_cost_bps)),
        max_position=float(env_cfg.get("max_position", 1.0)),
        reward_cfg=reward_cfg,
    )

    def _make_env():
        return ObservationStatsWrapper(TradingEnv(
            data=data_path,
            episode_length=env.episode_length,
            trading_cost_bps=env.trading_cost_bps,
            max_position=env.max_position,
            reward_cfg=reward_cfg,
        ))

    vec_env = DummyVecEnv([_make_env])

    train_cfg = cfg.get("training", {})
    total_steps = args.timesteps or int(train_cfg.get("total_timesteps", 100_000))
    output_dir = Path(args.output or train_cfg.get("checkpoint_dir", "models/policies"))
    output_dir.mkdir(parents=True, exist_ok=True)

    ppo_kwargs = {
        "learning_rate": float(train_cfg.get("learning_rate", 3e-4)),
        "batch_size": int(train_cfg.get("batch_size", 512)),
        "n_steps": int(train_cfg.get("n_steps", 2048)),
        "gamma": float(train_cfg.get("gamma", 0.99)),
        "gae_lambda": float(train_cfg.get("gae_lambda", 0.95)),
        "clip_range": float(train_cfg.get("clip_range", 0.2)),
        "ent_coef": float(train_cfg.get("ent_coef", 0.0)),
        "vf_coef": float(train_cfg.get("vf_coef", 0.5)),
        "seed": int(train_cfg.get("seed", 42)),
    }

    model = PPO("MlpPolicy", vec_env, verbose=1, **ppo_kwargs)
    model.learn(total_timesteps=total_steps)
    checkpoint = output_dir / "ppo_trading_env"
    model.save(checkpoint)
    print(f"Saved policy checkpoint to {checkpoint}")


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


if __name__ == "__main__":
    main()
