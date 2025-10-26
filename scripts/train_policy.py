"""Stage-2 PPO training script."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml
import torch

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

try:
    import mlflow  # type: ignore
except Exception:  # pragma: no cover
    mlflow = None  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO trading policy")
    parser.add_argument("--config", default="config/rl_policy.yaml", help="Path to RL policy YAML")
    parser.add_argument("--data", default=None, help="Override data path")
    parser.add_argument("--timesteps", type=int, default=None, help="Override total timesteps")
    parser.add_argument("--output", default=None, help="Override checkpoint directory")
    parser.add_argument("--symbol", default=None, help="Override symbol (uses config by default)")
    parser.add_argument("--device", default=None, help="torch device for PPO (cuda/cpu/auto)")
    return parser.parse_args()


def build_env_kwargs(env_cfg: Dict[str, Any], reward_cfg: RewardConfig, data_path: str, symbol_override: str | None):
    symbol = symbol_override or env_cfg.get("symbol")
    return dict(
        data=data_path,
        symbol=symbol,
        episode_length=int(env_cfg.get("episode_length", 256)),
        trading_cost_bps=float(env_cfg.get("trading_cost_bps", reward_cfg.trading_cost_bps)),
        max_position=float(env_cfg.get("max_position", 1.0)),
        reward_cfg=reward_cfg,
        market_config_path=env_cfg.get("market_config", "config/markets.yaml"),
        cost_config_path=env_cfg.get("cost_config", "config/costs.yaml"),
        execution_profile=env_cfg.get("execution_profile", "backtesting"),
        order_types=env_cfg.get("order_types"),
        limit_order_fill_bps=float(env_cfg.get("limit_order_fill_bps", 5.0)),
        limit_offset_bps=env_cfg.get("limit_offset_bps"),
        bracket_take_profit_bps=env_cfg.get("bracket_take_profit_bps"),
        bracket_stop_loss_bps=env_cfg.get("bracket_stop_loss_bps"),
        position_sizes=env_cfg.get("position_sizes"),
        include_position_in_obs=bool(env_cfg.get("include_position_in_obs", True)),
    )


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
    env_kwargs = build_env_kwargs(env_cfg, reward_cfg, data_path, args.symbol)
    env = TradingEnv(**env_kwargs)

    def _make_env():
        return ObservationStatsWrapper(TradingEnv(**env_kwargs))

    vec_env = DummyVecEnv([_make_env])

    train_cfg = cfg.get("training", {})
    total_steps = args.timesteps or int(train_cfg.get("total_timesteps", 100_000))
    output_dir = Path(args.output or train_cfg.get("checkpoint_dir", "models/policies"))
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or env_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "auto")
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
        "device": device,
    }

    # Initialize MLflow tracking if enabled
    use_mlflow = mlflow is not None and cfg.get("logging", {}).get("use_mlflow", False)
    if use_mlflow:
        experiment_name = cfg.get("logging", {}).get("experiment_name", "agv2_policy_training")
        mlflow.set_experiment(experiment_name)
        mlflow.start_run()
        # Log hyperparameters
        mlflow.log_params({
            "symbol": env.symbol,
            "episode_length": env.episode_length,
            "max_position": env.max_position,
            "total_timesteps": total_steps,
            **ppo_kwargs,
        })

    model = PPO("MlpPolicy", vec_env, verbose=1, **ppo_kwargs)
    model.learn(total_timesteps=total_steps)

    if use_mlflow:
        mlflow.end_run()

    # Save checkpoint with timestamp for versioning
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint = output_dir / f"ppo_trading_env_{timestamp}"
    model.save(checkpoint)
    print(f"Saved policy checkpoint to {checkpoint}")
    # Also save as latest for convenience
    latest = output_dir / "ppo_trading_env"
    model.save(latest)
    print(f"Saved latest checkpoint to {latest}")


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


if __name__ == "__main__":
    main()




