"""Evaluate policies via the trading environment and vectorized backtester."""
from __future__ import annotations

import argparse
from typing import Any, Dict

import yaml

from backtest import VectorizedBacktester
from rl import RewardConfig, TradingEnv

try:  # Optional dependency for loading PPO checkpoints
    from stable_baselines3 import PPO
except Exception:  # pragma: no cover - dependency optional
    PPO = None  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest a trained PPO policy")
    parser.add_argument("--config", default="config/backtest.yaml", help="Backtest YAML config")
    parser.add_argument("--model", default=None, help="Path to SB3 checkpoint (overrides config)")
    parser.add_argument("--episodes", type=int, default=None, help="Number of evaluation episodes")
    parser.add_argument("--random", action="store_true", help="Use random policy instead of loading a checkpoint")
    parser.add_argument("--symbol", default=None, help="Override symbol defined in config")
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
    args = parse_args()
    cfg = _load_yaml(args.config)
    env_cfg = cfg.get("environment", {})
    eval_cfg = cfg.get("evaluation", {})

    data_path = env_cfg.get("data_path")
    if not data_path:
        raise ValueError("environment.data_path must be provided")
    reward_cfg = RewardConfig(**(env_cfg.get("reward", {})))
    env_kwargs = build_env_kwargs(env_cfg, reward_cfg, data_path, args.symbol)
    env = TradingEnv(**env_kwargs)

    episodes = args.episodes or int(eval_cfg.get("episodes", 5))
    model_path = args.model or eval_cfg.get("model_path")
    random_policy = args.random or bool(eval_cfg.get("random_policy", False))

    policy = None
    if not random_policy:
        if PPO is None:
            raise ImportError(
                "stable-baselines3 is required to load PPO checkpoints. Install it via 'pip install stable-baselines3'."
            )
        if not model_path:
            raise ValueError("Model path must be specified via --model or evaluation.model_path")
        policy = PPO.load(model_path)

    positions, returns = _rollout(env, policy, episodes)

    # Determine annualization factor based on symbol type
    # Crypto (24/7): 365 days, Metals/Indices: 252 days
    annualization = 365.0 if env.symbol in ["ETH", "BTC", "SOL"] else 252.0

    bt = VectorizedBacktester(returns, trading_cost_bps=env.trading_cost_bps, annualization_factor=annualization)
    result = bt.run(positions)
    print("Backtest results:")
    print(f"  Symbol        : {env.symbol}")
    print(f"  Episodes      : {episodes}")
    print(f"  Total return  : {result.total_return:.4f}")
    print(f"  Sharpe        : {result.sharpe:.3f} (annualized with factor {annualization:.0f})")
    print(f"  Max drawdown  : {result.max_drawdown:.4f}")


def _rollout(env: TradingEnv, policy, episodes: int):
    positions: list[float] = []
    returns: list[float] = []
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        while not done:
            if policy is None:
                action = env.action_space.sample()
            else:
                action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            returns.append(float(info["price_return"]))
            positions.append(float(info["position"]))
            done = terminated or truncated
    return positions[: len(returns)], returns


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


if __name__ == "__main__":
    main()
