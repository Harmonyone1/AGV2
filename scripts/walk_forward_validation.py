"""Walk-forward validation for AGV2 trading system.

Implements time-series cross-validation with proper train/test splits:
1. Split data into rolling windows
2. Train encoder on train window
3. Generate embeddings
4. Train policy on embeddings
5. Evaluate on test window
6. Record out-of-sample metrics

This prevents lookahead bias and provides realistic performance estimates.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import polars as pl
import torch
import yaml

from data.price_action_dataset import PriceActionDataset
from models.encoder import StageOneEncoder
from rl import RewardConfig, TradingEnv
from backtest import VectorizedBacktester

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
except Exception:
    PPO = None
    DummyVecEnv = None

try:
    import mlflow
except Exception:
    mlflow = None


@dataclass
class WindowConfig:
    """Configuration for a single train/test window."""
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    window_id: int


@dataclass
class WindowResults:
    """Results from evaluating a single window."""
    window_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_samples: int
    test_samples: int
    encoder_loss: float
    policy_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    avg_trade_duration: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward validation for AGV2")
    parser.add_argument("--data", required=True, help="Path to parquet file with windowed features")
    parser.add_argument("--config", default="config/walk_forward.yaml", help="Walk-forward config")
    parser.add_argument("--encoder-config", default="config/encoder.yaml", help="Encoder config")
    parser.add_argument("--policy-config", default="config/rl_policy.yaml", help="Policy config")
    parser.add_argument("--output", default="results/walk_forward", help="Output directory")
    parser.add_argument("--device", default="auto", help="Device (cuda/cpu/auto)")
    return parser.parse_args()


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def create_time_windows(
    data_frame: pl.DataFrame,
    train_duration_days: int,
    test_duration_days: int,
    step_days: int,
) -> List[WindowConfig]:
    """Create rolling time windows for walk-forward validation."""
    if "timestamp" not in data_frame.columns:
        raise ValueError("DataFrame must contain 'timestamp' column")

    timestamps = data_frame["timestamp"].to_list()
    start_date = min(timestamps)
    end_date = max(timestamps)

    windows: List[WindowConfig] = []
    current_start = start_date
    window_id = 0

    while True:
        train_start = current_start
        train_end = train_start + timedelta(days=train_duration_days)
        test_start = train_end
        test_end = test_start + timedelta(days=test_duration_days)

        if test_end > end_date:
            break

        windows.append(WindowConfig(
            train_start=train_start.isoformat(),
            train_end=train_end.isoformat(),
            test_start=test_start.isoformat(),
            test_end=test_end.isoformat(),
            window_id=window_id,
        ))

        current_start += timedelta(days=step_days)
        window_id += 1

    return windows


def filter_by_time(df: pl.DataFrame, start: str, end: str) -> pl.DataFrame:
    """Filter dataframe by timestamp range."""
    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)
    return df.filter(
        (pl.col("timestamp") >= start_dt) & (pl.col("timestamp") < end_dt)
    )


def train_encoder_on_window(
    train_data: pl.DataFrame,
    encoder_config: Dict[str, Any],
    device: torch.device,
    output_dir: Path,
    window_id: int,
) -> float:
    """Train encoder on a specific time window."""
    print(f"\n[Window {window_id}] Training encoder...")

    # Create temporary dataset
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp_path = tmp.name
        train_data.write_parquet(tmp_path)

    arch_cfg = encoder_config.get("architecture", {})
    input_cfg = arch_cfg.get("input", {})
    window_len = int(input_cfg.get("window_length", 512))
    feature_dim = int(input_cfg.get("feature_dim", 12))

    dataset = PriceActionDataset(tmp_path, window_len, feature_dim, required_columns=["features"])

    # Simple train loop (reduced epochs for walk-forward)
    model = StageOneEncoder(encoder_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(encoder_config.get("training", {}).get("learning_rate", 3e-4)),
    )

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(encoder_config.get("training", {}).get("batch_size", 64)),
        shuffle=True,
    )

    num_epochs = int(encoder_config.get("training", {}).get("num_epochs", 5))
    best_loss = float("inf")

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in train_loader:
            features = batch["features"].to(device)
            optimizer.zero_grad()

            preds = model(features)
            # Simple MSE loss on embedding reconstruction
            if "masked_reconstruction" in preds:
                target = features[..., : preds["masked_reconstruction"].shape[-1]]
                loss = torch.nn.functional.mse_loss(preds["masked_reconstruction"], target)
            else:
                loss = torch.tensor(0.0, device=device)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(train_loader), 1)
        best_loss = min(best_loss, avg_loss)
        print(f"  Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f}")

    # Save encoder checkpoint
    checkpoint_path = output_dir / f"encoder_window_{window_id}.pt"
    torch.save({"state_dict": model.state_dict(), "config": encoder_config}, checkpoint_path)

    return best_loss


def generate_embeddings(
    model: StageOneEncoder,
    data: pl.DataFrame,
    device: torch.device,
) -> pl.DataFrame:
    """Generate embeddings for a dataset using trained encoder."""
    model.eval()

    features_list = data["features"].to_list()
    embeddings = []

    with torch.no_grad():
        for feat in features_list:
            # Reshape feature to (1, window_len, feature_dim)
            feat_array = np.array(feat, dtype=np.float32)
            window_len = len(feat_array) // 12  # Assume feature_dim=12
            feat_tensor = torch.tensor(feat_array.reshape(1, window_len, 12), device=device)

            outputs = model.backbone(feat_tensor)
            embedding = outputs.pooled.cpu().numpy()[0]
            embeddings.append(embedding.tolist())

    return data.with_columns(pl.Series("embedding", embeddings))


def train_policy_on_window(
    train_data: pl.DataFrame,
    policy_config: Dict[str, Any],
    device: torch.device,
    output_dir: Path,
    window_id: int,
) -> PPO:
    """Train PPO policy on embedded features."""
    print(f"\n[Window {window_id}] Training policy...")

    if PPO is None or DummyVecEnv is None:
        raise ImportError("stable-baselines3 required for policy training")

    # Create temporary parquet with embeddings
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp_path = tmp.name
        train_data.write_parquet(tmp_path)

    env_cfg = policy_config.get("environment", {})
    reward_cfg = RewardConfig(**env_cfg.get("reward", {}))

    def make_env():
        return TradingEnv(
            data=tmp_path,
            episode_length=int(env_cfg.get("episode_length", 256)),
            reward_cfg=reward_cfg,
            include_position_in_obs=bool(env_cfg.get("include_position_in_obs", True)),
        )

    vec_env = DummyVecEnv([make_env])

    train_cfg = policy_config.get("training", {})
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=float(train_cfg.get("learning_rate", 3e-4)),
        n_steps=int(train_cfg.get("n_steps", 2048)),
        batch_size=int(train_cfg.get("batch_size", 512)),
        device=str(device),
        verbose=0,
    )

    # Reduced timesteps for walk-forward (faster iterations)
    timesteps = int(train_cfg.get("total_timesteps", 50000))
    model.learn(total_timesteps=timesteps)

    # Save policy checkpoint
    checkpoint_path = output_dir / f"policy_window_{window_id}"
    model.save(checkpoint_path)

    return model


def evaluate_policy_on_window(
    policy: PPO,
    test_data: pl.DataFrame,
    policy_config: Dict[str, Any],
) -> Dict[str, float]:
    """Evaluate policy on test window."""
    print(f"Evaluating on test window...")

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp_path = tmp.name
        test_data.write_parquet(tmp_path)

    env_cfg = policy_config.get("environment", {})
    reward_cfg = RewardConfig(**env_cfg.get("reward", {}))

    env = TradingEnv(
        data=tmp_path,
        episode_length=int(env_cfg.get("episode_length", 256)),
        reward_cfg=reward_cfg,
        include_position_in_obs=bool(env_cfg.get("include_position_in_obs", True)),
    )

    # Run evaluation episodes
    num_episodes = 10
    all_positions = []
    all_returns = []
    trade_count = 0
    win_count = 0
    trade_durations = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_trades = 0
        prev_position = 0.0
        position_entry_step = 0

        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))

            all_returns.append(float(info["price_return"]))
            all_positions.append(float(info["position"]))

            # Track trades
            if abs(info["position"] - prev_position) > 1e-6:
                episode_trades += 1
                if prev_position != 0:
                    # Position closed
                    trade_durations.append(env.step_count - position_entry_step)
                    if reward > 0:
                        win_count += 1
                position_entry_step = env.step_count

            prev_position = info["position"]
            done = terminated or truncated

        trade_count += episode_trades

    # Calculate metrics
    symbol = test_data["symbol"][0] if "symbol" in test_data.columns else "UNKNOWN"
    annualization = 365.0 if symbol in ["ETH", "BTC", "SOL"] else 252.0

    bt = VectorizedBacktester(
        all_returns,
        trading_cost_bps=env.trading_cost_bps,
        annualization_factor=annualization,
    )
    result = bt.run(all_positions)

    return {
        "total_return": result.total_return,
        "sharpe_ratio": result.sharpe,
        "max_drawdown": result.max_drawdown,
        "win_rate": win_count / max(trade_count, 1),
        "total_trades": trade_count,
        "avg_trade_duration": np.mean(trade_durations) if trade_durations else 0.0,
    }


def run_walk_forward_validation(
    data_path: str,
    wf_config: Dict[str, Any],
    encoder_config: Dict[str, Any],
    policy_config: Dict[str, Any],
    output_dir: Path,
    device: torch.device,
) -> List[WindowResults]:
    """Execute complete walk-forward validation."""
    print("\n" + "=" * 80)
    print("WALK-FORWARD VALIDATION")
    print("=" * 80)

    # Load full dataset
    full_data = pl.read_parquet(data_path)
    print(f"\nLoaded dataset: {len(full_data)} samples")

    # Create time windows
    windows = create_time_windows(
        full_data,
        train_duration_days=int(wf_config.get("train_duration_days", 90)),
        test_duration_days=int(wf_config.get("test_duration_days", 30)),
        step_days=int(wf_config.get("step_days", 30)),
    )

    print(f"Created {len(windows)} walk-forward windows")

    results: List[WindowResults] = []

    for window in windows:
        print(f"\n{'=' * 80}")
        print(f"WINDOW {window.window_id + 1}/{len(windows)}")
        print(f"Train: {window.train_start} → {window.train_end}")
        print(f"Test:  {window.test_start} → {window.test_end}")
        print(f"{'=' * 80}")

        # Filter data
        train_data = filter_by_time(full_data, window.train_start, window.train_end)
        test_data = filter_by_time(full_data, window.test_start, window.test_end)

        print(f"Train samples: {len(train_data)}")
        print(f"Test samples:  {len(test_data)}")

        if len(train_data) < 100 or len(test_data) < 10:
            print("⚠️  Insufficient data, skipping window")
            continue

        # Train encoder
        encoder_loss = train_encoder_on_window(
            train_data, encoder_config, device, output_dir, window.window_id
        )

        # Load trained encoder
        checkpoint = torch.load(output_dir / f"encoder_window_{window.window_id}.pt")
        encoder = StageOneEncoder(encoder_config).to(device)
        encoder.load_state_dict(checkpoint["state_dict"])

        # Generate embeddings
        print(f"Generating embeddings...")
        train_data_emb = generate_embeddings(encoder, train_data, device)
        test_data_emb = generate_embeddings(encoder, test_data, device)

        # Train policy
        policy = train_policy_on_window(
            train_data_emb, policy_config, device, output_dir, window.window_id
        )

        # Evaluate on test set
        metrics = evaluate_policy_on_window(policy, test_data_emb, policy_config)

        # Record results
        result = WindowResults(
            window_id=window.window_id,
            train_start=window.train_start,
            train_end=window.train_end,
            test_start=window.test_start,
            test_end=window.test_end,
            train_samples=len(train_data),
            test_samples=len(test_data),
            encoder_loss=encoder_loss,
            policy_return=metrics["total_return"],
            sharpe_ratio=metrics["sharpe_ratio"],
            max_drawdown=metrics["max_drawdown"],
            win_rate=metrics["win_rate"],
            total_trades=metrics["total_trades"],
            avg_trade_duration=metrics["avg_trade_duration"],
        )
        results.append(result)

        print(f"\n✅ Window {window.window_id} Results:")
        print(f"   Return: {result.policy_return:.4f}")
        print(f"   Sharpe: {result.sharpe_ratio:.3f}")
        print(f"   Max DD: {result.max_drawdown:.4f}")
        print(f"   Win Rate: {result.win_rate:.2%}")

    return results


def save_results(results: List[WindowResults], output_dir: Path) -> None:
    """Save results to JSON and print summary."""
    # Save detailed results
    results_path = output_dir / "walk_forward_results.json"
    with open(results_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    print(f"\n✅ Saved detailed results to {results_path}")

    # Calculate aggregate statistics
    returns = [r.policy_return for r in results]
    sharpes = [r.sharpe_ratio for r in results]
    drawdowns = [r.max_drawdown for r in results]
    win_rates = [r.win_rate for r in results]

    summary = {
        "num_windows": len(results),
        "avg_return": np.mean(returns),
        "std_return": np.std(returns),
        "avg_sharpe": np.mean(sharpes),
        "std_sharpe": np.std(sharpes),
        "avg_max_drawdown": np.mean(drawdowns),
        "worst_drawdown": min(drawdowns),
        "avg_win_rate": np.mean(win_rates),
        "positive_windows": sum(1 for r in returns if r > 0),
        "negative_windows": sum(1 for r in returns if r <= 0),
    }

    summary_path = output_dir / "walk_forward_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("WALK-FORWARD VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Total Windows:      {summary['num_windows']}")
    print(f"Avg Return:         {summary['avg_return']:.4f} ± {summary['std_return']:.4f}")
    print(f"Avg Sharpe:         {summary['avg_sharpe']:.3f} ± {summary['std_sharpe']:.3f}")
    print(f"Avg Max Drawdown:   {summary['avg_max_drawdown']:.4f}")
    print(f"Worst Drawdown:     {summary['worst_drawdown']:.4f}")
    print(f"Avg Win Rate:       {summary['avg_win_rate']:.2%}")
    print(f"Positive Windows:   {summary['positive_windows']}/{summary['num_windows']}")
    print(f"Negative Windows:   {summary['negative_windows']}/{summary['num_windows']}")
    print("=" * 80)

    if mlflow:
        mlflow.log_metrics(summary)


def main() -> None:
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load configs
    wf_config = load_yaml(args.config) if Path(args.config).exists() else {}
    encoder_config = load_yaml(args.encoder_config)
    policy_config = load_yaml(args.policy_config)

    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Run validation
    results = run_walk_forward_validation(
        args.data,
        wf_config,
        encoder_config,
        policy_config,
        output_dir,
        device,
    )

    # Save and summarize
    save_results(results, output_dir)


if __name__ == "__main__":
    main()
