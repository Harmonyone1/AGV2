"""End-to-end training pipeline orchestration for AGV2.

Automates the complete workflow:
1. Feature engineering (with volatility pre-computation)
2. Encoder training (Stage 1)
3. Embedding generation
4. Policy training (Stage 2)
5. Backtesting and evaluation

Usage:
    python scripts/train_pipeline.py --symbol ETH --config config/pipeline.yaml
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import yaml


@dataclass
class PipelineConfig:
    """Configuration for the training pipeline."""
    symbol: str
    raw_data_path: str
    output_dir: str
    encoder_config: str
    policy_config: str
    backtest_config: str
    device: str
    skip_feature_engineering: bool
    skip_encoder_training: bool
    skip_embedding_generation: bool
    skip_policy_training: bool
    skip_backtesting: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AGV2 end-to-end training pipeline")
    parser.add_argument("--symbol", default="ETH", help="Symbol to train on")
    parser.add_argument("--config", default="config/pipeline.yaml", help="Pipeline config")
    parser.add_argument("--raw-data", help="Override raw data path")
    parser.add_argument("--device", default="auto", help="Device (cuda/cpu/auto)")
    parser.add_argument("--skip-features", action="store_true", help="Skip feature engineering")
    parser.add_argument("--skip-encoder", action="store_true", help="Skip encoder training")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip embedding generation")
    parser.add_argument("--skip-policy", action="store_true", help="Skip policy training")
    parser.add_argument("--skip-backtest", action="store_true", help="Skip backtesting")
    return parser.parse_args()


def load_yaml(path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    if not Path(path).exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*80}")
    print(f"STEP: {description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}\n")

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - start_time

    if result.returncode == 0:
        print(f"\n✅ {description} completed successfully ({elapsed:.1f}s)")
        return True
    else:
        print(f"\n❌ {description} failed with exit code {result.returncode}")
        return False


def build_pipeline_config(args: argparse.Namespace, cfg: Dict[str, Any]) -> PipelineConfig:
    """Build pipeline configuration from args and config file."""
    symbol = args.symbol or cfg.get("symbol", "ETH")

    # Paths
    output_dir = cfg.get("output_dir", f"experiments/{symbol.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    raw_data = args.raw_data or cfg.get("raw_data_path", f"data/raw/{symbol.lower()}_1m.parquet")

    return PipelineConfig(
        symbol=symbol,
        raw_data_path=raw_data,
        output_dir=output_dir,
        encoder_config=cfg.get("encoder_config", "config/encoder.yaml"),
        policy_config=cfg.get("policy_config", "config/rl_policy.yaml"),
        backtest_config=cfg.get("backtest_config", "config/backtest.yaml"),
        device=args.device,
        skip_feature_engineering=args.skip_features or cfg.get("skip_features", False),
        skip_encoder_training=args.skip_encoder or cfg.get("skip_encoder", False),
        skip_embedding_generation=args.skip_embeddings or cfg.get("skip_embeddings", False),
        skip_policy_training=args.skip_policy or cfg.get("skip_policy", False),
        skip_backtesting=args.skip_backtest or cfg.get("skip_backtest", False),
    )


def step_feature_engineering(config: PipelineConfig) -> tuple[bool, str]:
    """Step 1: Feature engineering with volatility pre-computation."""
    if config.skip_feature_engineering:
        print("\n⏭️  Skipping feature engineering (--skip-features)")
        # Return existing path
        return True, f"data/features/encoder_windows_{config.symbol.lower()}.parquet"

    output_path = f"{config.output_dir}/features/encoder_windows_{config.symbol.lower()}.parquet"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # TODO: This assumes a feature engineering script exists
    # For now, we'll document what it should do
    print(f"""
📝 Feature Engineering Step:

Expected input:  {config.raw_data_path}
Expected output: {output_path}

This step should:
1. Load raw OHLCV data
2. Engineer price features (log returns, wicks, gaps)
3. Pre-compute volatility (realized_vol_30, parkinson_vol_30)
4. Create sliding windows
5. Add regime labels and S/R heatmaps
6. Save to parquet

Since this script doesn't exist yet, please run manually or use existing data.
    """)

    # Check if output exists
    if Path(output_path).exists():
        print(f"✅ Found existing features at {output_path}")
        return True, output_path
    else:
        print(f"⚠️  Features not found, using default path")
        return True, f"data/features/encoder_windows_{config.symbol.lower()}.parquet"


def step_encoder_training(config: PipelineConfig, features_path: str) -> tuple[bool, str]:
    """Step 2: Train encoder (Stage 1)."""
    if config.skip_encoder_training:
        print("\n⏭️  Skipping encoder training (--skip-encoder)")
        return True, f"models/encoders/encoder_best.pt"

    output_path = f"{config.output_dir}/models/encoder_best.pt"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "scripts/train_encoder.py",
        "--data", features_path,
        "--config", config.encoder_config,
        "--device", config.device,
    ]

    success = run_command(cmd, "Encoder Training (Stage 1)")
    if success:
        # Copy to output dir
        import shutil
        shutil.copy("models/encoders/encoder_best.pt", output_path)

    return success, output_path if success else ""


def step_embedding_generation(config: PipelineConfig, features_path: str, encoder_path: str) -> tuple[bool, str]:
    """Step 3: Generate embeddings using trained encoder."""
    if config.skip_embedding_generation:
        print("\n⏭️  Skipping embedding generation (--skip-embeddings)")
        return True, f"data/features/encoder_windows_{config.symbol.lower()}_emb.parquet"

    output_path = f"{config.output_dir}/features/encoder_windows_{config.symbol.lower()}_emb.parquet"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Check if embed_windows.py exists
    if not Path("scripts/embed_windows.py").exists():
        print("""
⚠️  scripts/embed_windows.py not found

This script should:
1. Load trained encoder from {encoder_path}
2. Load features from {features_path}
3. Generate embeddings for all windows
4. Save embeddings to parquet

For now, skipping to next step.
        """)
        return True, features_path  # Use features without embeddings

    cmd = [
        sys.executable,
        "scripts/embed_windows.py",
        "--input", features_path,
        "--model", encoder_path,
        "--output", output_path,
    ]

    success = run_command(cmd, "Embedding Generation")
    return success, output_path if success else features_path


def step_policy_training(config: PipelineConfig, data_path: str) -> tuple[bool, str]:
    """Step 4: Train PPO policy (Stage 2)."""
    if config.skip_policy_training:
        print("\n⏭️  Skipping policy training (--skip-policy)")
        return True, "models/policies/ppo_trading_env.zip"

    output_path = f"{config.output_dir}/models/ppo_trading_env.zip"

    cmd = [
        sys.executable,
        "scripts/train_policy.py",
        "--config", config.policy_config,
        "--data", data_path,
        "--symbol", config.symbol,
        "--device", config.device,
    ]

    success = run_command(cmd, "Policy Training (Stage 2)")
    return success, "models/policies/ppo_trading_env.zip" if success else ""


def step_backtesting(config: PipelineConfig, policy_path: str) -> bool:
    """Step 5: Backtest the trained policy."""
    if config.skip_backtesting:
        print("\n⏭️  Skipping backtesting (--skip-backtest)")
        return True

    cmd = [
        sys.executable,
        "scripts/backtest_policy.py",
        "--config", config.backtest_config,
        "--model", policy_path,
        "--symbol", config.symbol,
        "--episodes", "50",
    ]

    return run_command(cmd, "Backtesting")


def main() -> None:
    print("""
╔════════════════════════════════════════════════════════════════╗
║              AGV2 END-TO-END TRAINING PIPELINE                 ║
╚════════════════════════════════════════════════════════════════╝
    """)

    args = parse_args()
    cfg = load_yaml(args.config)
    pipeline_config = build_pipeline_config(args, cfg)

    print(f"\n📊 Pipeline Configuration:")
    print(f"   Symbol:      {pipeline_config.symbol}")
    print(f"   Device:      {pipeline_config.device}")
    print(f"   Output Dir:  {pipeline_config.output_dir}")
    print(f"   Raw Data:    {pipeline_config.raw_data_path}")

    # Create output directory
    Path(pipeline_config.output_dir).mkdir(parents=True, exist_ok=True)

    # Save pipeline config
    config_path = Path(pipeline_config.output_dir) / "pipeline_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump({
            "symbol": pipeline_config.symbol,
            "timestamp": datetime.now().isoformat(),
            "raw_data_path": pipeline_config.raw_data_path,
            "output_dir": pipeline_config.output_dir,
        }, f)

    start_time = time.time()

    # Execute pipeline
    print(f"\n{'='*80}")
    print("STARTING PIPELINE EXECUTION")
    print(f"{'='*80}")

    # Step 1: Feature Engineering
    success, features_path = step_feature_engineering(pipeline_config)
    if not success:
        print("\n❌ Pipeline failed at feature engineering")
        sys.exit(1)

    # Step 2: Encoder Training
    success, encoder_path = step_encoder_training(pipeline_config, features_path)
    if not success:
        print("\n❌ Pipeline failed at encoder training")
        sys.exit(1)

    # Step 3: Embedding Generation
    success, embeddings_path = step_embedding_generation(pipeline_config, features_path, encoder_path)
    if not success:
        print("\n❌ Pipeline failed at embedding generation")
        sys.exit(1)

    # Step 4: Policy Training
    success, policy_path = step_policy_training(pipeline_config, embeddings_path)
    if not success:
        print("\n❌ Pipeline failed at policy training")
        sys.exit(1)

    # Step 5: Backtesting
    success = step_backtesting(pipeline_config, policy_path)
    if not success:
        print("\n❌ Pipeline failed at backtesting")
        sys.exit(1)

    # Success!
    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print("✅ PIPELINE COMPLETED SUCCESSFULLY")
    print(f"{'='*80}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Output directory: {pipeline_config.output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Review backtest results")
    print(f"  2. Run walk-forward validation:")
    print(f"     python scripts/walk_forward_validation.py --data {embeddings_path}")
    print(f"  3. If results are good, deploy to paper trading")


if __name__ == "__main__":
    main()
