"""ML inference engine for AGV2.

Handles loading encoder and policy models, generating predictions.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from stable_baselines3 import PPO

from models.encoder import StageOneEncoder

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Manages ML model loading and inference."""

    def __init__(
        self,
        policy_path: str | Path = "models/policies/ppo_trading_env.zip",
        encoder_path: str | Path = "models/encoders/encoder_best.pt",
        device: str = "cpu",
    ):
        """Initialize inference engine.

        Args:
            policy_path: Path to trained policy model
            encoder_path: Path to trained encoder model
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.policy_path = Path(policy_path)
        self.encoder_path = Path(encoder_path)
        self.device = device

        self.policy: Optional[PPO] = None
        self.encoder: Optional[StageOneEncoder] = None
        self.policy_version: str = "unknown"
        self.encoder_version: str = "unknown"

        logger.info(f"Initialized InferenceEngine (device={device})")

    async def load_models(self) -> None:
        """Load policy and encoder models asynchronously."""
        logger.info("Loading models...")

        # Load policy
        if self.policy_path.exists():
            try:
                self.policy = PPO.load(str(self.policy_path), device=self.device)
                self.policy_version = self._get_model_version(self.policy_path)
                logger.info(f"Loaded policy from {self.policy_path} (version={self.policy_version})")
            except Exception as e:
                logger.error(f"Failed to load policy: {e}")
                self.policy = None
        else:
            logger.warning(f"Policy not found at {self.policy_path}")

        # Load encoder
        if self.encoder_path.exists():
            try:
                checkpoint = torch.load(self.encoder_path, map_location=self.device)

                # Reconstruct encoder from config
                if "config" in checkpoint:
                    config = checkpoint["config"]
                    self.encoder = StageOneEncoder(config)
                    self.encoder.load_state_dict(checkpoint["model_state_dict"])
                    self.encoder.to(self.device)
                    self.encoder.eval()
                    self.encoder_version = self._get_model_version(self.encoder_path)
                    logger.info(
                        f"Loaded encoder from {self.encoder_path} (version={self.encoder_version})"
                    )
                else:
                    logger.warning("Encoder checkpoint missing config, skipping encoder load")
                    self.encoder = None
            except Exception as e:
                logger.error(f"Failed to load encoder: {e}")
                self.encoder = None
        else:
            logger.warning(f"Encoder not found at {self.encoder_path}")

    async def cleanup(self) -> None:
        """Cleanup resources on shutdown."""
        logger.info("Cleaning up inference engine...")
        self.policy = None
        self.encoder = None

    def is_ready(self) -> bool:
        """Check if models are loaded and ready."""
        return self.policy is not None

    def is_encoder_ready(self) -> bool:
        """Check if encoder is loaded."""
        return self.encoder is not None

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> tuple[float, Optional[dict]]:
        """Generate prediction from policy.

        Args:
            observation: Observation array (must match policy's observation space)
            deterministic: Whether to use deterministic policy (True for trading)

        Returns:
            Tuple of (action, extra_info)
            action: Predicted action value
            extra_info: Optional dict with additional info (e.g., value estimate)
        """
        if self.policy is None:
            raise RuntimeError("Policy not loaded. Call load_models() first.")

        # Ensure observation is correct shape
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)

        action, _states = self.policy.predict(observation, deterministic=deterministic)

        # Extract scalar action
        if isinstance(action, np.ndarray):
            action = float(action.item() if action.size == 1 else action[0])

        return action, None

    def encode_features(self, features: np.ndarray) -> np.ndarray:
        """Generate embeddings from raw features using encoder.

        Args:
            features: Raw feature array [seq_len, input_dim] or [batch, seq_len, input_dim]

        Returns:
            Encoded embeddings [emb_dim] or [batch, emb_dim]
        """
        if self.encoder is None:
            raise RuntimeError("Encoder not loaded. Call load_models() first.")

        was_2d = features.ndim == 2
        if was_2d:
            # Add batch dimension
            features = features[np.newaxis, ...]

        # Convert to tensor
        x = torch.from_numpy(features).float().to(self.device)

        with torch.no_grad():
            # StageOneEncoder returns a dict with multiple outputs
            outputs = self.encoder(x)
            # Extract pooled embedding for inference
            embeddings = outputs["embedding"]
            embeddings_np = embeddings.cpu().numpy()

        # Remove batch dimension if input was 2D
        if was_2d:
            embeddings_np = embeddings_np[0]

        return embeddings_np

    def _get_model_version(self, model_path: Path) -> str:
        """Extract version from model filename or modification time."""
        # Try to parse timestamp from filename (e.g., ppo_trading_env_20251025_143022.zip)
        name = model_path.stem
        parts = name.split("_")
        if len(parts) >= 3 and parts[-2].isdigit() and parts[-1].isdigit():
            # Has timestamp
            return f"{parts[-2]}_{parts[-1]}"
        else:
            # Use modification time
            mtime = model_path.stat().st_mtime
            dt = datetime.fromtimestamp(mtime)
            return dt.strftime("%Y%m%d_%H%M%S")


# Global instance (will be initialized in main.py lifespan)
inference_engine: Optional[InferenceEngine] = None


def get_inference_engine() -> InferenceEngine:
    """Get global inference engine instance."""
    if inference_engine is None:
        raise RuntimeError("Inference engine not initialized")
    return inference_engine


__all__ = ["InferenceEngine", "get_inference_engine", "inference_engine"]
