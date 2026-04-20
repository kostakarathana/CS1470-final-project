from pathlib import Path

import pytest
import torch

from madreamer.builders import build_modules
from madreamer.config import load_experiment_config
from madreamer.envs.factory import build_env
from madreamer.replay import MultiAgentReplayBuffer
from madreamer.trainers.ppo import PPOCollector


pytest.importorskip("torch")


def test_ppo_training_updates_policy_weights(tmp_path: Path) -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "ppo_smoke.yaml"
    cfg = load_experiment_config(config_path)
    cfg.training.log_dir = str(tmp_path / "logs")
    cfg.training.checkpoint_dir = str(tmp_path / "checkpoints")
    env = build_env(cfg)
    bundle = build_modules(cfg, env.agent_ids, env.observation_shape, env.action_dim, cfg.env.board_value_count)
    replay = MultiAgentReplayBuffer(capacity=128)
    trainer = PPOCollector(env, bundle, cfg, replay)

    before = {
        key: value.detach().clone()
        for key, value in bundle.ppo_policies["agent_0"].state_dict().items()
    }
    summary = trainer.run()
    after = bundle.ppo_policies["agent_0"].state_dict()

    assert summary.total_env_steps == cfg.training.total_steps
    assert any(not torch.allclose(before[key], after[key]) for key in before)
    assert Path(summary.latest_checkpoint_path or "").exists()
    env.close()
