import numpy as np

from madreamer.replay import MultiAgentReplayBuffer, ReplayStep, build_opponent_context


def _step(episode_id: int, step_index: int) -> ReplayStep:
    agent_ids = ("agent_0", "agent_1", "agent_2", "agent_3")
    observations = {agent_id: np.full((3, 3, 3), step_index, dtype=np.float32) for agent_id in agent_ids}
    actions = {agent_id: index % 6 for index, agent_id in enumerate(agent_ids)}
    alive = {agent_id: True for agent_id in agent_ids}
    return ReplayStep(
        episode_id=episode_id,
        observations=observations,
        actions=actions,
        opponent_actions={
            agent_id: build_opponent_context(agent_id, agent_ids, actions, alive, 6)
            for agent_id in agent_ids
        },
        rewards={agent_id: 0.0 for agent_id in agent_ids},
        raw_rewards={agent_id: 0.0 for agent_id in agent_ids},
        next_observations=observations,
        terminated={agent_id: False for agent_id in agent_ids},
        truncated={agent_id: False for agent_id in agent_ids},
        alive=alive,
        infos={agent_id: {} for agent_id in agent_ids},
        events={agent_id: {} for agent_id in agent_ids},
    )


def test_build_opponent_context_shape() -> None:
    context = build_opponent_context(
        "agent_0",
        ("agent_0", "agent_1", "agent_2", "agent_3"),
        {"agent_0": 0, "agent_1": 1, "agent_2": 2, "agent_3": 3},
        {"agent_0": True, "agent_1": True, "agent_2": False, "agent_3": True},
        6,
    )
    assert context.shape == (21,)


def test_replay_buffer_samples_sequence_batches() -> None:
    buffer = MultiAgentReplayBuffer(capacity=16)
    for step_index in range(4):
        buffer.add(_step(episode_id=0, step_index=step_index))
    for step_index in range(3):
        buffer.add(_step(episode_id=1, step_index=10 + step_index))

    batch = buffer.sample_sequences(2, 2, ("agent_0", "agent_1", "agent_2", "agent_3"))
    assert batch.agents["agent_0"].observations.shape == (2, 2, 3, 3, 3)
    assert batch.agents["agent_0"].opponent_actions.shape[-1] == 21
