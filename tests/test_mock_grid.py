from madreamer.envs.mock_grid import MockGridEnv


def test_mock_grid_reset_and_step() -> None:
    env = MockGridEnv(num_agents=4, grid_size=5, max_steps=8, task_type="cooperative", reward_preset="sparse")
    observations = env.reset(seed=0)

    assert set(observations) == {"agent_0", "agent_1", "agent_2", "agent_3"}
    assert observations["agent_0"].shape == (3, 5, 5)

    step = env.step({agent_id: 0 for agent_id in env.agent_ids})
    assert set(step.observations) == set(env.agent_ids)
    assert set(step.rewards) == set(env.agent_ids)
    assert set(step.raw_rewards) == set(env.agent_ids)
