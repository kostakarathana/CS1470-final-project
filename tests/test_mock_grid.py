from madreamer.envs.mock_grid import MockGridEnv


def test_mock_grid_reset_and_step() -> None:
    env = MockGridEnv(num_agents=2, grid_size=5, max_steps=8, task_type="cooperative")
    observations = env.reset(seed=0)

    assert set(observations) == {"agent_0", "agent_1"}
    assert observations["agent_0"].shape == (3, 5, 5)

    step = env.step({"agent_0": 0, "agent_1": 0})
    assert set(step.observations) == {"agent_0", "agent_1"}
    assert set(step.rewards) == {"agent_0", "agent_1"}
