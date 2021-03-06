from tensorforce import Agent, Environment
from rl_environments.rl_env import RLEnvironment
from rl_environments.kickoff_env import KickoffEnvironment
from rl_environments.shooting_env import ShootingEnvironment
from rl_environments.game_values import InputOptions, OutputOptions

from runner import Runner
from tensorforce.core.networks.auto import AutoNetwork
from rlbot.utils import public_utils, logging_utils

# Pre-defined or custom environment
# environment = Environment.create(
#     environment='gym', level='CartPole', max_episode_timesteps=500, visualize=True
# )

def create_agent():
    max_time = 5
    frames_per_sec = 20
    max_timesteps = RLEnvironment.get_max_timesteps(max_time, frames_per_sec)
    shooting_env = dict(
        # My custom kwargs 
        # Max time in seconds
        max_episode_timesteps=max_timesteps,
        max_time=max_time,
        frames_per_sec=frames_per_sec,
        name='shooting',
        message_throttle = 40,
        input_exclude=[InputOptions.BALL_POSITION, InputOptions.CAR_HEIGHT, InputOptions.CAR_POSITION],
        output_exclude=[],
    )
    kickoff_env = dict(
        environment=KickoffEnvironment,
        max_episode_timesteps=max_timesteps,
        # My custom kwargs 
        # Max time in seconds
        max_time=max_time,
        frames_per_sec=frames_per_sec,
        message_throttle = 40,
        input_exclude=[InputOptions.BALL_POSITION, InputOptions.CAR_HEIGHT, InputOptions.CAR_POSITION],
        output_exclude=[
            OutputOptions.JUMP,
            OutputOptions.PITCH,
            OutputOptions.ROLL,
            OutputOptions.BOOST,
            OutputOptions.E_BRAKE,
            OutputOptions.THROTTLE,
        ],
    )
    envs = [
        # Environment.create(environment=ShootingEnvironment, **kwargs),
        Environment.create(name='kickoff-1', **kickoff_env),
        # Environment.create(name='kickoff-2', **kickoff_env),
        # Environment.create(name='kickoff-3', **kickoff_env),
        # Environment.create(name='kickoff-4', **kickoff_env)
    ]
    num_envs = len(envs)

    # Instantiate a Tensorforce agent
    agent = dict(
        agent='double_dqn',
        memory=max_timesteps * 5,
        states=envs[0].states(),
        actions=envs[0].actions(),
        parallel_interactions=num_envs,
        # Automatically configured network
        # PPO optimization parameters
        network=[
            dict(type='dense', size=128, activation='relu'),
            dict(type='dropout', rate=0.5),
            dict(type='dense', size=64, activation='relu'),
            dict(type='dropout', rate=0.5),
            dict(type='dense', size=32, activation='relu'),
        ],
        batch_size=int(max_timesteps / 2), update_frequency=1, learning_rate=3e-4, 
        start_updating=max_timesteps,
        # Reward estimation
        discount=0.995, predict_terminal_values=False,
        # Baseline network and optimizer
        # baseline=dict(type='auto', size=32, depth=1),
        # baseline_optimizer=dict(optimizer='adam', learning_rate=1e-5, multi_step=10),
        # Regularization
        l2_regularization=0.0, entropy_regularization=0.0,
        # Preprocessing
        state_preprocessing="linear_normalization", reward_preprocessing=None,
        # Exploration
        exploration=0.0, variable_noise=0.0,
        # Default additional config values
        config=None,
        # Save agent every 10 updates and keep the 5 most recent checkpoints
        saver=dict(directory='model', frequency=1000, max_checkpoints=5),
        # Log all available Tensorboard summaries
        summarizer=dict(directory='summaries', summaries='all'),
        # Do not record agent-environment interaction trace
        recorder=None
    )
    # agent = Agent.load(directory='./model', format='checkpoint', environment=envs[0])
    runner = Runner(agent=agent, environments=envs, num_parallel=num_envs, max_episode_timesteps=max_timesteps)
    runner_gen = runner.run(num_episodes=25000, save_best_agent='best', batch_agent_calls=bool(num_envs - 1))

if __name__ == '__main__':
    create_agent()