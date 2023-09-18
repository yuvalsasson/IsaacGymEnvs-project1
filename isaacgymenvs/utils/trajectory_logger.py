import pandas
import os
from functools import wraps
from time import time


def measure(func):
    """ for measuring execution time"""
    @wraps(func)
    def _time_it(*args, **kwargs):
        start = int(round(time() * 1000))
        try:
            return func(*args, **kwargs)
        finally:
            end_ = int(round(time() * 1000)) - start
            print(f"Total execution time: {end_ if end_ > 0 else 0} ms")
    return _time_it


class Trajectory(object):
    def __init__(self, env_id):
        super(Trajectory, self).__init__()
        self.steps = []
        self.env_id = env_id

    def step(self, observations, actions, reward):
        self.steps.append((observations, actions, reward))

    def finish(self, is_reached_goal):
        return {
            'steps': self.steps,
            'step_count': len(self.steps),
            'total_reward': sum(step[2].values[0] for step in self.steps), # 2 is the reward and [0] is getting the only element
            'goal_reached': is_reached_goal
        }


class TrajectoryLogger(object):
    """ Save trajectory information for later usage """
    def __init__(self, num_env, output_dir='trajectories', prefix='prefix'):
        super(TrajectoryLogger, self).__init__()
        self.active_trajectories = [Trajectory(i) for i in range(num_env)]
        self.output_dir = output_dir
        os.makedirs(self.output_dir, 0o777, exist_ok=True)
        self.prefix = prefix
        self.num_envs = num_env
        self.completed_trajectories = None

    def add_completed_trajectories(self, trajectories):
        """ Add trajectories of completed episodes `self.completed_trajectories`.
         Evicts to disk if completed_trajectories is too large"""
        if self.completed_trajectories is None:
            self.completed_trajectories = pandas.DataFrame(trajectories)
        else:
            self.completed_trajectories = self.completed_trajectories.append(pandas.DataFrame(trajectories))

        if len(self.completed_trajectories.index) > 1024:
            self.completed_trajectories.to_pickle(f'{self.output_dir}/{self.prefix}_{time()}.pkl')
            self.completed_trajectories = None

    @measure
    def digest(self, observations, actions, rewards, reset_flag, reached_goal):
        altered_obs = {key: list(value.detach().cpu().numpy()) for key, value in observations.items()}
        observations_df = pandas.DataFrame.from_dict(altered_obs)
        actions_df = pandas.DataFrame(actions.detach().cpu().numpy())
        rewards_df = pandas.DataFrame(rewards.detach().cpu().numpy())
        reset_flag_df = pandas.DataFrame(reset_flag.detach().bool().cpu().numpy())
        reached_goal_df = pandas.DataFrame(reached_goal.detach().bool().cpu().numpy())

        completed_trajectories = []
        for env_id in range(self.num_envs):
            self.active_trajectories[env_id].step(observations_df.iloc[env_id], actions_df.iloc[env_id], rewards_df.iloc[env_id])
            if reset_flag_df.iloc[env_id].bool():
                completed_trajectories.append(self.active_trajectories[env_id].finish(reached_goal_df.iloc[env_id].values[0]))
                self.active_trajectories[env_id] = Trajectory(env_id)

        if completed_trajectories:
            self.add_completed_trajectories(completed_trajectories)
