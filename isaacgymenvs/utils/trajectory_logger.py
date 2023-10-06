import time
import torch
import numpy as np
import h5py
import sys

class Trajectory(object):
    def __init__(self, env_id):
        super(Trajectory, self).__init__()
        self.steps = []
        self.env_id = env_id
        self.terminal = False
        self.timeout = False

        self.data = {
            'observations': [],
            'next_observations': [],
            'actions': [],
            'rewards': [],
            'terminals': [],
            'timeouts': [],

            # additional info
            'initial_position': [],
            'target_position': [],
        }

    def step(self, observations, next_observations, actions, reward, terminal, timeout, info):
        self.data['observations'].append(observations)
        self.data['next_observations'].append(next_observations)
        self.data['actions'].append(actions)
        self.data['rewards'].append(reward)
        self.data['terminals'].append(terminal)
        self.data['timeouts'].append(timeout)

        # additional info
        self.data['initial_position'].append(info['initial_position'])
        self.data['target_position'].append(info['target_position'])

        self.terminal |= bool(terminal)
        self.timeout |= bool(timeout)

    def is_finished(self) -> bool:
        return self.terminal or self.timeout


class TrajectoryLogger(object):
    """ Save trajectory information for later usage """
    def __init__(self, num_env, sample_count, output_path='robotpush'):
        super(TrajectoryLogger, self).__init__()
        self.active_trajectories = [Trajectory(i) for i in range(num_env)]
        self.num_envs = num_env
        self.sample_count = sample_count
        self.output_path = output_path

        self.data = {
            'observations': [],
            'next_observations': [],
            'actions': [],
            'rewards': [],
            'terminals': [],
            'timeouts': [],

            # additional info
            'initial_position': [],
            'target_position': [],
        }

    def add_trajectory_data(self, trajectory : Trajectory):
        for key in self.data.keys():
            self.data[key].extend(trajectory.data[key])

        samples_collected = len(self.data['observations'])
        progress = samples_collected / self.sample_count
        print(f"collected {samples_collected} ouf of {self.sample_count}. Progress: {progress * 100}%")
        if progress >= 1:
            self.save_data()
            sys.exit()

    def save_data(self):
        typed_data = dict(
            observations=np.array(self.data['observations']).astype(np.float32),
            actions=np.array(self.data['actions']).astype(np.float32),
            next_observations=np.array(self.data['next_observations']).astype(np.float32),
            rewards=np.array(self.data['rewards']).astype(np.float32),
            terminals=np.array(self.data['terminals']).astype(np.bool),
            timeouts=np.array(self.data['timeouts']).astype(np.bool),
        )
        # Additional data
        typed_data['infos/initial_position'] = np.array(self.data['initial_position']).astype(np.float32)
        typed_data['infos/target_position'] = np.array(self.data['target_position']).astype(np.float32)

        # Save to file
        hfile = h5py.File(self.output_path + str(self.sample_count) + '.hdf', 'w')
        for k in self.data:
            if k not in ["observations", "next_observations", "rewards", "actions", "timeouts", "terminals"]:
                hfile.create_dataset(f"infos/{k}", data=self.data[k])
            else:
                hfile.create_dataset(k, data=self.data[k])


    def digest(self, observations, next_observations, actions, rewards, terminals, timeouts, info):
        observations_arr = observations.detach().clone().cpu().numpy()
        next_observations_arr = next_observations.detach().clone().cpu().numpy()
        actions_arr = actions.detach().clone().cpu().numpy()
        rewards_arr = rewards.detach().clone().cpu().numpy()
        terminals_arr = terminals.detach().clone().cpu().numpy()
        timeouts_arr = timeouts.detach().clone().cpu().numpy()
        info_per_env_id = [{key: value[env_id].cpu().numpy() for key, value in info.items()} for env_id in range(self.num_envs)]

        completed_trajectories = []
        for env_id in range(self.num_envs):
            self.active_trajectories[env_id].step(observations_arr[env_id],
                                                  next_observations_arr[env_id],
                                                  actions_arr[env_id],
                                                  rewards_arr[env_id],
                                                  terminals_arr[env_id],
                                                  timeouts_arr[env_id],
                                                  info_per_env_id[env_id])

            if self.active_trajectories[env_id].is_finished():
                # this may be because timeout or termination condition met
                completed_trajectories.append(self.active_trajectories[env_id])
                self.active_trajectories[env_id] = Trajectory(env_id)

        if completed_trajectories:
            for full_trajectory in completed_trajectories:
                self.add_trajectory_data(full_trajectory)
