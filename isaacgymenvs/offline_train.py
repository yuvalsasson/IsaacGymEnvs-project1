import logging
import os
from datetime import datetime

# noinspection PyUnresolvedReferences
import isaacgym
from collections import deque
import hydra
import torch

from isaacgymenvs.utils.rlgames_utils import multi_gpu_get_rank

from isaacgymenvs.pbt.pbt import PbtAlgoObserver, initial_pbt_check
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
from isaacgymenvs.tasks import isaacgym_task_map
from omegaconf import DictConfig, OmegaConf
import gym

from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
from isaacgymenvs.utils.utils import set_np_formatting, set_seed

from rl_games.torch_runner import  _restore, _override_sigma
import numpy as np

import numpy as np
import torch
import wandb
import random
from isaacgymenvs.tools.agent import CQLSAC
from torch.utils.data import DataLoader, TensorDataset
import h5py
from tqdm import tqdm

def preprocess_train_config(cfg, config_dict):
    """
    Adding common configuration parameters to the rl_games train config.
    An alternative to this is inferring them in task-specific .yaml files, but that requires repeating the same
    variable interpolations in each config.
    """

    train_cfg = config_dict['params']['config']

    train_cfg['device'] = cfg.rl_device

    train_cfg['population_based_training'] = cfg.pbt.enabled
    train_cfg['pbt_idx'] = cfg.pbt.policy_idx if cfg.pbt.enabled else None

    train_cfg['full_experiment_name'] = cfg.get('full_experiment_name')

    print(f'Using rl_device: {cfg.rl_device}')
    print(f'Using sim_device: {cfg.sim_device}')
    print(train_cfg)

    try:
        model_size_multiplier = config_dict['params']['network']['mlp']['model_size_multiplier']
        if model_size_multiplier != 1:
            units = config_dict['params']['network']['mlp']['units']
            for i, u in enumerate(units):
                units[i] = u * model_size_multiplier
            print(
                f'Modified MLP units by x{model_size_multiplier} to {config_dict["params"]["network"]["mlp"]["units"]}')
    except KeyError:
        pass

    return config_dict


@hydra.main(config_name="config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):
    if cfg.pbt.enabled:
        initial_pbt_check(cfg)

    from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver, MultiObserver, ComplexObsRLGPUEnv
    from isaacgymenvs.utils.wandb_utils import WandbAlgoObserver
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner
    from rl_games.algos_torch import model_builder
    from isaacgymenvs.learning import amp_continuous
    from isaacgymenvs.learning import amp_players
    from isaacgymenvs.learning import amp_models
    from isaacgymenvs.learning import amp_network_builder
    import isaacgymenvs

    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{cfg.wandb_name}_{time_str}"

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # set numpy formatting for printing only
    set_np_formatting()

    # global rank of the GPU
    global_rank = int(os.getenv("RANK", "0"))

    # sets seed. if seed is -1 will pick a random one
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=global_rank)

    def create_isaacgym_env(**kwargs):
        envs = isaacgymenvs.make(
            cfg.seed,
            cfg.task_name,
            cfg.task.env.numEnvs,
            cfg.sim_device,
            cfg.rl_device,
            cfg.graphics_device_id,
            cfg.headless,
            cfg.multi_gpu,
            cfg.capture_video,
            cfg.force_render,
            cfg,
            **kwargs,
        )

        return envs

    env_configurations.register('rlgpu', {
        'vecenv_type': 'RLGPU',
        'env_creator': lambda **kwargs: create_isaacgym_env(**kwargs),
    })

    ige_env_cls = isaacgym_task_map[cfg.task_name]
    dict_cls = ige_env_cls.dict_obs_cls if hasattr(ige_env_cls, 'dict_obs_cls') and ige_env_cls.dict_obs_cls else False

    if dict_cls:

        obs_spec = {}
        actor_net_cfg = cfg.train.params.network
        obs_spec['obs'] = {'names': list(actor_net_cfg.inputs.keys()),
                           'concat': not actor_net_cfg.name == "complex_net", 'space_name': 'observation_space'}
        if "central_value_config" in cfg.train.params.config:
            critic_net_cfg = cfg.train.params.config.central_value_config.network
            obs_spec['states'] = {'names': list(critic_net_cfg.inputs.keys()),
                                  'concat': not critic_net_cfg.name == "complex_net", 'space_name': 'state_space'}

        vecenv.register('RLGPU',
                        lambda config_name, num_actors, **kwargs: ComplexObsRLGPUEnv(config_name, num_actors, obs_spec,
                                                                                     **kwargs))
    else:

        vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))

    rlg_config_dict = omegaconf_to_dict(cfg.train)
    rlg_config_dict = preprocess_train_config(cfg, rlg_config_dict)

    observers = []

    # register new AMP network builder and agent
    def build_runner(algo_observer):
        runner = Runner(algo_observer)
        runner.algo_factory.register_builder('amp_continuous', lambda **kwargs: amp_continuous.AMPAgent(**kwargs))
        runner.player_factory.register_builder('amp_continuous',
                                               lambda **kwargs: amp_players.AMPPlayerContinuous(**kwargs))
        model_builder.register_model('continuous_amp', lambda network, **kwargs: amp_models.ModelAMPContinuous(network))
        model_builder.register_network('amp', lambda **kwargs: amp_network_builder.AMPBuilder())

        return runner

    # convert CLI arguments into dictionary
    # create runner and set the settings
    runner = build_runner(MultiObserver(observers))
    runner.load(rlg_config_dict)
    runner.reset()

    args = {
        'train': not cfg.test,
        'play': cfg.test,
        'checkpoint': cfg.checkpoint,
        'sigma': cfg.sigma if cfg.sigma != '' else None
    }

    player = runner.create_player()
    _restore(player, args)
    _override_sigma(player, args)

    env = player.env
    evaulate_agent.env = env

def evaulate_agent(agent):
    env = evaulate_agent.env

    # Reset all environments
    obses = env.reset()
    env.reset_stats()
    env.reset_idx(torch.tensor([env_id for env_id in range(env.num_envs)], device=env.device))

    # Run until all environments done once
    reward_from_episode_start = torch.zeros(env.num_envs, device=env.device)
    episodes_reward = torch.zeros(env.num_envs, device=env.device)
    env_dones = torch.zeros(env.num_envs, device=env.device)
    with torch.no_grad():
        for i in range(300):
            actions = torch.stack([agent.actor_local.get_det_action(obs) for obs in obses['obs']])
            obses, reward, done, info = env.step(actions)
            reward_from_episode_start += reward
            first_time_dones = torch.logical_and(done, torch.logical_not(env_dones))
            episodes_reward[first_time_dones] = reward_from_episode_start[first_time_dones]
            env_dones = torch.logical_or(env_dones, done)
            if torch.all(env_dones):
                return np.mean(episodes_reward.cpu().numpy()), env.goals_reached / env.games_played


def save(args, save_name, model, wandb, ep=None):
    import os
    save_dir = './trained_models/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    path = f"./{save_dir}/{args.run_name}_{save_name}.pth"
    torch.save(model.state_dict(), path)
    wandb.save(path)


def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys

def load_hdf(hdf5_path):
    data_dict = {}
    with h5py.File(hdf5_path, 'r') as dataset_file:
        for k in tqdm(get_keys(dataset_file), desc="load datafile"):
            try:  # first try loading as an array
                data_dict[k] = dataset_file[k][:]
            except ValueError as e:  # try loading as a scalar
                data_dict[k] = dataset_file[k][()]
    return data_dict


def prep_dataloader(hdf_path, batch_size=256, seed=1):
    dataset = load_hdf(hdf_path)
    tensors = {}
    for k, v in dataset.items():
        if k in ["actions", "observations", "next_observations", "rewards", "terminals"]:
            if k is not "terminals":
                tensors[k] = torch.from_numpy(v).float()
            else:
                tensors[k] = torch.from_numpy(v).long()

    tensordata = TensorDataset(tensors["observations"],
                               tensors["actions"],
                               tensors["rewards"][:, None],
                               tensors["next_observations"],
                               tensors["terminals"][:, None])
    dataloader = DataLoader(tensordata, batch_size=batch_size, shuffle=True)

    return dataloader

def process_learn(batches, agent, wandb, policy_loss, alpha_loss, bellmann_error1, bellmann_error2, cql1_loss, cql2_loss, current_alpha, lagrange_alpha_loss, lagrange_alpha, average_reward, average_win_ratio, episode, batch_idx):
    if batches % config.log_every == 0:
        wandb.log({
            "Average10Reward": np.mean(average_reward),
            "Average10WinRatio": np.mean(average_win_ratio),
            "Policy Loss": policy_loss,
            "Alpha Loss": alpha_loss,
            "Lagrange Alpha Loss": lagrange_alpha_loss,
            "CQL1 Loss": cql1_loss,
            "CQL2 Loss": cql2_loss,
            "Bellman error 1": bellmann_error1,
            "Bellman error 2": bellmann_error2,
            "Alpha": current_alpha,
            "Lagrange Alpha": lagrange_alpha,
            "Batches": batches,
            "Episode": episode
        }, step=batches)

    if batches % config.eval_every == 0:
        eval_reward, win_ratio = evaulate_agent(agent)
        wandb.log({
            "Test Reward": eval_reward,
            "Test Win ratio": win_ratio,
            "Episode": episode,
            "Batches": batches,
        }, step=batches)

        print(f"Episode.Batch: {episode}.{batch_idx} | Reward: {eval_reward} | Win Ratio: {win_ratio} | Policy Loss: {policy_loss}")
        average_reward.append(eval_reward)
        average_win_ratio.append(win_ratio)

        if eval_reward > process_learn.max_reward:
            process_learn.max_reward = eval_reward
            print(f"new best reward: {eval_reward}")
            save(config, save_name="best net", model=agent.actor_local, wandb=wandb)

    if batches % config.save_every == 0:
        save(config, save_name=f"net{episode}.{batch_idx}", model=agent.actor_local, wandb=wandb)

process_learn.max_reward = 0

def train(config):
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # create env
    launch_rlg_hydra()

    average10reward = deque(maxlen=10)
    average10winratio = deque(maxlen=10)

    dataloader = prep_dataloader(config.dataset_path, batch_size=config.batch_size)
    batches = 0
    with wandb.init(project="CQL-offline", name=config.run_name, config=config):
        agent = CQLSAC(state_size=config.observation_space_size,
                       action_size=config.action_space_size,
                       tau=config.tau,
                       hidden_size=config.hidden_size,
                       learning_rate=config.learning_rate,
                       temp=config.temperature,
                       with_lagrange=config.with_lagrange,
                       cql_weight=config.cql_weight,
                       target_action_gap=config.target_action_gap,
                       device=device)

        for i in tqdm(range(1, config.episodes + 1), position=0):
            for batch_idx, experience in enumerate(tqdm(dataloader, position=1, leave=False)):
                states, actions, rewards, next_states, dones = experience
                states = states.to(device)
                actions = actions.to(device)
                rewards = rewards.to(device)
                next_states = next_states.to(device)
                dones = dones.to(device)reset_
                policy_loss, alpha_loss, bellmann_error1, bellmann_error2, cql1_loss, cql2_loss, current_alpha, lagrange_alpha_loss, lagrange_alpha = agent.learn(
                    (states, actions, rewards, next_states, dones))
                process_learn(batches, agent, wandb, policy_loss, alpha_loss, bellmann_error1, bellmann_error2,
                              cql1_loss, cql2_loss, current_alpha, lagrange_alpha_loss, lagrange_alpha, average10reward,
                              average10winratio, i, batch_idx)
                batches += 1

class FrankaPushCube():
    def __init__(self):
        self.seed = 0
        self.tau = 0.95
        self.learning_rate = 5e-4
        self.eval_every = 10
        self.log_every = 20
        self.save_every = 1000
        self.observation_space_size = 24
        self.action_space_size = 2
        self.hidden_size = 4096
        self.temperature = 0.2
        self.with_lagrange = False
        self.cql_weight = 0.5
        self.target_action_gap = 0.2
        self.batch_size = 256
        self.run_name = 'franka_push_cube_offline'
        self.episodes = 5
        self.dataset_path = '/home/user_119/Project/IsaacGymEnvs-project1/isaacgymenvs/robotpush10000000.hdf'

if __name__ == '__main__':
    config = FrankaPushCube()
    train(config)