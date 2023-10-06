import numpy as np
import torch
import wandb
import random
from agent import CQLSAC
from torch.utils.data import DataLoader, TensorDataset
import h5py
from tqdm import tqdm

def save(args, save_name, model, wandb, ep=None):
    import os
    save_dir = './trained_models/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if ep is not None:
        torch.save(model.state_dict(), save_dir + args.run_name + save_name + str(ep) + ".pth")
        wandb.save(save_dir + args.run_name + save_name + str(ep) + ".pth")
    else:
        torch.save(model.state_dict(), save_dir + args.run_name + save_name + ".pth")
        wandb.save(save_dir + args.run_name + save_name + ".pth")


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

def train(config):
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
                states, actions, rewards, next_states, dones = experience # experience is trajectories but agent expects tensor of steps
                states = states.to(device)
                actions = actions.to(device)
                rewards = rewards.to(device)
                next_states = next_states.to(device)
                dones = dones.to(device)
                policy_loss, alpha_loss, bellmann_error1, bellmann_error2, cql1_loss, cql2_loss, current_alpha, lagrange_alpha_loss, lagrange_alpha = agent.learn(
                    (states, actions, rewards, next_states, dones))
                batches += 1

            wandb.log({
                #"Average10": np.mean(average10),
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
                "Episode": i
            })

            save(config, save_name="IQL", model=agent.actor_local, wandb=wandb, ep=str(i))

class FrankaPushCube():
    def __init__(self):
        self.seed = 0
        self.tau = 0.95
        self.learning_rate = 5e-4
        self.hidden_size = 400
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