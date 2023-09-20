import pickle
import glob
import sys
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict


class TrajectoryDataset(Dataset):
    def __init__(self, trajectory_dir, transform=None):
        self.trajectory_dir = trajectory_dir
        self.transform = transform
        self.trajectory_count_per_file = self._fill_trajectory_metadata()
        self.total_count = sum(self.trajectory_count_per_file.values())

    def _fill_trajectory_metadata(self) -> OrderedDict:
        """ Returns an OrderedDict with entries for each trajectory pickle file.
        Each entry contains the starting index for trajectories within the file and number within the file
        """
        trajectory_filenames = glob.glob(sys.path.join(self.trajectory_dir, "*"))
        trajectory_file_metadata = OrderedDict()
        for filename in trajectory_filenames:
            trajectory_file_metadata[filename] = self._get_trajectory_count(filename)
        # consider adding starting index at each file to allow binary search
        return trajectory_file_metadata

    def _get_trajectory_count(self, pickle_path):
        with open(pickle_path) as pkl:
            trajectory_df = pickle.load(pkl)
            return len(trajectory_df.index)

    def __len__(self):
        return self.total_count

    def __getitem__(self, idx):
        """ this returns a trajectory at index i"""
        curr_index = 0
        selected_trajectory_path = None
        for trajectory_path, number_of_entries in self._fill_trajectory_metadata():
            if curr_index <= idx < curr_index + number_of_entries:
                selected_trajectory_path = trajectory_path
                break
            curr_index += number_of_entries
        if selected_trajectory_path is None:
            raise IndexError("idx is not within TrajectoryDataset bounds")
        idx_within_file = idx - curr_index
        with open(selected_trajectory_path) as trajectory_file:
            trajectory_df = pickle.load(trajectory_file)
            traj = trajectory_df.iloc[idx_within_file]
            if self.transform:
                traj = self.transform(traj)
            return traj

def get_dataloader(trajectory_dir="trajectories", batch_size=256):
    dataset = TrajectoryDataset(trajectory_dir=trajectory_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train(config):
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataloader = get_dataloader(config.trajectory_dir, batch_size=config.batch_size)
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

        for i in range(1, config.episodes + 1):
            for batch_idx, experience in enumerate(dataloader):
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
                "Average10": np.mean(average10),
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

if __name__ == '__main__':
    train()