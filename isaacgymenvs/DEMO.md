# Online Agent
```commandline
export LD_LIBRARY_PATH=/home/user_119/anaconda3/envs/rlgpu/lib
cd /home/user_119/Project/IsaacGymEnvs-project1/isaacgymenvs
conda activate rlgpu
python train.py  task=FrankaPushCube test=True num_envs=4 task.env.draw_targets=True checkpoint=demo/online_agent.pth
```

# Offline agent
```commandline
export LD_LIBRARY_PATH=/home/user_119/anaconda3/envs/rlgpu/lib
cd /home/user_119/Project/IsaacGymEnvs-project1/isaacgymenvs
conda activate rlgpu
python train.py task=FrankaPushCube test=True num_envs=4 task.env.draw_targets=True task.offline.test=True task.offline.network_path=demo/offline_agent.pth
```