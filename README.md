# humanoid_sim_teleop

## Installation

```bash
# Create environemnt
conda create -n teleop python=3.8 numpy tqdm 
conda activate dpi

# Install package
pip install -r requirements.txt
```

## Run
```bash
cd scripts
# run teleop script
python3 teleop_sim_mujoco_updated.py --embodiment h1_2_inspire_sim --tasktype pepsi --expid 1061_sim_pepsi_grasp_h1_2_inspire --scene_description "Pick up the pepsi bottle from the right hand and put it in the gray basket"
```