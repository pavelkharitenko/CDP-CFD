# Installation of Aerocae Robotics

Author: Jiwei Wang (王际维)

## Part I. Simulator (for Ubuntu 22.04)
Use the following commands to install Aerocae Robotics simulation software and install your product key:
```
sudo dpkg -i aerocae_robotics-1.6.3-ubuntu2204-amd64.deb

sh install-key.sh
```

After installation, you should be able to find "Aerocae Robotics" in your application list, or you can run it from command line: `aerocae-robotics`

#### Important Notes
1. The `install-key.sh` file contains the key to activate the software that is **exclusive for you**. Please be careful not to redistribute it.
2. Please make sure you are using an NVIDIA GPU and the **driver up to date** (supports CUDA >= 12.4). 

#### Quickstart Guide
- You need to load a **config file (in YAML format)** before starting simulation.
- Once a config is loaded, you can navigate in the world. **Holding right mouse button** anywhere in the scene view, use WASDQE to move and move your mouse to look around.
- You need to select the `Mode` on top of the window to `MessageControlled` in order to control the simulation from *Python* (see part III).
- Press `Run` to start the simulation.


## Part II. World Config
An example world config `config.yml` is given inside `example-world` folder, along with files it needs. You can load it in Aerocae Robotics.

#### Editing config file
Open `config.yml` in any text editor, you can edit the world. However, you may not want to do this at the beginning.

You can modify objects, fluid simulation settings, actuators and sensors in the config file.


## Part III. Python Control Script
A minimum example for controlling the simulation from Python is given in `control-example.py`. You can read the codes and comments in it to learn how it works, and modify it to insert your own control policy later.

Before running the script, please make sure you have installed **python >= 3.9 and pip**, then run the following commands to **install dependencies**:

```
python3 -m pip install -i https://test.pypi.org/simple/ --upgrade --no-deps simexp-controller

python3 -m pip install numpy
```

After you start running the script, you should be able to see the simulation process with fluid visualizations in the Aerocae Robotics window.

