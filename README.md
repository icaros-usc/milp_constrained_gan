# Video Game Level Repair via Mixed Integer Linear Programming

### Setup Instructions
- Install [CPLEX Optimization Studio and it Python API](https://developer.ibm.com/docloud/blog/2019/07/04/cplex-optimization-studio-for-students-and-academics/)
    - You may need academic version to solve large programs. 
- Clone the repo
- git submodule init update
- pip install -r requirements.txt

### Zelda domains
Training a GAN that learns the distribution of *Zelda* levels
- in the root folder of the project
- python launchers/zelda_gan_training.py

Generate fixed levels
- in the root folder of the project
- python launchers/zelda_gan_generate.py --output_folder=\<path to save the generated levels\> --network_path=\<path to the save generator network\>

### Pac-Man domains
Training a GAN that learns the distribution of *Pac-Man* levels
- in the root folder of the project
- python launchers/zelda_gan_training.py

Generate fixed levels
- in the root folder of the project
- python launchers/zelda_gan_generate.py --output_folder=\<path to save the generated levels\> --network_path=\<path to the save generator network\>

