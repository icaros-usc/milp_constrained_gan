# Video Game Level Repair via Mixed Integer Linear Programming

### Setup Instructions
- Install [CPLEX Optimization Studio and it Python API](https://developer.ibm.com/docloud/blog/2019/07/04/cplex-optimization-studio-for-students-and-academics/)
    - You need full version to solve large programs. (Researchers can use academic version).
- Unzip the file in a folder
- pip install -r requirements.txt
- Add the folder path to your PYTHONPATH

### Zelda domain
Training a GAN that learns the distribution of *Zelda* levels
- in the root folder of the project
- python launchers/zelda_gan_training.py --gan_experiment=\<path to save the samples and models\> --lvl_data=\<path to human authored levels>\
- If you have GPUs, you can use --cuda to enable GPUs.

Generate fixed levels
- in the root folder of the project
- python launchers/zelda_gan_generate.py --output_folder=\<path to save the generated levels\> --network_path=\<path to the save generator network\>
    - We have a pretrained model in default_samples
    
Visualize the generated levels
- in the root folder of the project
- python launchers/zelda_grid_visualize.py --lvl_path=\<path to the generated levels\> --output_folder=\<path to save the visualized levels\>


### PacMan domain
Training a GAN that learns the distribution of *PacMan* levels
- in the root folder of the project
- python launchers/pacman_gan_training.py --experiment=\<path to save the samples and models\>
- If you have GPUs, you can use --cuda to enable GPUs.

Generate fixed levels
- in the root folder of the project
- python launchers/pacman_gan_generate.py
    
Visualize the generated levels
- in the root folder of the project
- python launchers/pacman_grid_visualize.py


### End-to-end training
**Here the code only works with pytorch==1.2.0**
- in the root folder of the project
- python launchers/zelda_gan_partial_lp_end2end_generate.py --lvl_data=\<path to human authored levels>\ --gan_experiment=\<path to save the samples and models>\ --mipaal_experiment=\<path to save the mipaal samples and models>\
