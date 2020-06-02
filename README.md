# Video Game Level Repair via Mixed Integer Linear Programming

Setup Instructions
- Clone the repo
- git submodule init update
- pip install -r requirements.txt

Training a GAN that learns the distribution of *Zelda* levels
- in the root folder of the project
- python launchers/zelda_gan_training.py

Generate fixed levels
- in the root folder of the project
- python launchers/zelda_gan_generate.py --output_folder=\<path to save the generated levels\> --network_path=\<path to the save generator network\>

