# deeplexicon

Signal based nanopore RNA demultiplexing with convolutional neural networks

# Installation

### For Ubuntu 16.04
#### add python 3.6 repo (not on default 16.04 ppa repos)

    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt-get update
    sudo apt-get install python3.6 python3.6-venv


### Linux with python3.6
##### (other python3 versions not yet tested)
#### Create environtment

    python3.6 -m venv ./Deeplexicon/


#### clone git repository

    git clone https://github.com/Psy-Fer/deeplexicon.git

#### source and install requirements
    source Deeplexicon/bin/activate
    pip3 install Keras Tensorflow Pandas PyTs Scikit-learn numba==0.45.0

## Run

    python3 cmd_line_deeplexicon_caller.py -p ~/top/fast5/path/ -t multi -m /model/path/pAmps-final-actrun_newdata_nanopore_UResNet20v2_model.030.h5 > output.tsv
