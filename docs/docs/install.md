# How to install DeePlexiCon

**DeePlexiCon** is built to work in python 3.7 and is quite sensitive to python versions, as well as library versions such as `PyTs`.

Linux is also prefered, and no support for MacOS or Windows will be provided.
All options excepty `train` should work on any OS if set up correctly, however we strongly advise using `train` on Ubuntu.


## Getting python3.7 and setting up environments

Many systems won't have python3.7, so here is how to get it, and create environments with it. (on Ubuntu <=16.04, 3.7 isn't in the default ppa repos)

    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt-get update
    sudo apt install python3.7 python3.7-dev python3.7-venv

#### Create environtment

    python3.7 -m venv ./Deeplexicon/

#### clone git repository

    git clone https://github.com/Psy-Fer/deeplexicon.git

#### source and install requirements (CPU)
Keep in mind, these versions are crucial to expected operation.

    source Deeplexicon/bin/activate
    pip install h5py Keras==2.2.4 pandas PyTs==0.8.0 Scikit-learn numba==0.45.0 TensorFlow==1.13.1


#### Done!

That's it. To simplify running things, you can add `deeplexicon.py` to your $PATH
Just add the following to your ~/.bashrc or run in your current shell

    export PATH="/path/to/deeplexicon:$PATH"

<!-- TODO: add deeplexicon to pip and make executable -->
<!-- TODO: aff tesnorflow-gpu instruction -->
