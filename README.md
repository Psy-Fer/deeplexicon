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

#### source and install requirements CPU
    source Deeplexicon/bin/activate
    pip3 install Keras Tensorflow Pandas PyTs Scikit-learn numba==0.45.0

#### Source and install requirements GPU

    source Deeplexicon/bin/activate
    pip3 install Keras tensorflow-gpu Pandas PyTs Scikit-learn numba==0.45.0

## Run

    python3 cmd_line_deeplexicon_caller.py -p ~/top/fast5/path/ -t multi -m /model/path/pAmps-final-actrun_newdata_nanopore_UResNet20v2_model.030.h5 > output.tsv


## Publication extra info

Full library versions used in publications for CPU calling

    absl-py==0.7.1
    astor==0.8.0
    cycler==0.10.0
    gast==0.2.2
    google-pasta==0.1.7
    grpcio==1.22.0
    h5py==2.9.0
    joblib==0.13.2
    Keras==2.2.4
    Keras-Applications==1.0.8
    Keras-Preprocessing==1.1.0
    kiwisolver==1.1.0
    llvmlite==0.29.0
    Markdown==3.1.1
    matplotlib==3.1.1
    numba==0.45.0
    numpy==1.17.0
    pandas==0.25.0
    protobuf==3.9.1
    pyparsing==2.4.2
    python-dateutil==2.8.0
    pyts==0.8.0
    pytz==2019.2
    PyYAML==5.1.2
    scikit-learn==0.21.3
    scipy==1.3.1
    six==1.12.0
    tensorboard==1.14.0
    tensorflow==1.14.0
    tensorflow-estimator==1.14.0
    termcolor==1.1.0
    Werkzeug==0.15.5
    wrapt==1.11.2
